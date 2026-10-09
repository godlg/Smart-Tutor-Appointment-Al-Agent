"""Real SQLite storage regressions; every database is temporary or in-memory."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import sys
import types

import pytest
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError

from config.identity import Identity
from db.models import (
    ChatSession, LongTermMemory, SessionCheckpoint, SessionMessage, SessionSummary,
)
from services.conversation_service import ConversationService
from services.identity_service import IdentityError, IdentityService
from services.long_term_memory_service import LongTermMemoryService


@pytest.fixture
def storage(tmp_path):
    service = ConversationService(db_path=f"sqlite:///{tmp_path / 'storage.db'}")
    yield service
    service.close()


def ident(user="owner", student=None, session="session"):
    return Identity(user_id=user, student_id=student, session_id=session)


OPERATIONS = [
    lambda s, i: s.restore_history(i),
    lambda s, i: s.message_count(i),
    lambda s, i: s.summary_watermark(i),
    lambda s, i: s.get_summaries(i),
    lambda s, i: s.recent_summary_text(i),
    lambda s, i: s.load_checkpoint(i),
    lambda s, i: s.record_human(i, "intrusion"),
    lambda s, i: s.record_ai(i, "intrusion"),
    lambda s, i: s.save_rollup(i, 1, 1, "intrusion"),
    lambda s, i: s.save_checkpoint(i, "TRIAL", {"secret": "changed"}),
]


class TestOwnership:
    @pytest.mark.parametrize("operation", OPERATIONS)
    def test_every_identity_operation_checks_persisted_owner(self, storage, operation):
        storage.record_human(ident(), "private")
        with pytest.raises(IdentityError) as error:
            operation(storage, ident(user="other"))
        assert error.value.status_code == 403
        assert storage.conversation_repo.get_messages("session")[0]["content"] == "private"

    @pytest.mark.parametrize("operation", OPERATIONS)
    def test_same_owner_cannot_change_student_scope(self, storage, operation):
        storage.record_human(ident(student="child-a"), "private")
        with pytest.raises(IdentityError) as error:
            operation(storage, ident(student="child-b"))
        assert error.value.status_code == 409

    def test_family_scope_cannot_be_promoted_to_student(self, storage):
        storage.record_human(ident(), "family")
        with pytest.raises(IdentityError) as error:
            storage.restore_history(ident(student="child-a"))
        assert error.value.status_code == 409

    def test_ensure_session_returns_bound_identity(self, storage):
        requested = ident(student="child-a")
        assert storage.ensure_session(requested) == requested
        owner = storage.db_router.identity_repo.get_session("session")
        assert owner["user_id"] == "owner"
        assert owner["student_id"] == "child-a"

    @pytest.mark.parametrize("model,values", [
        (SessionMessage, {"seq": 1, "role": "human", "content": "legacy"}),
        (SessionSummary, {"start_seq": 1, "end_seq": 2, "content": "legacy"}),
        (SessionCheckpoint, {"state": "TRIAL"}),
        (LongTermMemory, {"memory_type": "summary", "content": "legacy"}),
    ])
    def test_orphan_artifacts_cannot_be_claimed(self, storage, model, values):
        with storage.db_router.session_manager.session_scope() as session:
            session.add(model(session_id="orphan", user_id="legacy", **values))
        # Even the artifact's asserted user does not establish ownership.
        for user in ("new-owner", "legacy"):
            with pytest.raises(IdentityError) as error:
                storage.restore_history(ident(user=user, session="orphan"))
            assert error.value.status_code == 403
        assert storage.db_router.identity_repo.get_session_owner("orphan") is None

    def test_repository_conflict_rechecks_owner_and_student(self, storage):
        repo = storage.db_router.identity_repo
        repo.create_session("session", "owner")
        with pytest.raises(ValueError) as error:
            repo.create_session("session", "other")
        assert error.value.status_code == 403
        with pytest.raises(ValueError) as error:
            repo.create_session("session", "owner", student_id="child-a")
        assert error.value.status_code == 409

    def test_identity_service_ignores_stale_preflight_owner(self, tmp_path, monkeypatch):
        service = IdentityService(db_path=f"sqlite:///{tmp_path / 'id.db'}")
        try:
            service.create_session(ident(), "session")
            monkeypatch.setattr(service.repo, "get_session_owner", lambda sid: None)
            with pytest.raises(IdentityError) as error:
                service.assert_session_owned(ident(user="other"), "session")
            assert error.value.status_code == 403
        finally:
            service.close()

    def test_checkpoint_upsert_never_rebinds_owner(self, storage):
        repo = storage.conversation_repo
        repo.save_checkpoint("session", "owner", state="TRIAL")
        with pytest.raises(ValueError):
            repo.save_checkpoint("session", "other", state="OTHER")
        assert repo.load_checkpoint("session")["user_id"] == "owner"
        assert repo.load_checkpoint("session")["state"] == "TRIAL"

    def test_session_race_exactly_one_owner_succeeds(self, storage):
        repo = storage.db_router.identity_repo
        def claim(user):
            try:
                repo.create_session("race", user)
                return user
            except ValueError as error:
                assert error.status_code == 403
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(claim, ["alice", "bob"]))
        successful = [r for r in results if r is not None]
        assert successful == [repo.get_session_owner("race")]


class TestRecovery:
    def test_inline_thought_does_not_discard_reply(self):
        from services.conversation_service import strip_thought_output
        assert strip_thought_output("[THOUGHT]secret[REPLY]visible") == "[REPLY]visible"
        assert strip_thought_output("plain reply") == "plain reply"
        assert strip_thought_output("[THOUGHT]secret[ERROR]visible error") == "[ERROR]visible error"

    def test_restore_preserves_provider_and_messages_objects(self, storage):
        storage.record_human(ident(), "old")
        provider = storage.get_history("session")
        messages = provider.messages
        restored = storage.restore_history(ident())
        assert restored is provider
        assert restored.messages is messages
        storage.restore_history(ident())
        assert [m.content for m in messages] == ["old"]

    def test_first_record_after_restart_restores_previous_segment(self, tmp_path):
        url = f"sqlite:///{tmp_path / 'restart.db'}"
        first = ConversationService(db_path=url)
        first.record_human(ident(), "old human")
        first.record_ai(ident(), "old partial", partial=True)
        first.close()
        second = ConversationService(db_path=url)
        try:
            provider = second.get_history("session")
            second.record_human(ident(), "new human")
            assert [m.content for m in provider.messages] == [
                "old human", "old partial", "new human",
            ]
            assert provider.messages[1].additional_kwargs["status"] == "partial"
            second.restore_history(ident())
            assert len(provider.messages) == 3
            assert provider.messages[1].additional_kwargs["status"] == "partial"
        finally:
            second.close()


class TestStudentMemory:
    def test_true_user_and_student_scope_exactly_filtered(self, storage):
        memory = LongTermMemoryService(db_path="sqlite:///:memory:")
        try:
            for user, student, content in [
                ("owner", None, "family"), ("owner", "a", "child a"),
                ("owner", "b", "child b"), ("other", "a", "other family"),
            ]:
                memory.add_memory(user, "preference", content, embedding=[1, 0],
                                  auto_embed=False, student_id=student)
            for student, expected in [(None, "family"), ("a", "child a"), ("b", "child b")]:
                rows = memory.get_user_memories("owner", student_id=student)
                assert [r["content"] for r in rows] == [expected]
                assert rows[0]["user_id"] == "owner"
                assert rows[0]["student_id"] == student
                found = memory.retrieve_top_k("math", "owner", query_embedding=[1, 0],
                                              student_id=student)
                assert [r["content"] for r in found] == [expected]
        finally:
            memory.db_router.close()

    def test_legacy_encoded_scope_still_supported(self, storage):
        memory = LongTermMemoryService(db_path="sqlite:///:memory:")
        try:
            scope = ident(student="a").memory_scope()
            memory.add_memory(scope, "preference", "legacy", auto_embed=False)
            assert [m["content"] for m in memory.get_user_memories(scope)] == ["legacy"]
            assert memory.get_user_memories("owner") == []
            assert memory.get_user_memories("owner", student_id="a") == []
        finally:
            memory.db_router.close()


class TestRollup:
    def test_rollup_shared_router_atomic_archive_and_idempotent_span(self, storage, monkeypatch):
        no_embedding = types.SimpleNamespace(embed_input=lambda text: pytest.fail("embedding called"))
        monkeypatch.setitem(sys.modules, "services.text_embedding", no_embedding)
        storage.record_human(ident(student="a"), "math constraints")
        assert storage.memory_service.db_router is storage.db_router
        calls = []
        original = storage.memory_service.save_rollup_summary
        def spy(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)
        monkeypatch.setattr(storage.memory_service, "save_rollup_summary", spy)
        assert storage.save_rollup(ident(student="a"), 1, 1, "math summary") is True
        assert storage.save_rollup(ident(student="a"), 1, 1, "math summary") is False
        assert calls[0]["auto_embed"] is False
        memories = storage.memory_service.get_user_memories("owner", "summary", student_id="a")
        assert len(memories) == 1
        assert memories[0]["content"] == "math summary"
        assert memories[0]["embedding"] is None
        assert storage.summary_watermark(ident(student="a")) == 1

    def test_memory_insert_failure_rolls_back_summary_and_watermark(self, storage):
        storage.record_human(ident(), "must remain")
        def fail_memory(conn, cursor, statement, parameters, context, executemany):
            if statement.lower().startswith("insert into long_term_memories"):
                raise RuntimeError("injected archive failure")
        event.listen(storage.db_router.session_manager.engine, "before_cursor_execute", fail_memory)
        try:
            with pytest.raises(RuntimeError, match="injected archive failure"):
                storage.save_rollup(ident(), 1, 1, "summary")
        finally:
            event.remove(storage.db_router.session_manager.engine, "before_cursor_execute", fail_memory)
        assert storage.summary_watermark(ident()) == 0
        assert storage.get_summaries(ident()) == []
        assert storage.memory_service.get_user_memories("owner") == []
        assert [m.content for m in storage.restore_history(ident()).messages] == ["must remain"]
        assert storage.save_rollup(ident(), 1, 1, "summary") is True

    def test_three_rollups_watermark_survives_restart(self, tmp_path):
        url = f"sqlite:///{tmp_path / 'rollup.db'}"
        first = ConversationService(db_path=url)
        for number in range(1, 7):
            first.record_human(ident(), str(number))
        for start, end in [(1, 2), (3, 4), (5, 6)]:
            assert first.save_rollup(ident(), start, end, f"span {start}-{end}")
            assert first.summary_watermark(ident()) == end
        first.close()
        second = ConversationService(db_path=url)
        try:
            assert second.summary_watermark(ident()) == 6
            assert second.save_rollup(ident(), 5, 6, "span 5-6") is False
            assert len(second.memory_service.get_user_memories("owner", "summary")) == 3
        finally:
            second.close()


class TestSequence:
    def test_same_explicit_sequence_different_message_is_rejected(self, storage):
        repo = storage.conversation_repo
        assert repo.append_message("s", "owner", "human", "first", seq=1) == 1
        assert repo.append_message("s", "owner", "human", "first", seq=1) == 1
        for changed in [
            {"content": "second"}, {"user_id": "other"},
            {"role": "ai"}, {"status": "partial"},
        ]:
            kwargs = dict(session_id="s", user_id="owner", role="human", content="first", seq=1)
            with pytest.raises(ValueError):
                repo.append_message(**{**kwargs, **changed})
        assert repo.count_messages("s") == 1

    def test_sequence_allocated_atomically_across_instances(self, tmp_path):
        url = f"sqlite:///{tmp_path / 'concurrent.db'}"
        left = ConversationService(db_path=url)
        right = ConversationService(db_path=url)
        def write(pair):
            index, repo = pair
            return repo.append_message("s", "owner", "human", f"message {index}")
        try:
            pairs = [(i, (left if i % 2 else right).conversation_repo) for i in range(40)]
            with ThreadPoolExecutor(max_workers=4) as pool:
                sequence = list(pool.map(write, pairs))
            rows = left.conversation_repo.get_messages("s")
            assert sorted(sequence) == list(range(1, 41))
            assert [r["seq"] for r in rows] == list(range(1, 41))
            assert len({r["content"] for r in rows}) == 40
        finally:
            left.close()
            right.close()


class TestMigration:
    @pytest.mark.parametrize("model,values", [
        (SessionMessage, {"seq": 1, "role": "human", "content": "same"}),
        (SessionSummary, {"start_seq": 1, "end_seq": 2, "content": "same"}),
    ])
    def test_orm_inmemory_unique_constraints(self, model, values):
        service = ConversationService(db_path="sqlite:///:memory:")
        try:
            with pytest.raises(IntegrityError):
                with service.db_router.session_manager.session_scope() as session:
                    session.add(model(session_id="s", user_id="u", **values))
                    session.add(model(session_id="s", user_id="u", **values))
        finally:
            service.close()

    def test_idempotent_additive_migration_preserves_old_tables_and_records(self, tmp_path):
        from db.base.schema_migrator import run_migration
        path = tmp_path / "old.db"
        with sqlite3.connect(path) as connection:
            connection.executescript("""
                CREATE TABLE technicians (id INTEGER PRIMARY KEY, name TEXT);
                INSERT INTO technicians VALUES (1, 'preserve');
                CREATE TABLE technician_schedules (id INTEGER PRIMARY KEY);
                CREATE TABLE chat_sessions (id INTEGER PRIMARY KEY, session_id TEXT, user_id TEXT);
                INSERT INTO chat_sessions VALUES (1, 'old-session', 'old-user');
                CREATE TABLE session_messages (id INTEGER PRIMARY KEY, session_id TEXT,
                    user_id TEXT, seq INTEGER);
                CREATE TABLE session_summaries (id INTEGER PRIMARY KEY, session_id TEXT,
                    user_id TEXT, start_seq INTEGER, end_seq INTEGER);
                CREATE TABLE long_term_memories (id INTEGER PRIMARY KEY, user_id TEXT,
                    content TEXT);
                INSERT INTO long_term_memories VALUES (1, 'default_user', 'legacy memory');
            """)
        assert run_migration(str(path))
        assert run_migration(str(path))
        with sqlite3.connect(path) as connection:
            for table in ["chat_sessions", "session_messages", "session_summaries", "long_term_memories"]:
                columns = [r[1] for r in connection.execute(f"PRAGMA table_info({table})")]
                assert columns.count("student_id") == 1
            assert connection.execute("SELECT name FROM technicians").fetchone() == ("preserve",)
            assert connection.execute("SELECT user_id, student_id FROM chat_sessions").fetchone() == ("old-user", None)
            assert connection.execute("SELECT content, student_id FROM long_term_memories").fetchone() == ("legacy memory", None)
