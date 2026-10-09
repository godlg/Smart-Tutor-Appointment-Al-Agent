"""Offline, disposable proof of the real phase-1 memory reliability pipeline.

Run: .venv/Scripts/python.exe scripts/demo_memory_reliability.py
Only model/embedding factories are fake; repositories, SQLite transactions,
ContextManager, scoped runtime, business agents and chat entry are real.
Checkpoints restore allowlisted requirement drafts, NOT executable operations.
"""
from __future__ import annotations

import os
import sys

# These must precede every application/LangChain import, including in the child.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_TRACING_V2"] = "false"
os.environ["LANGSMITH_OTEL_ENABLED"] = "false"
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.dont_write_bytecode = True
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="strict")

import asyncio
from contextlib import closing, contextmanager
import ipaddress
import json
import logging
from pathlib import Path
import socket
import subprocess
from tempfile import TemporaryDirectory, gettempdir
from types import ModuleType
from typing import Any, Iterator
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
REPLY = "离线咨询建议：初二数学先理解概念，再整理错题；仅供参考。"
SAFE_DRAFT = {
    "subject": "数学", "grade_level": "初中", "preferred_teacher_name": "周敏",
    "preferred_style": "温和", "student_situation": "应用题薄弱",
    "start_time": "2030-10-03 19:00", "duration_minutes": "45",
}
CHECKS = [
    "offline_guards", "scope_isolation", "durable_rollups", "archive_failure",
    "hard_budget", "real_consultation", "fresh_process_recovery",
]


def require(condition: bool, message: str) -> None:
    """Unlike assert, demo acceptance remains active under python -O."""
    if not condition:
        raise AssertionError(message)


def local_host(host: Any) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="strict")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(str(host).split("%", 1)[0]).is_loopback
    except ValueError:
        return False


@contextmanager
def offline_network() -> Iterator[None]:
    """Block external DNS, TCP and UDP; keep loopback for asyncio socketpairs."""
    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex
    sendto = socket.socket.sendto
    getaddrinfo = socket.getaddrinfo
    gethostbyname = socket.gethostbyname
    gethostbyname_ex = socket.gethostbyname_ex
    gethostbyaddr = socket.gethostbyaddr
    getnameinfo = socket.getnameinfo

    def check_address(sock: socket.socket, address: Any) -> None:
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            if not isinstance(address, tuple) or not local_host(address[0]):
                raise PermissionError("offline demo blocks external sockets")

    def guarded_connect(sock: socket.socket, address: Any) -> Any:
        check_address(sock, address)
        return connect(sock, address)

    def guarded_connect_ex(sock: socket.socket, address: Any) -> Any:
        check_address(sock, address)
        return connect_ex(sock, address)

    def guarded_sendto(sock: socket.socket, data: bytes, *args: Any) -> Any:
        check_address(sock, args[-1])
        return sendto(sock, data, *args)

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if not local_host(host):
            raise PermissionError("offline demo blocks external DNS")
        return getaddrinfo(host, *args, **kwargs)

    def local_resolver(resolve: Any) -> Any:
        def guarded(host: Any, *args: Any) -> Any:
            if not local_host(host):
                raise PermissionError("offline demo blocks external DNS")
            return resolve(host, *args)
        return guarded

    def guarded_getnameinfo(address: Any, flags: int) -> Any:
        if not isinstance(address, tuple) or not local_host(address[0]):
            raise PermissionError("offline demo blocks external reverse DNS")
        return getnameinfo(address, flags | socket.NI_NUMERICHOST | socket.NI_NUMERICSERV)

    with patch.object(socket.socket, "connect", guarded_connect), \
            patch.object(socket.socket, "connect_ex", guarded_connect_ex), \
            patch.object(socket.socket, "sendto", guarded_sendto), \
            patch.object(socket, "getaddrinfo", guarded_getaddrinfo), \
            patch.object(socket, "gethostbyname", local_resolver(gethostbyname)), \
            patch.object(socket, "gethostbyname_ex", local_resolver(gethostbyname_ex)), \
            patch.object(socket, "gethostbyaddr", local_resolver(gethostbyaddr)), \
            patch.object(socket, "getnameinfo", guarded_getnameinfo):
        yield


def verify_network_guard() -> None:
    for call in (
        lambda: socket.getaddrinfo("offline.invalid", 443),
        lambda: socket.gethostbyname("offline.invalid"),
        lambda: socket.gethostbyname_ex("offline.invalid"),
        lambda: socket.gethostbyaddr("192.0.2.1"),
        lambda: socket.getnameinfo(("192.0.2.1", 443), 0),
    ):
        try:
            call()
        except PermissionError:
            pass
        else:
            raise AssertionError("external networking was not blocked")
    with socket.socket() as blocked:
        for connect in (blocked.connect, blocked.connect_ex):
            try:
                connect(("192.0.2.1", 443))
            except PermissionError:
                pass
            else:
                raise AssertionError("external connection was not blocked")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as blocked:
        try:
            blocked.sendto(b"offline", ("192.0.2.1", 443))
        except PermissionError:
            pass
        else:
            raise AssertionError("external datagram was not blocked")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        with socket.create_connection(listener.getsockname(), timeout=2):
            pass
    print("PASS offline_guards: dotenv disabled; external DNS/socket blocked; loopback allowed")


def install_fake_factories() -> dict[str, list]:
    """Replace provider module BEFORE import, never loading providers/auth/secrets."""
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    evidence: dict[str, list] = {"prompts": [], "embeddings": []}

    class CapturePrompts(BaseCallbackHandler):
        def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
            evidence["prompts"].append("\n".join(msg.content for msg in messages[0]))

    class OfflineEmbedding:
        def embed_query(self, text: str) -> list[float]:
            evidence["embeddings"].append(text)
            return [1.0, 0.0, 0.0]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_query(text) for text in texts]

    def create_chat_model(**kwargs: Any) -> Any:
        responses = ["YES", REPLY] if kwargs.get("temperature") == 0.3 else ["consultation"]
        return FakeListChatModel(responses=responses, callbacks=[CapturePrompts()])

    provider = ModuleType("config.model_provider")
    provider.create_chat_model = create_chat_model
    provider.create_embedding_model = lambda **kwargs: OfflineEmbedding()
    sys.modules[provider.__name__] = provider
    return evidence


@contextmanager
def disposable_database(root: Path) -> Iterator[dict[str, Any]]:
    """Inject every default business DB; reject even direct sqlite escape attempts."""
    import sqlite3
    db_file = (root / "memory.sqlite3").resolve()
    require(db_file.parent == root.resolve(), "database symlink escaped temporary directory")
    db_url = f"sqlite:///{db_file.as_posix()}"
    os.environ["DATABASE_URL"] = db_url
    os.environ["RETRIEVAL_MODE"] = "dense"
    os.environ["RERANK_ENABLED"] = "false"
    # Reproducible budgets, regardless of the caller's environment.
    for name, value in {"CONTEXT_MAX_TOKENS": "8192", "CONTEXT_OUTPUT_TOKENS": "1024",
                        "CONTEXT_KEEP_RECENT_ROUNDS": "4", "CONTEXT_ROLLUP_ROUNDS": "4",
                        "CONTEXT_SUMMARY_MAX_CHARS": "600"}.items():
        os.environ[name] = value

    def authorize_sql(action: int, *args: Any) -> int:
        if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def sqlite_guard(event_name: str, args: tuple) -> None:
        if event_name == "sqlite3.connect":
            path = Path(str(args[0])).resolve()
            require(path == db_file, "SQLite attempted to leave disposable database")

    sys.addaudithook(sqlite_guard)
    connect_sqlite = sqlite3.connect

    def guarded_sqlite_connect(*args: Any, **kwargs: Any) -> Any:
        connection = connect_sqlite(*args, **kwargs)
        try:
            # The CPython connect/handle audit event precedes full initialization;
            # authorizers must instead be attached after the factory returns.
            connection.set_authorizer(authorize_sql)
        except BaseException:
            connection.close()
            raise
        return connection

    # Redundant with PYTHON_DOTENV_DISABLED for old python-dotenv versions.
    import dotenv
    with patch.object(dotenv, "load_dotenv", lambda *args, **kwargs: False), \
            patch.object(sqlite3, "connect", guarded_sqlite_connect), \
            patch.object(sqlite3.dbapi2, "connect", guarded_sqlite_connect):
        # Direct sqlite, not a repository mock. Even an attached in-memory DB is
        # rejected, without ever attempting to open an external filename.
        with closing(sqlite3.connect(str(db_file))) as connection:
            try:
                connection.execute("ATTACH DATABASE ':memory:' AS forbidden")
            except sqlite3.DatabaseError:
                pass
            else:
                raise AssertionError("SQLite ATTACH escaped the connection guard")
        print("PASS SQLite containment: direct connections restricted; ATTACH denied; symlink escape rejected")
        evidence = install_fake_factories()
        from db.base.session_manager import SessionManager
        from sqlalchemy import event
        from sqlalchemy.engine import make_url

        original = SessionManager.__init__
        managers = []
        operations = []

        def no_business_execution(conn: Any, cursor: Any, statement: str,
                                  parameters: Any, context: Any, many: bool) -> None:
            import re
            if re.match(r"\s*(?:INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+[\"`\[]?"
                        r"(?:lessons|lesson_series|course_packages)\b", statement, re.I):
                operations.append(statement.split()[0])
                raise AssertionError("demo must never execute booking/payment/change operations")

        def isolated_init(manager: Any, db_path: str = "sqlite:///data/smart_appointment.db") -> None:
            url = make_url(db_path)
            require(url.get_backend_name() == "sqlite", "non-SQLite business database rejected")
            requested = Path(url.database or "").resolve()
            production = (PROJECT_ROOT / "data").resolve()
            if requested.is_relative_to(production):
                db_path = db_url
            else:
                require(requested == db_file, "non-disposable business database rejected")
            try:
                original(manager, db_path)
            except BaseException:
                # create_all can fail after opening a pooled Windows SQLite handle.
                registry = getattr(manager, "Session", None)
                if registry is not None:
                    registry.remove()
                engine = getattr(manager, "engine", None)
                if engine is not None:
                    engine.dispose()
                raise
            managers.append(manager)
            event.listen(manager.engine, "before_cursor_execute", no_business_execution)

        with patch.object(SessionManager, "__init__", isolated_init):
            try:
                yield {"url": db_url, "evidence": evidence, "operations": operations}
            finally:
                for manager in reversed(managers):
                    manager.close()


def runtime_for(service: Any, identity: Any, include_memory: bool = False) -> Any:
    from agents.context.runtime import build_runtime_context
    history = service.restore_history(identity)
    return build_runtime_context(identity, history, service, include_memory=include_memory)


def scope_isolation(service: Any) -> Any:
    from config.identity import Identity
    from services.identity_service import IdentityError, IdentityService

    identities = IdentityService(db_router=service.db_router)
    owner = identities.issue_new_identity()
    other = identities.issue_new_identity()
    child_a = Identity(owner.user_id, student_id="child-a", session_id="rollup-session")
    child_b = Identity(owner.user_id, student_id="child-b", session_id="child-b-session")
    family = owner.with_session("family-session")
    stranger = Identity(other.user_id, student_id="other-child", session_id="other-session")
    seeds = [(child_a, "ONLY_CHILD_A"), (child_b, "ONLY_CHILD_B"),
             (family, "ONLY_FAMILY"), (stranger, "ONLY_OTHER_IDENTITY")]
    for ident, marker in seeds:
        service.record_human(ident, marker)
        service.memory_service.add_memory(
            ident.user_id, "preference", marker, student_id=ident.student_id,
            session_id=ident.session_id, embedding=[1.0, 0.0, 0.0], auto_embed=False,
        )
    # Move the seeded child's isolation exchange into its own session, keeping the
    # rollup fixture itself exactly twenty sequential messages.
    rollup = Identity(owner.user_id, student_id="child-a", session_id="durable-rollup")
    for ident, marker in seeds:
        rows = service.memory_service.get_user_memories(ident.user_id, student_id=ident.student_id)
        require([row["content"] for row in rows] == [marker], "student/user exact filtering failed")
        recalled = service.memory_service.retrieve_top_k(
            "学习方法", ident.user_id, student_id=ident.student_id,
            query_embedding=[1.0, 0.0, 0.0],
        )
        require([row["content"] for row in recalled] == [marker], "vector recall crossed scope")
        prompt = runtime_for(service, ident, include_memory=True).build(task="学习方法")
        require(marker in prompt, "scoped runtime did not read its own memory")
        require(all(others not in prompt for _, others in seeds if others != marker),
                "runtime leaked another scope")
    same_owner_new_session = Identity(owner.user_id, student_id="child-a", session_id="cross-session")
    cross_prompt = runtime_for(service, same_owner_new_session, True).build(task="学习方法")
    require("ONLY_CHILD_A" in cross_prompt, "same identity cross-session memory missing")
    require(service.message_count(same_owner_new_session) == 0, "new session inherited old messages")
    try:
        service.restore_history(Identity(other.user_id, student_id="child-a", session_id=child_a.session_id))
    except IdentityError as exc:
        require(exc.status_code == 403, "foreign session must be denied as 403")
    else:
        raise AssertionError("foreign owner claimed persisted session")
    print("PASS scope_isolation: two issued anonymous identities; cross-session memory; exact student/family scopes; 403")
    print("Boundary: anonymous identity/student namespace is not authenticated parent/student ownership.")
    return rollup


def durable_rollups(service: Any, identity: Any, evidence: dict) -> list[int]:
    for index in range(10):
        service.record_human(identity, f"原文用户{index:02d}：确认初二数学，周敏老师，周三19:00，只能线上")
        service.record_ai(identity, f"原文顾问{index:02d}：已记录约束，待办仍须家长重新确认")
    manager = runtime_for(service, identity)
    manager.rollup_rounds = 2
    manager.rollup_window_ratio = 0.01
    embedding_count = len(evidence["embeddings"])
    watermarks = []
    for expected in (4, 8, 12):
        prompt = manager.build(task="后续学习安排", max_tokens=10000)
        watermarks.append(service.summary_watermark(identity))
        require(watermarks[-1] == expected and manager.last_rollup_triggered, "rollup watermark did not progress")
        require("【近期滚动摘要】" in prompt, "archive was not explicitly injected")
    summary_rows = service.get_summaries(identity)
    require([(s["start_seq"], s["end_seq"]) for s in summary_rows] == [(1, 4), (5, 8), (9, 12)],
            "rollup spans were not unique and contiguous")
    memories = [m for m in service.memory_service.get_user_memories(
        identity.user_id, "summary", student_id=identity.student_id)
                if m["session_id"] == identity.session_id]
    require(len(memories) == 3 and len({m["id"] for m in memories}) == 3, "duplicate long-term archives")
    require(all(m["embedding"] is None for m in memories), "text fallback must not require embedding")
    require(len(evidence["embeddings"]) == embedding_count, "rollup incurred embedding calls")
    for summary in summary_rows:
        require(len(summary["content"]) <= manager.summary_max_chars, "unbounded summary")
        require("extractive" in summary["content"], "summary fallback not identified")
    require(service.save_rollup(identity, 9, 12, summary_rows[-1]["content"]) is False,
            "same span was not idempotent")
    prompt = manager.build(task="后续学习安排", max_tokens=10000)
    require(not manager.last_rollup_triggered and len(service.get_summaries(identity)) == 3,
            "no-new-span build created another archive")
    require("原文用户09" in prompt, "recent original dialogue disappeared")
    require(all(s["content"] in prompt for s in summary_rows), "summary text missing without vector recall")
    print(f"PASS durable_rollups: watermarks={watermarks}; spans=1-4/5-8/9-12; unique text-only memories=3")
    print("Summary is bounded deterministic extractive fallback, not intelligent semantic summarization.")
    return watermarks


def archive_failure(service: Any, identity: Any) -> None:
    from sqlalchemy import event
    failure_identity = identity.with_session("archive-failure")
    for index in range(6):
        service.record_human(failure_identity, f"不可丢失原文{index:02d}：数学时间约束")
        service.record_ai(failure_identity, f"失败测试回答{index:02d}")
    manager = runtime_for(service, failure_identity)
    manager.rollup_rounds = 2
    manager.rollup_window_ratio = 0.01

    def fail_insert(conn: Any, cursor: Any, statement: str,
                    parameters: Any, context: Any, many: bool) -> None:
        if statement.lstrip().lower().startswith("insert into long_term_memories"):
            raise RuntimeError("injected long-term archive failure")

    engine = service.db_router.session_manager.engine
    event.listen(engine, "before_cursor_execute", fail_insert)
    try:
        prompt = manager.build(task="检查原文", max_tokens=10000)
    finally:
        event.remove(engine, "before_cursor_execute", fail_insert)
    require(service.summary_watermark(failure_identity) == 0, "failed archive advanced durable watermark")
    conv = next(context for context in manager.contexts if context.name == "conversation")
    require(conv._rolled_up_count(failure_identity.session_id) == 0, "failed archive hid hot history")
    require(not manager.last_rollup_triggered and not service.get_summaries(failure_identity),
            "partial transaction left summary artifacts")
    require(all(f"不可丢失原文{index:02d}" in prompt for index in range(6)), "failed archive lost original input")
    prompt = manager.build(task="恢复后继续", max_tokens=10000)
    require(service.summary_watermark(failure_identity) == 4, "archive did not recover after failure removal")
    require("不可丢失原文00" in prompt and "【近期滚动摘要】" in prompt, "recovered archive text unreadable")
    service.drop_history(failure_identity.session_id)
    require(len(service.restore_history(failure_identity).messages) == 12, "durable original history lost")
    print("PASS archive_failure: real SQL insert failure; atomic rollback; watermark=0 and originals visible; recovery=4")


def hard_budget() -> None:
    from agents.context import ContextManager, SystemPromptContext
    from agents.context.types import ContextBudgetExceeded

    manager = ContextManager([SystemPromptContext()], token_counter=len, output_tokens=100)
    question = "初二数学怎样复习？"
    contract = '{"teacher_id":77,"must_confirm":true}'
    prompt = manager.build(task=question, max_tokens=1400,
                           extra_contexts={"knowledge": "超长知识" * 4000, "instructions": contract})
    require(len(prompt) <= 1300, "output reserve/final hard budget exceeded")
    require(question in prompt and contract in prompt and "K12" in prompt, "mandatory prompt was truncated")
    try:
        manager.build(task="不能删除的必需问题" * 1000, max_tokens=1400)
    except ContextBudgetExceeded as exc:
        require(exc.used > exc.budget, "rejection was not based on required budget")
    else:
        raise AssertionError("oversized mandatory question was silently cut")
    print(f"PASS hard_budget: len-counter prompt={len(prompt)} <= 1300 (1400-100 output); oversized required question rejected")
    print("Counter is a conservative/injected engineering budget, not a provider billing-token claim.")


async def collect(stream: Any) -> str:
    return "".join([token async for token in stream])


def real_consultation(service: Any, identity: Any, evidence: dict) -> None:
    from api import chat_handler
    chat_handler._conversation_service = service
    before = service.message_count(identity)
    output = asyncio.run(collect(chat_handler.ProcessUserInput_stream(
        "初二数学怎样整理错题？", identity=identity,
    )))
    require(REPLY in output, "real consultation entry did not reach fake model boundary")
    rows = service.conversation_repo.get_messages(identity.session_id)
    require(len(rows) == before + 2, "normal entry did not persist exactly one exchange")
    require([row["role"] for row in rows[-2:]] == ["human", "ai"], "incorrect unified role writes")
    require("[THOUGHT]" not in rows[-1]["content"] and REPLY in rows[-1]["content"],
            "visible AI persistence leaked thoughts or lost reply")
    require(rows[-1]["status"] == "complete", "normal reply was marked partial")
    agent = chat_handler.get_task_agent(identity=identity)
    manager = agent.consultant_agent.context_manager
    require(manager.identity == identity and manager.conversation_service is service and manager.rollup_enabled,
            "normal consultant not wired to real scoped runtime")
    prompt = manager._last_build
    require("【近期滚动摘要】" in prompt and "原文用户09" in prompt and "ONLY_CHILD_A" in prompt,
            "real consultant prompt missed scoped summary/recent history/memory")
    require(any(prompt == text for text in evidence["prompts"]), "runtime prompt never reached actual model call")
    require("ONLY_CHILD_B" not in prompt and "ONLY_OTHER_IDENTITY" not in prompt, "consultation prompt leaked scope")
    behavior = service.db_router.user_behavior_repo.get_user_behaviors(identity.memory_scope(), "consultation")
    require(len(behavior) == 1 and behavior[0]["session_id"] == identity.session_id,
            "real scoped consultation behavior missing")
    require(agent.consultant_agent.knowledge_retriever.knowledge_service.db_router.session_manager.engine.url.database
            == service.db_router.session_manager.engine.url.database, "knowledge escaped temporary database")
    print("PASS real_consultation: ProcessUserInput_stream -> real classifier/consultant/RAG/runtime; persisted human+visible ai+scoped behavior")
    # Capture real business agent state, but do NOT run the seeded old operation.
    from agents.task_classification.state_manager import FlowState
    from services.session_checkpoint import capture_checkpoint
    agent.state.value = FlowState.TRIAL_CONFIRM
    agent.trial_agent.trial_history = {
        **agent.trial_agent.trial_history, **SAFE_DRAFT,
        "teacher_id": 7, "confirmed": True, "awaiting_confirmation": True,
        "pending_trial": {"teacher_id": 7, "start_time": SAFE_DRAFT["start_time"]},
    }
    payload = capture_checkpoint(agent)
    require(payload["drafts"]["trial"] == SAFE_DRAFT, "checkpoint not exact allowlist")
    require(payload["requires_reconfirmation"], "unsafe old operation was not invalidated")
    service.save_checkpoint(identity, payload["source_flow"], payload,
                            pending_ops=[{"op": "book_trial", "teacher_id": 7, "confirmed": True}])
    print("Checkpoint: allowlisted draft persisted; intentionally stale pending operation must be discarded in fresh process.")


def child_recovery(root: Path, user: str, student: str, session: str) -> None:
    # Factories/DB guards must be installed BEFORE even importing the real entry.
    with disposable_database(root) as sandbox:
        from config.identity import Identity
        from services.conversation_service import ConversationService
        from services.session_checkpoint import restore_checkpoint
        from config.constants import StateEnum
        from api import chat_handler

        identity = Identity(user, student_id=student, session_id=session)
        service = ConversationService(db_path=sandbox["url"])
        chat_handler._conversation_service = service
        expected_count = 22
        history = service.restore_history(identity)
        require(len(history.messages) == expected_count, "fresh process did not recover ordered conversation")
        snapshot = [(message.type, message.content, message.additional_kwargs) for message in history.messages]
        for _ in range(2):
            restored = service.restore_history(identity)
            require(restored is history and [(m.type, m.content, m.additional_kwargs) for m in restored.messages] == snapshot,
                    "repeat restore duplicated or replaced history")
        require(service.message_count(identity) == expected_count, "restore wrote duplicate DB messages")
        require(service.summary_watermark(identity) == 12, "fresh process lost summary watermark")
        manager = runtime_for(service, identity)
        prompt = manager.build(task="重新确认前查看草稿")
        require("【近期滚动摘要】" in prompt and "原文用户09" in prompt, "fresh runtime could not read summary/history")
        require(len(service.get_summaries(identity)) == 3, "restore duplicated summaries")
        agent = chat_handler.get_task_agent(identity=identity)
        row = service.load_checkpoint(identity)
        require(row is not None, "fresh process lost checkpoint")
        for _ in range(2):
            result = restore_checkpoint(agent, row)
            require(result["drafts"]["trial"] == SAFE_DRAFT and result["requires_reconfirmation"],
                    "safe checkpoint draft recovery failed")
            require("重新确认" in result["notice"], "missing re-confirmation notice")
            require(agent.state.value == StateEnum.CLASSIFY, "stale execution flow survived")
            require(agent.trial_agent.trial_history["start_time"] is None, "time was restored as executable state")
            require(not agent.trial_agent.trial_history.get("pending_trial")
                    and not agent.trial_agent.trial_history.get("teacher_id")
                    and not agent.trial_agent.trial_history.get("confirmed"), "unsafe operation survived")
            require(agent.trial_agent.trial_history["subject"] == "数学", "safe requirement lost")
        require(service.message_count(identity) == expected_count and len(history.messages) == expected_count,
                "checkpoint restore mutated shared conversation")
        calls_before = len(sandbox["evidence"]["prompts"])
        for _ in range(2):
            output = asyncio.run(collect(chat_handler.ProcessUserInput_stream("确认", identity=identity)))
            require("重新确认" in output and "不会自动执行" in output,
                    "old bare confirmation did not request explicit re-confirmation")
        require(len(sandbox["evidence"]["prompts"]) == calls_before, "old confirmation called a model/business route")
        require(service.message_count(identity) == expected_count + 4, "entry did not preserve both repeated confirmation exchanges")
        service.restore_history(identity)
        service.restore_history(identity)
        require(len(history.messages) == expected_count + 4, "post-entry repeat restore duplicated history")
        require(not sandbox["operations"], "old business operations were executed")
        print("CHILD_RESULT " + json.dumps({"pid": os.getpid(), "restored_messages": expected_count,
                                           "watermark": 12, "repeat_restore_no_duplicates": True,
                                           "safe_draft": SAFE_DRAFT, "no_old_execution": True}, ensure_ascii=False))


def fresh_process(root: Path, identity: Any) -> int:
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--restore", str(root),
         identity.user_id, identity.student_id, identity.session_id],
        cwd=PROJECT_ROOT, env={**os.environ, "PYTHON_DOTENV_DISABLED": "1"},
        capture_output=True, text=True, encoding="utf-8", timeout=90,
    )
    require(process.returncode == 0, "fresh child failed:\n" + process.stdout + process.stderr)
    lines = [line.removeprefix("CHILD_RESULT ") for line in process.stdout.splitlines()
             if line.startswith("CHILD_RESULT ")]
    require(len(lines) == 1, "missing child recovery evidence")
    report = json.loads(lines[0])
    require(report["pid"] != os.getpid() and report["watermark"] == 12
            and report["safe_draft"] == SAFE_DRAFT and report["no_old_execution"], "invalid child evidence")
    print(f"PASS fresh_process_recovery: sys.executable child pid={report['pid']}; history=22; watermark=12; duplicate restore=0; old executions=0")
    print("Checkpoint policy: 白名单需求草稿恢复；待执行预约/支付/调课安全失效，重新询问老师和时间并重新确认；不是完整操作恢复。")
    return report["pid"]


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    with offline_network():
        if len(sys.argv) > 1:
            require(len(sys.argv) == 6 and sys.argv[1] == "--restore", "unsupported demo arguments")
            root = Path(sys.argv[2]).resolve()
            require(root.parent == Path(gettempdir()).resolve() and root.name.startswith("memory-reliability-")
                    and (root / "memory.sqlite3").is_file(), "child must use the parent's disposable database")
            child_recovery(root, *sys.argv[3:])
            return
        verify_network_guard()
        with TemporaryDirectory(prefix="memory-reliability-") as temporary:
            root = Path(temporary).resolve()
            with disposable_database(root) as sandbox:
                from services.conversation_service import ConversationService
                service = ConversationService(db_path=sandbox["url"])
                identity = scope_isolation(service)
                watermarks = durable_rollups(service, identity, sandbox["evidence"])
                archive_failure(service, identity)
                hard_budget()
                real_consultation(service, identity, sandbox["evidence"])
                require(not sandbox["operations"], "parent executed a business operation")
                service.close()
            # All parent connections are disposed before launching the new process.
            child_pid = fresh_process(root, identity)
        require(not root.exists(), "temporary database cleanup failed")
        print("DEMO_RESULT " + json.dumps({
            "checks": CHECKS, "watermarks": watermarks, "parent_pid": os.getpid(),
            "child_pid": child_pid, "temporary_directory": str(root),
            "checkpoint_policy": "allowlisted_draft_safe_invalidation",
        }, ensure_ascii=False))
        print("PASS all checks; TemporaryDirectory removed; offline demo completed (exit 0)")


if __name__ == "__main__":
    main()
