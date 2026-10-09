"""Phase 1 记忆可靠性 —— 身份 / 会话 / 学员隔离（Slice 1）

覆盖:
- Identity 匿名身份原语(memory_scope 家庭级 / 学员级分开、legacy 不冒充)
- IdentityService 服务端持久绑定匿名身份 + session ownership(403/409) + 学员 namespace 归属
- 长期记忆按学员 scope 过滤(不同学员 namespace 不混,default_user 不暴露给匿名身份)

框架约束:本仓库无 pytest-asyncio;异步用例用 _run(asyncio.run) 包装。
DB 一律用临时库 / :memory:,不碰生产库。
"""
import asyncio
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run(coro):
    return asyncio.run(coro)


# ======================================================================
# 1. Identity 原语
# ======================================================================

class TestIdentity:

    def test_family_scope_is_user_id(self):
        from config.identity import Identity
        ident = Identity(user_id="anon_x")
        assert ident.memory_scope() == "anon_x"

    def test_student_scope_separated_from_family(self):
        from config.identity import Identity
        family = Identity(user_id="anon_x")
        child_a = Identity(user_id="anon_x", student_id="s1")
        child_b = Identity(user_id="anon_x", student_id="s2")
        # 同一身份下不同学员 namespace 不混
        assert child_a.memory_scope() != child_b.memory_scope()
        assert child_a.memory_scope() != family.memory_scope()
        assert child_a.memory_scope().startswith("anon_x")
        assert "s1" in child_a.memory_scope()

    def test_anonymous_ids_unpredictable_and_distinct(self):
        from config.identity import new_anonymous_user_id
        a = new_anonymous_user_id()
        b = new_anonymous_user_id()
        assert a != b
        assert a.startswith("anon_") and len(a) > 20

    def test_legacy_identity_uses_default_user_and_flagged(self):
        from config.identity import legacy_identity, LEGACY_DEFAULT_USER_ID
        ident = legacy_identity()
        assert ident.legacy is True
        assert ident.user_id == LEGACY_DEFAULT_USER_ID

    def test_with_session_preserves_user_and_student(self):
        from config.identity import Identity
        ident = Identity(user_id="anon_x", student_id="s1")
        bound = ident.with_session("sess_1")
        assert bound.session_id == "sess_1"
        assert bound.user_id == "anon_x"
        assert bound.student_id == "s1"


# ======================================================================
# 2. IdentityService:匿名身份 + session ownership + 学员归属
# ======================================================================

@pytest.fixture
def id_service(tmp_path):
    from services.identity_service import IdentityService
    db_url = f"sqlite:///{tmp_path / 'identity.db'}"
    svc = IdentityService(db_path=db_url)
    yield svc
    svc.close()


class TestSessionOwnership:

    def test_resolve_or_create_new_identity(self, id_service):
        ident = id_service.resolve_identity(cookie_user_id=None)
        assert ident.user_id.startswith("anon_")
        assert ident.legacy is False

    def test_resolve_existing_identity_from_cookie(self, id_service):
        first = id_service.resolve_identity(cookie_user_id=None)
        again = id_service.resolve_identity(cookie_user_id=first.user_id)
        assert again.user_id == first.user_id

    def test_unknown_cookie_user_id_is_not_trusted_blindly(self, id_service):
        """客户端塞一个未签发过的 user_id → 服务端不认,改发新身份(不冒充他人)。"""
        forged = id_service.resolve_identity(cookie_user_id="default_user")
        assert forged.user_id != "default_user"
        assert forged.user_id.startswith("anon_")

    def test_new_session_bound_to_identity(self, id_service):
        ident = id_service.resolve_identity(None)
        sid = id_service.create_session(ident)
        assert sid
        owner = id_service.session_owner(sid)
        assert owner == ident.user_id

    def test_same_identity_can_reuse_session(self, id_service):
        ident = id_service.resolve_identity(None)
        sid = id_service.create_session(ident)
        # 不抛异常即通过
        id_service.assert_session_owned(ident, sid)

    def test_other_identity_cannot_claim_session(self, id_service):
        owner = id_service.resolve_identity(None)
        intruder = id_service.resolve_identity(None)
        sid = id_service.create_session(owner)
        with pytest.raises(id_service.SessionOwnershipError) as ei:
            id_service.assert_session_owned(intruder, sid)
        # 403 语义,而不是泄露/静默抢占
        assert ei.value.status_code == 403

    def test_unknown_session_id_is_claimable_by_first_identity(self, id_service):
        """全新、库里不存在的 session_id:首次出现即绑定为当前身份(便于前端生成 id)。"""
        ident = id_service.resolve_identity(None)
        sid = id_service.assert_session_owned(ident, "client-made-session-xyz")
        assert sid == "client-made-session-xyz"
        assert id_service.session_owner(sid) == ident.user_id

    def test_ownership_survives_rebuild(self, tmp_path):
        """重建 service(独立实例)后,ownership 仍生效(持久化,而非仅内存)。"""
        from services.identity_service import IdentityService
        db_url = f"sqlite:///{tmp_path / 'identity.db'}"
        svc1 = IdentityService(db_path=db_url)
        owner = svc1.resolve_identity(None)
        sid = svc1.create_session(owner)
        owner_id = owner.user_id
        svc1.close()

        svc2 = IdentityService(db_path=db_url)
        try:
            other = svc2.resolve_identity(None)
            with pytest.raises(svc2.SessionOwnershipError):
                svc2.assert_session_owned(other, sid)
            # 原身份可从持久库恢复身份
            back = svc2.resolve_identity(cookie_user_id=owner_id)
            assert back.user_id == owner_id
            svc2.assert_session_owned(back, sid)
        finally:
            svc2.close()


class TestStudentNamespaceOwnership:

    def test_identity_can_claim_student_namespace(self, id_service):
        ident = id_service.resolve_identity(None)
        id_service.claim_student(ident, "stu-001")

    def test_same_student_reclaim_by_same_identity_is_idempotent(self, id_service):
        ident = id_service.resolve_identity(None)
        id_service.claim_student(ident, "stu-002")
        id_service.claim_student(ident, "stu-002")  # 幂等,不抛

    def test_other_identity_cannot_claim_same_student(self, id_service):
        a = id_service.resolve_identity(None)
        b = id_service.resolve_identity(None)
        id_service.claim_student(a, "stu-003")
        with pytest.raises(id_service.StudentClaimError) as ei:
            id_service.claim_student(b, "stu-003")
        assert ei.value.status_code == 409

    def test_use_student_requires_claim(self, id_service):
        ident = id_service.resolve_identity(None)
        # 未认领的 student namespace → 不允许用于读业务数据
        with pytest.raises(id_service.StudentClaimError):
            id_service.assert_student_owned(ident, "unclaimed-stu")


# ======================================================================
# 3. 长期记忆按学员 scope 过滤
# ======================================================================

class TestStudentScopedMemory:

    @pytest.fixture
    def mem(self, tmp_path):
        from services.long_term_memory_service import LongTermMemoryService
        svc = LongTermMemoryService(db_path=f"sqlite:///{tmp_path / 'mem.db'}")
        yield svc
        svc.db_router.close()

    def test_student_scopes_do_not_mix(self, mem):
        from config.identity import Identity
        a = Identity(user_id="anon_a", student_id="s1").memory_scope()
        b = Identity(user_id="anon_a", student_id="s2").memory_scope()
        mem.add_memory(a, "preference", "学员1偏数学", embedding=[1.0, 0.0], auto_embed=False)
        mem.add_memory(b, "preference", "学员2偏英语", embedding=[1.0, 0.0], auto_embed=False)
        got_a = [m["content"] for m in mem.get_user_memories(a)]
        got_b = [m["content"] for m in mem.get_user_memories(b)]
        assert got_a == ["学员1偏数学"]
        assert got_b == ["学员2偏英语"]

    def test_default_user_not_visible_to_anonymous_identity(self, mem):
        from config.identity import Identity, LEGACY_DEFAULT_USER_ID
        mem.add_memory(LEGACY_DEFAULT_USER_ID, "preference", "旧默认用户隐私",
                       embedding=[1.0, 0.0], auto_embed=False)
        anon = Identity(user_id="anon_fresh").memory_scope()
        assert mem.get_user_memories(anon) == []
