"""身份 / 会话归属服务(记忆可靠性 Phase 1)

职责与边界(勿夸大):
- 本系统**无登录系统**。它只在服务端签发并持久绑定**不可预测的匿名身份**,
  不能声称账户级真实家长认证,也不能证明 student_id 的真实归属。
- 客户端请求体里的 user_id **不**被当作授权;只有服务端签发过(cookie 命中
  identities 表)的 user_id 才被接受,否则改发新身份(防止冒充他人)。
- session_id 必须校验 ownership(含从持久库恢复的会话):他人不得抢占/重绑,
  返回 403 语义,而不是泄露消息或静默覆盖。
- student_id 只作该身份下的子 namespace:未被认领可首次认领(便于前端生成),
  已被他人认领则 409,避免用任意 student_id 读取别人学员数据。
"""

from typing import Optional

from config.identity import (
    IDENTITY_COOKIE_NAME,
    Identity,
    new_anonymous_user_id,
    new_session_id,
)
from db.db_router import DatabaseRouter
from db.repositories.identity_repository import SessionBindingError


class IdentityError(Exception):
    """身份相关错误基类(带 HTTP 语义状态码,供 API 层映射)。"""

    status_code = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class SessionOwnershipError(IdentityError):
    """会话不属于当前身份(403:不允许抢占/重绑他人会话)。"""

    status_code = 403


class StudentClaimError(IdentityError):
    """学员 namespace 归属冲突(409)或未认领(403)。"""

    def __init__(self, message: str, status_code: int = 409):
        super().__init__(message)
        self.status_code = status_code


class IdentityService:
    """匿名身份 / 会话归属 / 学员 namespace 校验。"""

    def __init__(self, db_path: str = "sqlite:///data/smart_appointment.db",
                 db_router: Optional[DatabaseRouter] = None):
        self._owns_router = db_router is None
        self.db_router = db_router or DatabaseRouter(db_path)
        self.repo = getattr(self.db_router, "identity_repo", None)
        if self.repo is None:  # 兼容旧 DatabaseRouter 未挂 identity_repo 的情况
            from db.repositories.identity_repository import IdentityRepository
            self.repo = IdentityRepository(self.db_router.session_manager)

    # 暴露给 API 层映射用(避免 API 直接 import 具体异常类)
    SessionOwnershipError = SessionOwnershipError
    StudentClaimError = StudentClaimError

    def close(self) -> None:
        if self._owns_router:
            self.db_router.close()

    # ------------------------------------------------------------------
    # 身份解析
    # ------------------------------------------------------------------

    def resolve_identity(self, cookie_user_id: Optional[str]) -> Identity:
        """从 cookie 里的 user_id 解析身份;未签发/非法则签发新身份。

        关键:**不盲信** cookie 里的任意 user_id —— 只有本服务登记过(cookie 命中
        identities 表)才复用,否则视为伪造,改发新身份,防止冒充 default_user 等。
        """
        if cookie_user_id:
            # 允许的形态:anon_* 且已登记。legacy 的 default_user 不在此接受。
            if self.repo.identity_exists(cookie_user_id):
                return Identity(user_id=cookie_user_id, legacy=False)
        new_id = new_anonymous_user_id()
        self.repo.create_identity(new_id)
        return Identity(user_id=new_id, legacy=False)

    def issue_new_identity(self) -> Identity:
        """显式签发一个全新匿名身份(用于首次访问无 cookie)。"""
        new_id = new_anonymous_user_id()
        self.repo.create_identity(new_id)
        return Identity(user_id=new_id, legacy=False)

    @property
    def cookie_name(self) -> str:
        return IDENTITY_COOKIE_NAME

    def new_session_id(self) -> str:
        return new_session_id()

    # ------------------------------------------------------------------
    # 会话归属
    # ------------------------------------------------------------------

    def create_session(self, identity: Identity, session_id: Optional[str] = None) -> str:
        """创建一个新会话并绑定到身份;返回 session_id。"""
        sid = session_id or new_session_id()
        try:
            self.repo.create_session(sid, identity.user_id, student_id=identity.student_id)
        except SessionBindingError as exc:
            if exc.status_code == 409:
                raise StudentClaimError(str(exc), status_code=409) from exc
            raise SessionOwnershipError(str(exc)) from exc
        return sid

    def session_owner(self, session_id: str) -> Optional[str]:
        return self.repo.get_session_owner(session_id)

    def assert_session_owned(self, identity: Identity, session_id: str) -> str:
        """校验会话归属:属于当前身份(或全新可绑定)则通过并返回 session_id。

        - 库里不存在该 session_id → 首次出现,绑定为当前身份(便于前端生成 id)
        - 已属于当前身份 → 通过
        - 属于他人 → raise SessionOwnershipError(403)
        """
        if not session_id:
            raise SessionOwnershipError("缺少 session_id")
        return self.create_session(identity, session_id)

    # ------------------------------------------------------------------
    # 学员 namespace
    # ------------------------------------------------------------------

    def claim_student(self, identity: Identity, student_id: str) -> None:
        """认领学员 namespace:未认领可认领;已被他人认领 → 409。"""
        if not student_id:
            raise StudentClaimError("缺少 student_id", status_code=400)
        ok = self.repo.claim_student(identity.user_id, student_id)
        if not ok:
            raise StudentClaimError(
                f"student {student_id} 已被其它身份认领", status_code=409
            )

    def assert_student_owned(self, identity: Identity, student_id: str) -> str:
        """校验学员 namespace 已归属当前身份(用于读该学员业务数据)。"""
        owner = self.repo.get_student_owner(student_id)
        if owner is None:
            raise StudentClaimError(
                f"student {student_id} 尚未认领,不可访问其业务数据",
                status_code=403,
            )
        if owner != identity.user_id:
            raise StudentClaimError(
                f"student {student_id} 不属于当前身份", status_code=403
            )
        return student_id
