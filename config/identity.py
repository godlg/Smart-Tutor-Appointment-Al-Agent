"""会话身份与隔离(Phase 1 记忆可靠性)

本模块是**叶子依赖**:config 包不依赖 agents / services / db,任何层都可安全导入,
用于在 API 入口 → 主控 Agent → 业务 Agent → MemoryContext → 摘要/行为写入之间
传递"可验证的匿名身份"。

设计边界(重要,勿夸大):
- 本系统**没有登录系统**。Identity.user_id 是服务端签发给某个浏览器的**不可预测匿名身份**,
  它**不能**证明账户级真实家长身份,也**不能**证明 student_id 的真实归属。
- student_id 只作为该匿名身份下的**子 namespace 提示**;是否允许使用,由服务端校验
  (见 services.identity_service):若 student_id 不属于当前身份,则拒绝,避免用任意
  student_id 读取别人的学员业务数据。
- 同一匿名身份跨会话可保留长期记忆;不同浏览器身份互不可读。
"""
from __future__ import annotations

import secrets
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

# 匿名身份 cookie 名(HttpOnly,服务端持久绑定)
IDENTITY_COOKIE_NAME = "tutor_uid"

# 旧内部 demo / 单用户路径使用的默认用户,仅 legacy,不向匿名 web 身份暴露
LEGACY_DEFAULT_USER_ID = "default_user"

# 请求级当前身份:由 web 入口在调用主控前 set,主控/业务 Agent/记忆读写读取。
# 用 ContextVar 而非参数,是为保持既有内部函数签名(如 ProcessUserInput_stream
# 的 (message, session_id=...) )向后兼容,同时把身份贯穿全链路。
_current_identity: ContextVar[Optional["Identity"]] = ContextVar(
    "current_identity", default=None
)


def set_current_identity(identity: Optional["Identity"]):
    """设置当前请求身份,返回用于重置的 token。"""
    return _current_identity.set(identity)


def reset_current_identity(token) -> None:
    _current_identity.reset(token)


def get_current_identity() -> Optional["Identity"]:
    """取当前请求身份(未设置返回 None)。"""
    return _current_identity.get()


def new_anonymous_user_id() -> str:
    """生成不可预测的匿名用户 id(32 字节 URL-safe 随机串,前缀便于识别)。"""
    return "anon_" + secrets.token_urlsafe(24)


def new_session_id() -> str:
    """生成不可预测的会话 id。"""
    return "sess_" + secrets.token_urlsafe(18)


@dataclass(frozen=True)
class Identity:
    """一次请求可验证的匿名身份上下文。

    - user_id: 服务端认可的匿名身份(不可由客户端 user_id 参数冒充)
    - student_id: 可选,该身份下的学员子 namespace 提示(服务端需校验归属)
    - session_id: 本次会话 id(服务端需校验 ownership)
    - legacy: 是否为旧的无身份内部/ demo 路径(此时不得把 default_user 记忆暴露给 web)
    """

    user_id: str
    student_id: Optional[str] = None
    session_id: Optional[str] = None
    legacy: bool = False

    def with_session(self, session_id: Optional[str]) -> "Identity":
        """返回绑定到指定 session 的副本(保持 user/student/cookie 来源不变)。"""
        return Identity(
            user_id=self.user_id,
            student_id=self.student_id,
            session_id=session_id,
            legacy=self.legacy,
        )

    def memory_scope(self) -> str:
        """长期记忆的归属 namespace。

        家庭级(student_id=None)与学员级分开:同一身份下不同 student namespace 不混。
        格式: `<user_id>` 或 `<user_id>::student::<student_id>`。
        """
        if self.student_id:
            return f"{self.user_id}::student::{self.student_id}"
        return self.user_id


# 向后兼容别名:spec 里两种叫法(Identity / SessionContext)指同一概念
SessionContext = Identity


def legacy_identity(session_id: Optional[str] = None) -> Identity:
    """构造旧内部路径(demo / 单进程无身份)的 legacy 身份。

    legacy=True 时,记忆读写落在 LEGACY_DEFAULT_USER_ID namespace,且**不会**被
    匿名 web 身份读到(匿名身份的 user_id 是 anon_*,与之不同)。
    """
    return Identity(
        user_id=LEGACY_DEFAULT_USER_ID,
        student_id=None,
        session_id=session_id,
        legacy=True,
    )
