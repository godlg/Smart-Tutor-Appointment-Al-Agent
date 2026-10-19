"""控制台账号与鉴权服务(机构后台分面 A)

职责与边界(勿夸大):
- 只服务**机构内部后台**(admin / support / academic 三种角色)的登录与会话。
- 与匿名家长身份(services.identity_service)是两套独立体系:本服务的 token
  不能证明任何家长身份,匿名家长 cookie 也不能换取控制台权限。
- 密码哈希只用 stdlib(`hashlib.pbkdf2_hmac`),不引入 passlib / bcrypt 等新依赖。
- 存储格式固定为 `pbkdf2_sha256$120000$<salt_hex>$<digest_hex>`,
  校验失败(格式非法/长度不符/算法不同)一律返回 False,不抛异常。
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from config.console_auth import CONSOLE_SESSION_TTL_HOURS
from db.base.session_manager import DEFAULT_DB_PATH, SessionManager
from db.models import ConsoleSession, ConsoleUser, Teacher

# 老师账号角色(与 api/core/console_auth.py 的 ROLE_TEACHER 取值一致;services 不反向依赖 api)
TEACHER_ROLE = "teacher"
DEFAULT_TEACHER_PASSWORD = "teacher12345"  # 仅本地开发种子口令;生产部署必须全部更换

# 密码哈希参数(格式串必须与 _PBKDF2_PREFIX 一致,测试会断言前缀)
_PBKDF2_ALGORITHM = "sha256"
_PBKDF2_ITERATIONS = 120000
_PBKDF2_PREFIX = f"pbkdf2_sha256${_PBKDF2_ITERATIONS}$"
_SALT_BYTES = 16

# token_urlsafe(32) 约 43 字符,列宽 64,入库前断言,防止未来改长度撑爆列
_TOKEN_BYTES = 32
_TOKEN_MAX_LEN = 64


class AuthError(Exception):
    """控制台鉴权错误(登录失败统一走这个,不区分具体原因)。"""


def hash_password(raw: str) -> str:
    """把明文口令哈希成 `pbkdf2_sha256$<iters>$<salt_hex>$<digest_hex>`。"""
    salt = secrets.token_bytes(_SALT_BYTES)
    derived = hashlib.pbkdf2_hmac(_PBKDF2_ALGORITHM, raw.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"{_PBKDF2_PREFIX}{salt.hex()}${derived.hex()}"


def verify_password(stored: str, raw: str) -> bool:
    """校验明文口令是否匹配存储串;格式非法/长度不符返回 False(不抛异常)。"""
    if not isinstance(stored, str) or not isinstance(raw, str):
        return False
    parts = stored.split("$")
    if len(parts) != 4:
        return False
    algorithm, iterations, salt_hex, digest_hex = parts
    if algorithm != "pbkdf2_sha256":
        return False
    try:
        rounds = int(iterations)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except (TypeError, ValueError):
        return False
    if rounds <= 0 or not salt or not expected:
        return False
    try:
        derived = hashlib.pbkdf2_hmac(_PBKDF2_ALGORITHM, raw.encode("utf-8"), salt, rounds)
    except (TypeError, ValueError):
        return False
    return hmac.compare_digest(derived, expected)


class ConsoleAuthService:
    """控制台账号登录 / 会话签发与校验。"""

    # 默认开发账号(明文口令仅用于本地开发;生产部署必须全部更换)
    DEFAULT_USERS: List[Dict[str, Any]] = [
        {"username": "admin", "password": "admin12345", "role": "admin", "display_name": "教务主管"},
        {"username": "support", "password": "support12345", "role": "support", "display_name": "客服小美"},
        {"username": "academic", "password": "academic12345", "role": "academic", "display_name": "教研老师"},
    ]

    def __init__(self, db_path: str | None = None):
        self.db = SessionManager(db_path or DEFAULT_DB_PATH)

    def close(self) -> None:
        self.db.close()

    # ------------------------------------------------------------------
    # 默认账号
    # ------------------------------------------------------------------

    def ensure_default_users(self) -> int:
        """按 username 查缺补漏地种默认账号;返回本次**新插入**的条数。

        幂等判据是「逐个 username 是否存在」,不是「表非空就整体跳过」——
        否则手动删掉某个账号后永远补不回来。
        """
        created = 0
        with self.db.session_scope() as session:
            existing = {row[0] for row in session.query(ConsoleUser.username).all()}
            for spec in self.DEFAULT_USERS:
                if spec["username"] in existing:
                    continue
                session.add(ConsoleUser(
                    username=spec["username"],
                    password_hash=hash_password(spec["password"]),
                    role=spec["role"],
                    display_name=spec["display_name"],
                    is_active=True,
                ))
                created += 1
        return created

    def ensure_default_teacher_accounts(self) -> int:
        """为 teachers 表里每位老师按 username(`teacher{id}`) 查缺补漏地建账号;返回新插入条数。

        幂等判据同 ensure_default_users:逐个 username 判断,不得「表非空就跳过」。
        """
        created = 0
        with self.db.session_scope() as session:
            existing = {row[0] for row in session.query(ConsoleUser.username).all()}
            for teacher_id, name in session.query(Teacher.id, Teacher.name).all():
                username = f"teacher{teacher_id}"
                if username in existing:
                    continue
                session.add(ConsoleUser(
                    username=username,
                    password_hash=hash_password(DEFAULT_TEACHER_PASSWORD),
                    role=TEACHER_ROLE,
                    display_name=name,
                    is_active=True,
                    teacher_id=teacher_id,
                ))
                created += 1
        return created

    # ------------------------------------------------------------------
    # 登录 / 注销
    # ------------------------------------------------------------------

    def login(self, username: str, password: str) -> Dict[str, Any]:
        """校验口令并签发会话;失败统一 raise AuthError(不泄露用户是否存在)。"""
        with self.db.session_scope() as session:
            user = session.query(ConsoleUser).filter(ConsoleUser.username == username).first()
            if user is None or not user.is_active or not verify_password(user.password_hash, password):
                raise AuthError("用户名或密码错误")

            token = secrets.token_urlsafe(_TOKEN_BYTES)
            if len(token) > _TOKEN_MAX_LEN:
                raise AuthError("会话令牌长度超出存储列宽")

            expires_at = datetime.now() + timedelta(hours=CONSOLE_SESSION_TTL_HOURS)
            session.add(ConsoleSession(
                token=token,
                user_id=user.id,
                role=user.role,
                expires_at=expires_at,
            ))
            return {
                "token": token,
                "username": user.username,
                "role": user.role,
                "display_name": user.display_name,
                "teacher_id": user.teacher_id,
                "expires_at": expires_at.isoformat(),
            }

    def logout(self, token: Optional[str]) -> None:
        """删除该 token 的会话;token 为空或不存在都不报错(幂等)。"""
        if not token:
            return
        with self.db.session_scope() as session:
            session.query(ConsoleSession).filter(ConsoleSession.token == token).delete()

    # ------------------------------------------------------------------
    # 会话解析
    # ------------------------------------------------------------------

    def resolve(self, token: Optional[str]) -> Optional[Dict[str, Any]]:
        """把 token 解析为控制台用户;无效返回 None,过期则删除该行后返回 None。"""
        if not token:
            return None
        with self.db.session_scope() as session:
            record = session.query(ConsoleSession).filter(ConsoleSession.token == token).first()
            if record is None:
                return None
            if record.expires_at is None or record.expires_at <= datetime.now():
                session.delete(record)
                return None

            user = session.query(ConsoleUser).filter(ConsoleUser.id == record.user_id).first()
            if user is None or not user.is_active:
                # 账号被删除/停用 → 旧会话立即失效
                session.delete(record)
                return None
            return {
                "user_id": user.id,
                "username": user.username,
                "role": record.role,          # 用签发时快照,与依赖矩阵的鉴权语义一致
                "display_name": user.display_name,
                "teacher_id": user.teacher_id,  # 仅老师账号有值,其余为 None
                "expires_at": record.expires_at.isoformat(),
            }
