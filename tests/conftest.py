"""全局离线测试隔离：collection 前禁 dotenv、外网和默认生产 SQLite。

默认业务服务使用每个测试独立的临时库，显式临时路径保持原样。
不存在 TEST_USE_PROD_DB 绕过开关；子进程必须自行传临时 DB 并禁 dotenv。
"""
from __future__ import annotations

import os
import socket
import ipaddress

import pytest


_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex
_original_getaddrinfo = socket.getaddrinfo


def _is_local_address(host):
    if host in (None, "", "localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _offline_connect(self, address):
    if isinstance(address, tuple) and not _is_local_address(address[0]):
        raise RuntimeError("offline tests prohibit external socket connections")
    return _original_connect(self, address)


def _offline_connect_ex(self, address):
    if isinstance(address, tuple) and not _is_local_address(address[0]):
        raise RuntimeError("offline tests prohibit external socket connections")
    return _original_connect_ex(self, address)


def _offline_getaddrinfo(host, *args, **kwargs):
    if not _is_local_address(host):
        raise RuntimeError("offline tests prohibit external DNS lookups")
    return _original_getaddrinfo(host, *args, **kwargs)


socket.socket.connect = _offline_connect
socket.socket.connect_ex = _offline_connect_ex
socket.getaddrinfo = _offline_getaddrinfo


# 此测试套件永不连接默认生产库,包括 collection 阶段的模块级 service。
# 临时文件库和显式 :memory: 保持原样;不能用 TEST_USE_PROD_DB 绕开隔离。
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# 仅用于构造SDK对象的占位值,不是密钥;测试的网络边界由各用例mock。
os.environ["MODEL_PROVIDER"] = "qwen"
os.environ["LLM_API_KEY"] = "offline-test-placeholder"
os.environ["LLM_BASE_URL"] = "https://offline.invalid/v1"
os.environ["LLM_MODEL"] = "offline-test"
os.environ["EMBEDDING_API_KEY"] = "offline-test-placeholder"
os.environ["EMBEDDING_BASE_URL"] = "https://offline.invalid/v1"
from db.base.session_manager import SessionManager

_TEST_DEFAULT_DB = "sqlite:///:memory:"

_session_manager_init = SessionManager.__init__


def _isolated_session_manager_init(self, db_path="sqlite:///data/smart_appointment.db"):
    from pathlib import Path
    from sqlalchemy.engine import make_url
    url = make_url(db_path)
    if url.get_backend_name() == "sqlite" and url.database not in (None, "", ":memory:"):
        production_dir = (Path(__file__).resolve().parents[1] / "data").resolve()
        if Path(url.database).resolve().is_relative_to(production_dir):
            db_path = _TEST_DEFAULT_DB
    _session_manager_init(self, db_path)


SessionManager.__init__ = _isolated_session_manager_init
_PATCH_ENABLED = True


@pytest.fixture(autouse=True)
def _isolated_application_database(tmp_path, monkeypatch, request):
    """默认业务服务共享每个test独立的临时库,不借生产种子/残留数据。"""
    monkeypatch.setattr(__import__(__name__, fromlist=["_"]), "_TEST_DEFAULT_DB",
                        f"sqlite:///{tmp_path / 'application.db'}")
    if request.node.path.name == "test_api_endpoints.py":
        from services.teacher_service import TeacherService
        from services.student_profile_service import StudentProfileService
        from services.package_service import PackageService
        TeacherService().initialize_default_teachers()
        StudentProfileService().initialize_default_student()
        PackageService().initialize_default_packages()


@pytest.fixture(autouse=_PATCH_ENABLED)
def _force_knowledge_service_in_memory(monkeypatch):
    """autouse 默认开:每个 test 强制 KnowledgeService 走 :memory:。

    用 monkeypatch 保证 test 结束后自动还原,不影响 prod 进程行为。
    """
    from services import knowledge_service as ks_mod

    original_init = ks_mod.KnowledgeService.__init__

    def _patched_init(self, db_path: str = "sqlite:///:memory:"):
        # 显式传 db_path 的调用方仍生效(测试里有用 in_memory_db fixture)
        original_init(self, db_path=db_path)

    monkeypatch.setattr(ks_mod.KnowledgeService, "__init__", _patched_init)


@pytest.fixture(autouse=_PATCH_ENABLED)
def _force_identity_and_conversation_in_memory(monkeypatch):
    """autouse 默认开:强制 IdentityService / ConversationService 默认走 :memory:。

    记忆可靠性 Phase 1 新增的两个服务默认连生产 SQLite。测试里走 FastAPI 路由时
    (如 /chat/stream 会解析匿名身份 + 记录会话历史)必须隔离,不能读写 data/。
    显式传 db_path 的调用方仍生效(测试用 tmp_path 建独立临时库)。
    """
    from services import identity_service as id_mod
    from services import conversation_service as conv_mod

    id_orig = id_mod.IdentityService.__init__
    conv_orig = conv_mod.ConversationService.__init__

    def _patched_id_init(self, db_path=None, **kwargs):
        id_orig(self, db_path=db_path or _TEST_DEFAULT_DB, **kwargs)

    def _patched_conv_init(self, db_path=None, **kwargs):
        conv_orig(self, db_path=db_path or _TEST_DEFAULT_DB, **kwargs)

    monkeypatch.setattr(id_mod.IdentityService, "__init__", _patched_id_init)
    monkeypatch.setattr(conv_mod.ConversationService, "__init__", _patched_conv_init)

    # 会话服务单例在 api.chat_handler 里被缓存;测试间清掉,避免串库。
    from api import chat_handler
    monkeypatch.setattr(chat_handler, "_conversation_service", None, raising=False)
    monkeypatch.setattr(chat_handler, "_task_agent", None, raising=False)
    for name in ("_session_agents", "_session_chat_histories", "_session_async_locks",
                 "_active_sessions", "_session_last_access"):
        monkeypatch.setattr(chat_handler, name, {}, raising=False)
    from api import identity as api_identity
    monkeypatch.setattr(api_identity, "_identity_service", None, raising=False)