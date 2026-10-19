"""
人工接管队列页面（web 层）测试

覆盖:
- GET /console/human-handoff 返回 200,HTML 含页面标题关键字「人工接管队列」
- 页面自包含状态过滤(待处理/已处理/全部)与表格列关键字
- 前端 fetch 依赖的 API 路径在页面中出现(认领 / 回复 / 标记已处理 / 拉取对话)

约束:本仓库无 pytest-asyncio;不触网;不改后端,仅验证 web 层页面路由与模板。

面迁移（控制台分面 B）:页面从家长面 `/human-handoff` 搬到机构后台面
`/console/human-handoff` 并挂上控制台鉴权(需 support / admin),所以本文件
每个用例都要先登录拿 cookie,否则全是 401。页面内 `fetch` 的 API 路径不变。
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HANDOFF_PAGE = "/console/human-handoff"


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from app import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def support_login(client, monkeypatch, tmp_path):
    """后台面迁移后页面需 support / admin 会话;用 admin 默认账号登录。"""
    from api.core import console_auth as console_dep
    from config.console_auth import CONSOLE_COOKIE_NAME
    from services.console_auth_service import ConsoleAuthService

    # 控制台鉴权服务显式指向本用例临时文件库:TestClient 在独立线程跑 app,
    # `:memory:` 每线程一库,登录写入的会话解析不到。
    console_db = f"sqlite:///{(tmp_path / 'console_web.db').as_posix()}"
    original_init = ConsoleAuthService.__init__

    def _patched_init(self, db_path=None, *args, **kwargs):
        original_init(self, db_path=db_path or console_db)

    monkeypatch.setattr(ConsoleAuthService, "__init__", _patched_init)
    monkeypatch.setattr(console_dep, "_console_auth_service", None, raising=False)

    ConsoleAuthService(db_path=console_db).ensure_default_users()
    response = client.post("/api/console/login",
                           json={"username": "admin", "password": "admin12345"})
    assert response.status_code == 200, response.text
    assert CONSOLE_COOKIE_NAME in response.cookies


class TestHumanHandoffPage:

    def test_page_returns_200(self, client):
        r = client.get(HANDOFF_PAGE)
        assert r.status_code == 200, r.text
        assert "text/html" in r.headers.get("content-type", "")

    def test_page_contains_title_keyword(self, client):
        r = client.get(HANDOFF_PAGE)
        assert "人工接管队列" in r.text

    def test_page_contains_status_filter_options(self, client):
        r = client.get(HANDOFF_PAGE)
        assert "待处理" in r.text
        assert "已处理" in r.text
        assert "全部" in r.text

    def test_page_contains_table_headers(self, client):
        r = client.get(HANDOFF_PAGE)
        for header in ("ID", "会话 ID", "触发类型", "无关轮数", "创建时间", "状态"):
            assert header in r.text, f"表格应包含列: {header}"

    def test_page_contains_trigger_type_labels(self, client):
        r = client.get(HANDOFF_PAGE)
        assert "踢回阈值" in r.text
        assert "关键词触发" in r.text
        assert "客服手动" in r.text

    def test_page_contains_action_labels(self, client):
        r = client.get(HANDOFF_PAGE)
        assert "认领" in r.text
        assert "回复" in r.text
        assert "标记已处理" in r.text

    def test_page_references_handoff_api(self, client):
        r = client.get(HANDOFF_PAGE)
        assert "/api/handoffs" in r.text
        assert "/claim" in r.text
        assert "/reply" in r.text
        assert "/resolve" in r.text
        assert "/messages" in r.text

    def test_page_default_filter_is_pending(self, client):
        r = client.get(HANDOFF_PAGE)
        assert 'value="pending" selected' in r.text

    def test_page_links_back_to_console_home(self, client):
        """后台页只回后台首页,不再链回家长面 `/parent-center`。"""
        r = client.get(HANDOFF_PAGE)
        assert 'href="/console"' in r.text
        assert "/parent-center" not in r.text

    def test_page_requires_console_login(self, client):
        from config.console_auth import CONSOLE_COOKIE_NAME

        issued = client.cookies.get(CONSOLE_COOKIE_NAME)
        try:
            client.cookies.delete(CONSOLE_COOKIE_NAME)
            assert client.get(HANDOFF_PAGE).status_code == 401
        finally:
            client.cookies.set(CONSOLE_COOKIE_NAME, issued)
