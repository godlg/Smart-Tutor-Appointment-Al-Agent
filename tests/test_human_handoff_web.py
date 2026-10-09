"""
人工接管队列页面（web 层）测试

覆盖:
- GET /human-handoff 返回 200,HTML 含页面标题关键字「人工接管队列」
- 页面自包含状态过滤(待处理/已处理/全部)与表格列关键字
- 前端 fetch 依赖的 API 路径在页面中出现(认领 / 回复 / 标记已处理 / 拉取对话)

约束:本仓库无 pytest-asyncio;不触网;不改后端,仅验证 web 层页面路由与模板。
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from app import app
    return TestClient(app)


class TestHumanHandoffPage:

    def test_page_returns_200(self, client):
        r = client.get("/human-handoff")
        assert r.status_code == 200, r.text
        assert "text/html" in r.headers.get("content-type", "")

    def test_page_contains_title_keyword(self, client):
        r = client.get("/human-handoff")
        assert "人工接管队列" in r.text

    def test_page_contains_status_filter_options(self, client):
        r = client.get("/human-handoff")
        assert "待处理" in r.text
        assert "已处理" in r.text
        assert "全部" in r.text

    def test_page_contains_table_headers(self, client):
        r = client.get("/human-handoff")
        for header in ("ID", "会话 ID", "触发类型", "无关轮数", "创建时间", "状态"):
            assert header in r.text, f"表格应包含列: {header}"

    def test_page_contains_trigger_type_labels(self, client):
        r = client.get("/human-handoff")
        assert "踢回阈值" in r.text
        assert "关键词触发" in r.text
        assert "客服手动" in r.text

    def test_page_contains_action_labels(self, client):
        r = client.get("/human-handoff")
        assert "认领" in r.text
        assert "回复" in r.text
        assert "标记已处理" in r.text

    def test_page_references_handoff_api(self, client):
        r = client.get("/human-handoff")
        assert "/api/handoffs" in r.text
        assert "/claim" in r.text
        assert "/reply" in r.text
        assert "/resolve" in r.text
        assert "/messages" in r.text

    def test_page_default_filter_is_pending(self, client):
        r = client.get("/human-handoff")
        assert 'value="pending" selected' in r.text
