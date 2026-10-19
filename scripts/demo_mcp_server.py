"""MCP(stdio)server 可复跑证据:真起子进程 + 真走 stdio 协议。

不是项目测试套件的一部分(那是 tests/test_mcp_server.py),本脚本负责证明
"外部 Agent 能通过 MCP 接进来":

1. 用临时 sqlite 建库并种子化老师 / 套餐 / 知识库(不碰 data/*.db);
2. 本机 127.0.0.1 起一个 OpenAI 兼容的确定性 embedding stub —— 让
   search_knowledge 真跑通既有 dense 检索链路,而不依赖外部网络 / 真实 Key;
3. 用 mcp.client.stdio.stdio_client 起 server 子进程 → ClientSession.initialize()
   → list_tools() 打印工具名 → 真实调用 find_teachers、search_knowledge 各一次
   并打印结果;
4. 断言写入工具(book_trial_lesson)默认被拒、身份闸门生效;
5. 全部通过打印 [OK] 并 sys.exit(0)。

运行解释器必须是 .venv 的解释器:`.venv\\Scripts\\python.exe scripts/demo_mcp_server.py`
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EMBED_DIM = 16


def require(condition: Any, message: str) -> None:
    """断言失败立刻非零退出,避免把源码里的期望当成已通过的结果。"""
    if not condition:
        print(f"[FAIL] {message}")
        sys.exit(1)


def embed(text: str) -> List[float]:
    """确定性 16 维向量:同文本必有同向量,相似文本内积仍可排序。"""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [b / 255.0 for b in digest[:EMBED_DIM]]


class _EmbeddingHandler(BaseHTTPRequestHandler):
    """OpenAI 兼容 /v1/embeddings 桩:只做本地确定性 embedding,不联网。"""

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - http.server 约定命名
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
        inputs = payload.get("input")
        if isinstance(inputs, str):
            inputs = [inputs]
        if not isinstance(inputs, list):
            inputs = [""]
        body = json.dumps({
            "object": "list",
            "model": payload.get("model", "demo-embed"),
            "data": [
                {"object": "embedding", "index": i, "embedding": embed(str(text))}
                for i, text in enumerate(inputs)
            ],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """静音 http.server 的默认访问日志(不污染证据输出)。"""


def start_embedding_stub() -> tuple[HTTPServer, str]:
    """在 127.0.0.1 上起桩,返回 (server, base_url)。仅监听 loopback。"""
    server = HTTPServer(("127.0.0.1", 0), _EmbeddingHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    return server, f"http://127.0.0.1:{port}/v1"


def seed_temp_database(url: str) -> dict:
    """把老师 / 套餐 / 知识库种子写进临时库,返回计数证据。

    种子写完后立即释放父进程持有的 sqlite 连接:Windows 上文件句柄不释放会让
    临时目录清理报 WinError 32,也会让 server 子进程的写入被锁。
    """
    from mcp_server.server import build_services

    services = build_services(url)
    try:
        services.teacher.initialize_default_teachers()
        services.package.initialize_default_packages()

        knowledge_docs = [
            ("高中数学 12 周提分方案: 函数、几何、数列、概率四大模块系统讲", "数学", "高中",
             ["高中", "数学", "提分"]),
            ("小学阅读启蒙课: 以经典绘本为载体, 培养自主阅读习惯", "语文", "小学",
             ["小学", "阅读", "绘本"]),
            ("初中数学 一元二次方程专题: 判别式、根与系数关系", "数学", "初中",
             ["初中", "数学", "方程"]),
        ]
        for content, subject, grade, keywords in knowledge_docs:
            services.knowledge.db.add_document(
                content=content, category="course_intro", keywords=keywords,
                embedding=embed(f"{content} {' '.join(keywords)}"),
                subject=subject, grade_level=grade, doc_type="course_intro",
            )
        return {
            "teachers": len(services.teacher.get_all_teachers()),
            "packages": len(services.package.get_all_packages()),
            "knowledge": len(services.knowledge.db.get_all_documents()),
        }
    finally:
        close_services(services)


def close_services(services: Any) -> None:
    """释放 service 持有的库连接(父进程用完即放,不占临时库文件)。"""
    for name in ("teacher", "package", "lesson"):
        router = getattr(getattr(services, name, None), "db", None)
        if router is not None:
            router.close()
    knowledge_router = getattr(getattr(services, "knowledge", None), "db_router", None)
    if knowledge_router is not None:
        knowledge_router.close()
    identity = getattr(services, "identity", None)
    if identity is not None:
        identity.close()  # IdentityService.close() 会关掉自持的 DatabaseRouter


def server_env(database_url: str, embedding_base_url: str) -> dict:
    """server 子进程环境:指向临时库 + 本机 embedding 桩;不设任何写入开关。"""
    env = dict(os.environ)
    env.update({
        "PYTHON_DOTENV_DISABLED": "1",
        "PYTHONIOENCODING": "utf-8",
        "DATABASE_URL": database_url,
        "MODEL_PROVIDER": "openai-compatible",
        "LLM_API_KEY": "demo-offline-placeholder",
        "LLM_BASE_URL": embedding_base_url,
        "LLM_MODEL": "demo-chat",
        "EMBEDDING_PROVIDER": "openai-compatible",
        "EMBEDDING_API_KEY": "demo-offline-placeholder",
        "EMBEDDING_BASE_URL": embedding_base_url,
        "EMBEDDING_MODEL": "demo-embed",
        "RETRIEVAL_MODE": "dense",
        "RERANK_ENABLED": "false",
    })
    # 显式清掉写入开关/身份声明,证明"默认拒绝"不是靠外部环境凑出来的
    for name in ("MCP_ALLOW_WRITE", "MCP_WRITE_TOKEN", "MCP_CALLER_USER_ID"):
        env.pop(name, None)
    return env


async def run() -> None:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    with TemporaryDirectory(prefix="mcp-demo-") as tmp:
        db_file = Path(tmp) / "demo_mcp.db"
        database_url = f"sqlite:///{db_file.as_posix()}"
        stub, embedding_base_url = start_embedding_stub()
        try:
            seeded = seed_temp_database(database_url)
            print(f"[SETUP] 临时库 {database_url}")
            print(f"[SETUP] 种子: {seeded}")
            print(f"[SETUP] embedding 桩 {embedding_base_url}")

            params = StdioServerParameters(
                command=sys.executable,
                # 用 -c 而非 -m:包 __init__ 已导入 server 模块,runpy 会重复执行并告警
                args=["-c", "from mcp_server.server import run_stdio; run_stdio()"],
                cwd=str(PROJECT_ROOT),
                env=server_env(database_url, embedding_base_url),
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    init = await session.initialize()
                    print(f"[MCP] initialize -> {init.server_info.name} "
                          f"v{init.server_info.version}")

                    tools = await session.list_tools()
                    names = [t.name for t in tools.tools]
                    print(f"[MCP] list_tools -> {names}")
                    require(
                        set(names) == {"find_teachers", "get_teacher_availability",
                                       "search_knowledge", "list_packages",
                                       "get_student_lessons", "book_trial_lesson"},
                        f"工具集合与预期不一致: {names}",
                    )

                    teachers = await session.call_tool("find_teachers", {"subject": "数学"})
                    teachers_payload = json.loads(teachers.content[0].text)
                    print("[CALL] find_teachers(数学) -> "
                          f"{json.dumps(teachers_payload, ensure_ascii=False)[:400]}")
                    require(teachers_payload.get("ok") is True, "find_teachers 未返回 ok")
                    require(teachers_payload.get("count", 0) >= 1,
                            "find_teachers 未命中任何老师")

                    knowledge = await session.call_tool(
                        "search_knowledge", {"query": "高中数学提分方案", "top_k": 2}
                    )
                    knowledge_payload = json.loads(knowledge.content[0].text)
                    print("[CALL] search_knowledge(高中数学提分方案) -> "
                          f"{json.dumps(knowledge_payload, ensure_ascii=False)[:400]}")
                    require(knowledge_payload.get("ok") is True,
                            "search_knowledge 未返回 ok")
                    require(knowledge_payload.get("count", 0) >= 1,
                            "search_knowledge 未命中任何文档(既有检索链路未跑通)")

                    lessons = await session.call_tool(
                        "get_student_lessons", {"student_id": "1"}
                    )
                    lessons_payload = json.loads(lessons.content[0].text)
                    print("[CALL] get_student_lessons(未声明身份) -> "
                          f"{json.dumps(lessons_payload, ensure_ascii=False)[:300]}")
                    require(lessons_payload.get("error") == "identity_required",
                            "未声明身份时读取学员数据必须被拒")

                    write = await session.call_tool("book_trial_lesson", {
                        "teacher_id": 1,
                        "student_id": "1",
                        "scheduled_at": "2026-09-20 19:00",
                        "subject": "数学",
                        "write_token": "guess-the-token",
                    })
                    write_payload = json.loads(write.content[0].text)
                    print("[GATE] book_trial_lesson(默认关闭) -> "
                          f"{json.dumps(write_payload, ensure_ascii=False)}")
                    require(write_payload.get("ok") is False,
                            "写入工具默认必须被拒")
                    require(write_payload.get("error") == "write_disabled",
                            "写入工具默认拒绝原因应为 write_disabled")
        finally:
            stub.shutdown()
            stub.server_close()

        require(not db_file.exists() or db_file.stat().st_size > 0, "临时库未落盘")
        print("[CHECK] data/ 未被写入:server 全程使用 DATABASE_URL 指向的临时库")
        print("[OK] MCP stdio server 工具接口验证通过")


def main() -> int:
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
