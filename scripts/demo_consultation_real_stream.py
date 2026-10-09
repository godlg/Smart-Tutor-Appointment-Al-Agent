"""离线实际 HTTP 咨询流验真：临时 SQLite + loopback 临时端口 + 事件屏障。

运行：.venv/Scripts/python.exe scripts/demo_consultation_real_stream.py
只替换 LLM/embedding 边界；使用真实 Web、主入口、router、上下文和 repositories。
不导入 app、不运行生产 lifespan；复用已验收 demo 的 dotenv/网络/SQLite containment。
"""
from __future__ import annotations

import os
import sys

# 必须早于任何应用导入；不读取 dotenv 或 SDK 认证配置。
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["TUTOR_IDENTITY_COOKIE_SECURE"] = "false"
os.environ["PYTHONUTF8"] = "1"
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="strict")

import asyncio
from contextlib import closing
import json
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
from typing import Any

from demo_memory_reliability import (
    disposable_database, offline_network, require, verify_network_guard,
)

FIRST = "先理解概念。"
LAST = "再整理错题。"
PREFIX = "[REPLY][咨询机器人]"
PATHS = ["/chat/stream", "/chat"]


class BarrierConsultationModel:
    """模型首正文后必须等待客户端放行，绝不模拟最终全文 ainvoke。"""

    def __init__(self, mode: str):
        self.mode = mode
        self.release = asyncio.Event()
        self.waiting = asyncio.Event()
        self.closed = asyncio.Event()
        self.done = False
        self.prompt = ""
        self.identities = []
        self.invoke_count = 0
        self.stream_count = 0

    async def ainvoke(self, messages: list, **kwargs: Any) -> Any:
        from langchain_core.messages import AIMessage
        require("YES" in messages[0]["content"], "answer attempted ainvoke")
        self.invoke_count += 1
        return AIMessage(content="YES")

    async def astream(self, messages: list, **kwargs: Any) -> Any:
        from langchain_core.messages import AIMessageChunk
        from config.identity import get_current_identity
        self.stream_count += 1
        self.prompt = messages[0]["content"]
        try:
            self.identities.append(get_current_identity())
            yield AIMessageChunk(content="", additional_kwargs={"reasoning_content": "PRIVATE_REASONING"})
            yield AIMessageChunk(content=FIRST)
            self.waiting.set()
            await self.release.wait()
            self.identities.append(get_current_identity())
            if self.mode == "midstream":
                raise RuntimeError("SECRET_RAW_MODEL_ERROR")
            yield AIMessageChunk(content=[
                {"type": "reasoning", "text": "PRIVATE_REASONING"},
                {"type": "text", "text": LAST},
            ], response_metadata={"private": "PRIVATE_METADATA"})
            self.done = True
        finally:
            self.identities.append(get_current_identity())
            self.closed.set()


def install_barrier_factory(models: dict) -> None:
    """disposable_database 已装离线 provider；在应用导入前替换其模型工厂。"""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from config.identity import get_current_identity

    def factory(**kwargs: Any) -> Any:
        if kwargs.get("temperature") != 0.3:
            return FakeListChatModel(responses=["consultation"])
        identity = get_current_identity()
        require(identity is not None, "model constructed without scoped identity")
        mode = "midstream" if identity.session_id == "http-midstream" else "normal"
        model = BarrierConsultationModel(mode)
        models[identity.session_id] = model
        return model

    sys.modules["config.model_provider"].create_chat_model = factory


async def first_body(response: Any, model_lookup: Any) -> tuple[str, Any, Any]:
    require(response.status_code == 200, f"HTTP status {response.status_code}")
    iterator = response.aiter_text()
    body = ""
    while FIRST not in body:
        body += await asyncio.wait_for(anext(iterator), 10)
    model = model_lookup()
    require(not model.done and not model.release.is_set(), "model finished before first body")
    require(PREFIX + FIRST in body, "only a protocol label arrived, not model body")
    require("PRIVATE" not in body, "metadata/reasoning leaked")
    await asyncio.wait_for(model.waiting.wait(), 5)
    require(model.stream_count == 1 and model.invoke_count == 1, "wrong LLM calls")
    return body, iterator, model


def verify_saved(chat: Any, sid: str, status: str, body: str, successful: bool) -> None:
    from config.identity import Identity
    service = chat.get_conversation_service()
    rows = service.conversation_repo.get_messages(sid)
    require([row["role"] for row in rows] == ["human", "ai"], "duplicate/missing history")
    require(rows[1]["status"] == status and body in rows[1]["content"], "wrong saved response")
    require("[THOUGHT]" not in rows[1]["content"], "internal routing saved")
    binding = service.db_router.identity_repo.get_session(sid)
    identity = Identity(binding["user_id"], student_id=binding["student_id"], session_id=sid)
    require(len(service.restore_history(identity).messages) == 2, "history restore duplicated")
    records = service.db_router.user_behavior_repo.get_user_behaviors(
        identity.memory_scope(), action_type="consultation")
    records = [row for row in records if row["session_id"] == sid]
    require(len(records) == int(successful), "failed answer recorded as successful consultation")
    require(not chat._get_session_lock(sid).locked(), "session lock not released")
    require(sid not in chat._active_sessions, "active session leaked")


async def exercise_http(base_url: str, models: dict) -> dict:
    import httpx
    from api import chat_handler as chat
    from config.identity import get_current_identity

    async with httpx.AsyncClient(base_url=base_url, trust_env=False, timeout=15) as client:
        for index, path in enumerate(PATHS):
            sid = f"http-normal-{index}"
            async with client.stream("POST", path, json={
                "message": "初二数学怎么学", "session_id": sid,
            }) as response:
                body, iterator, model = await first_body(response, lambda: models[sid])
                print(f"FIRST_BODY_BEFORE_MODEL_DONE=True path={path}")
                model.release.set()
                body += "".join([text async for text in iterator])
            require(model.done and model.closed.is_set(), "model did not finish")
            require(body.count(PREFIX) == 1 and body.endswith(FIRST + LAST), "truncated final body")
            require("PRIVATE" not in body and "[ERROR]" not in body, "unexpected final payload")
            require(all(ident.session_id == sid for ident in model.identities), "model identity leaked")
            verify_saved(chat, sid, "complete", FIRST + LAST, True)
            print(f"COMPLETE_BODY=True path={path} body={FIRST + LAST}")

        sid = "http-midstream"
        async with client.stream("POST", PATHS[0], json={
            "message": "初二数学怎么学", "session_id": sid,
        }) as response:
            body, iterator, model = await first_body(response, lambda: models[sid])
            model.release.set()
            body += "".join([text async for text in iterator])
        require(body.count("[ERROR]") == 1 and body.count(PREFIX) == 1, "missing/duplicate failure")
        require(FIRST in body and LAST not in body and "SECRET_RAW" not in body, "failure payload unsafe")
        verify_saved(chat, sid, "partial", FIRST, False)
        print("MIDSTREAM_PARTIAL=True prior_body_preserved=True raw_error_hidden=True")

        sid = "http-cancel"
        async with client.stream("POST", PATHS[0], json={
            "message": "初二数学怎么学", "session_id": sid,
        }) as response:
            _, _, model = await first_body(response, lambda: models[sid])
            # 关闭真实 TCP 响应，不放行模型，触发服务端 disconnect/cancellation。
        await asyncio.wait_for(model.closed.wait(), 10)
        require(not model.done and not model.release.is_set(), "disconnect wrongly completed model")
        require(all(ident.session_id == sid for ident in model.identities), "cancel lost scoped identity")
        verify_saved(chat, sid, "partial", FIRST, False)
        require(get_current_identity() is None, "client task inherited server identity")
        print("CANCEL_CLEANUP=True model_closed=True partial=True lock_released=True")
    return {"transport": "loopback_tcp_http", "paths": PATHS,
            "first_body_before_model_done": True, "complete_body": True,
            "midstream_partial": True, "cancel_cleanup": True}


async def run_server(models: dict) -> dict:
    import uvicorn
    from fastapi import FastAPI
    from web.routes import router

    ready = asyncio.Event()

    class ReadyServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            if self.started:
                ready.set()

    application = FastAPI()
    application.include_router(router)
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = ReadyServer(uvicorn.Config(
            application, host="127.0.0.1", port=port, lifespan="off",
            access_log=False, log_level="warning", ws="none"))
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            await asyncio.wait_for(ready.wait(), 10)
            result = await exercise_http(f"http://127.0.0.1:{port}", models)
        finally:
            for model in models.values():
                model.release.set()
            server.should_exit = True
            try:
                await asyncio.wait_for(task, 10)
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        require(task.done() and not task.cancelled(), "server failed to stop normally")
        require(task.exception() is None, "server task failed")
        return {**result, "server_stopped": True}


def main() -> None:
    models: dict[str, BarrierConsultationModel] = {}
    with offline_network():
        verify_network_guard()
        with TemporaryDirectory(prefix="consultation-real-stream-") as temporary:
            with disposable_database(Path(temporary)) as sandbox:
                install_barrier_factory(models)
                result = asyncio.run(run_server(models))
                require(sandbox["operations"] == [], "business operation occurred")
            result = {**result, "temporary_directory": temporary}
        require(not Path(temporary).exists(), "temporary database was not removed")
    print("STREAM_DEMO_RESULT " + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
