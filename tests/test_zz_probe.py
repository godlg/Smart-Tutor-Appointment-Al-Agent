from __future__ import annotations

import asyncio
import time

import pytest


def _make(tmp_path, monkeypatch):
    from api import chat_handler
    from agents import consultant_agent, task_classification_agent
    from services.conversation_service import ConversationService
    from services import text_embedding
    from config.identity import Identity
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    monkeypatch.setattr(text_embedding, "create_embedding_model", lambda: type(
        "Embedding", (), {"embed_query": lambda self, text: [1.0, 0.0, 0.0]})())
    monkeypatch.setattr(task_classification_agent, "create_chat_model",
                        lambda **kwargs: FakeListChatModel(responses=["consultation"]))
    service = ConversationService(db_path=f"sqlite:///{tmp_path / 'stream.db'}")
    monkeypatch.setattr(chat_handler, "_conversation_service", service)
    return chat_handler, service, consultant_agent, Identity, task_classification_agent


@pytest.mark.parametrize("orch", ["legacy"])
def test_cold_first_body(tmp_path, monkeypatch, orch):
    monkeypatch.setenv("ORCHESTRATOR", orch)
    from test_consultation_real_stream import BarrierModel, FIRST
    from langchain_core.messages import AIMessage

    class M(BarrierModel):
        async def ainvoke(self, messages, **kwargs):
            self.invoke_calls.append(messages)
            return AIMessage(content="YES")

    chat_handler, service, cagent, Identity, tca = _make(tmp_path, monkeypatch)
    model = M(blocked=True)
    sid = "real-stream"
    identity = service.ensure_session(Identity("owner-a", student_id="child-a", session_id=sid))
    history = service.restore_history(identity)
    cagent.create_chat_model = lambda **kwargs: model
    consultant = cagent.ConsultantAgent(
        session_id=sid, identity=identity, conversation_service=service,
        session_chat_history=history)
    agent = tca.TaskClassificationAgent(
        consultant_agent=consultant, session_id=sid,
        session_chat_history=history, context_manager=consultant.context_manager)
    agent.identity = identity
    chat_handler._session_agents[sid] = agent

    async def run():
        t0 = time.monotonic()
        stream = chat_handler.ProcessUserInput_stream("初二数学怎么学", identity=identity)
        try:
            while True:
                tok = await asyncio.wait_for(anext(stream), 30)
                dt = time.monotonic() - t0
                print(f"[{orch}] +{dt:.2f}s {tok!r}", flush=True)
                if FIRST in tok:
                    print(f"[{orch}] FIRST BODY at {dt:.2f}s", flush=True)
                    return dt
        except StopAsyncIteration:
            print(f"[{orch}] STOP", flush=True)
            return None
        finally:
            model.release.set()
            await stream.aclose()
    dt = asyncio.run(run())
    print(f"[{orch}] RESULT {dt}", flush=True)
