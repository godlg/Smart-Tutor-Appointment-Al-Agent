"""咨询真实模型流：事件屏障证明正文早于完成，禁止 ainvoke 伪流式。"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from agents.consultant.response_generator import ResponseGenerator
from agents.consultant.consultation_processor import ConsultationProcessor

PREFIX = "[REPLY][咨询机器人]"
FIRST = "先理解概念。"
LAST = "再整理错题。"


class BarrierModel:
    """只在正文第一块后等待放行；ainvoke 仅允许分类/非流式测试。"""

    def __init__(self, chunks=None, *, blocked=False, fail=None):
        self.chunks = chunks if chunks is not None else [FIRST, LAST]
        self.release = asyncio.Event()
        if not blocked:
            self.release.set()
        self.waiting = asyncio.Event()
        self.closed = asyncio.Event()
        self.done = False
        self.fail = fail
        self.stream_calls = 0
        self.invoke_calls = []
        self.identities = []
        self.prompts = []

    async def ainvoke(self, messages, **kwargs):
        self.invoke_calls.append(messages)
        raise AssertionError("answer generation must use astream")

    async def astream(self, messages, **kwargs):
        from config.identity import get_current_identity
        self.stream_calls += 1
        self.prompts.append(messages)
        try:
            for index, content in enumerate(self.chunks):
                if self.fail == index:
                    raise RuntimeError("SECRET_RAW_MODEL_ERROR")
                self.identities.append(get_current_identity())
                yield content if hasattr(content, "content") else SimpleNamespace(content=content)
                if index == 0:
                    self.waiting.set()
                    await self.release.wait()
            if self.fail == len(self.chunks):
                raise RuntimeError("SECRET_RAW_MODEL_ERROR")
            self.done = True
        finally:
            self.identities.append(get_current_identity())
            self.closed.set()


async def collect(stream):
    return "".join([part async for part in stream])


def processor(model):
    retriever = SimpleNamespace(search_knowledge=AsyncMock(return_value=[]))
    proc = ConsultationProcessor(retriever, None, ResponseGenerator(model))
    proc._record_consultation_behavior = AsyncMock()
    return proc


def test_first_body_arrives_before_model_is_released():
    async def scenario():
        model = BarrierModel(blocked=True)
        stream = ResponseGenerator(model).generate_response_stream("数学怎么学", [])
        try:
            assert await anext(stream) == PREFIX
            assert await anext(stream) == FIRST
            assert not model.done and not model.release.is_set()
            tail = asyncio.create_task(collect(stream))
            await asyncio.wait_for(model.waiting.wait(), 2)
            assert not tail.done()
            model.release.set()
            assert await tail == LAST
            assert model.done and model.closed.is_set()
            assert model.stream_calls == 1 and model.invoke_calls == []
        finally:
            model.release.set()
            await stream.aclose()
    asyncio.run(scenario())


@pytest.mark.parametrize("chunks,expected", [
    (["甲", "乙"], "甲乙"),
    ([AIMessageChunk(content="甲", additional_kwargs={"reasoning_content": "PRIVATE"},
                     response_metadata={"private": "PRIVATE"}), "乙"], "甲乙"),
    ([[{"type": "text", "text": "甲"}, {"type": "reasoning", "text": "PRIVATE"},
       {"type": "tool_call", "text": "PRIVATE"}, {"type": "text", "text": "乙"}]], "甲乙"),
    ([None, "", [], [{"type": "text", "text": None}, {"type": "image", "text": "PRIVATE"}],
      {"text": "PRIVATE"}, "正文"], "正文"),
    ([["PRIVATE", {"text": "PRIVATE"}, {"type": "text", "text": 123}], "正文"], "正文"),
])
def test_visible_content_only_and_one_protocol_prefix(chunks, expected):
    model = BarrierModel(chunks)
    result = asyncio.run(collect(ResponseGenerator(model).generate_response_stream("Q", [])))
    assert result == PREFIX + expected
    assert result.count(PREFIX) == 1
    assert model.invoke_calls == []


def test_normal_completion_records_consultation_exactly_once():
    model = BarrierModel()
    proc = processor(model)
    result = asyncio.run(collect(proc.process_consultation_stream("Q", "s")))
    assert result == PREFIX + FIRST + LAST
    proc._record_consultation_behavior.assert_awaited_once_with("Q", [], "s")


@pytest.mark.parametrize("fail", [0, 1, 2])
def test_model_failure_is_explicit_preserves_sent_body_and_never_records_success(fail):
    model = BarrierModel(fail=fail)
    proc = processor(model)
    result = asyncio.run(collect(proc.process_consultation_stream("Q", "s")))
    assert result.count(PREFIX) == 1
    assert result.count("[ERROR]") == 1
    assert "SECRET_RAW_MODEL_ERROR" not in result
    assert "请稍后重试" in result
    assert FIRST in result if fail >= 1 else FIRST not in result
    assert LAST in result if fail >= 2 else LAST not in result
    proc._record_consultation_behavior.assert_not_awaited()
    assert model.closed.is_set()


@pytest.mark.parametrize("chunks", [[], [None, "", []],
                                    [[{"type": "reasoning", "text": "PRIVATE"}]]])
def test_empty_output_is_failure_not_success(chunks):
    proc = processor(BarrierModel(chunks))
    result = asyncio.run(collect(proc.process_consultation_stream("Q", "s")))
    assert result.count(PREFIX) == 1 and result.count("[ERROR]") == 1
    assert "PRIVATE" not in result
    proc._record_consultation_behavior.assert_not_awaited()


def test_nonstreaming_still_uses_ainvoke():
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content="原非流式回答")),
                            astream=lambda *args: pytest.fail("nonstreaming called astream"))
    result = asyncio.run(ResponseGenerator(model).generate_response("Q", []))
    assert result == "原非流式回答"
    model.ainvoke.assert_awaited_once()


def test_budget_refusal_still_skips_model_and_success_behavior():
    from agents.context.types import ContextBudgetExceeded
    model = BarrierModel()
    proc = processor(model)
    proc.response_generator._build_prompt = lambda *args: (_ for _ in ()).throw(
        ContextBudgetExceeded("budget"))
    result = asyncio.run(collect(proc.process_consultation_stream("Q", "s")))
    assert result == PREFIX + "问题或必需上下文过长，请缩短输入后重试。"
    assert model.stream_calls == 0 and model.invoke_calls == []
    proc._record_consultation_behavior.assert_not_awaited()
