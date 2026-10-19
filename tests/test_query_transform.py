"""Query 变换(多查询检索 + 改写, 三档 off/rule/llm)单元测试。

覆盖契约(逐条守住):
1. off → [query]; 2. rule 术语归一命中; 3. rule 无命中 → [query]
4. 指代补全产年级学科变体; 5. 上下文抽不到 → 不产(不许编)
6. 原 query 恒在 [0]; 7. 去重; 8. 上限 max_variants 生效
9. llm 正常(桩 LLM 返 JSON 数组); 10. llm 超时回退 rule; 11. llm 抛异常回退
12. llm 返回垃圾文本回退; 13. 变体过长 / 等于原 query 被丢弃; 14. 非法 mode 回退 rule

零网络 / 零 LLM:LLM 档一律用桩对象, 不触发 create_chat_model。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402
from services import query_transform as qt  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset_transform_settings(monkeypatch):
    """每个用例回到默认档, 避免用例间互相污染(settings 是模块级单例)。"""
    monkeypatch.setattr(settings, "query_transform_mode", "rule")
    monkeypatch.setattr(settings, "query_transform_timeout", 3.0)
    monkeypatch.setattr(settings, "query_transform_max_variants", 3)


def _assert_contract(result: List[str], query: str, max_variants: int = 3) -> None:
    """契约断言: 非空 / [0]==query / 无重复 / 不超上限。"""
    assert result, "返回值必须非空"
    assert result[0] == query, "原始 query 必须恒在 [0]"
    assert len(result) == len(set(result)), f"不应有重复元素: {result}"
    assert len(result) <= max_variants, f"超出上限: {result}"


class _StubLLM:
    """桩 LLM: 记录收到的 prompt, 返回固定文本(或抛异常)。"""

    def __init__(self, reply: Any = "[]", exc: Exception | None = None,
                 delay: float = 0.0) -> None:
        self.reply = reply
        self.exc = exc
        self.delay = delay
        self.prompts: List[str] = []

    async def ainvoke(self, prompt: str):
        self.prompts.append(prompt)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc is not None:
            raise self.exc
        return self.reply


class _Msg:
    """模拟 langchain AIMessage(.content 字段)。"""

    def __init__(self, content: str) -> None:
        self.content = content


# ----------------------------------------------------------------------
# 同步档: off / rule
# ----------------------------------------------------------------------

class TestOffMode:
    def test_off_returns_query_only(self):
        """1. mode=off → [query], 不产生任何变体。"""
        result = qt.variants("试听多少钱", mode="off")
        assert result == ["试听多少钱"]


class TestRuleMode:
    def test_term_normalization_hit(self):
        """2. 术语归一命中 → 含规范词变体, 原串仍在 [0]。"""
        query = "试听一节课多少钱"
        result = qt.variants(query, mode="rule")
        _assert_contract(result, query)
        joined = " ".join(result[1:])
        assert "试讲" in joined, f"应产出「试讲」变体: {result}"
        assert "收费标准 价格" in joined, f"应产出「收费标准 价格」变体: {result}"

    def test_no_hit_returns_query_only(self):
        """3. 无规则命中 → [query], 不硬造变体。"""
        query = "高中的课程安排是怎样的"
        assert qt.variants(query, mode="rule") == [query]

    def test_pronoun_completion_uses_context(self):
        """4. 指代补全: context 有「初二/数学」→ 产「年级学科：query」变体。"""
        query = "这个怎么样"
        result = qt.variants(
            query, context_text="用户: 我家孩子初二, 数学不太好", mode="rule",
        )
        _assert_contract(result, query)
        assert "初中数学：这个怎么样" in result, result

    def test_pronoun_without_extractable_context_produces_nothing(self):
        """5. context 抽不到学科/年级 → 不产变体(不许编)。"""
        query = "这个怎么样"
        result = qt.variants(query, context_text="用户: 随便聊聊", mode="rule")
        assert result == [query]

    def test_pronoun_with_only_grade_produces_nothing(self):
        """5b. 只抽到年级、抽不到学科 → 仍不产(两个字段都要有)。"""
        query = "那个课程咋样"
        result = qt.variants(query, context_text="用户: 我家孩子初二", mode="rule")
        assert result == [query]

    def test_original_query_always_first(self):
        """6. 原 query 恒在 [0](术语归一 + 指代补全都命中时也一样)。"""
        query = "试听这个怎么样"
        result = qt.variants(
            query, context_text="初二 数学 需要补课", mode="rule",
        )
        _assert_contract(result, query)

    def test_no_duplicates(self):
        """7. 去重: 重复命中的同一变体只出现一次。"""
        query = "试听试课体验课"
        result = qt.variants(query, mode="rule")
        _assert_contract(result, query)
        assert result.count(result[-1]) == 1

    def test_max_variants_cap(self, monkeypatch):
        """8. 上限生效: max_variants=2 时最多 2 条(含原串)。"""
        monkeypatch.setattr(settings, "query_transform_max_variants", 2)
        query = "试听这个怎么样"
        result = qt.variants(
            query, context_text="初二 数学 需要补课", mode="rule",
        )
        assert len(result) == 2
        _assert_contract(result, query, max_variants=2)

    def test_more_variants_when_cap_raised(self):
        """8b. 上限放开到 3 时, 术语 + 指代两条变体都能进。"""
        query = "试听这个怎么样"
        result = qt.variants(
            query, context_text="初二 数学 需要补课", mode="rule",
        )
        assert len(result) == 3
        _assert_contract(result, query)


# ----------------------------------------------------------------------
# 异步 LLM 档
# ----------------------------------------------------------------------

class TestLLMMode:
    def test_llm_variants_merged_and_query_first(self, monkeypatch):
        """9. llm 正常(桩返 JSON 数组)→ 变体进列表且原 query 仍在 [0]。

        上限调到 3 且 query 不走 rule(无口语词/指代词), 保证两条 LLM 变体都能进。
        """
        monkeypatch.setattr(settings, "query_transform_max_variants", 3)
        llm = _StubLLM(reply='["初二数学 一对一 收费标准", "初中数学 提分方案"]')
        result = _run(qt.variants_async(
            "课程安排", mode="llm", llm=llm, timeout=1.0,
        ))
        _assert_contract(result, "课程安排")
        assert "初二数学 一对一 收费标准" in result
        assert "初中数学 提分方案" in result
        assert llm.prompts, "应真的调用了 LLM"

    def test_rule_variants_take_priority_over_llm(self):
        """9d. 合并顺序: rule 变体排在 LLM 变体之前, 原串恒第 0 位。"""
        llm = _StubLLM(reply='["LLM 变体"]')
        result = _run(qt.variants_async(
            "补课多少钱", mode="llm", llm=llm, timeout=1.0,
        ))
        _assert_contract(result, "补课多少钱")
        assert result[0] == "补课多少钱"
        assert result[1] == "辅导收费标准 价格"  # rule 变体优先占位

    def test_llm_timeout_falls_back_to_rule(self):
        """10. llm 超时 → 回退 rule 结果, 不抛异常。"""
        rule_expected = qt.variants("试听多少钱", mode="rule")
        llm = _StubLLM(reply='["不该出现"]', delay=0.5)
        result = _run(qt.variants_async(
            "试听多少钱", mode="llm", llm=llm, timeout=0.05,
        ))
        _assert_contract(result, "试听多少钱")
        assert result == rule_expected
        assert "不该出现" not in result

    def test_llm_exception_falls_back_to_rule(self):
        """11. llm 抛异常 → 回退 rule 结果, 不抛异常。"""
        rule_expected = qt.variants("试听多少钱", mode="rule")
        llm = _StubLLM(exc=RuntimeError("LLM 挂了"))
        result = _run(qt.variants_async(
            "试听多少钱", mode="llm", llm=llm, timeout=1.0,
        ))
        _assert_contract(result, "试听多少钱")
        assert result == rule_expected

    def test_llm_garbage_text_falls_back_to_rule(self):
        """12. llm 返回不可解析文本 → 回退 rule 结果, 不抛异常。"""
        rule_expected = qt.variants("试听多少钱", mode="rule")
        for garbage in ("抱歉我不知道", "[不是 json", '{"k": "v"}', "[]", '[""]'):
            llm = _StubLLM(reply=garbage)
            result = _run(qt.variants_async(
                "试听多少钱", mode="llm", llm=llm, timeout=1.0,
            ))
            _assert_contract(result, "试听多少钱")
            assert result == rule_expected, f"garbage={garbage!r} → {result}"

    def test_llm_overlong_and_identical_variants_dropped(self):
        """13. 变体过长(>60 字)/ 等于原 query → 被丢弃。"""
        long_variant = "长" * 61
        query = "试听多少钱"
        llm = _StubLLM(reply=(
            '["' + long_variant + '", "' + query + '", "合法变体", 123, ""]'
        ))
        result = _run(qt.variants_async(
            query, mode="llm", llm=llm, timeout=1.0,
        ))
        _assert_contract(result, query)
        assert long_variant not in result
        assert "合法变体" in result
        assert result.count(query) == 1

    def test_llm_message_object_content_parsed(self):
        """9b. 兼容 langchain AIMessage(.content)响应体。"""
        llm = _StubLLM(reply=_Msg('```json\n["初一数学 补课方案"]\n```'))
        result = _run(qt.variants_async(
            "补课", mode="llm", llm=llm, timeout=1.0,
        ))
        _assert_contract(result, "补课")
        assert "初一数学 补课方案" in result

    def test_llm_mode_never_exceeds_cap(self, monkeypatch):
        """9c. LLM 档同样受 max_variants 上限约束。"""
        monkeypatch.setattr(settings, "query_transform_max_variants", 2)
        llm = _StubLLM(reply='["变体一", "变体二"]')
        result = _run(qt.variants_async(
            "补课多少钱", mode="llm", llm=llm, timeout=1.0,
        ))
        assert len(result) == 2
        _assert_contract(result, "补课多少钱", max_variants=2)


# ----------------------------------------------------------------------
# mode 解析
# ----------------------------------------------------------------------

class TestModeResolution:
    def test_illegal_mode_falls_back_to_rule(self):
        """14. 非法 mode → 回退 rule 且不影响返回契约。"""
        rule_expected = qt.variants("试听多少钱", mode="rule")
        result = qt.variants("试听多少钱", mode="bogus-mode")
        _assert_contract(result, "试听多少钱")
        assert result == rule_expected

    def test_illegal_mode_via_settings_falls_back_to_rule(self, monkeypatch):
        """14b. settings 里配了非法值时同样回退 rule。"""
        monkeypatch.setattr(settings, "query_transform_mode", "LLM-typo")
        assert qt.variants("试听多少钱") == qt.variants("试听多少钱", mode="rule")

    def test_mode_none_reads_settings(self, monkeypatch):
        """mode 未显式传 → 读 settings(off 档同样只有原串)。"""
        monkeypatch.setattr(settings, "query_transform_mode", "off")
        assert qt.variants("试听多少钱") == ["试听多少钱"]

    def test_async_llm_mode_falls_back_when_llm_unavailable(self, monkeypatch):
        """llm=None 且工厂抛错 → 回退 rule, 不抛异常。"""
        def _boom(**kwargs):
            raise RuntimeError("create_chat_model 不可用")

        monkeypatch.setattr("config.model_provider.create_chat_model", _boom)
        result = _run(qt.variants_async("试听多少钱", mode="llm"))
        _assert_contract(result, "试听多少钱")
        assert result == qt.variants("试听多少钱", mode="rule")

    def test_max_variants_floor_keeps_query_only(self, monkeypatch):
        """max_variants 配成 0/负值 → 至少保留原串一条, 不产变体。"""
        for value in (0, -1):
            monkeypatch.setattr(settings, "query_transform_max_variants", value)
            assert qt.variants("试听多少钱", mode="rule") == ["试听多少钱"]
