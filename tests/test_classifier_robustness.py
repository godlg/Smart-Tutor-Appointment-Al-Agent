"""
分类器输出解析健壮性测试。

覆盖修复：TaskClassifier.classify_task 对 LLM 输出的脏格式容错。
旧实现只做 strip().lower() 后精确比对，LLM 输出 "reschedule。" / '"cancel"' /
"reschedule - 调课" 等带尾巴时会被归一成 unrelated → 导致随机拒答。

注意：本仓库未安装 pytest-asyncio，异步协程统一用同步包装 `_run` 驱动
（与本仓库 test_reschedule_routing.py 的约定一致）。
"""
import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from agents.task_classification.task_classifier import TaskClassifier


def _run(coro):
    return asyncio.run(coro)


def _clf(content):
    """构造一个不触发 LCEL 链构建的分类器，仅替换分类结果。"""
    clf = TaskClassifier.__new__(TaskClassifier)
    clf.context_manager = None  # 可空注入契约:未注入时走旧 PromptTemplate 链
    msg = Mock()
    msg.content = content
    clf.chain = Mock()
    clf.chain.ainvoke = AsyncMock(return_value=msg)
    return clf


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("reschedule", "reschedule"),
        ("reschedule。", "reschedule"),                       # 中文句号尾巴
        ('"cancel"', "cancel"),                               # 引号包裹
        ("'trial'", "trial"),
        ("  consultation  ", "consultation"),                 # 首尾空白
        ("Reschedule", "reschedule"),                         # 大写
        ("reschedule - 调课(调整已有课时时间)", "reschedule"),  # 子串+说明
        ("6. lesson", "lesson"),                              # 编号前缀
        ("progress", "progress"),
        ("unrelated", "unrelated"),
        ("完全无关的一句话", "unrelated"),                     # 无合法词 → 兜底
        ("", "unrelated"),                                    # 空输出
        ("```cancel```", "cancel"),                           # markdown 代码块
    ],
)
def test_classify_robustness(raw, expected):
    clf = _clf(raw)
    assert _run(clf.classify_task("随便")) == expected
