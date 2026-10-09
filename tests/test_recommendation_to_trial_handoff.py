"""
推荐确认 → 试讲 跨 Agent 上下文平滑交接测试

背景:推荐确认(『是』)后,家长回复『试讲』时,TrialAgent 拿到的输入只有
 "试讲" 一词,上游的学科/年级/老师上下文全丢,导致 TrialInputParser 提取出
 subject="未知" 进而 LLM 误判 unrelated=true,被踢回主路由(死循环风险)。

修复:
1. RecommendationAgent 确认『是』时,引导消息带上学科/年级/老师ID,
   让下游 TrialAgent 从"消息携带"拿到完整字段(非实例引用,避免反向依赖)。
2. TrialInputParser 的 prompt 新增 {context} 变量,parse_stream 接收 context,
   并把上一轮 AI 输出自动注入,使 LLM 提取时能看到学科/年级。

本文件:
- test_confirmation_message_includes_subject_grade :『是』→ yield 含学科/年级/老师ID
- test_trial_parser_prompt_contains_context         :prompt 含 {context} 变量(静态断言)
- test_trial_parser_handles_context_with_subject    :context 注入 → subject/grade 不再为"未知"

异步用例统一用 _run(_collect) 同步包装,不依赖 pytest-asyncio(本仓库未安装)。
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.chat_history import InMemoryChatMessageHistory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.recommendation_agent import RecommendationAgent  # noqa: E402
from agents.trial.trial_processor import TrialInputParser   # noqa: E402


def _run(coro):
    """同步执行协程/异步生成器"""
    return asyncio.run(coro)


async def _drain(agen):
    """把异步生成器收集成字符串"""
    out = []
    async for token in agen:
        out.append(token)
    return "".join(out)


def _collect(agen):
    """_run + _drain 的快捷封装"""
    return _run(_drain(agen))


# ----------------------------------------------------------------------
# 固定会话数据:一次完整推荐(学科=数学,年级=初中,进入确认态)
# ----------------------------------------------------------------------

_RECS = [
    {"teacher": {"id": 7, "name": "周敏"}, "score": 92, "reason": "擅长初二数学"},
    {"teacher": {"id": 9, "name": "张明"}, "score": 80, "reason": "覆盖初中"},
    {"teacher": {"id": 11, "name": "高远"}, "score": 70, "reason": "耐心细致"},
]
_COMPLETE_INPUT = {
    "subject": "数学", "grade_level": "初中", "preferred_style": "温和",
    "preferred_teacher_name": "未知", "mode": "未知",
    "info_complete": True, "missing_info": [],
}


def _make_confirm_agent():
    """绕过 __init__(避免真实 LLM),构造一个可跑确认状态机的 RecommendationAgent。"""
    agent = RecommendationAgent.__new__(RecommendationAgent)
    agent.session_id = "s-handoff"
    agent.chats_by_session_id = {}
    agent.history = {}
    agent.message_builder = RecommendationAgent(session_id="b").message_builder
    agent.input_parser = MagicMock()
    agent._recommendation_service = MagicMock()
    agent._student_service = MagicMock()
    return agent


def _recommend(agent, sid="s-handoff"):
    """跑一轮完整推荐:字段全 → 打分 → 存 last_recommendations + subject/grade,进入确认态。"""
    agent.input_parser.extract.return_value = _COMPLETE_INPUT
    agent._recommendation_service.recommend_teachers.return_value = _RECS
    agent._student_service.initialize_default_student.return_value = {"id": 1}
    _collect(agent.run_stream("初二数学", session_id=sid))


class TestConfirmationMessageCarriesContext:
    """推荐确认『是』时,引导消息必须带上学科/年级/老师ID(下游试讲提取用)。"""

    def test_confirmation_message_includes_subject_grade(self):
        agent = _make_confirm_agent()
        _recommend(agent)
        # 先选定老师
        _collect(agent.run_stream("周敏", session_id="s-handoff"))
        # 再确认『是』
        out = _collect(agent.run_stream("是", session_id="s-handoff"))

        assert "已为您确认 周敏 老师" in out
        assert "学科: 数学" in out
        assert "年级: 初中" in out
        # 老师 ID 一并带上,下游能精确定位老师
        assert "老师ID: 7" in out
        # 确认完成后清空确认态,避免下一轮误进
        hist = agent.history["s-handoff"]
        assert hist["confirmed_teacher"] is None
        assert hist["awaiting_confirmation"] is False


class TestTrialParserContextPathVariable:
    """TrialInputParser.prompt 必须具备 {context} 变量(静态断言)。"""

    def test_trial_parser_prompt_contains_context(self):
        parser = TrialInputParser(MagicMock(), context_manager=None)
        assert "context" in parser.prompt.input_variables
        assert "{context}" in parser.prompt.template


class TestTrialParserContextHandlesSubject:
    """注入 context(上游已确认的学科/年级)后,LLM 提取的 subject/grade 不再为"未知"。"""

    def test_trial_parser_handles_context_with_subject(self):
        llm = MagicMock()
        parser = TrialInputParser(llm, context_manager=None)
        chat_history = InMemoryChatMessageHistory()

        captured = {}

        def fake_stream(input_dict):
            # 拦截 chain 输入,断言 context 透传;并按上下文返回带学科/年级的 JSON
            captured.update(input_dict)
            return iter([
                '{"subject":"数学","grade_level":"初中","start_time":"未知",'
                '"unrelated":false,"info_complete":false,"missing_info":["start_time"]}',
            ])

        parser.chain = MagicMock()
        parser.chain.stream.side_effect = fake_stream

        tokens = list(parser.parse_stream(
            "试讲", chat_history, context="周敏老师(数学,初中)"
        ))
        data = parser.parse_data("".join(tokens))

        # context 原样进了 chain 输入,而非丢弃
        assert captured.get("context") == "周敏老师(数学,初中)"
        # 语义:context 让 subject/grade 不再是"未知",从而避免误判 unrelated
        assert data["subject"] == "数学"
        assert data["subject"] != "未知"
        assert data["grade_level"] == "初中"
        assert data["grade_level"] != "未知"
        # 无 context 时保持原行为(不影响已有链路)
        assert captured.get("history", "").count("用户：试讲") >= 1