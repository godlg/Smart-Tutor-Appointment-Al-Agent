"""
LLM JSON 布尔字段字符串陷阱 + unrelated 递归防护 + chat_handler wiring 测试

背景:LLM(deepseek-chat)字段提取常输出 `"unrelated": "false"`(带引号字符串),
json.loads 后得到字符串 "false",而 Python 中非空字符串恒为真 bool("false")==True:
- trial/lesson 把"无关"误判成真 → 踢回主路由 → 主路由重分类仍命中本站 → 无限递归
- recommendation 把"信息不齐全"误判成"齐全" → 跳过追问、拿"未知"直接打分推荐

修复(本文件覆盖):
1. trial/lesson/recommendation 三处 parse 入口对 unrelated/info_complete 做字符串→真布尔归一化
2. TrialAgent / LessonAgent 的 unrelated 踢回加递归防护(连续两次命中则直接兜底)
3. api/chat_handler._build_task_agent 只给接收 session_chat_history 的 Agent 传参,服务能起

异步用例统一用 _run(_collect) 同步包装,不依赖 pytest-asyncio(本仓库未安装)。
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.trial.trial_processor import TrialInputParser                   # noqa: E402
from agents.lesson.lesson_processor import LessonInputParser                # noqa: E402
from agents.recommendation_agent import RecommendationInputParser            # noqa: E402
from agents.trial.trial_agent import TrialAgent                             # noqa: E402


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


def _new_parser(cls):
    """跳过 __init__(避免创建真实 LLM),直接构造只用得到 parse_data 的解析器。"""
    return object.__new__(cls)


# ----------------------------------------------------------------------
# Fix 1 — 布尔归一化
# ----------------------------------------------------------------------

class TestTrialParseDataCoercion:
    """TrialInputParser.parse_data 对 unrelated / info_complete 的归一化。"""

    def test_false_strings_become_false(self):
        parser = _new_parser(TrialInputParser)
        result = parser.parse_data(
            '{"subject": "数学", "grade_level": "初中", "start_time": "2026-09-12 19:00",'
            ' "unrelated": "false", "info_complete": "false"}'
        )
        assert result["unrelated"] is False
        assert result["info_complete"] is False

    def test_true_strings_become_true(self):
        parser = _new_parser(TrialInputParser)
        result = parser.parse_data(
            '{"subject": "数学", "grade_level": "初中", "start_time": "2026-09-12 19:00",'
            ' "unrelated": "true", "info_complete": "true"}'
        )
        assert result["unrelated"] is True
        assert result["info_complete"] is True

    def test_real_booleans_are_idempotent(self):
        parser = _new_parser(TrialInputParser)
        result = parser.parse_data(
            '{"subject": "数学", "unrelated": false, "info_complete": false}'
        )
        assert result["unrelated"] is False
        assert result["info_complete"] is False

    def test_missing_keys_not_added(self):
        parser = _new_parser(TrialInputParser)
        result = parser.parse_data('{"subject": "数学"}')
        assert "unrelated" not in result
        assert "info_complete" not in result

    def test_invalid_json_keeps_real_bool_fallbacks(self):
        parser = _new_parser(TrialInputParser)
        result = parser.parse_data("not json at all")
        assert result["unrelated"] is False
        assert result["info_complete"] is False


class TestLessonParseDataCoercion:
    """LessonInputParser.parse_data 对 unrelated / info_complete 的归一化。"""

    def test_false_strings_become_false(self):
        parser = _new_parser(LessonInputParser)
        result = parser.parse_data(
            '{"subject": "数学", "grade_level": "初中", "start_time": "2026-09-12 19:00",'
            ' "unrelated": "false", "info_complete": "false"}'
        )
        assert result["unrelated"] is False
        assert result["info_complete"] is False

    def test_true_strings_become_true(self):
        parser = _new_parser(LessonInputParser)
        result = parser.parse_data(
            '{"subject": "数学", "unrelated": "true", "info_complete": "true"}'
        )
        assert result["unrelated"] is True
        assert result["info_complete"] is True

    def test_real_booleans_are_idempotent(self):
        parser = _new_parser(LessonInputParser)
        result = parser.parse_data(
            '{"subject": "数学", "unrelated": false, "info_complete": false}'
        )
        assert result["unrelated"] is False
        assert result["info_complete"] is False

    def test_true_string_variants(self):
        parser = _new_parser(LessonInputParser)
        result = parser.parse_data('{"unrelated": "TRUE", "info_complete": " True "}')
        assert result["unrelated"] is True
        assert result["info_complete"] is True


class TestRecommendationExtractCoercion:
    """RecommendationInputParser.extract 对 info_complete 的归一化。"""

    def _parser_for(self, json_content):
        parser = _new_parser(RecommendationInputParser)
        parser.chain = MagicMock()
        msg = MagicMock()
        msg.content = json_content
        parser.chain.invoke.return_value = msg
        return parser

    def test_false_string_becomes_false(self):
        parser = self._parser_for(
            '{"subject": "数学", "grade_level": "初中",'
            ' "info_complete": "false", "missing_info": []}'
        )
        result = parser.extract("初一数学")
        assert result["info_complete"] is False

    def test_true_string_becomes_true(self):
        parser = self._parser_for(
            '{"subject": "数学", "grade_level": "初中",'
            ' "info_complete": "true", "missing_info": []}'
        )
        result = parser.extract("初一数学")
        assert result["info_complete"] is True

    def test_real_false_is_idempotent(self):
        parser = self._parser_for(
            '{"subject": "数学", "grade_level": "初中",'
            ' "info_complete": false, "missing_info": []}'
        )
        result = parser.extract("初一数学")
        assert result["info_complete"] is False

    def test_invalid_json_keeps_false_fallback(self):
        parser = self._parser_for("boom")
        parser.chain.invoke.side_effect = ValueError("bad")
        result = parser.extract("初一数学")
        assert result["info_complete"] is False


# ----------------------------------------------------------------------
# Fix 2 — TrialAgent 递归防护
# ----------------------------------------------------------------------

def _make_trial_agent(parse_data_result, callback, calls):
    """绕过 __init__(避免创建真实 LLM),直接构造最小可用 TrialAgent。

    仅填充 _run_stream_impl 用到的属性。handle_unrelated_request 用真实异步生成器
    模拟:回调非 None 时会调用它(记入 calls),从而能断言"二次命中不再调 callback"。
    """
    agent = object.__new__(TrialAgent)
    agent.processor = MagicMock()
    agent.unrelated_callback = callback
    agent.state = None
    agent.trial_history = {
        "subject": None, "grade_level": None, "start_time": None,
        "duration_minutes": "30", "student_situation": None,
        "preferred_teacher_name": None, "preferred_style": None,
        # 新增递归防护计数必须显式初始化,否则 __new__ 缺属性
        "unrelated_escalations": 0,
    }
    agent.chat_history = None
    agent.session_id = "test-session"

    agent.processor.input_parser.parse_stream.return_value = ['{"unrelated": true}']
    agent.processor.input_parser.parse_data.return_value = parse_data_result
    agent.processor.update_history_from_data.return_value = False

    async def _unrelated_gen(user_input, unrelated_callback, state):
        if unrelated_callback is not None:
            calls.append(user_input)
        yield "[UNRELATED]"

    async def _incomplete_gen(data, trial_history):
        yield "[INCOMPLETE]"

    agent.processor.handle_unrelated_request = MagicMock(side_effect=_unrelated_gen)
    agent.processor.handle_incomplete_info = MagicMock(side_effect=_incomplete_gen)
    return agent


class TestTrialRecursionGuard:
    """同一实例连续两次 unrelated=true 时,第二次不再调 callback、不死循环。"""

    @staticmethod
    def _unrelated_entry(subject="未知"):
        return {
            "subject": subject, "grade_level": "未知", "start_time": "未知",
            "duration_minutes": "30", "student_situation": "",
            "preferred_teacher_name": "未知", "preferred_style": "未知",
            "confirmation": "未知", "info_complete": False, "unrelated": True,
            "missing_info": ["所有信息"],
        }

    def test_callback_called_once_then_fallback(self):
        calls = []
        agent = _make_trial_agent(self._unrelated_entry(), callback=None, calls=calls)

        # 第一次:踢回主路由,should_be 调用 callback(模拟主路由再路由回本站)
        out1 = _collect(agent.run_stream(user_input="今天天气怎么样"))
        assert "[UNRELATED]" in out1
        assert agent.trial_history["unrelated_escalations"] == 1

        # 第二次:不再调 callback(不触发处理函数),直接兜底,不死循环不抛异常
        out2 = _collect(agent.run_stream(user_input="今天天气怎么样"))
        assert "[UNRELATED]" in out2
        # 兜底走了 handle_unrelated_request(以 None 作回调),整个过程无异常
        assert out2  # 有输出
        # 计数被重置,为下一次重新计数
        assert agent.trial_history["unrelated_escalations"] == 0

    def test_real_callback_only_invoked_once_after_escalation(self):
        """mock unrelated_callback 为真实异步函数,确认只被调用一次。"""
        calls = []
        async def _callback(user_input):
            calls.append(user_input)
        agent = _make_trial_agent(self._unrelated_entry(), callback=_callback, calls=calls)

        _collect(agent.run_stream(user_input="今天天气怎么样"))
        assert len(calls) == 1   # 第一次调了 callback

        _collect(agent.run_stream(user_input="今天天气怎么样"))
        assert len(calls) == 1   # 第二次不再调 callback → 递归被切断


# ----------------------------------------------------------------------
# Fix 3 — chat_handler wiring
# ----------------------------------------------------------------------

class TestBuildTaskAgentWiring:
    """_build_task_agent 不再给不接收 session_chat_history 的 Agent 传参,构造不炸。

    必须走 get_task_agent(持 _session_lock)而非 _build_task_agent 直调:
    get_task_agent 持锁期间会再进 get_session_chat_history(同锁),
    普通 Lock 会同一线程自锁死(已用 RLock 修复),直调 _build_task_agent
    绕过了锁,测不出该回归。
    """

    def test_build_task_agent_does_not_raise(self):
        from api import chat_handler
        session_id = "test-bool-coercion-wiring"
        try:
            agent = chat_handler.get_task_agent(session_id)
            assert agent is not None
        finally:
            # 显式释放会话 agent,避免测试间互相污染
            chat_handler.drop_session_agent(session_id)