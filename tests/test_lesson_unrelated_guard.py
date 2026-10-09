"""
LessonAgent unrelated 误判防御测试

背景:输入"代数，一周三次，无偏好"时,LLM 提取出的 subject="代数"(不在枚举
 且非"未知"),但可能把 unrelated=true;若此时走 handle_unrelated_request 踢回
 主路由,主路由再分类又落到 lesson,造成死循环。

修复:run_stream 里一旦 subject 提取出非空且非"未知"的值(含"代数"这类口语化
 别名),即使 unrelated=true 也强制跳过踢回、走 handle_incomplete_info 追问。

本文件:
- subject=数学 + unrelated=true  → 不走踢回,走 handle_incomplete_info
- subject=未知 + unrelated=true  → 仍走踢回(保留原行为)

异步用例统一用 _run(_collect) 同步包装,不依赖 pytest-asyncio(本仓库未安装)。
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.lesson.lesson_agent import LessonAgent  # noqa: E402


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


def _make_agent(parse_data_result):
    """绕过 __init__(避免创建真实 LLM / 触碰 DB),直接构造最小可用 LessonAgent。

    仅填充 run_stream 用到的属性:processor(unrelated_callback/state/lesson_history/
    chat_history)。
    """
    agent = object.__new__(LessonAgent)
    agent.processor = MagicMock()
    agent.unrelated_callback = None
    agent.state = None
    agent.lesson_history = {
        "subject": None,
        "grade_level": None,
        "start_time": None,
        "duration_minutes": "60",
        "preferred_teacher_name": None,
        "preferred_style": None,
        "package_id": None,
    }
    agent.chat_history = None

    # parse_stream 模拟 LLM 流式返回一段文本(具体内容无意义,parse_data 已被 patch)
    agent.processor.input_parser.parse_stream.return_value = ['{"subject": "x"}']
    agent.processor.input_parser.parse_data.return_value = parse_data_result
    # finished 由 update_history_from_data 决定,统一返回 False → 落追问分支
    agent.processor.update_history_from_data.return_value = False

    async def _unrelated_gen(user_input, unrelated_callback, state):
        yield "[UNRELATED]"

    async def _incomplete_gen(data, lesson_history):
        yield "[INCOMPLETE]"

    # 用真实异步生成器作为 side_effect,既保证 async for 可迭代,又可用 MagicMock 断言调用
    agent.processor.handle_unrelated_request = MagicMock(side_effect=_unrelated_gen)
    agent.processor.handle_incomplete_info = MagicMock(side_effect=_incomplete_gen)
    return agent


def _missing_entry(subject):
    """构造 parse_data 返回的"半结构化缺字段"数据"""
    return {
        "subject": subject,
        "grade_level": "未知",
        "start_time": "未知",
        "duration_minutes": "60",
        "preferred_teacher_name": "未知",
        "preferred_style": "未知",
        "package_id": "未知",
        "confirmation": "未知",
        "info_complete": False,
        "unrelated": True,
        "missing_info": ["grade_level", "start_time"],
    }


class TestUnrelatedGuard:
    """subject 非"未知"时,unrelated=true 不应踢回主路由"""

    def test_subject_present_but_missing_fields_goes_to_incomplete(self):
        """案例:"代数，一周三次，无偏好" 场景 —— subject=数学,unrelated=true。

        subject 非"未知" → 跳过踢回,落 handle_incomplete_info 追问缺的年级/时间。
        """
        agent = _make_agent(_missing_entry("数学"))

        out = _collect(agent.run_stream(user_input="代数，一周三次，无偏好", intent="lesson"))

        # 走追问分支,产出 INCOMPLETE 标记
        assert "INCOMPLETE" in out
        # 断言:踢回未调用,追问调用一次
        agent.processor.handle_unrelated_request.assert_not_called()
        agent.processor.handle_incomplete_info.assert_called_once()

    def test_colloquial_subject_alias_treated_as_signal(self):
        """口语化别名(如"代数",不在 9 科枚举但也非"未知")同样被视为信号。"""
        agent = _make_agent(_missing_entry("代数"))

        out = _collect(agent.run_stream(user_input="代数，一周三次，无偏好", intent="lesson"))

        assert "INCOMPLETE" in out
        agent.processor.handle_unrelated_request.assert_not_called()
        agent.processor.handle_incomplete_info.assert_called_once()

    def test_truly_unrelated_still_kicks_back(self):
        """反向用例:subject=未知 + unrelated=true → 仍走踢回,保留原行为。

        这是防御逻辑的关键底线:不能误伤真正的无关问题(如"今天天气怎么样")。
        """
        agent = _make_agent(_missing_entry("未知"))

        out = _collect(agent.run_stream(user_input="今天天气怎么样", intent="lesson"))

        assert "UNRELATED" in out
        agent.processor.handle_unrelated_request.assert_called_once()
        agent.processor.handle_incomplete_info.assert_not_called()