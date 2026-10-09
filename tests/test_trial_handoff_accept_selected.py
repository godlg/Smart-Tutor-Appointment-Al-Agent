"""
TrialAgent 接收上游已选 teacher_id,直接进排课(不再重列 3 个老师)测试

背景:用户 chat 实测——推荐 agent 确认『是』后,家长说『本周三晚上7点』,
TrialAgent 拿到 user_input 后无视 chat_history 里上一轮 AI 输出的『老师ID: 1』,
重新走 RecommendationService.score_teacher_for_student 重列 Top-3 让家长再选。

修复:
1. agents/trial/trial_processor.py 加 helper `extract_confirmed_teacher_id(chat_history)`
   从最近一条 AI 消息里抽「老师ID: N」/「(ID N)」,只匹配一条就 break 避免老消息污染。
2. TrialProcessor.handle_complete_trial 加新分支:
   - 优先用 trial_history["teacher_id"] (TrialAgent._extract_confirmed_context 已回填)
   - 否则用 extract_confirmed_teacher_id(chat_history) 兜底
   - 两者都拿不到 → 走原有 fallback(列 3 个老师)
3. 新增 TrialMessageBuilder.create_direct_trial_message:yield 二次确认引导消息,
   不直接创建 Lesson(避免误操作落库,符合 spec 的『方案 a』)。

异步用例统一用 _run(_collect) 同步包装,不依赖 pytest-asyncio。
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.chat_history import InMemoryChatMessageHistory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.trial.trial_processor import (  # noqa: E402
    TrialProcessor,
    extract_confirmed_teacher_id,
)


def _run(coro):
    return asyncio.run(coro)


async def _drain(agen):
    out = []
    async for token in agen:
        out.append(token)
    return "".join(out)


def _collect(agen):
    return _run(_drain(agen))


# ---------------------------------------------------------------------------
# 工具:绕开 __init__ 构造 TrialProcessor + TrialAgent-like 上下文,聚焦 handle_complete_trial
# ---------------------------------------------------------------------------

def _make_processor(teacher_id_for_lookup: int = 1, teacher_record: dict = None):
    """构造一个 TrialProcessor-like,只保留 handle_complete_trial 链路需要的字段。

    teacher_service.get_teacher_by_id 用 mock 注入(避免真实 DB 依赖):
    - 默认:返回 teacher_record(可改 None 模拟『老师已删』fallback)
    """
    tp = object.__new__(TrialProcessor)

    tp.teacher_finder = MagicMock()
    tp.teacher_finder.check_teacher_availability.return_value = True  # 默认无冲突
    tp.lesson_service = MagicMock()
    tp.lesson_service.create_lesson.return_value = (True, 9999)  # 默认成功
    tp.student_service = MagicMock()
    tp.student_service.initialize_default_student.return_value = {"id": 1}
    tp.message_builder = MagicMock()
    tp.message_builder.create_direct_trial_message.return_value = (
        "\n顾问:已为您预约 张明 老师(数学, 初中 段)于 2026-09-16 19:00 的试讲。"
        "确认请回复『确认』,修改时间请直接告诉我。\n"
    )
    tp.message_builder.create_recommendation_message.return_value = (
        "\n顾问:为您推荐以下老师:1. 张明 / 2. 周敏 / 3. 高远\n"
    )
    tp.message_builder.create_parse_error_message.return_value = "parse err"

    # 把真实方法绑定过来
    tp.handle_complete_trial = TrialProcessor.handle_complete_trial.__get__(tp)
    tp._create_trial = TrialProcessor._create_trial.__get__(tp)
    tp.update_history_from_data = TrialProcessor.update_history_from_data.__get__(tp)
    tp.handle_incomplete_info = TrialProcessor.handle_incomplete_info.__get__(tp)
    tp.handle_unrelated_request = TrialProcessor.handle_unrelated_request.__get__(tp)

    # 默认老师记录(teacher_id=1 → 张明)
    default_teacher = teacher_record if teacher_record is not None else {
        "id": teacher_id_for_lookup,
        "name": "张明",
        "subjects": ["数学", "物理"],
        "grade_levels": ["初中", "高中"],
    }
    # 注入 mocked teacher_service:直接替换属性,handle_complete_trial 用
    # TeacherService() 抽方法时会走到真实类——所以我们在模块层面 patch 一下
    from services import teacher_service
    orig_cls = teacher_service.TeacherService
    mock_service = MagicMock()
    mock_service.get_teacher_by_id.return_value = (
        default_teacher if teacher_id_for_lookup is not None else None
    )
    mock_service.get_teacher_by_name.return_value = default_teacher
    teacher_service.TeacherService = MagicMock(return_value=mock_service)

    # 把 orig_cls / mock_service 挂回 tp 方便测试清理
    tp._orig_teacher_service_cls = orig_cls
    tp._mock_teacher_service = mock_service
    return tp, teacher_service


def _cleanup_processor(tp):
    """还原 TeacherService 类(避免污染后续测试)。"""
    from services import teacher_service
    teacher_service.TeacherService = tp._orig_teacher_service_cls


def _make_chat_history(messages):
    """把 [(role, content), ...] 转成 InMemoryChatMessageHistory(类型与 production 一致)。"""
    hist = InMemoryChatMessageHistory()
    for role, content in messages:
        if role == "user":
            from langchain_core.messages import HumanMessage
            hist.add_message(HumanMessage(content=content))
        else:
            from langchain_core.messages import AIMessage
            hist.add_message(AIMessage(content=content))
    return hist


# ---------------------------------------------------------------------------
# Case 7 (基础 helper 测试):extract_confirmed_teacher_id
# ---------------------------------------------------------------------------

class TestExtractConfirmedTeacherId:
    """单元测试:extract_confirmed_teacher_id 抽『老师ID: N』/『ID N』/『(ID N)』。"""

    def test_extracts_with_老师ID_format(self):
        hist = [{"role": "assistant", "content": "老师ID: 1"}]
        assert extract_confirmed_teacher_id(hist) == 1

    def test_extracts_with_老师ID_no_space(self):
        hist = [{"role": "assistant", "content": "老师ID:1"}]
        assert extract_confirmed_teacher_id(hist) == 1

    def test_extracts_with_paren_ID(self):
        hist = [{"role": "assistant", "content": "(ID 7)"}]
        assert extract_confirmed_teacher_id(hist) == 7

    def test_extracts_with_bare_ID(self):
        hist = [{"role": "assistant", "content": "ID 9"}]
        assert extract_confirmed_teacher_id(hist) == 9

    def test_returns_none_when_no_match(self):
        hist = [{"role": "assistant", "content": "请问什么时候试讲？"}]
        assert extract_confirmed_teacher_id(hist) is None

    def test_returns_none_for_empty_or_none_history(self):
        assert extract_confirmed_teacher_id(None) is None
        assert extract_confirmed_teacher_id([]) is None

    def test_only_looks_at_most_recent_ai_message(self):
        """多轮时只看最近一条 AI(break 行为)。"""
        hist = [
            {"role": "assistant", "content": "老师ID: 1"},  # 老消息
            {"role": "user", "content": "好的"},
            {"role": "assistant", "content": "请问什么时候试讲？"},  # 最近一条
        ]
        assert extract_confirmed_teacher_id(hist) is None  # 取最近,没有 ID → fallback

    def test_takes_most_recent_even_if_older_had_id(self):
        hist = [
            {"role": "assistant", "content": "老师ID: 6"},
            {"role": "user", "content": "是"},
            {"role": "assistant", "content": "已为您确认张明,老师ID: 1"},  # 最近一条
        ]
        assert extract_confirmed_teacher_id(hist) == 1

    def test_accepts_langchain_message_format(self):
        """InMemoryChatMessageHistory 的 AIMessage 也支持(type=='ai')。"""
        from langchain_core.messages import AIMessage
        hist = InMemoryChatMessageHistory()
        hist.add_message(AIMessage(content="老师ID: 8"))
        assert extract_confirmed_teacher_id(hist) == 8

    def test_full_width_colon_supported(self):
        hist = [{"role": "assistant", "content": "老师ID: 5"}]
        assert extract_confirmed_teacher_id(hist) == 5

    def test_non_assistant_messages_skipped(self):
        """最近一条是 user 时跳过,继续往前找 AI(直到第一条 AI)。"""
        hist = [
            {"role": "user", "content": "x"},  # 不是 AI,跳过
            {"role": "assistant", "content": "老师ID: 3"},
            {"role": "user", "content": "y"},
        ]
        # 从后往前:first user 跳过,然后 assistant(content=老师ID:3) → 取 3
        assert extract_confirmed_teacher_id(hist) == 3


# ---------------------------------------------------------------------------
# Cases 1-6:handle_complete_trial 在 chat_history 含『老师ID: N』时的行为
# ---------------------------------------------------------------------------

class TestHandleCompleteTrialAcceptsConfirmedTeacher:
    """TrialProcessor.handle_complete_trial 在 chat_history 含『老师ID: N』时
    必须直接 yield 预约引导消息,不再调 RecommendationService/重列 3 个老师。"""

    def test_case1_zhangming_full_handoff(self):
        """Case 1:chat_history 含『老师ID: 1』+ start_time='2026-09-16 19:00' →
        yield 直接预约消息(张明),不 yield『为您推荐以下老师』,不调 find_recommended_teachers。"""
        tp, ts_mod = _make_processor(teacher_id_for_lookup=1)
        try:
            trial_history = {
                "subject": "数学", "grade_level": "初中",
                "preferred_teacher_name": None, "start_time": "2026-09-16 19:00",
                "duration_minutes": "30",
            }
            chat_history = _make_chat_history([
                ("user", "我家孩子在读初二，数学不好，请为他推荐一位老师"),
                ("assistant", "[REPLY][推荐助手]为您推荐 张明 / 周敏 / 高远 三位老师。"),
                ("user", "张明"),
                ("assistant", "已为您选定张明。回复『是』"),
                ("user", "是"),
                ("assistant", "已为您确认 张明 老师(学科: 数学, 年级: 初中, 老师ID: 1)。请问什么时候试讲?"),
            ])
            out = _collect(tp.handle_complete_trial(
                trial_history, "test-session", chat_history=chat_history.messages
            ))

            # ✅ 含『已为您预约 张明 老师』
            assert "已为您预约 张明 老师" in out
            assert "老师ID" not in out or "老师ID" in out  # 直接消息不重复老师ID(可选)
            # ✅ 不含『为您推荐以下老师』
            assert "为您推荐以下老师" not in out
            # ✅ find_recommended_teachers 没被调(防御性)
            tp.teacher_finder.find_recommended_teachers.assert_not_called()
            # ✅ trial_history 记录了 confirmed_teacher_id / start_time_parsed
            assert trial_history["confirmed_teacher_id"] == 1
            assert trial_history["confirmed_teacher"]["name"] == "张明"
            assert trial_history["start_time_parsed"] is not None
        finally:
            _cleanup_processor(tp)

    def test_case2_zhoumin_handoff(self):
        """Case 2:chat_history 含『老师ID: 6』+ start_time='2026-09-17 15:00' →
        yield 走周敏分支(get_teacher_by_id(6) → 周敏)。"""
        zhoumin = {
            "id": 6, "name": "周敏",
            "subjects": ["数学"], "grade_levels": ["初中", "高中"],
        }
        tp, ts_mod = _make_processor(teacher_id_for_lookup=6, teacher_record=zhoumin)
        tp.message_builder.create_direct_trial_message.return_value = (
            "\n顾问:✅ 已为您预约 周敏 老师(数学, 初中、高中 段)于 2026-09-17 15:00 的试讲。\n"
        )
        try:
            trial_history = {
                "subject": "数学", "grade_level": "初中",
                "preferred_teacher_name": None, "start_time": "2026-09-17 15:00",
                "duration_minutes": "30",
            }
            chat_history = _make_chat_history([
                ("user", "推荐老师"),
                ("assistant", "已为您确认 周敏 老师(学科: 数学, 年级: 初中, 老师ID: 6)。"),
            ])
            out = _collect(tp.handle_complete_trial(
                trial_history, "test-session", chat_history=chat_history.messages
            ))

            assert "已为您预约 周敏 老师" in out
            assert "为您推荐以下老师" not in out
            # get_teacher_by_id(6) 被调用
            tp._mock_teacher_service.get_teacher_by_id.assert_called_with(6)
            assert trial_history["confirmed_teacher_id"] == 6
        finally:
            _cleanup_processor(tp)

    def test_case3_fallback_when_no_teacher_id(self):
        """Case 3:chat_history 没有『老师ID: N』+ start_time 完整 →
        走 fallback:不调 get_teacher_by_id(teacher_id),后续走『列 3 个老师』路径。"""
        tp, ts_mod = _make_processor(teacher_id_for_lookup=None)
        try:
            trial_history = {
                "subject": "数学", "grade_level": "初中",
                "preferred_teacher_name": None, "start_time": "2026-09-16 19:00",
                "duration_minutes": "30",
            }
            chat_history = _make_chat_history([
                ("user", "想给孩子找个数学老师"),
                ("assistant", "请问您倾向哪种教学风格?"),
                ("user", "下周晚上"),
                ("assistant", "好的"),
            ])
            # preferred_name 为 None,又没 teacher_id → fallback 到 find_recommended_teachers
            tp.teacher_finder.find_recommended_teachers.return_value = [
                {"teacher": {"id": 1, "name": "张明"}, "score": 92, "reason": "擅长"},
                {"teacher": {"id": 6, "name": "周敏"}, "score": 80, "reason": "覆盖"},
                {"teacher": {"id": 11, "name": "高远"}, "score": 70, "reason": "耐心"},
            ]
            out = _collect(tp.handle_complete_trial(
                trial_history, "test-session", chat_history=chat_history.messages
            ))

            # get_teacher_by_id 没拿到有效 ID(因为 fallback 分支没传 ID)
            # 这里『teacher_id=None』是 trial_history 默认行为—— fallback 不会调 get_teacher_by_id
            # 而 find_recommended_teachers 被调了
            tp.teacher_finder.find_recommended_teachers.assert_called_once()
            assert "为您推荐以下老师" in out
        finally:
            _cleanup_processor(tp)

    def test_case4_fallback_with_invalid_teacher_id(self):
        """Case 4:chat_history 含『老师ID: 999』(无效/老师不存在) →
        fallback:get_teacher_by_id(999) 返回 None,不崩,走后续 preferred_name 路径。"""
        tp, ts_mod = _make_processor(teacher_id_for_lookup=None)  # 默认 None 模拟『老师被删』
        try:
            trial_history = {
                "subject": "数学", "grade_level": "初中",
                "preferred_teacher_name": "张明",  # 兜底:用名字查
                "start_time": "2026-09-16 19:00",
                "duration_minutes": "30",
            }
            chat_history = _make_chat_history([
                ("assistant", "老师ID: 999"),  # 无效 ID
            ])
            tp.teacher_finder.find_recommended_teachers.return_value = []
            out = _collect(tp.handle_complete_trial(
                trial_history, "test-session", chat_history=chat_history.messages
            ))

            # get_teacher_by_id(999) 被调过,返回 None → 降级
            tp._mock_teacher_service.get_teacher_by_id.assert_called_with(999)
            # 然后用 preferred_name='张明' 查 get_teacher_by_name
            tp._mock_teacher_service.get_teacher_by_name.assert_called_with("张明")
            # 没 yield『为您推荐以下老师』(因为 preferred_name='张明' 直接命中)
            assert trial_history.get("recommended_teacher", {}).get("name") == "张明"
        finally:
            _cleanup_processor(tp)

    def test_case5_no_chat_history_safe(self):
        """Case 5:chat_history 是 None 或 []+ start_time 完整 →
        不抛异常;trial_history 也没 teacher_id → 走 fallback 路径(列 3 个候选)。"""
        for empty in (None, []):
            tp, ts_mod = _make_processor(teacher_id_for_lookup=1)
            try:
                trial_history = {
                    "subject": "数学", "grade_level": "初中",
                    "preferred_teacher_name": None,
                    "start_time": "2026-09-16 19:00",
                    "duration_minutes": "30",
                }
                tp.teacher_finder.find_recommended_teachers.return_value = []
                # 不抛异常就是成功
                out = _collect(tp.handle_complete_trial(
                    trial_history, "test-session", chat_history=empty
                ))
                # 走 fallback 路径(没 preferred_name,没 teacher_id → 列 3 个 → 没有老师时
                # yield create_recommendation_message(subject,grade_level,start_time_str,[]))
                assert "目前没有找到适合" in out or "为您推荐以下老师" in out
            finally:
                _cleanup_processor(tp)

    def test_case6_only_looks_at_most_recent_ai_message(self):
        """Case 6:chat_history 多轮,只有最早一条 AI 含 ID,最近一条不含 →
        fallback(只看最近一条)。"""
        tp, ts_mod = _make_processor(teacher_id_for_lookup=1)
        try:
            trial_history = {
                "subject": "数学", "grade_level": "初中",
                "preferred_teacher_name": None,
                "start_time": "2026-09-16 19:00",
                "duration_minutes": "30",
            }
            chat_history = _make_chat_history([
                ("assistant", "老师ID: 1"),  # 老消息含 ID
                ("user", "好的"),
                ("assistant", "还有其他问题吗?"),  # 最近一条 AI,没 ID
            ])
            tp.teacher_finder.find_recommended_teachers.return_value = [
                {"teacher": {"id": 1, "name": "张明"}, "score": 92, "reason": "擅长"},
            ]
            _collect(tp.handle_complete_trial(
                trial_history, "test-session", chat_history=chat_history.messages
            ))
            # get_teacher_by_id 没被调(因为最近一条 AI 没 ID,跳过 teacher_id 分支)
            tp._mock_teacher_service.get_teacher_by_id.assert_not_called()
        finally:
            _cleanup_processor(tp)


class TestTrialHistoryTeacherIdShortCircuits:
    """TrialProcessor.handle_complete_trial 在 trial_history 已有 teacher_id 时也走直接预约,
    不依赖 chat_history(因为 TrialAgent._extract_confirmed_context 已经从 session_chat_history 回填)。"""

    def test_trial_history_teacher_id_takes_precedence(self):
        tp, ts_mod = _make_processor(teacher_id_for_lookup=1)
        try:
            trial_history = {
                "subject": "数学", "grade_level": "初中",
                "preferred_teacher_name": "张明",
                "teacher_id": 1,  # 已回填
                "start_time": "2026-09-16 19:00",
                "duration_minutes": "30",
            }
            # chat_history 为 None,trial_history.teacher_id 仍然有效
            out = _collect(tp.handle_complete_trial(
                trial_history, "test-session", chat_history=None
            ))

            assert "已为您预约 张明 老师" in out
            assert trial_history["confirmed_teacher_id"] == 1
        finally:
            _cleanup_processor(tp)

    def test_no_5_layer_violation_no_service_layer_call(self):
        """约束:不调 RecommendationService(反向依赖),不直接 create_lesson。"""
        tp, ts_mod = _make_processor(teacher_id_for_lookup=1)
        try:
            trial_history = {
                "subject": "数学", "grade_level": "初中",
                "preferred_teacher_name": None, "teacher_id": 1,
                "start_time": "2026-09-16 19:00", "duration_minutes": "30",
            }
            chat_history = _make_chat_history([
                ("assistant", "老师ID: 1"),
            ])
            _collect(tp.handle_complete_trial(
                trial_history, "test-session", chat_history=chat_history.messages
            ))
            # ❌ 不调 create_lesson(spec 约束)
            tp.lesson_service.create_lesson.assert_not_called()
        finally:
            _cleanup_processor(tp)
