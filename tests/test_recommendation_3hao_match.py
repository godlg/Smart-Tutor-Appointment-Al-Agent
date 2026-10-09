"""
修复推荐确认态 "3号老师" 匹配 bug 的测试

背景:家长在推荐列表(展示顺序 张明 id1 / 周敏 id6 / 高远 id7)后回复口语化序号
"就选3号老师吧" / "3号" / "3号老师" / "选2号",应正确选中**列表第 N 位**老师,
而非把 "3号" 的 3 误当老师真实 ID(=3)去匹配 → 找不到 → 卡"您说的是哪位老师?"。

修复点:
1. _match_teacher:"N号" 先按老师真实 ID 匹配;若该 ID 不在本轮推荐 → 再按列表第 N 位(1-based)回退。
2. _handle_confirmation 未确认分支:先 _match_teacher,命中直接选定;未命中再判 _is_positive / _is_negative
   (修 "好的,就选3号老师" 被 "好" 前缀误判成正向确认、默认选第一顺位张明的假成功)。

异步用例统一用 _run(_collect) 同步包装(本仓库未装 pytest-asyncio)。
"""
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.recommendation_agent import RecommendationAgent  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


async def _drain(agen):
    out = []
    async for token in agen:
        out.append(token)
    return "".join(out)


def _collect(agen):
    return _run(_drain(agen))


# 与推荐 Agent 真实场景一致:第 1 位 张明(id1)、第 2 位 周敏(id6)、第 3 位 高远(id7)
def _recommendations():
    return [
        {"teacher": {"id": 1, "name": "张明"}, "score": 92, "reason": "擅长初二数学"},
        {"teacher": {"id": 6, "name": "周敏"}, "score": 90, "reason": "经验丰富"},
        {"teacher": {"id": 7, "name": "高远"}, "score": 88, "reason": "逻辑清晰"},
    ]


def _agent():
    """绕过 __init__(避免真实 LLM)构造 RecommendationAgent。"""
    return RecommendationAgent.__new__(RecommendationAgent)


def _match(user_input, recs=None):
    """直接单测 _match_teacher。"""
    recs = recs if recs is not None else _recommendations()
    history = {"last_recommendations": recs}
    return _agent()._match_teacher(history, user_input)


class TestMatchTeacher3Hao:
    """_match_teacher 直接单测:口语序号 → 选列表第 N 位"""

    def test_jiazhang_juxuan_3_hao(self):
        matched = _match("就选3号老师吧")
        assert matched is not None
        assert matched["teacher"]["name"] == "高远"
        assert matched["teacher"]["id"] == 7

    def test_3_hao(self):
        matched = _match("3号")
        assert matched["teacher"]["name"] == "高远"

    def test_3_hao_laoshi(self):
        matched = _match("3号老师")
        assert matched["teacher"]["name"] == "高远"

    def test_hao_de_juxuan_3_hao(self):
        matched = _match("好的,就选3号老师")
        assert matched["teacher"]["name"] == "高远"

    def test_xuan_2_hao(self):
        matched = _match("选2号")
        assert matched["teacher"]["name"] == "周敏"
        assert matched["teacher"]["id"] == 6

    def test_id_7(self):
        matched = _match("ID 7")
        assert matched["teacher"]["name"] == "高远"

    def test_7_hao_matches_real_id(self):
        """'7号' → 真实 ID 7 存在 → 按 teacher ID 命中 高远"""
        matched = _match("7号")
        assert matched["teacher"]["name"] == "高远"

    def test_bare_number_still_list_pos(self):
        """裸数字 '2' 仍按列表第 2 位 周敏"""
        matched = _match("2")
        assert matched["teacher"]["name"] == "周敏"

    def test_name_substring_unchanged(self):
        """'张明' 姓名子串匹配不受影响"""
        matched = _match("张明")
        assert matched["teacher"]["name"] == "张明"

    def test_no_match_returns_none(self):
        assert _match("不") is None
        assert _match("是") is None
        assert _match("随便说说") is None


def _unconfirmed_history():
    """尚未圈定老师、awaiting_confirmation=True 的历史状态(贴近真实确认态)。"""
    return {
        "last_recommendations": _recommendations(),
        "subject": "数学",
        "grade_level": "初中",
        "preferred_style": "未知",
        "awaiting_confirmation": True,
        "confirmed_teacher": None,
        "recommendation_declined": False,
    }


class TestHandleConfirmationUnconfirmed:
    """_handle_confirmation 未确认分支:先匹配,再判正/负"""

    def test_hao_de_juxuan_3_hao_selects_3(self):
        """'好的,就选3号老师' 必须选 高远,不能因 '好' 前缀默认选第一顺位 张明"""
        history = _unconfirmed_history()
        out = _collect(_agent()._handle_confirmation("好的,就选3号老师", history))
        assert "高远" in out
        assert "张明" not in out
        assert history["confirmed_teacher"]["teacher"]["name"] == "高远"

    def test_juxuan_3_hao_selects_3(self):
        history = _unconfirmed_history()
        out = _collect(_agent()._handle_confirmation("就选3号老师吧", history))
        assert "高远" in out
        assert history["confirmed_teacher"]["teacher"]["name"] == "高远"

    def test_xuan_2_hao_selects_zhoumin(self):
        history = _unconfirmed_history()
        out = _collect(_agent()._handle_confirmation("选2号", history))
        assert "周敏" in out
        assert history["confirmed_teacher"]["teacher"]["name"] == "周敏"

    def test_shi_defaults_first(self):
        """'是' 仍默认选第一顺位 张明(回归,语义不变)"""
        history = _unconfirmed_history()
        out = _collect(_agent()._handle_confirmation("是", history))
        assert "张明" in out
        assert history["confirmed_teacher"]["teacher"]["name"] == "张明"

    def test_hao_defaults_first(self):
        """'好' 单独 → 正向确认默认选第一顺位"""
        history = _unconfirmed_history()
        _collect(_agent()._handle_confirmation("好", history))
        assert history["confirmed_teacher"]["teacher"]["name"] == "张明"

    def test_negative_declines(self):
        """'不' → 负向确认,清 awaiting、置 declined"""
        history = _unconfirmed_history()
        out = _collect(_agent()._handle_confirmation("不", history))
        assert "不再需要推荐" in out
        assert history["awaiting_confirmation"] is False
        assert history["confirmed_teacher"] is None
        assert history["recommendation_declined"] is True

    def test_unmatched_reasks(self):
        """匹配不到 / 非正 / 非负 → 重说 '您说的是哪位老师?'"""
        history = _unconfirmed_history()
        out = _collect(_agent()._handle_confirmation("随便说说", history))
        assert "您说的是哪位老师" in out
        assert history["confirmed_teacher"] is None