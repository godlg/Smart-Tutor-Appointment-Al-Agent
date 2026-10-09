"""LangGraph P0-A 迁移实跑演示（离线，无外部依赖，exit 0）。

打印 `.claude/tasks/langgraph-migrate-p0a.md` 要求的 9 个场景判定结果：

  1. A1 record_input 被调用（含无关轮）
  2. A2 已转人工 → 短路且不分类
  3. A3 主动关键词 → 触发 handoff
  4. B1/B7 匿名校验（真实 Identity(legacy=False) + set_current_identity）
  5. B2/B3/B4 续轮态跳过分类器直连对应 agent（分类器 0 调用）
  6. B5 档期白名单 → 不调分类器、不改 flow_state
  7. B6 分类器抛 ClassificationUnavailable → BUSY 文案且踢回计数不增长
  8. D 踢回 3 次 → HANDOFF；1-2 次 → 能力列表
  9. C 四条主路由各落对节点

另打印文案逐字一致性（_VERIFICATION_REPLY / _BUSY_REPLY / HANDOFF）。

运行：python scripts/demo_langgraph_migrate.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.graph import build_graph, new_state  # noqa: E402
from agents.graph.nodes import _BUSY_REPLY, _VERIFICATION_REPLY  # noqa: E402
from agents.task_classification.classification_processor import (  # noqa: E402
    _BUSY_REPLY as LEGACY_BUSY_REPLY,
    _VERIFICATION_REPLY as LEGACY_VERIFICATION_REPLY,
)
from agents.task_classification.task_classifier import ClassificationUnavailable  # noqa: E402
from config.identity import (  # noqa: E402
    Identity,
    reset_current_identity,
    set_current_identity,
)
from services.human_handoff_service import (  # noqa: E402
    HANDED_OFF_REPLY,
    HANDOFF_REPLY,
)

ALL_BUSINESS = ("consultation", "recommendation", "trial", "lesson", "reschedule", "cancel")


# ----------------------------------------------------------------------
# 替身
# ----------------------------------------------------------------------

class FakeClassifier:
    def __init__(self, mapping=None, default="unrelated"):
        self.mapping = mapping or {}
        self.default = default
        self.calls = []

    async def classify_task(self, task):
        self.calls.append(task)
        for key, intent in self.mapping.items():
            if key in task:
                return intent
        return self.default


class RaisingClassifier:
    def __init__(self):
        self.calls = 0

    async def classify_task(self, task):
        self.calls += 1
        raise ClassificationUnavailable("AccountQuotaExceeded: 429")


class FakeAgent:
    def __init__(self, name, state=None):
        self.name = name
        self.calls = []
        self.intents = []
        self.state = state

    async def run_stream(self, user_input=None):
        self.calls.append(user_input)
        self.intents.append(None)
        yield f"[REPLY][{self.name}]已处理：{user_input}"


class LessonLikeAgent(FakeAgent):
    async def run_stream(self, user_input=None, intent=None):
        self.calls.append(user_input)
        self.intents.append(intent)
        yield f"[REPLY][{self.name}]已处理(intent={intent})：{user_input}"


class FakeAvailabilityHandler:
    def __init__(self):
        self.calls = []

    async def handle(self, task):
        self.calls.append(task)
        yield "[REPLY]张明老师最近可预约:周一 9:00、周二 10:00"


class FakeHandoffService:
    def __init__(self, handed=False, keyword=None, threshold=3):
        self.handed = handed
        self.keyword = keyword
        self.threshold = threshold
        self.records = []
        self.kickbacks = 0
        self.record_kickback_calls = 0
        self.triggered = []

    def record_input(self, session_id, content):
        self.records.append((session_id, content))

    def is_handed_off(self, session_id):
        return self.handed

    def parent_message(self, session_id, content):
        self.records.append(("parent", session_id, content))

    def match_keyword(self, task):
        return self.keyword if (self.keyword and task and self.keyword in task) else None

    def record_kickback(self, session_id):
        self.record_kickback_calls += 1
        self.kickbacks += 1
        return self.kickbacks >= self.threshold

    def get_count(self, session_id):
        return self.kickbacks

    def trigger_handoff(self, session_id, trigger_type, **kwargs):
        self.triggered.append((trigger_type, kwargs))

    @staticmethod
    def handoff_reply():
        return HANDOFF_REPLY

    @staticmethod
    def handed_off_reply():
        return HANDED_OFF_REPLY

    def build_transcript(self, session_id, messages):
        return None


def _agents():
    return {
        "consultation": FakeAgent("咨询"),
        "recommendation": FakeAgent("推荐"),
        "trial": FakeAgent("试讲"),
        "lesson": LessonLikeAgent("正式课"),
        "reschedule": LessonLikeAgent("调课"),
        "cancel": LessonLikeAgent("退课"),
    }


async def _collect(graph, payload, cfg):
    out = []
    async for token in graph.astream(payload, cfg, stream_mode="custom"):
        out.append(token)
    return out


async def _drain(graph, payload, cfg):
    return "".join(await _collect(graph, payload, cfg))


def _graph(classifier=None, agents=None, handoff=None, availability=None):
    return build_graph(
        classifier=classifier or FakeClassifier({}, default="unrelated"),
        agents=agents if agents is not None else _agents(),
        handoff_service=handoff,
        availability_handler=availability,
    )


def _cfg(tid):
    return {"configurable": {"thread_id": tid}}


RESULTS = []


def _check(label, ok, detail=""):
    RESULTS.append(ok)
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f"  ({detail})" if detail else ""))
    return ok


# ----------------------------------------------------------------------
# 场景
# ----------------------------------------------------------------------

async def scenario_a1():
    hs = FakeHandoffService()
    graph = _graph(classifier=FakeClassifier({}, default="unrelated"), handoff=hs)
    await _drain(graph, new_state("今天天气怎么样", "a1"), _cfg("a1"))
    _check("A1 record_input 被调用(含无关轮)", hs.records == [("a1", "今天天气怎么样")],
           f"records={hs.records}")


async def scenario_a2():
    hs = FakeHandoffService(handed=True)
    classifier = FakeClassifier({}, default="trial")
    graph = _graph(classifier=classifier, handoff=hs)
    out = await _drain(graph, new_state("再问一句", "a2"), _cfg("a2"))
    ok = HANDED_OFF_REPLY in out and classifier.calls == [] and ("parent", "a2", "再问一句") in hs.records
    _check("A2 已转人工 → 短路且不分类", ok,
           f"classifier_calls={len(classifier.calls)} parent落库={'parent' in [r[0] for r in hs.records if isinstance(r, tuple) and len(r) == 3]}")


async def scenario_a3():
    hs = FakeHandoffService(keyword="人工")
    classifier = FakeClassifier({}, default="trial")
    graph = _graph(classifier=classifier, handoff=hs)
    out = await _drain(graph, new_state("我要转人工", "a3"), _cfg("a3"))
    ok = ("[REPLY][HANDOFF]" in out and HANDOFF_REPLY in out
          and classifier.calls == [] and hs.triggered and hs.triggered[0][0] == "keyword")
    _check("A3 关键词 → 触发 handoff", ok, f"trigger={hs.triggered[0][0] if hs.triggered else None}")


async def scenario_verification():
    classifier = FakeClassifier({}, default="trial")
    agents = _agents()
    graph = _graph(classifier=classifier, handoff=FakeHandoffService(), agents=agents)

    token = set_current_identity(Identity("anon_parent", student_id="1"))
    try:
        out = await _drain(graph, new_state("我要约课", "v1"), _cfg("v1"))
    finally:
        reset_current_identity(token)
    ok = (_VERIFICATION_REPLY in out and agents["trial"].calls == []
          and _VERIFICATION_REPLY == LEGACY_VERIFICATION_REPLY)
    _check("B1/B7 匿名校验 → 校验文案且不调业务 agent", ok,
           f"文案逐字一致={_VERIFICATION_REPLY == LEGACY_VERIFICATION_REPLY}")


async def scenario_continuation():
    ok = True
    detail = []
    for flow_state, target in (("recommendation_confirm", "recommendation"),
                               ("trial_confirm", "trial"),
                               ("trial_convert", "trial")):
        classifier = FakeClassifier({}, default="unrelated")
        agents = _agents()
        graph = _graph(classifier=classifier, handoff=FakeHandoffService(), agents=agents)
        tid = f"resume-{flow_state}"
        await _drain(graph, new_state("确认", tid, flow_state=flow_state), _cfg(tid))
        hit = classifier.calls == [] and agents[target].calls == ["确认"]
        ok = ok and hit
        detail.append(f"{flow_state}→{target}:{hit}")
    _check("B2/B3/B4 续轮态跳过分类器直连对应 agent", ok, ", ".join(detail))


async def scenario_availability():
    classifier = FakeClassifier({}, default="consultation")
    availability = FakeAvailabilityHandler()
    graph = _graph(classifier=classifier, handoff=FakeHandoffService(), availability=availability)
    tid = "b5"
    out = await _drain(graph, new_state("张明老师什么时候有空", tid), _cfg(tid))
    values = (await graph.aget_state(_cfg(tid))).values
    ok = (availability.calls == ["张明老师什么时候有空"] and classifier.calls == []
          and "可预约" in out and values["flow_state"] == "classify")
    _check("B5 档期白名单 → 不调分类器、不改 flow_state", ok,
           f"classifier_calls={len(classifier.calls)} flow_state={values['flow_state']}")


async def scenario_busy():
    classifier = RaisingClassifier()
    hs = FakeHandoffService()
    graph = _graph(classifier=classifier, handoff=hs)
    out = await _drain(graph, new_state("我想给孩子约一节数学试听课", "b6"), _cfg("b6"))
    ok = (_BUSY_REPLY in out and hs.record_kickback_calls == 0 and hs.triggered == []
          and _BUSY_REPLY == LEGACY_BUSY_REPLY)
    _check("B6 分类器不可用 → BUSY 文案且踢回计数不增长", ok,
           f"kickback_calls={hs.record_kickback_calls} 文案逐字一致={_BUSY_REPLY == LEGACY_BUSY_REPLY}")


async def scenario_unrelated():
    hs = FakeHandoffService(threshold=3)
    graph = _graph(classifier=FakeClassifier({}, default="unrelated"), handoff=hs)
    tid = "d"
    r1 = await _drain(graph, new_state("天气一", tid), _cfg(tid))
    r2 = await _drain(graph, new_state("天气二", tid), _cfg(tid))
    r3 = await _drain(graph, new_state("天气三", tid), _cfg(tid))
    low_ok = "超出我的能力范围" in r1 and "超出我的能力范围" in r2 and "[HANDOFF]" not in r1 + r2
    high_ok = "[REPLY][HANDOFF]" in r3 and HANDOFF_REPLY in r3
    _check("D 踢回 1-2 次 → 能力列表", low_ok)
    _check("D 踢回 3 次 → HANDOFF", high_ok,
           f"trigger={hs.triggered[0][0] if hs.triggered else None}")


async def scenario_business_routes():
    cases = [
        ("孩子数学怎么学", "consultation", "consultation"),
        ("想约一节试听课", "trial", "trial"),
        ("帮我预约明天下午3点的课", "lesson", "lesson"),
        ("把周三的课调到周五", "reschedule", "reschedule"),
        ("帮我取消明天的课", "cancel", "cancel"),
        ("请给我推荐一位老师", "recommendation", "recommendation"),
    ]
    all_ok = True
    for text, intent, node in cases:
        classifier = FakeClassifier({}, default=intent)
        agents = _agents()
        graph = _graph(classifier=classifier, handoff=FakeHandoffService(), agents=agents)
        tid = f"c-{node}"
        await _drain(graph, new_state(text, tid), _cfg(tid))
        hit = agents[node].calls == [text]
        all_ok = all_ok and hit
        print(f"        路由 {intent:<13} → {node}_node  命中={hit}")
    _check("C 四条主路由(含 cancel/recommendation)各落对节点", all_ok)


async def main():
    print("=" * 72)
    print("LangGraph P0-A 迁移实跑：9 场景判定")
    print("=" * 72)
    print("\n[A. 入口层]")
    await scenario_a1()
    await scenario_a2()
    await scenario_a3()
    print("\n[B. 分类前 / 分类后]")
    await scenario_verification()
    await scenario_continuation()
    await scenario_availability()
    await scenario_busy()
    print("\n[D. 兜底]")
    await scenario_unrelated()
    print("\n[C. 主路由]")
    await scenario_business_routes()

    print("\n" + "=" * 72)
    passed = sum(1 for r in RESULTS if r)
    print(f"结论：{passed}/{len(RESULTS)} 场景通过")
    print("=" * 72)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
