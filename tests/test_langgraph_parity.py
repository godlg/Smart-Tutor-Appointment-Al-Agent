"""LangGraph P0-B 双跑对拍：legacy 与 graph 逐字等价性。

对应 `.claude/tasks/langgraph-migrate-p0b.md` 第二节。

思路
----
- 每条**多轮对话脚本**在两条链路上各跑一遍，逐轮收集输出文本：
  - legacy：真实 `TaskClassificationAgent`（仅把 `create_chat_model` 换掉避免建 LLM），
    业务 agent / handoff / availability 全部注入替身。
  - graph：`build_graph(...)` 编译后 `astream(stream_mode="custom")`，每轮按
    `flow_state` 从 checkpointer 恢复（与 `api/chat_handler._process_graph_stream`
    的驱动方式一致），thread_id 独立。
- 两条链路注入**同一套替身规格**：
  - `FakeClassifier`：同一实例被两条链路共享（无状态，仅记录调用），保证分类输入一致。
  - 业务 agent / handoff_service / availability_handler：**同一替身类 + 同一配置**，
    但每条链路各持一份实例 —— 它们承载跨轮可变状态（踢回计数、handoff、flow_state），
    若字面共享会互相污染，对拍就失去意义。替身文案由同一模板生成，输出仍逐字可比。
- 比较前做 `_normalize`：剔除已知可接受的差异行
  `[THOUGHT][LangGraph] 意图识别为 X。`（P0-A 图路径额外的可观测性标注），
  其余**逐字**比较，一致率须 100%。

覆盖脚本（≥14）：**全部 14 类 spec 场景 + 3 条续轮态扩展**，见 `PARITY_SCRIPTS`。

约束：无 pytest-asyncio，异步统一 `asyncio.run` 同步包装；替身不连真实 LLM / DB。
"""

from __future__ import annotations

import asyncio
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from agents.graph import build_graph, new_state  # noqa: E402
from agents.task_classification.state_manager import FlowState, SharedState  # noqa: E402
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


# ----------------------------------------------------------------------
# 规范化：剔除「已知可接受差异」，其余逐字比较
# ----------------------------------------------------------------------

# 图分类节点额外外发的可观测性标注（legacy 无此行，P0-A 引入，接受为可忽略差异）。
_LANGGRAPH_THOUGHT_RE = re.compile(r"\[THOUGHT\]\[LangGraph\] 意图识别为 [^。]*。")


def normalize(text: str) -> str:
    """剔除已知可接受差异行后返回剩余文本（其余原样，逐字可比）。"""
    return _LANGGRAPH_THOUGHT_RE.sub("", text)


# ----------------------------------------------------------------------
# 替身（同一套规格；每条链路各持一份有状态实例）
# ----------------------------------------------------------------------

class FakeClassifier:
    """子串映射分类器；两条链路共享同一实例（无状态，仅记录调用）。"""

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
    """固定抛 `ClassificationUnavailable`（模拟 LLM 故障）。"""

    def __init__(self):
        self.calls = []

    async def classify_task(self, task):
        self.calls.append(task)
        raise ClassificationUnavailable("AccountQuotaExceeded: 429")


class FakeSharedState(SharedState):
    """可指名初始 flow_state 的共享状态容器。"""

    def __init__(self, value=None):
        super().__init__()
        self.value = value or FlowState.CLASSIFY


class FakeBusinessAgent:
    """业务 agent 替身：同形态转发 + 可写共享状态 + 异步上下文（咨询用）。

    - 文案由 `_text` 统一生成，两条链路的替身配置相同 → 输出逐字可比。
    - `run_stream` / `consult_stream` 都产出同一文本：legacy 咨询走 `consult_stream`，
      graph 咨询节点走 `run_stream`，两者必须一致。
    - `post_state`：本轮结束后把共享状态推进到该 flow_state（模拟真实 agent
      自行 transition，如 TrialAgent → TRIAL_CONFIRM）。
    """

    def __init__(self, name: str, post_state: Optional[str] = None):
        self.name = name
        self.post_state = post_state
        self.state: Optional[SharedState] = None
        self.shared_state: Optional[SharedState] = None
        self.calls: List[str] = []
        self.intents: List[Optional[str]] = []
        self.entered = 0
        self.exited = 0
        self.kb_initialized = False

    def set_shared_state(self, shared_state):
        self.state = shared_state
        self.shared_state = shared_state

    def _text(self, user_input, intent):
        suffix = f"(intent={intent})" if intent else ""
        return f"[REPLY][{self.name}]{suffix}已处理：{user_input}"

    def _apply_post_state(self):
        if self.post_state and self.state is not None:
            self.state.value = FlowState(self.post_state)

    async def run_stream(self, user_input=None, intent=None):
        self.calls.append(user_input)
        self.intents.append(intent)
        yield self._text(user_input, intent)
        self._apply_post_state()

    async def consult_stream(self, user_input=None):
        self.calls.append(user_input)
        self.intents.append(None)
        yield self._text(user_input, None)
        self._apply_post_state()

    async def __aenter__(self):
        self.entered += 1
        self.kb_initialized = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exited += 1


class FakeAvailabilityHandler:
    """档期白名单替身：直接产出固定时段文案（不连 Teacher/LessonService）。"""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: List[str] = []

    async def handle(self, task):
        self.calls.append(task)
        yield self.reply


class FakeHandoffService:
    """handoff 替身：进程内计数 + 触发记录，不落 DB。

    有状态（kickback 计数 / handed_off / keyword），故每条链路各持一份实例。
    """

    def __init__(self, keyword=None, handed=False, threshold=3):
        self.keyword = keyword
        self.handed = handed
        self.threshold = threshold
        self.kickbacks = 0
        self.record_kickback_calls = 0
        self.records = []
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
        self.handed = True

    @staticmethod
    def handoff_reply():
        return HANDOFF_REPLY

    @staticmethod
    def handed_off_reply():
        return HANDED_OFF_REPLY

    def build_transcript(self, session_id, messages):
        return None


# ----------------------------------------------------------------------
# 场景 / 脚本定义
# ----------------------------------------------------------------------

DEFAULT_AVAILABILITY_REPLY = (
    "[REPLY][归类机器人]\n顾问：张明老师最近可预约:周一 9:00、周二 10:00…请告诉我要哪个时间。\n"
)


@dataclass
class Scenario:
    """一条多轮脚本的替身规格与轮次。"""

    sid: str
    name: str
    turns: List[str]
    default: str = "unrelated"
    mapping: dict = field(default_factory=dict)
    classifier_kind: str = "mapping"      # mapping | raising
    post_states: dict = field(default_factory=dict)  # 业务 key -> 本轮结束后的 flow_state
    keyword: Optional[str] = None
    handed: bool = False
    threshold: int = 3
    availability: Optional[str] = None
    identity: Optional[Identity] = None
    note: str = ""

    def make_classifier(self):
        if self.classifier_kind == "raising":
            return RaisingClassifier()
        return FakeClassifier(self.mapping, default=self.default)


# 覆盖 spec 第二节列出的全部 14 类场景（1-14），另加 3 条续轮态扩展（15-17）。
PARITY_SCRIPTS: List[Scenario] = [
    Scenario("1", "consultation", ["孩子初二数学怎么提高"],
             default="consultation", post_states={"consultation": "classify"},
             note="咨询：首轮分流行 + 咨询 agent 产出"),
    Scenario("2", "trial", ["想约一节数学试听课"],
             default="trial", post_states={"trial": "trial"},
             note="试讲：transition_to_trial + 分流行"),
    Scenario("3", "lesson", ["帮我预约明天下午3点的数学课"],
             default="lesson", post_states={"lesson": "lesson"},
             note="正式课：transition_to_lesson + 分流行"),
    Scenario("4", "reschedule", ["把周三的课调到周五"],
             default="reschedule", note="调课：intent=reschedule + 分流行"),
    Scenario("5", "cancel", ["帮我取消明天的课"],
             default="cancel", note="退课：intent=cancel + 分流行"),
    Scenario("6", "recommendation", ["请给我推荐一位数学老师"],
             default="recommendation", note="推荐：分流行 + 推荐 agent 产出"),
    Scenario("7", "trial_confirm_continuation", ["想约一节数学试听课", "确认"],
             default="trial", post_states={"trial": "trial_confirm"},
             note="TRIAL_CONFIRM 续轮：跳过分类器、无分流行"),
    Scenario("8", "trial_convert_continuation", ["试讲很满意，想继续", "继续"],
             default="trial", post_states={"trial": "trial_convert"},
             note="TRIAL_CONVERT 续轮：跳过分类器、无分流行"),
    Scenario("9", "anonymous_business_verification", ["我要约一节试听课"],
             default="trial", identity=Identity("anon_parent", student_id="1"),
             note="匿名 + 业务意图 → 校验文案（两条链路均不落业务 agent）"),
    Scenario("10", "availability_whitelist", ["张明老师什么时候有空"],
             default="unrelated", availability=DEFAULT_AVAILABILITY_REPLY,
             note="档期白名单：规则直答、不调分类器"),
    Scenario("11", "kickback_threshold_3",
             ["今天天气怎么样", "随便聊聊", "再问个无关的"],
             default="unrelated", threshold=3,
             note="踢回 1/2 次 → 能力列表；第 3 次 → 转人工"),
    Scenario("12", "classification_unavailable", ["我想给孩子约一节数学试听课"],
             classifier_kind="raising",
             note="LLM 故障 → BUSY 且不踢回"),
    Scenario("13", "keyword_handoff", ["我要转人工"],
             keyword="人工", note="关键词转人工"),
    Scenario("14", "handed_off_short_circuit", ["转人工", "我想约课"],
             keyword="人工", note="已转人工后继续输入 → 短路"),
    # ---- 扩展：其余续轮态，同样要求 100% ----
    Scenario("15", "recommendation_confirm_continuation", ["请推荐一位老师", "就选3号"],
             default="recommendation", post_states={"recommendation": "recommendation_confirm"},
             note="RECOMMENDATION_CONFIRM 续轮：跳过分类器、无分流行"),
    Scenario("16", "lesson_continuation", ["帮我预约明天的课", "再确认下时间"],
             default="lesson", post_states={"lesson": "lesson"},
             note="LESSON 续轮：跳过分类器、无分流行"),
    Scenario("17", "consultation_continuation", ["孩子数学怎么提高", "再讲讲"],
             default="consultation", post_states={"consultation": "consultation"},
             note="CONSULTATION 续轮：经 async with 进入、无分流行"),
]

REQUIRED_SPEC_IDS = {str(i) for i in range(1, 15)}


def _dummy_model():
    """占位 chat model：仅让 legacy 构造函数跑通（实际分类器被替身替换）。"""
    return RunnableLambda(lambda _value: AIMessage(content="unrelated"))


def _build_fakes(scenario: Scenario):
    """按同一规格构造一套有状态替身（每条链路各一份）。"""
    post = scenario.post_states
    fakes = {}
    for key, label in (("consultation", "咨询"), ("trial", "试讲"),
                       ("lesson", "正式课"), ("recommendation", "推荐")):
        fakes[key] = FakeBusinessAgent(label, post_state=post.get(key))
    handoff = FakeHandoffService(keyword=scenario.keyword, handed=scenario.handed,
                                 threshold=scenario.threshold)
    availability = (FakeAvailabilityHandler(scenario.availability)
                    if scenario.availability is not None else None)
    return fakes, handoff, availability


# ----------------------------------------------------------------------
# 两条链路驱动器
# ----------------------------------------------------------------------

def _drain(stream):
    async def _go():
        parts = []
        async for token in stream:
            parts.append(token)
        return "".join(parts)

    return asyncio.run(_go())


@dataclass
class LegacyChain:
    """legacy 链路：真实 TaskClassificationAgent + 注入替身（不建 LLM/DB）。"""

    scenario: Scenario
    classifier: object
    session_id: str
    agent: object = None
    handoff: object = None
    fakes: dict = None

    def __post_init__(self):
        self.fakes, self.handoff, self._availability = _build_fakes(self.scenario)
        with patch("agents.task_classification_agent.create_chat_model",
                   lambda **kw: _dummy_model()):
            from agents.task_classification_agent import TaskClassificationAgent

            self.agent = TaskClassificationAgent(
                consultant_agent=self.fakes["consultation"],
                trial_agent=self.fakes["trial"],
                lesson_agent=self.fakes["lesson"],
                recommendation_agent=self.fakes["recommendation"],
                session_id=self.session_id,
                handoff_service=self.handoff,
            )
        # 用共享替身替换真实分类器：两条链路拿到同一分类行为。
        self.agent.classification_processor.task_classifier = self.classifier
        self.agent.task_classifier = self.classifier
        if self._availability is not None:
            self.agent.classification_processor.availability_handler = self._availability

    def turn(self, text: str) -> str:
        return _drain(self.agent.classify_task_stream(text))


@dataclass
class GraphChain:
    """graph 链路：build_graph 编译后 astream(custom)，按轮恢复 flow_state。"""

    scenario: Scenario
    classifier: object
    thread_id: str
    graph: object = None
    handoff: object = None
    flow_state: Optional[str] = None

    def __post_init__(self):
        fakes, self.handoff, availability = _build_fakes(self.scenario)
        for key in ("consultation", "trial", "lesson", "recommendation"):
            fakes[key].set_shared_state(FakeSharedState())
        self.graph = build_graph(
            classifier=self.classifier,
            agents={
                "consultation": fakes["consultation"],
                "recommendation": fakes["recommendation"],
                "trial": fakes["trial"],
                "lesson": fakes["lesson"],
                "reschedule": fakes["lesson"],
                "cancel": fakes["lesson"],
            },
            checkpointer=InMemorySaver(),
            handoff_service=self.handoff,
            availability_handler=availability,
            session_chat_history=None,
        )
        self.cfg = {"configurable": {"thread_id": self.thread_id}}

    def turn(self, text: str) -> str:
        payload = new_state(text, self.thread_id, flow_state=self.flow_state)

        async def _go():
            parts = []
            async for token in self.graph.astream(payload, self.cfg, stream_mode="custom"):
                if isinstance(token, str):
                    parts.append(token)
            snapshot = await self.graph.aget_state(self.cfg)
            return "".join(parts), (snapshot.values or {}).get("flow_state")

        out, self.flow_state = asyncio.run(_go())
        return out


# ----------------------------------------------------------------------
# 对拍执行
# ----------------------------------------------------------------------

@dataclass
class TurnCompare:
    scenario_id: str
    turn_index: int
    user_input: str
    legacy_raw: str
    graph_raw: str
    legacy_normalized: str
    graph_normalized: str
    matched: bool


@dataclass
class ScriptResult:
    scenario_id: str
    name: str
    note: str
    turns: List[TurnCompare]

    @property
    def matched(self) -> int:
        return sum(1 for t in self.turns if t.matched)

    @property
    def total(self) -> int:
        return len(self.turns)


def run_script(scenario: Scenario) -> ScriptResult:
    """一条脚本在两条链路上各跑一遍，逐轮规范化比较。"""
    classifier = scenario.make_classifier()   # 同一分类器实例被两条链路共享
    legacy = LegacyChain(scenario, classifier, session_id=f"parity:{scenario.sid}:legacy")
    graph = GraphChain(scenario, classifier, thread_id=f"parity:{scenario.sid}:graph")

    token = (set_current_identity(scenario.identity)
             if scenario.identity is not None else None)
    try:
        turns = []
        for index, text in enumerate(scenario.turns, 1):
            legacy_out = legacy.turn(text)
            graph_out = graph.turn(text)
            legacy_norm = normalize(legacy_out)
            graph_norm = normalize(graph_out)
            turns.append(TurnCompare(
                scenario_id=scenario.sid, turn_index=index, user_input=text,
                legacy_raw=legacy_out, graph_raw=graph_out,
                legacy_normalized=legacy_norm, graph_normalized=graph_norm,
                matched=legacy_norm == graph_norm,
            ))
        return ScriptResult(scenario.sid, scenario.name, scenario.note, turns)
    finally:
        if token is not None:
            reset_current_identity(token)


def run_parity(scripts: Optional[List[Scenario]] = None) -> List[ScriptResult]:
    """对拍全部脚本，返回逐轮比较结果。"""
    return [run_script(s) for s in (scripts or PARITY_SCRIPTS)]


def consistency_rate(results: List[ScriptResult]) -> float:
    """总一致率 = 命中轮数 / 总轮数。"""
    total = sum(r.total for r in results)
    matched = sum(r.matched for r in results)
    return matched / total if total else 1.0


# ----------------------------------------------------------------------
# 测试
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def parity_results():
    return run_parity()


def test_normalize_strips_only_known_acceptable_diff():
    graph = "[THOUGHT][LangGraph] 意图识别为 trial。[THOUGHT][归类机器人] 这是一个试讲预约请求，转给试讲顾问处理。[REPLY]ok"
    legacy = "[THOUGHT][归类机器人] 这是一个试讲预约请求，转给试讲顾问处理。[REPLY]ok"
    assert normalize(graph) == normalize(legacy) == legacy
    assert normalize("[THOUGHT][LangGraph] 意图识别为 consultation。").strip() == ""


def test_scenario_coverage_covers_all_spec_categories():
    ids = {s.sid for s in PARITY_SCRIPTS}
    assert REQUIRED_SPEC_IDS <= ids, f"缺少场景: {sorted(REQUIRED_SPEC_IDS - ids)}"
    assert len(PARITY_SCRIPTS) >= 14


def test_script_count_is_at_least_14(parity_results):
    assert len(parity_results) >= 14


def test_consistency_rate_is_100_percent(parity_results):
    rate = consistency_rate(parity_results)
    mismatches = [
        (r.scenario_id, r.name, t.turn_index, t.user_input,
         t.legacy_normalized, t.graph_normalized)
        for r in parity_results for t in r.turns if not t.matched
    ]
    assert rate == 1.0, f"一致率 {rate:.4f}，残留差异: {mismatches}"


@pytest.mark.parametrize("scenario", PARITY_SCRIPTS, ids=[s.sid for s in PARITY_SCRIPTS])
def test_script_legacy_and_graph_are_verbatim_equal(scenario):
    result = run_script(scenario)
    for turn in result.turns:
        assert turn.matched, (
            f"[{result.sid} {result.name}] 第 {turn.turn_index} 轮不一致\n"
            f"  输入     : {turn.user_input!r}\n"
            f"  legacy   : {turn.legacy_normalized!r}\n"
            f"  graph    : {turn.graph_normalized!r}"
        )


def test_legacy_and_graph_share_same_classifier_object():
    """两条链路必须注入同一分类器实例（保证分类行为可比）。"""
    scenario = PARITY_SCRIPTS[0]
    classifier = scenario.make_classifier()
    legacy = LegacyChain(scenario, classifier, session_id="share:legacy")
    graph = GraphChain(scenario, classifier, thread_id="share:graph")
    assert legacy.agent.classification_processor.task_classifier is classifier
    assert graph.graph is not None


def test_doubles_avoid_real_services():
    """对拍全程不构造真实 LLM / DB：legacy 的 llm 是占位模型的 RunnableSequence。"""
    scenario = PARITY_SCRIPTS[0]
    legacy = LegacyChain(scenario, scenario.make_classifier(), session_id="offline:legacy")
    assert isinstance(legacy.handoff, FakeHandoffService)
    assert isinstance(legacy.fakes["trial"], FakeBusinessAgent)
    # 占位模型是 RunnableLambda 组合，不是真实 ChatOpenAI/AzureChatOpenAI。
    assert type(legacy.agent.llm).__module__.startswith("langchain_core")
