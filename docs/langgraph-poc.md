# PoC：用 LangGraph 表达主控编排

**最后更新：2026-10-06**
**范围：** 对应 `.claude/tasks/langgraph-poc.md`。结论依据实跑，不改现有业务 Agent，默认走旧链路。

---

## 0. 第 0 步：API 核实结论

实跑环境：项目内 `.venv`（Python），**未新增任何第三方依赖**。

| 包 | 版本 |
| --- | --- |
| langgraph | 1.2.11 |
| langgraph-checkpoint | 4.2.0 |
| langgraph-prebuilt | 1.1.0 |

核实命令与实际结论：

```powershell
# 1) langgraph 顶层没有 __version__（任务文档给的命令会报 AttributeError）；
#    版本用 pip show 读取，符号用 langgraph.graph 列出。
.\.venv\Scripts\python.exe -c "import langgraph.graph as g; print([n for n in dir(g) if n[0].isupper()])"
# -> ['END', 'MessageGraph', 'MessagesState', 'START', 'StateGraph']

# 2) 内存 checkpointer 路径与任务文档一致
.\.venv\Scripts\python.exe -c "from langgraph.checkpoint.memory import InMemorySaver; print('InMemorySaver OK')"
# -> InMemorySaver OK

# 3) 人机接管符号路径与任务文档一致
.\.venv\Scripts\python.exe -c "from langgraph.types import interrupt, Command; print('interrupt/Command OK')"
# -> interrupt/Command OK

# 4) 消息归约器（任务文档指定用 add_messages）
.\.venv\Scripts\python.exe -c "from langgraph.graph.message import add_messages; print('add_messages OK')"
# -> add_messages OK

# 5) 逐 token 外发（保持 [THOUGHT]/[REPLY] 流式协议）
.\.venv\Scripts\python.exe -c "from langgraph.config import get_stream_writer; print('get_stream_writer OK')"
# -> get_stream_writer OK
```

**实际可用 import 路径（1.2.11 已核实）：**

- `from langgraph.graph import StateGraph, START, END`
- `from langgraph.graph.message import add_messages`
- `from langgraph.checkpoint.memory import InMemorySaver`
- `from langgraph.types import interrupt, Command`
- `from langgraph.config import get_stream_writer`

**运行语义（实跑确认）：**

- 编译产物类型为 `CompiledStateGraph`，有 `astream / ainvoke / aget_state`。
- `interrupt(payload)` 未恢复时：`ainvoke` 返回值带 `__interrupt__`；`astream(stream_mode="custom")` 在该节点无 token 输出，并在该节点暂停。
- 同一个 `thread_id` 连续两轮 → 第二轮能读到第一轮写入的 `messages`（`add_messages` 累计）。
- `aget_state(cfg).tasks[i].interrupts` 非空 → 该会话停在人工接管点；此时下一轮用 `Command(resume=...)` 恢复并继续 `-> END`。

---

## 1. 改动文件清单

**新增（旁路，默认不启用）：**

- `agents/graph/__init__.py` — 对外暴露 `build_graph / route_by_intent / TutorState / new_state`。
- `agents/graph/state.py` — `TutorState(TypedDict)`：`messages`(add_messages) / `flow_state`(复用 `StateEnum.value`) / `intent` / `slots` / `session_id` / `handoff`。
- `agents/graph/nodes.py` — 只做转发的节点：`classify_node` + `consultation_node / trial_node / lesson_node / reschedule_node` + `unrelated_node`；`emit()` 用 `get_stream_writer()` 逐 token 外发。
- `agents/graph/build.py` — `route_by_intent`（纯函数）、`build_graph(classifier, agents, checkpointer)`、`TaskClassifierAdapter`（惰性兜底，不缺依赖）。
- `tests/test_langgraph_poc.py` — 新增 26 个用例。
- `scripts/demo_langgraph_poc.py` — 离线实跑 demo，三组证据、exit 0。

**修改（仅 1 处，且默认关闭）：**

- `api/chat_handler.py` — `ProcessUserInput_stream` 增加 `ORCHESTRATOR` 开关分支（默认 `legacy`）；新增 `_process_graph_stream`、`_build_orchestrator_graph`、`_build_graph_deps`、`_ConsultStreamAdapter`、`reload_graph` 等旁路辅助函数。legacy 函数体**零改动**。

**未修改：** `agents/task_classification/*`、`agents/task_classification_agent.py`、`config/*`、各业务 Agent、web 层、既有任何测试。

依赖方向：`agents/graph/` 只 import `agents/` + `config/`，**未**反向依赖 `api/`（`api/chat_handler.py` 单向调用图）。

---

## 2. 新增测试数

**26 个**（`tests/test_langgraph_poc.py`），分组：

| 组 | 数量 | 覆盖 |
| --- | --- | --- |
| 图编译 | 2 | `build_graph().compile(checkpointer=...)` 成功；默认依赖无需初始化 KB/LLM/DB |
| `route_by_intent` 纯函数 | 11 | 参数化 4 类 intent + 6 类兜底 + 空串；纯函数无副作用 |
| 4 条路径落到预期节点 | 5 | 4 条各一次 + unrelated 兜底，替身 agent 记录被调用 |
| checkpointer 跨轮持久 | 2 | 同线程第二轮读到第一轮 messages；不同 thread_id 隔离 |
| interrupt + resume | 2 | 转人工 interrupt → `Command(resume=...)` → END；正常试讲不 interrupt |
| 入口开关 | 3 | 默认 legacy；默认路径不走图；`ORCHESTRATOR=graph` 才走图 |

全量：**1115 → 1141 passed**，增量恰为新增 26，无既有测试需要修改。

---

## 3. demo 实跑输出

```text
D:...> .\.venv\Scripts\python.exe scripts/demo_langgraph_poc.py
====================================================================
证据 1：4 条路由各走对节点
====================================================================
  [OK] 孩子数学怎么学        -> intent=consultation  命中 agent=consultation  产出=[THOUGHT][LangGraph] 意图识别为 consultation。...
  [OK] 想约一节试听课        -> intent=trial         命中 agent=trial  产出=[THOUGHT][LangGraph] 意图识别为 trial。[THOUGH...
  [OK] 帮我预约明天的课       -> intent=lesson        命中 agent=lesson  产出=[THOUGHT][LangGraph] 意图识别为 lesson。[THOUG...
  [OK] 把周三的课调到周五      -> intent=reschedule    命中 agent=reschedule  产出=[THOUGHT][LangGraph] 意图识别为 reschedule。[T...
  [OK] 今天天气如何         -> intent=unrelated      命中 agent=unrelated_node

====================================================================
证据 2：跨轮状态恢复（同一 thread_id，checkpointer=InMemorySaver）
====================================================================
  第一轮后 messages 数：2
  第二轮后 messages 数：4
  第二轮读到第一轮内容：True
  线程内累计轮次（agent 调用次数）：2

====================================================================
证据 3：interrupt() 人工接管 + Command(resume=...) 恢复
====================================================================
  第一轮暂停：pending interrupt = {'reason': 'human_handoff'}
  Command(resume='客服已介入') -> [REPLY][人工接管] 已为您转接人工客服。恢复输入：客服已介入
  最终状态 handoff = True

====================================================================
结论：路由=True  跨轮持久=True  interrupt 恢复=True
====================================================================
# exit code = 0
```

---

## 4. `pytest tests/ -q` 完整结果原文

```text
1141 passed, 2879 warnings in 188.23s (0:03:08)
```

（相对任务给出的基线 **1115 passed**，增量 26 = 本次新增用例数；无既有用例被修改或跳过。warnings 为项目既有 DeprecationWarning，与本 PoC 无关。）

---

## 5. 判断："这套结构能否承载全量迁移"

**结论：能承载，但需要补齐两块，因此"值不值得全量迁"是工程取舍而非可行性问题。**

逐项对照现有自研编排：

1. **续轮态能否用条件边表达？——能。**
   现有 `StateManager` 的 7 个态（`CLASSIFY / CONSULTATION / RECOMMENDATION_CONFIRM / TRIAL / TRIAL_CONFIRM / TRIAL_CONVERT / LESSON`）都是"上一轮结论决定本轮入口"。LangGraph 里这就是**入口条件边**：`route_by_intent` 已证明纯函数式路由可行，续轮态只需在读 checkpointer 得到的 `flow_state` 上做同样的映射（如 `TRIAL_CONFIRM -> trial_node`）。PoC 的 `flow_state` 字段与 `StateEnum.value` 同构，迁移时不改枚举语义。

2. **interrupt 能否替代自研 handoff？——能，且更规范。**
   自研 handoff（`services/human_handoff_service.py` + `[REPLY][HANDOFF]` 协议 + transcript 快照）本质是"暂停等人工"，LangGraph 的 `interrupt()` + `Command(resume=...)` 原生表达同一语义，且 checkpointer 自动保存暂停点，无需自定义快照。PoC 已实跑：暂停 → 检测 pending interrupt → resume → 到 END。

3. **checkpointer 能否替代 StateManager + session_checkpoint？——能，是最大收益点。**
   现有 `FlowState` 是进程内枚举，`services/session_checkpoint.py` 手写草稿快照；checkpointer 提供同一 thread_id 的原子状态持久化（PoC 已证明跨轮 messages 持久）。迁移后 `StateManager` 的内部状态可退化为图状态的一部分，`session_checkpoint` 的手写序列化可以删除。**但**：PoC 用 `InMemorySaver`，进程重启即丢——生产要持久化必须引入 `SqliteSaver`（需新增依赖 `langgraph-checkpoint-sqlite`，**未擅自安装，待确认**）。这是全量迁移的前置条件，也是主要成本。

4. **风险 / 成本（迁移前需确认）：**
   - `agents/context/*` 的预算控制、`ContextBudgetExceeded`、匿名身份校验 `_requires_business_verification` 等**当前在 `ClassificationProcessor` 内**，迁到图时要决定这些横切职责放哪个节点，避免业务逻辑漏搬。
   - 现有 `[THOUGHT]/[REPLY]` 文本协议必须逐字保持——PoC 通过 `get_stream_writer()` + `stream_mode="custom"` 已验证可原样外发。
   - `TaskClassifier` 的分类归一必须复用（PoC 已通过 `_build_graph_deps` 复用真实 `TaskClassifier`，保证图/legacy 分类结果一致）。

**总体判断：** 结构可行、收益明确（白拿 checkpointer + interrupt，可删 `StateManager`/`session_checkpoint` 的手写持久化）。是否全量迁，取决于两点：是否接受为 SQLite/Postgres checkpointer 引入依赖，以及是否愿意把 `ClassificationProcessor` 里的横切逻辑（预算/身份校验/档期白名单）重新分配到图节点。PoC 阶段**默认 legacy**，可零风险并行验证，再决定迁移节奏。
