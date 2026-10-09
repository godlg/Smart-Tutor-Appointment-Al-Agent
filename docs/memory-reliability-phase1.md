# 第一阶段：记忆可靠性闭环

**最后更新：2026-10-03**  
**范围：** 对应 `.claude/tasks/memory-reliability-phase1.md` 第 62 行的架构与使用边界文档。本文依据当前实现说明行为，不代替测试或 demo 的实际执行报告。

## 1. 实际架构与职责

```text
app.py 挂载 web/routes.py
  POST /chat 或 POST /chat/stream
    → api/identity.py：解析匿名 cookie，持久绑定/校验 owner、student namespace
    → 请求级 Identity / SessionContext（ContextVar）
    → api/chat_handler.py：ProcessUserInput_stream
        → 会话 active 保护 + 每 session 的 async 锁
        → ConversationService：先记录 human
        → 重建/复用主控及业务 Agent，安全恢复 checkpoint
        → classify_task_stream：分类、咨询、公开老师匹配等
        → 过滤 [THOUGHT]，保存 complete / partial ai，保存草稿 checkpoint

agents/context/runtime.py
  → ContextManager（实际聊天启用预算及滚动摘要）
      → ConversationContext：读取上述唯一可见历史 provider
      → system / tools / structured_output：必需指令
      → MemoryContext：咨询路径启用，按 identity/student scope 读取记忆
      → 近期滚动摘要：显式注入，不依赖向量碰巧召回

ConversationService
  → 同一个 DatabaseRouter / SQLAlchemy SessionManager
      → IdentityRepository：身份、namespace 与不可变会话绑定
      → ConversationRepository：顺序消息、摘要区间、checkpoint
      → LongTermMemoryService：经同 router 原子归档摘要及长期记忆
```

关键实现位于 `web/routes.py`、`api/identity.py`、`api/chat_handler.py`、`services/identity_service.py`、`services/conversation_service.py`、`services/session_checkpoint.py`、`agents/context/runtime.py`、`agents/context/manager.py`、`agents/context/contexts/{conversation,memory}.py`，以及对应的 `db/repositories/`、`db/base/schema_migrator.py`。

### 正常入口与 legacy 的历史职责

- 正常主控入口是用户可见消息的唯一写入者。human 在处理前持久化；ai 在回复收尾时保存，路由调试 `[THOUGHT]` 不进入可见历史。不能以文本相同为理由删除用户再次发送的消息。
- 服务返回 `ManagedChatHistory`，其 `entry_managed=True` 使 Trial / Recommendation 等业务 Agent 不再重复写共享历史。分类、咨询及业务上下文从同一 provider 读取上游对话。
- 业务 Agent 的局部 `chat_history`、字段提取 JSON、工作状态仍有各自职责，不等于用户看到的问答，也不应混入统一可见历史。重置局部提取状态不应清空共享对话。
- 旧内部 Python 调用可不传 identity，按 legacy 路径使用 `default_user`；未注入 managed provider 的旧 Agent 仍保留原有局部/共享历史行为。**正常网页不会降级到这个身份，也不会自动继承其记忆。**
- `ConversationContext` / `ContextManager` 独立构造时仍保留兼容默认；正常主控通过 runtime 装配才启用本阶段的统一存储、预算和 rollup，不能把独立 legacy 构造与正常入口混为一谈。

## 2. 匿名身份与两路 Web 入口

### cookie 的实际含义

`Identity`（别名 `SessionContext`）携带 `user_id`、可选 `student_id`、`session_id`、`legacy`。网页身份来自服务器以 `secrets` 生成的不可预测 `anon_*` 随机值，并登记在 `identities` 表；cookie 名为 `tutor_uid`。缺失或未登记的 cookie 会触发新身份签发。请求体中的 `user_id` **不作为授权依据**。

cookie 设置为 HttpOnly、SameSite=Lax、Path=/，有效期一年（365 天）。HTTPS 请求始终设置 Secure。`TUTOR_IDENTITY_COOKIE_SECURE` 取 `1/true/yes/on`（忽略大小写、两端空白）时可强制开启 Secure，适用于 HTTPS 终止代理后方；其余值不会关闭 HTTPS 自带的 Secure。实现不直接信任客户端转发头。反向代理部署需要正确配置服务器感知的协议或强制 Secure；本地 HTTP 默认不带 Secure。

**这是匿名浏览器凭据，不是已登录家长、学生或客服账号。** HttpOnly 不意味着凭据无法被窃取或重放；持有同一有效 cookie 即持有同一匿名身份。清除 cookie 或换浏览器通常得到新身份，当前没有账号级找回、跨设备认证或真实家长关系证明。

### 会话与学员 namespace

- `/chat` 与 `/chat/stream` 共用同一身份/ownership 接线。持久绑定校验在构造 `StreamingResponse` 前完成；同步 SQLite 预处理在线程池进行。Web 不重复记录问答。
- 未传 `session_id` 时服务器生成 `sess_*`，不会回落到 global/legacy 会话。正常聊天响应以 **`X-Tutor-Session-Id`** 返回实际绑定的 id，客户端应保留并在续聊时发送。
- 新会话绑定 `(session_id, user_id, student_id)`；已有会话的 owner 与 student scope 均不可更改。跨身份访问或抢占已有会话返回 **403**；同 owner 改变既有 student scope、或 namespace 已被另一身份认领返回 **409**。存档存在但缺少可信 owner、归属混杂时拒绝访问，不能重新认领以泄露旧消息。
- `session_id` 必须为 1–256 字符，首字符为 ASCII 字母/数字，其余仅允许 ASCII 字母、数字及 `_ . : -`。不安全 id（包括控制字符、换行等）由请求模型返回 **422**，早于身份绑定及 header 输出。`student_id` 有 1–256 字符的长度校验，但不具有真实业务学员授权语义。
- 续聊省略 `student_id` 时，仅对同 owner 的既有会话恢复其 namespace；显式 `null` 也属于 scope 选择，不能借此清除原学员绑定。切换 scope 应创建新会话。
- 长期记忆按 `(user_id, student_id)` 精确过滤。`student_id=None` 是独立家庭级范围；学生级不会自动合并家庭级，也不会混入另一学生级记忆。同匿名身份、同 scope 的不同会话可以读取该范围的长期记忆，但不会把不同 session 的原始对话历史合并。
- `identity_students` 目前对 namespace 标识做全局唯一认领，重名冲突会拒绝；**认领只是记忆隔离规则，不是认领真实学员。** 它不查询业务学员表，也不能授权读取 source-of-truth 学员业务数据。即使 namespace 恰好等于某个业务 student id，也没有新增业务访问权。

### 未登录聊天的敏感业务路由

`agents/task_classification/classification_processor.py` 对非 legacy 身份显式拦截 `trial`、`lesson`、`reschedule`、`cancel`、`progress` 分类，以及已进入 `trial` / `trial_confirm` / `trial_convert` / `lesson` 的续轮状态。返回要求先验证家长身份及学员归属的说明，不继续调用这些敏感业务路由；匿名用户仍可咨询课程或匹配公开老师。

当前没有供该匿名身份升级为真实已认证家长的登录流程。**只有 legacy 内部原流程保留兼容执行行为，不能据此宣称 Web 用户已经认证。** 因而正常网页虽然仍能发消息、咨询和选择老师，但原先的“推荐→确认→预约”聊天闭环不再可用，真实家长使用匿名网页也会被拦截；验证提示不是已经存在的登录升级能力。这是明确的业务兼容影响，不是无损接入。

此保护针对上述聊天主控路径，不等于所有独立业务 API 都已实施全量认证/RBAC。`/agentic/chat`、直接试讲/课时/系列课等 API 尚未统一接入该身份与对象归属校验；排课日历仍直接调用系列课 API。因此不能宣称“全站匿名预约已阻断”或“整站已安全”，公网部署前仍须单独处理这些入口的访问控制。预约、课时、付款及真实学员进度等业务数据库仍是 source of truth，摘要和 namespace 都不能代替业务授权或业务事实。本轮只修 7 个验收失败，不新增认证系统或改造这些独立端点。

## 3. 配置默认值与预算口径

`config/memory.py` 只读取进程环境，不自行加载 `.env`；本阶段不要求修改 `.env`。下列整数变量接受正整数，缺失、无效或非正值回退到默认值。

| 配置 | 默认值 | 正常 runtime 的用途 |
| --- | ---: | --- |
| `CONTEXT_MAX_TOKENS` | 8192 | 总上下文预算，包含输出预留 |
| `CONTEXT_OUTPUT_TOKENS` | 1024 | 先从总预算扣除的输出预留 |
| `CONTEXT_KEEP_RECENT_ROUNDS` | 4 | rollup 保留近期消息及压缩的优先窗口 |
| `CONTEXT_ROLLUP_ROUNDS` | 4 | 每次滚动摘要的分段规模 |
| `CONTEXT_SUMMARY_MAX_CHARS` | 600 | 单段确定性摘要及近期摘要视图的字符上限 |
| `SESSION_CACHE_TTL_SECONDS` | 1800 | 闲置缓存 TTL，单位秒 |
| `SESSION_CACHE_MAX_ENTRIES` | 128 | 闲置会话淘汰时的容量目标 |
| `TUTOR_IDENTITY_COOKIE_SECURE` | 未设置 | 可强制 Secure；不能降低 HTTPS 的 Secure |

`ContextManager` 的 rollup 窗口比例默认是 0.6，通过构造参数可注入，并非上述环境变量之一。这里的“轮”以每轮两条消息换算，默认分段 8 条、保留最近 8 条，不保证 partial/异常后的消息恰好构成完整问答轮。

### 硬预算

- 默认 `token_counter` 是 `agents/context/types.py` 的 **UTF-8 字节数**：常用汉字通常按 3 计。它是离线、偏保守的纯文本计数，**不是模型账单 token 数**，也不覆盖模型消息封装或特殊 token。可注入匹配目标模型的计数器。
- 默认最终 prompt 可用额度为 `8192 - 1024 = 7168`，按同一计数器口径最终重新计数。输出预留是 prompt 装配策略，不等于已验证供应商生成输出的实际计费长度。
- system、当前问题、tools、structured output 及未被识别为可选内容的额外业务指令是必需项，不通过截断 JSON 指令或删除当前问题来满足预算。必需项自身超限抛 `ContextBudgetExceeded`；主入口转换为明确的 `[ERROR]`，要求缩短本次输入，错误回复不标为完整业务成功。
- 可选内容初始份额为 conversation 45%、summary 20%、knowledge 20%、memory 15%（按实际存在的块归一化）。先压缩较旧对话，再按类别额度截取；未用额度优先再分给近期对话、摘要、知识、记忆。最终仍超限时依次移除 knowledge、memory、summary、conversation 可选块，再做最终计数。
- 近期轮次是优先保留窗口，**不是任意极小预算下的不可裁剪保证**；超长单条历史、RAG 或摘要仍可能被截取/移除。必需业务指令与当前问题不参与这些裁剪。
- 每次 build 使用已收集的块及历史快照完成拟合，不为反复裁剪重新调用记忆 build、对同一 query 重复做 embedding/召回。咨询检索前的历史视图也有界，读取近期摘要及对话，不自行增加一次记忆 embedding。

## 4. 持久化、迁移与恢复保证

### 存储与事务

服务默认使用 `sqlite:///data/smart_appointment.db`，并可注入 `db_path` 或共享 `DatabaseRouter`。测试与 demo 应显式使用临时文件库；文件库用于证明服务重建/独立进程恢复，不能只用进程内 `:memory:` 声称跨进程持久化。

| 存储 | 作用/关键约束 |
| --- | --- |
| `identities`、`identity_students` | 随机匿名身份登记、记忆 namespace 认领 |
| `chat_sessions` | 会话 owner、student scope；session 唯一且绑定不可变 |
| `session_messages` | `(session_id, seq)` 唯一；role、content、scope、complete/partial 状态 |
| `session_summaries` | `(session_id, start_seq, end_seq)` 唯一；已提交区间决定水位 |
| `long_term_memories` | 同 scope 长期文本；rollup span 唯一，与摘要同时归档 |
| `session_checkpoints` | session 唯一；来源状态及 JSON 草稿包，不是执行日志 |

SQLite 写事务使用 `BEGIN IMMEDIATE`，使 owner 检查、seq 分配与写入位于同一写事务。消息 seq 在会话内递增；指定相同 seq 的精确重放可幂等，但同 seq 不同内容/scope/status 会拒绝。服务通常为每次新输入分配新 seq，**没有新增 HTTP 请求幂等键或业务 exactly-once 保证**。

热层只是持久库的缓存。正常记录按已知 seq 水位增量同步缺失后缀；显式 `restore_history` 按持久 seq 重建内容，但保持 provider/history/list 引用，保留 seq 与 partial metadata。重复恢复不重复追加消息；`drop_session` 后仍可从持久库恢复。

human 在业务处理前落库。取消/异常时 human 保留，ai 仅保存已产生且过滤后的可用内容为 partial，或者没有 ai；不能伪造完整答案。ai 或 checkpoint 保存失败有日志，但已返回给用户的回复不因此自动获得持久化保证。一个聊天轮次、业务写入与 checkpoint **不是一个总事务**。

### SQLite 幂等迁移与冲突库

`SessionManager` 先调用 ORM `create_all`，对文件型 SQLite 再调用现有 `db/base/schema_migrator.py`。迁移采用幂等建表、缺列检查后追加 `student_id` 及长期记忆的 `start_seq/end_seq`，并建立/核查核心唯一约束；不自动 backfill 未知身份，不把旧 `default_user` 数据授予新匿名用户，也不清除旧业务数据。

核心唯一索引创建失败或关键唯一约束缺失会使 `run_migration` 返回 False，文件型 SQLite 的 `SessionManager` 拒绝启动，而不是在缺少可靠约束时继续服务。**重复 seq/span、坏 schema 或归属冲突的旧库需要人工核查与修复；不会自动删除重复记录。** 幂等升级不意味着兼容任意损坏数据库；部署前应备份并在副本检查升级，不能用 stash/reset/clean 等方式处理数据库冲突。

### 重启与独立进程的边界

持久 cookie 登记、session owner/scope、按序消息、已提交摘要及水位可在重建服务或另一个 Python 进程中恢复，前提是访问同一文件库并提供同一认可身份。归属校验仍先于读取。水位来自已提交 `SessionSummary.end_seq` 的最大值，不依赖原进程的内存计数。

恢复不是重新执行历史请求，不保证重建原来的 LLM 内部状态，也不代表预约/支付已经成功。每 session 的 async 锁与 Agent 缓存是**进程内**机制；SQLite 事务保护写入，不等价于多 worker 间整段聊天推理的分布式串行锁。本阶段不宣称多进程并发业务 exactly-once。

## 5. 滚动摘要、长期记忆与失败安全

正常 runtime 启用 rollup。当输入窗口达到可用预算的默认 60%，且当前水位之后有足够的旧消息段时，一次 build 推进一个新区间，保留近期窗口。后续 build 从新水位继续推进，不会一直重复摘要最早一段；没有可归档新段时不写重复摘要。

默认 summarizer 是 **deterministic extractive fallback（确定性有界摘录）**：按句切分，以确认/约束、时间、老师、年级、学科、待办等关键词及日期时间形态评分，选取并截取原文片段。它不调用新增外部 LLM，不是智能语义总结，不保证理解指代、消歧、解决冲突或完整保留每个事实。可注入 summarizer，但扩展能力及效果不属于当前默认保证。

`ConversationService.save_rollup` 经共享 `LongTermMemoryService` / router 调用 `archive_rollup`，在**同一事务**写入 session 摘要与同 scope 长期 summary 文本。水位只从提交后的摘要推导。相同 span 的精确重复归档不新增记录；同 span 内容/scope 不同或只存在一半存档时拒绝，而不是静默推进。只有持久化成功才更新内存折叠水位。

归档失败时记录可观察日志，水位不推进，未存档原文不被当成已摘要内容隐藏；该次 build 将相关历史提升为必需块。如果原文因此无法满足硬预算，会明确报超限，而不是以丢弃它伪装成功。数据库里的原始消息不会因摘要或 prompt 裁剪而删除。

- `ContextManager` 从当前 session 的已提交摘要显式构造 `【近期滚动摘要】`，正文有字符上限及独立预算，**无需 embedding**。
- 咨询路径启用 `MemoryContext`；除向量召回外，还按相同 user/student scope 单独读取近期长期 summary 文本，文本最多 3 条，向量最多 5 条，已召回同 id 不重复展示。文本配额不被满额向量 Top-5 挤占，再受最终 memory 预算限制。
- 正常 rollup 使用 `auto_embed=False`，不新增归档 embedding 依赖。其他记忆写入或召回的 embedding 失败可降级/记录日志，不能阻断正常聊天；有文本摘要仍显式可读。摘要读取故障是可选依赖故障，ownership/scope 失败则必须拒绝，不能降级读取私有历史。

## 6. checkpoint：恢复草稿，安全失效执行状态

`services/session_checkpoint.py` 的 v1 checkpoint 仅保存可验证、限长、可 JSON 序列化的白名单草稿。主要需求字段为学科、年级、风格、偏好老师姓名、学生情况；时间和时长仅作为展示字段。姓名和时间不证明老师仍可预约，也不代表业务 id 有效。

恢复策略如下。

1. 先清除旧工作状态并回到分类态，重置 Trial / Lesson / Recommendation 的执行状态；局部清理不应影响共享可见历史。
2. 仅回填白名单需求字符串。`start_time`、`duration_minutes` 的旧值留在返回草稿供参考，不回填为旧执行参数；须重新说明时间、查询老师/档期并重新确认。
3. 不恢复可执行的 `pending_trial`、支付/套餐选择、调退课命令、旧确认、已选业务 teacher/package id 等；旧 `pending_ops` 仅识别是否有待办，绝不重放，新的主控 checkpoint 保存空执行列表。
4. 重建 Agent 首次加载待确认草稿或失效 checkpoint 时，该轮只返回恢复/失效 notice，**不把本次输入交给业务流程执行**。之后默认保留重确认门闩：问候、致谢、确认变体及缺少具体需求的输入仍提示重新说明，不进入业务流程，也不清除门闩。当前使用保守确定性规则，要求请求/学科与日期、时间或年级两类信号同时出现；确认、引用之前安排或致谢等输入默认不能解除，仅当输入开头明确重新发起“重新预约/安排/排课/调课/查询”等请求（可带已识别的礼貌引导或“我要/我想/请”）并具备上述详情时可重新分类；引用旧对话中的“重新预约”或被否定的请求不作为解除依据。例如“请重新确认之前说好的明天3点数学试讲”仍被拦截，而“好的，重新预约初中数学试讲，明天下午3点”可以重新查询。规则不是语义理解，部分合理表达会被要求重述；重新分类仍须重新查询、重新确认并受匿名业务验证限制。
5. checkpoint 读取或恢复抛错时不保存未恢复的 Agent 状态覆盖旧草稿，也不标记已加载；异常向调用方传播，下轮可重新加载。恢复成功后才允许该轮收尾保存 checkpoint。
6. 未知版本、坏 JSON、结构不符或任一工作状态清理失败，会使检查点安全失效：清空旧执行状态、提示重新说明，不把旧命令恢复成“可直接确认”。没有待确认需求的空检查点不必产生恢复提示。

**这是需求草稿恢复与安全重确认，不是完整 pending 断点续跑。** 持久化 `source_flow` 只用于判断历史风险，不表示恢复后仍处于旧确认态，也不能证明业务已执行成功。业务操作最终结果仍须查询业务数据库。

## 7. 会话并发与缓存

同 session 的输入在进程内 async 锁下串行，收尾持久化仍处于该锁内；不同 session 不通过一个包住 LLM 调用的全局锁串行。入口的每次流推进和关闭分别配对设置/重置 ContextVar，向调用方 yield 前恢复调用方身份，token 不跨任务沿用；关闭业务流后再完成持久化及释放锁。异常/取消或跨任务 aclose 后不遗留请求身份。

active 计数在排队前增加，正在运行及等待同 session 锁的请求均受保护。默认闲置 TTL 为 1800 秒、容量目标 128；请求进入/退出触发闲置淘汰，没有新增后台清理调度器。active 会话不淘汰，所以全活跃时可以暂时超过容量目标，并非绝对 128 硬上限。

`drop_session` 在非 active 时清理 Agent、共享历史、锁、访问时间及服务热层水位，但保留持久存档。重新进入时可重建历史及安全恢复草稿；这不是删除聊天数据的 API，也没有实现持久记录 TTL/账号注销清理。

## 8. 离线验证入口与未做项

以下命令由主执行者在项目根目录使用仓库虚拟环境运行。各轮实际 passed/warnings/耗时、RED/GREEN、demo 和未完成项以 `.claude/tasks/memory-reliability-phase1-result.md` 的真实执行记录为准；本文仅说明实现保证及边界。

```text
.venv/Scripts/python.exe -m pytest tests/test_memory_reliability.py -q
.venv/Scripts/python.exe -m pytest tests/ -q -k memory_reliability
.venv/Scripts/python.exe scripts/demo_memory_reliability.py
.venv/Scripts/python.exe -m pytest tests/ -q
```

第二条用于包含拆分后的 `test_memory_reliability_*` 用例，不能仅跑最初单文件就代表全部新增覆盖。`tests/conftest.py` 在 collection 前禁用 dotenv，禁外部 DNS/socket 连接（允许本地 loopback），关闭追踪和在线模型资源加载，并拦截/重定向默认生产 SQLite；显式临时路径保留，每个 test 默认应用库隔离。没有允许测试改用生产库的绕过开关。测试需要在 LLM/embedding 边界提供替身；独立子进程不自动继承 pytest monkeypatch，必须自行禁 dotenv/外网并注入临时库。

`scripts/demo_memory_reliability.py` 的离线入口使用 UTF-8 输出、TemporaryDirectory 及独立子进程恢复验证，检查两身份隔离/同身份跨会话/学生隔离、至少三次水位推进、归档失败、预算和旧 pending 安全失效。实际运行的输出与退出码记录在 `.claude/tasks/memory-reliability-phase1-result.md`，不能将脚本源码中的预期输出视为已经通过的结果。

本阶段没有新增或完成以下能力。

- AutoDream 自动记忆整理、偏好版本/冲突策略、智能语义摘要。
- 全量账号登录、真实家长/学员关系认证、账号找回与跨设备认证。
- 人工客服认证/RBAC，以及现有业务 API 的全量访问控制；不能据本阶段宣称整站安全。
- 真正模型 token 级流式改造；使用 StreamingResponse 或 `_stream` 命名不代表底层模型已完成真流式。
- 推荐/清理等调度重构、Redis/分布式会话锁、框架迁移。
- 完整业务 pending 自动续跑、预约/支付等执行重放或 exactly-once，以及任意损坏旧库的自动修复。

本文件只说明记忆可靠性的当前实现与边界；验收结论、真实命令摘要及未完成项另见结果报告。本轮不修改 `.env` 或生产数据库。
