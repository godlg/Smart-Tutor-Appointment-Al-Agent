# 咨询真实模型流式（增量）

## 范围与调用链

仅改变咨询回答生成和该路径的流生命周期，不改变分类、JSON 提取、预约执行、匿名业务权限、记忆预算或 checkpoint 恢复策略。

```text
/chat/stream 或 /chat
  → StreamingResponse.token_generator
  → ProcessUserInput_stream（每次推进/关闭配对身份 ContextVar）
  → _process_session_stream（会话锁、唯一 human/ai 写入、checkpoint）
  → TaskClassificationAgent.classify_task_stream
  → ClassificationProcessor.process_task_stream
  → AgentRouter.route_to_consultation / route_by_state 的咨询分支
  → ConsultantAgent.consult_stream
  → ConsultationProcessor.process_consultation_stream（检索）
  → ResponseGenerator.generate_response_stream（上下文/预算 → llm.astream）
```

上述咨询转发层使用 `contextlib.aclosing` 显式关闭内层异步生成器；`api/chat_handler.py` 的身份、partial、锁和 checkpoint 实现未改动。

非流式 `ResponseGenerator.generate_response` 仍调用 `ainvoke`。主控意图分类和咨询 YES/NO 分类也仍为非流式调用，不把 JSON 或有业务副作用的执行改成 token 流。

## 输出协议和可见内容

- 模型正文通过 `llm.astream` 到达后逐块立即 `yield`，不先等待全文、不人为拆字、不引入后台缓冲队列。
- 第一块非空正文前输出一次 `[REPLY][咨询机器人]`；标签不是“正文首达”的证明。
- `chunk.content` 是字符串时直接取正文。
- `chunk.content` 是 list 时仅拼接本块内 `type == "text"` 且 `text` 为字符串的内容；不序列化整个块或其他字段。
- 忽略 None、空串、空 list、未知块、非字符串 text、reasoning/tool-call 块以及 `additional_kwargs`、`response_metadata` 等元数据，不输出 `str(None)` 或字典。
- 本过滤只认 SDK 标明的正文类型，不解释或显示思考字段；不声称能辨别供应商错误地塞入正文 text 的内部内容。
- 前端 `web/templates/index.html` 已用 `res.body.getReader()` 循环读取，每次 decode 后调用 `updateLastBotMessage`，没有等待全文才更新。本次不修改 UI。

## 错误与取消语义

| 情况 | 对用户输出 | 成功 consultation 行为 | 主入口 AI 历史 |
| --- | --- | --- | --- |
| 有正文且模型正常结束 | 单一 REPLY 标签 + 按块正文 | 一次 | 一次 complete |
| 首正文前模型异常 | REPLY 标签 + 单一 `[ERROR]` 安全提示 | 不写 | partial |
| 正文中途/结尾抛错 | 保留已发正文，追加单一 `[ERROR]` 安全提示，不重发 REPLY | 不写 | partial，保留已发部分 |
| 正常结束但没有任何正文 | 明确 `[ERROR]` 失败提示 | 不写 | partial |
| 取消或客户端断连 | 不补发“成功”或新的错误全文 | 不写 | 已有可用内容为 partial，保留 human |
| 原有上下文预算拒绝 | 保持原“请缩短输入后重试”协议 | 不写 | 沿用原有预算拒绝处理，不改策略 |

模型生成层把普通异常转换为 `ConsultationStreamError`，不输出上游原始异常。咨询处理器是生成失败转用户 `[ERROR]` 的单一边界；自身检索失败也用相同安全提示。日志仅记录异常类型，不记录模型异常的原始消息或 traceback。

`CancelledError` 与 `GeneratorExit` 不属于这里捕获的 `Exception`，会继续向外传播。`aclosing` 让入口关闭立即传至模型流，不依赖事件循环退出时的异步生成器垃圾收尾。正常完成之前关闭，不执行后续成功 consultation 写入。主入口原有 `[ERROR]` 检测将流式失败保存为 partial；同一会话锁在收尾后释放。checkpoint 保存失败仍由既有安全边界处理，不伪造完成、不重复历史。

## 首正文指标和验真

定义：请求开始至客户端实际读到**来自模型的非空正文**。HTTP header、归类 `[THOUGHT]` 或 `[REPLY]` 标签均不计为正文首达。

确定性验收使用可控事件屏障：fake model 发出第一块正文，后续生成必须等待测试/HTTP 客户端设置 release 事件。客户端在 release 未设置、模型 done 为 False 时读到第一正文，之后才释放屏障验证完整收尾。超时只用于防止测试死锁，不通过“睡眠少于某阈值”判定真流式。

```powershell
.venv/Scripts/python.exe -m pytest tests/test_consultation_real_stream.py tests/test_consultation_real_stream_chain.py tests/test_consultation_real_stream_demo.py -q
.venv/Scripts/python.exe scripts/demo_consultation_real_stream.py
.venv/Scripts/python.exe -m pytest tests/ -q
```

### 实际 HTTP demo

`scripts/demo_consultation_real_stream.py`：

- `TemporaryDirectory` 中的真实 SQLite、真实 repositories 和身份 cookie/session 绑定。
- 复用 `demo_memory_reliability.py` 已验收的离线 containment：关闭 dotenv，禁止外部 DNS/TCP/UDP，SQLite 只允许指定临时库，禁止 ATTACH 和预约/支付/调退课执行。
- 只替换 LLM/embedding 边界；咨询、主控、上下文、检索、存储和 Web 路由均为真实实现。
- 单独创建无生产 lifespan 的 FastAPI 实例，仅挂 Web router；不导入或启动 `app.py`。
- uvicorn 在 `127.0.0.1:0` 的已绑定 socket 启动，使用系统临时端口；httpx 用真实 TCP HTTP stream，`trust_env=False`，不是 ASGITransport/TestClient。
- 两路接口都在模型屏障释放前读到正文；另外验证中途错误 partial 和真实客户端断连时模型关闭、partial、锁释放。
- 服务器 ready/模型等待/关闭用事件同步；正常收尾关闭服务器、socket、所有 DB engine 并删除临时目录，无遗留子进程。
- 结果行：`FIRST_BODY_BEFORE_MODEL_DONE=True`、`COMPLETE_BODY=True`、`MIDSTREAM_PARTIAL=True`、`CANCEL_CLEANUP=True` 和 `STREAM_DEMO_RESULT` JSON。

这证明应用和本地 HTTP 链路不再等待模型全文，不是在线供应商 TTFT 测量。真实外部模型是否及时发送正文、部署反向代理是否缓冲、公网耗时等仍需在另行授权的在线环境验收；本次未读取密钥、未访问外网。

## 首正文之前仍存在的耗时/阻塞点（未扩 scope）

1. **身份与历史恢复**：Web `_chat_response` 的同步身份事务在线程池中执行；但主入口 `ensure_session`、`record_human`、懒构建 agent/恢复历史、加载 checkpoint 等仍有同步 SQLite 查询和事务，可能阻塞其所在事件循环任务。
2. **同 session 串行锁**：`asyncio.Lock` 等待本身不是同步阻塞，但排在同会话上一个请求后，增加首正文等待；本次不绕过锁。
3. **主控分类与咨询分类**：仍须等待各自 `ainvoke` 的完整意图/YES-NO 结果。这是前置网络等待，不是咨询回答的伪流式；本次不改分类协议。
4. **进入咨询的知识初始化**：`ConsultantAgent.__aenter__ → KnowledgeRetriever.initialize → KnowledgeService.initialize` 尽管入口是 async，其内部默认知识写入、同步 `embed_input`、DB 文档加载和 FAISS 索引构建仍是同步工作。
5. **检索前上下文与画像**：`_build_rag_query_context` 的身份校验、摘要读取、最近历史组装，以及 `_build_memory_profile_text` 的画像/行为读取为同步工作；学科/年级规则抽取也同步执行。
6. **RAG 检索**：dense/hybrid 接口虽然 async，内部 SQL 候选过滤、同步 query embedding、FAISS 查询、逐条取文档仍可阻塞；hybrid 的 BM25 构建/分词、RRF、可选 reranker 推理也同步。本离线 demo 固定 dense 并关闭 rerank，不声称覆盖线上 hybrid 性能。
7. **回答 prompt 构建和预算**：`ResponseGenerator._build_prompt → ContextManager.build` 同步执行身份约束、记忆向量召回/文本摘要/画像读取、token 估算和分配；触发 rollup 时摘要构造、归档事务和水位写入也在开始 `astream` 前完成。预算及归档失败保护保持原样。
8. **正文后的收尾**：成功行为写入、AI complete/partial 存储、checkpoint 保存仍有同步 DB 操作，可能影响最终 EOF/锁释放或其他任务。这里不是首正文前置耗时，但未在本增量中改成后台持久化。

真实流式不等于所有前置工作都已异步化，也不保证固定毫秒级首正文。本增量只去掉“咨询模型全文完成后才能发正文”的等待。
