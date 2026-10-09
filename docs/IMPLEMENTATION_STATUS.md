# 实现状态对照表 (Promise vs Reality)

> 最后核对: 2026-08-25
>
> 这份文档明确告诉读者 **README 中承诺的每项能力,实际落到代码的进度**。
> 已经实现的不重复说明;只列**有承诺但未实现/部分实现/已删除承诺**的条目。

## 全局说明

- 项目是早期学习项目 (README §"项目价值" 自述 2025 年起做,2026 年改写场景),
  架构分层是真的,RAG/多 Agent/Repository 这些核心能跑通;
- 但 README 中一些"听起来很美"的扩展能力(天气、短信、MCP、自我反思等)
  **从来没有真正接入**,属于"先在文档里写出来"的占位表述。
- 本表目的:读者/面试官打开 README 时,能立即知道**哪些是真功能,哪些是文档承诺**。

## 状态矩阵

| README 位置 | 承诺 | 实际状态 | 备注 |
|---|---|---|---|
| 核心能力 · 个性化提醒 | "结合实时天气、近期考试节点... '明天降温,记得让孩子带件外套'" | ❌ **未实现** | `.env.example` 有 `OPENWEATHER_API_KEY` 但 `grep -r openweather` 在 agents/services 里 0 命中;`requirements.txt` 有 `mcp>=1.0` 但 `grep -r "from mcp"` 在源码里 0 命中 |
| 核心能力 · 续费与流失预警 | `RenewalDetector` + `ChurnPredictor` | ✅ **已实现** | `agents/user_behavior/`,有完整单测 (`test_renewal_detector.py`) |
| 核心能力 · 多 Agent 协作 | "任务分类 Agent、咨询 Agent、试讲 Agent..." | ✅ **已实现** | 但**没有 Agent-to-Agent 直连**,全部经 `TaskClassificationAgent` 路由 |
| 核心能力 · Embedding 缓存优化 | "通过数据库缓存和文件缓存减少重复向量计算" | ⚠️ **部分实现** | DB 缓存有(embedding 字段落库);**没有文件缓存** |
| 系统架构 · 严格 5 层 | Web → API → Agents → Services → DB | ✅ **基本符合** | `db/KnowledgeDBRouter` 兼容类是死代码;其他层都单向 |
| Agent 设计 · Appointment Agent | "负责课程预约相关流程" | ⚠️ **分散实现** | 没有独立 `AppointmentAgent` 类,功能由 `TaskClassificationAgent` → `LessonAgent` 拆分完成 |
| Agent 设计 · 自我反思 | "让系统能够评估回答质量" | ❌ **未实现** | 后续规划 §"更强的 Agent 自主能力" 提到,从未做 |
| Agent 设计 · 后续规划 · Agent-to-Agent | "减少所有任务都依赖主分类器转发" | ❌ **未实现** | 当前 100% 走主分类器 |
| 后续规划 · 家长登录/权限 | "增加家长登录、权限控制和数据隔离" | ❌ **未实现** | 整个系统只有 1 个 `default_user` + 1 个默认家长;数据不隔离 |
| 后续规划 · Docker 部署 | "支持 Docker 部署" | ❌ **未实现** | 无 Dockerfile / docker-compose.yml / CI workflow |
| 试讲 Agent · 转化漏斗 | "推荐适合的套餐并可一键转为正式课" | ✅ **已实现** | `api/trial.py /convert` + `ConversionLogic` |
| Lesson Agent · 周固定系列课 | `LessonSeries` 按 `weekly_pattern` 自动展开 | ✅ **已实现** | `ScheduleGenerator` + 测试 14 个用例全过 |
| Lesson Agent · 调课级联 | `this_and_following` mode | ✅ **已实现** | `RescheduleHandler` + 测试 |
| 试讲成功提示 | "试讲前 1 小时会有短信提醒" | ❌ **不存在** | 已在 `trial_processor.py` 修复 #轮中删除该虚假承诺 |
| 试讲成功提示 | "试讲前 1 小时会有短信提醒" | ❌ **已删除** | 见 `agents/trial/trial_processor.py` 行 209 注释 |
| 老师匹配 · 价格适配 | "价格适配 ×0.5~1.0" | ⚠️ **未接 parent.budget_max** | `score_teacher_for_student` 中明确标 TODO |
| 后台调度器 | "推荐调度服务每天 9/14/19 点跑" | ✅ **修后真实落库** | `RecommendationService.generate_recommendations_job` 现在写 `user_recommendations` 表,带 20h 限频 |
| 拉取推荐 | 无文档承诺 | ✅ **新增** | `GET /api/user-behavior/recommendations` 端点,家长中心可拉 |

## 关于 README 的建议

- §"核心能力" 中**凡是有 ❌ 标记的条目**,建议在 README 里删除或加 `(规划中)` 标注。
- §"后续规划" 中已经实现的能力 (转化漏斗、周固定系列课、调课级联) 可以从"规划"挪到"已实现"。
- §"试讲前 1 小时短信提醒" 这类 UI 文案中嵌入的承诺,已统一在代码里删除。

## 自动核对

可以用以下命令快速核对这份表:

```bash
# 1. 天气 / 短信 / MCP 实际未接入
grep -r "openweather\|weather" agents/ services/ api/ 2>/dev/null | head -5
grep -r "from mcp\|import mcp" agents/ services/ api/ 2>/dev/null | head -5

# 2. 自我反思 / Agent-to-Agent 实际未实现
grep -r "self_reflect\|reflection\|agent_to_agent" agents/ 2>/dev/null | head -5

# 3. 推荐调度器落库已接通
grep -A3 "create_recommendation" services/recommendation_service.py
```

如果未来某项承诺变成了"✅ 已实现",直接改这张表 + 把 README 对应条目挪到"已实现"。
