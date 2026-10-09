# Smart Tutor Appointment AI Agent

Smart Tutor Appointment AI Agent 是一个面向家教与课外辅导场景的智能预约与学习咨询系统。项目基于 FastAPI、LangChain、FAISS、SQLite 和多 Agent 协作架构，实现了意图识别、RAG 知识问答、家教老师智能匹配、课程预约管理、学员行为分析和个性化学习提醒等能力。

这个项目的核心目标不是只做一个普通的课程预约表单，而是尝试把家教机构前台 / 教务老师日常需要处理的高频工作自动化：理解家长或学员是在咨询课程、预约老师，还是反馈学习情况，判断学科、年级与学习偏好，匹配合适的家教老师，检查可用时间，生成预约结果，并在必要时结合天气、历史学习行为和偏好数据给出更贴近学员实际情况的提醒与推荐。

## 项目背景

在一次和家教机构负责人的交流中，我注意到教务前台每天都要同时处理大量复杂事务：老师排班、学员预约、课程咨询、价格计算、临时调课、老师偏好记录和续费确认等。随着老师数量和学员量增加，传统人工前台模式很容易出现沟通成本高、信息遗漏、排课冲突和学习体验不稳定的问题。

而且在家教这个场景里，用户咨询的内容往往非常专业——不同年级、不同学科的考点、学习方法、适合的教材、考前冲刺策略等，这些都很难靠一句模板回答解决；同时，学员的个体差异非常大，小学三年级数学基础薄弱和高二物理冲刺的需求完全不同，需要系统能够记录并理解这些差异。

因此，本项目尝试用 AI Agent 的方式重构这一流程：让系统能够像一个智能教务前台一样主动理解家长和学员的需求，并把不同任务分发给对应的专业 Agent 处理。它既适用于 K12 一对一上门家教，也可以扩展到语言培训、考研辅导、留学规划和兴趣班等需要老师排班、课程预约和智能客服的教育服务行业。

## 核心能力

- **智能任务分类**：自动识别用户是在咨询课程、预约老师、查询学习问题，还是触发学习行为分析，并将请求路由到对应 Agent。
- **多 Agent 协作**：通过任务分类 Agent、咨询 Agent、试讲 Agent、正式课 Agent、学员行为 Agent、续费/流失分析子模块分工处理复杂流程，减少单个模块的职责膨胀。
- **RAG 知识咨询**：使用 FAISS 向量索引检索学科知识库、课程介绍、学习方法等内容，结合大模型生成自然语言回答，支持流式输出，适合家长在课前课后随时提问。
- **试讲 → 转化漏斗**：独立 TrialAgent 负责试讲前的需求收集与信息追问；试讲结束后收集 1-5 星反馈、是否继续上课和补充学科，自动推荐适合的套餐并可一键转为正式课或系列课。
- **正式课与系列课**：支持单次试讲 / 单次正式课 / 周固定系列课 / 补课四种课时类型；周固定系列课由 `LessonSeries` 按 `weekly_pattern` 自动展开为多节具体 `Lesson`，并支持调课、退课、冲突检测。
- **老师匹配**：综合学科、年级、教学风格、家长预算、历史评分、价格适配系数对候选老师打分（见 `RecommendationService.score_teacher_for_student`），支持家长中心主动推荐与 Agent 自动调用。
- **学员行为分析**：记录学员的咨询内容、上课记录、薄弱知识点和兴趣偏好，分析学习模式，并用于后续推荐和个性化反馈。
- **续费与流失预警**：`RenewalDetector` 在套餐剩余 ≤ 3 课时或有效期不足 14 天时触发续费提醒；`ChurnPredictor` 根据 `engagement_score` 评估流失风险。
- **个性化提醒**：在课程预约完成后，可结合实时天气、近期考试节点等外部信息生成更贴近实际场景的上课提醒，例如“明天降温，记得让孩子带件外套”。
- **Embedding 缓存优化**：通过数据库缓存和文件缓存减少重复向量计算，提高知识检索性能。
- **数据管理能力**：支持学科知识库、老师信息、学员档案、套餐和学习进度的增删改查，并在数据变化后自动维护索引。
- **日志与兜底机制**：保留关键处理过程日志，在信息不足或异常情况下提供更稳定的降级处理，例如自动追问年级或学科信息。

> 详细业务模型、数据流与扩展规划见 [`docs/TUTOR_DESIGN.md`](./docs/TUTOR_DESIGN.md)。

## 系统架构

项目采用严格的五层架构，核心原则是：**下层不能反向调用上层**。这样可以避免循环依赖，让业务逻辑、数据访问和接口编排保持清晰边界。

```text
Web & Application Layer
    ↓  app.py, web/：页面、路由入口、系统启动
API Layer
    ↓  api/：外部接口、请求编排、响应封装
Agents Layer
    ↓  agents/：AI Agent、任务路由、对话流程控制
Services Layer
    ↓  services/：业务逻辑、推荐算法、向量处理
DB Layer
    ↓  db/：数据模型、数据库连接、Repository
```

### 允许的调用方向

- Web 层调用 API 层
- API 层调用 Agents 层或 Services 层
- Agents 层调用 Services 层
- Services 层调用 DB 层

### 禁止的调用方式

- 下层反向调用上层
- Web 层绕过 API 直接访问 Services 或 DB
- Agents 层绕过 Services 直接访问 DB
- Services 层调用 Agents、API 或 Web

## Agent 设计

### Task Classification Agent

任务分类 Agent 是系统的主调度器，负责分析用户输入、判断任务类型，并把请求分发给合适的专业 Agent。在家教场景下，它需要识别的是：家长是在咨询课程、咨询学科问题、预约老师、调整已有课程，还是在询问学员近期的学习表现。

```text
用户输入 → 意图分析 → Agent 路由 → 响应协调
```

主要职责：

- 判断用户意图（咨询 / 预约 / 行为分析 / 其他）
- 维护对话状态
- 控制不同 Agent 之间的切换
- 处理无法分类或超出能力范围的问题，例如明确表示不接 K12 之外的需求时礼貌回退

### Consultation Agent

咨询 Agent 负责学科知识问答和学习规划场景，使用 RAG 流程从知识库中检索相关内容，再结合大模型生成回答。知识库里通常会放各年级各学科的考点总结、学习方法、常见错题分析、升学政策等内容，避免大模型“自由发挥”给出不适合学员的答案。

```text
任务分类 → 知识检索 → FAISS 相似度搜索 → 流式回答
```

主要职责：

- 区分咨询问题类型（学科知识 / 学习方法 / 课程介绍 / 政策信息）
- 从知识库检索相关内容
- 结合学员年级、学科等上下文构建提示词
- 生成自然语言回答

### Appointment Agent

预约 Agent 负责课程预约相关流程，包括解析家长输入、匹配老师、检查老师档期、确认价格与课时、生成预约确认消息等。在家教场景里，它需要重点处理信息缺失的情况——家长常常只说“想给孩子补一下数学”，这时候系统需要主动追问年级、目标、上课时间和地点等关键信息。

```text
任务分类 → 解析预约需求 → 老师匹配 → 预约确认
```

主要职责：

- 提取学员年级、学科、学习目标、期望时间、上课地点等信息
- 根据学科、年级、风格偏好匹配合适的老师
- 处理信息缺失时的追问
- 检查老师档期冲突
- 生成预约结果和上课提醒

### User Behavior Agent

学员行为 Agent 更偏向后台智能分析，不完全依赖家长或学员的显式请求。它会根据交互记录、预约历史、咨询内容、薄弱知识点等数据分析学员学习行为，为后续推荐提供依据。

```text
行为记录 → 模式分析 → 偏好更新 → 个性化推荐
```

主要职责：

- 记录学员咨询和预约行为
- 分析偏弱学科、偏好老师风格、学习时间段等模式
- 生成推荐依据，例如“这位学员更适合耐心型老师”
- 支持主动反馈和个性化服务

附带的两个后台子模块（位于 `agents/user_behavior/`）：

- **RenewalDetector**：根据套餐剩余课时（≤ 3 节）或距过期天数（< 14 天）触发续费提醒，并生成个性化续费推荐语。
- **ChurnPredictor**：根据 `StudentProfile.engagement_score` 与交互间隔评估流失风险，写回 `churn_risk` 字段，供顾问主动跟进。

### Trial Agent

试讲 Agent 处理"试讲预约 → 试讲反馈 → 转正式课/套餐"这条主线，是 K12 家教场景的核心转化漏斗。

```text
任务分类 → 收集试讲需求(年级/学科/时段) → 创建试讲 Lesson → 课后反馈 → 推荐套餐 → 转化
```

主要职责：

- 多轮对话补全试讲信息（学科、年级、期望时间、老师风格）
- 试讲前创建 `LessonTypeEnum.TRIAL` 类型的 Lesson
- 收集 1-5 星评分、是否继续、补充学科等 `TrialFeedback`
- 根据反馈与学员画像通过 `ConversionLogic` 推荐匹配的 CoursePackage
- 一键调用 `/api/trial/{lesson_id}/convert` 转为正式课或周固定系列课

### Lesson Agent

正式课 Agent（`agents/lesson/`）负责单次正式课与周固定系列课的完整生命周期。

```text
任务分类 → 收集上课需求 → ScheduleGenerator 排课 → ConflictDetector 冲突检测
        → RescheduleHandler 调课 → CancelHandler 退课 → 课后 ProgressRecord
```

主要职责：

- 单次课排课：补全学科 / 年级 / 时段 / 老师偏好等字段
- 周固定系列课生成：根据 `weekly_pattern` 与 `validity_days` 自动展开多节 Lesson
- 冲突检测：避免同一老师同一时段被重复排课
- 调课与退课：保留原 Lesson 的审计字段，不直接删除
- 课后进度记录：触发 `ProgressRecord` 创建，由 `progress_service` 与 `renewal_detector` 串联

## 核心设计思想

### 1. 用任务分类降低系统复杂度

系统并不让一个 Agent 处理所有事情，而是先判断家长或学员意图，再分发给对应模块。这样可以让咨询、预约、行为分析等逻辑保持独立，也更容易扩展新的 Agent，例如后续可以加入“续费提醒 Agent”或“学习报告 Agent”。

### 2. 用 RAG 解决专业知识回答

学科相关的考点、学习方法、升学政策等内容更适合通过知识库维护。RAG 能让回答基于可控知识来源，而不是完全依赖大模型自由生成，避免出现“小学阶段讲微积分”或“数学题答案错误”这类硬伤。

### 3. 用学员行为让推荐更个性化

系统会记录学员的咨询、上课和偏好信息。后续在推荐老师或课程时，可以结合历史行为，而不是每次都从零开始询问家长。例如系统可以记住“这位学员不喜欢男老师”或“数学基础偏弱，适合讲解细致的老师”。

### 4. 用分层架构保证可维护性

Agent 负责智能流程，Service 负责业务逻辑，Repository 负责数据访问。每层只关心自己的职责，减少后期修改时的连锁影响——比如换一套数据库、换一套 Embedding 服务，都不会影响到 Agent 层的对话逻辑。

### 5. 为真实业务场景预留扩展空间

项目目前以本地 SQLite 和单体服务为主，但架构上预留了模型提供商切换、MCP 外部服务接入（例如天气、节假日提醒）、后台任务、缓存优化和云端部署的扩展方向，非常适合作为家教机构内部教务系统原型继续迭代。

## 架构图

详细架构(5 层调用关系 + 跨层时序图)见 [`docs/ARCHITECTURE_MAP.md`](./docs/ARCHITECTURE_MAP.md)。

### 5 层结构概览

```text
Web & Application Layer   app.py, web/routes.py
    ↓
API Layer                  api/ (orchestration, REST + chat)
    ↓
Agents Layer               agents/ (TaskClassification → Consultant/Trial/Lesson/UserBehavior)
    ↓
Services Layer             services/ (teacher / lesson / package / progress / knowledge / recommendation)
    ↓
DB Layer                   db/ (SQLAlchemy models + Repository 模式)
```

调用方向是**单向**: 下层不能反向调用上层。 当前唯一例外是 `services/lesson_service.py` 在函数体内 lazy import `agents.lesson.ScheduleGenerator`,见 `docs/REPO_HEALTH.md` T2。

## 技术栈

- **后端框架**：FastAPI、Uvicorn
- **AI 框架**：LangChain
- **大模型接入**：兼容 OpenAI 格式的模型提供商，例如 Qwen、DeepSeek、Zhipu、OpenAI、Azure OpenAI
- **向量检索**：FAISS
- **数据库**：SQLite、SQLAlchemy
- **RAG 能力**：Embedding、向量索引、知识库检索、提示词构建
- **流式响应**：Python AsyncGenerator
- **前端页面**：Jinja2 模板、静态 CSS
- **外部服务扩展**：MCP，用于天气、节假日等外部信息接入
- **配置管理**：python-dotenv
- **后台任务**：schedule

## 项目结构

```text
Smart Tutor Appointment AI Agent/
├── agents/                              # 多 Agent 智能层
│   ├── task_classification_agent.py     # 任务分类与主路由
│   ├── consultant_agent.py              # RAG 咨询 Agent
│   ├── trial/                           # 试讲预约/反馈/转化
│   │   ├── trial_agent.py               # 试讲主控制器
│   │   ├── trial_processor.py           # 试讲对话处理器
│   │   ├── trial_feedback.py            # 试讲反馈收集
│   │   └── conversion_logic.py          # 试讲转套餐决策
│   ├── lesson/                          # 正式课与系列课
│   │   ├── lesson_agent.py              # 正式课主控制器
│   │   ├── lesson_processor.py
│   │   ├── schedule_generator.py        # 周固定系列课展开
│   │   ├── conflict_detector.py         # 老师时段冲突检测
│   │   ├── reschedule_handler.py        # 调课
│   │   └── cancel_handler.py            # 退课
│   ├── user_behavior_agent.py           # 学员行为分析 Agent
│   ├── consultant/                      # 知识检索、提示词、回答生成
│   │   ├── consultation_processor.py
│   │   ├── consultation_classifier.py
│   │   ├── knowledge_retriever.py
│   │   ├── prompt_builder.py
│   │   └── response_generator.py
│   ├── task_classification/             # 意图识别、状态管理、路由逻辑
│   │   ├── task_classifier.py
│   │   ├── state_manager.py
│   │   ├── agent_router.py
│   │   ├── classification_processor.py
│   │   └── unrelated_handler.py
│   └── user_behavior/                   # 行为记录、偏好管理、模式分析
│       ├── behavior_recorder.py
│       ├── preference_manager.py
│       ├── pattern_analyzer.py
│       ├── renewal_detector.py          # 续费信号
│       └── churn_predictor.py           # 流失风险
├── api/                                 # API 编排层
│   ├── chat_handler.py                  # 流式聊天处理
│   ├── task.py                          # 任务分类接口
│   ├── knowledge.py                     # 知识库管理接口
│   ├── teacher.py                       # 老师管理接口
│   ├── lesson.py                        # 课时 + 系列课接口
│   ├── trial.py                         # 试讲预约/反馈/转化接口
│   ├── student.py                       # 学员档案接口
│   ├── package.py                       # 套餐接口
│   ├── progress.py                      # 课后进度接口
│   ├── user_behavior_analysis.py        # 学员行为分析接口
│   └── core/                            # 统一响应模型与异常处理
│       ├── response_models.py
│       └── exceptions.py
├── services/                            # 业务逻辑层
│   ├── teacher_service.py               # 老师 CRUD + 筛选 + 默认数据
│   ├── lesson_service.py                # 课时 CRUD + 冲突 + 系列课
│   ├── package_service.py               # 套餐管理 + 默认数据
│   ├── student_profile_service.py       # 学员档案 + 默认数据
│   ├── progress_service.py              # 课后进度记录
│   ├── knowledge_service.py             # 知识库管理 + FAISS
│   ├── text_embedding.py                # Embedding 与向量处理
│   ├── user_behavior_service.py         # 学员行为服务
│   └── recommendation_service.py        # 老师打分 + 后台调度
├── db/                                  # 数据持久化层
│   ├── models.py                        # SQLAlchemy 模型（13 张表）
│   ├── db_router.py                     # 数据库路由 + Repository 聚合
│   ├── base/
│   │   ├── session_manager.py           # Session 生命周期管理
│   │   └── interfaces.py                # Repository 抽象接口
│   └── repositories/                    # Repository 数据访问封装
│       ├── knowledge_repository.py
│       ├── teacher_repository.py
│       ├── student_repository.py
│       ├── lesson_repository.py
│       ├── package_repository.py
│       ├── progress_repository.py
│       └── user_behavior_repository.py
├── config/                              # 配置模块
│   ├── constants.py                     # 常量与枚举
│   ├── database.py                      # 数据库配置
│   ├── model_provider.py                # 模型与 Embedding Provider 工厂
│   ├── settings.py                      # 应用配置
│   └── time_config.py                   # 时间与排课配置
├── web/                                 # Web 页面层
│   ├── routes.py                        # 页面路由
│   ├── templates/                       # HTML 模板（11 个页面）
│   │   ├── index.html                   # 聊天入口
│   │   ├── teacher_list.html / teacher_detail.html
│   │   ├── packages.html
│   │   ├── schedule_calendar.html / lesson_series.html
│   │   ├── progress_dashboard.html
│   │   ├── student_profile.html / parent_center.html
│   │   ├── knowledge_management.html
│   │   └── user_behavior_analysis.html
│   └── static/styles.css                # 全站样式
├── docs/
│   └── TUTOR_DESIGN.md                  # 详细业务模型与数据流设计
├── tests/                               # 单元测试（核心算法）
│   ├── test_schedule_generator.py
│   ├── test_conflict_detector.py
│   ├── test_cancel_handler.py
│   ├── test_conversion_logic.py
│   └── test_renewal_detector.py
├── data/                                # SQLite 数据库与 FAISS 索引
├── app.py                               # 应用入口
├── requirements.txt                     # Python 依赖
├── .env.example                         # 环境变量模板
└── README.md
```

## 快速开始

### 1. 创建虚拟环境

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows CMD：

```cmd
.venv\Scripts\activate.bat
```

macOS 或 Linux：

```bash
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制环境变量模板：

```bash
cp .env.example .env
```

Windows PowerShell 可以使用：

```powershell
Copy-Item .env.example .env
```

然后在 `.env` 中填写模型和数据库配置。项目支持 OpenAI 兼容格式的大模型与 Embedding 服务。

```env
MODEL_PROVIDER=qwen
LLM_API_KEY=your_llm_api_key_here
LLM_BASE_URL=your_openai_compatible_chat_base_url_here
LLM_MODEL=your_chat_model_name_here

EMBEDDING_PROVIDER=qwen
EMBEDDING_API_KEY=your_embedding_api_key_here
EMBEDDING_BASE_URL=your_openai_compatible_embedding_base_url_here
EMBEDDING_MODEL=your_embedding_model_name_here

DATABASE_URL=sqlite:///./data/smart_appointment.db

DEBUG=True
LOG_LEVEL=INFO
```

常见配置方向：

- Qwen：使用阿里云百炼或 DashScope 的模型、Base URL 和 API Key，适合国内访问。
- DeepSeek：可用于聊天模型，Embedding 可搭配其他兼容服务。
- Zhipu：可配置智谱的聊天模型和向量模型。
- Azure OpenAI：将 `MODEL_PROVIDER` 设置为 `azure`，并补充对应的 Azure OpenAI 环境变量。

### 4. 启动服务

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

如果 8000 端口已被占用，可以换成 8001：

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8001 --reload
```

启动后可以访问：

- Web 页面：http://127.0.0.1:8000
- API 文档：http://127.0.0.1:8000/docs
- ReDoc 文档：http://127.0.0.1:8000/redoc

## 测试

运行全部测试：

```bash
pytest
```

运行单个测试文件：

```bash
pytest tests/test_task_classification_agent.py
```

## 主要页面

| 路径 | 说明 |
|---|---|
| `/` | 首页：聊天入口，集成任务分类、咨询、试讲、正式课的统一对话界面 |
| `/teachers` | 老师库列表，支持按学科 / 年级筛选 |
| `/teachers/{id}` | 老师详情页 + 预约试讲 |
| `/packages` | 课时套餐列表与购买 |
| `/schedule-calendar` | 排课日历视图 |
| `/lesson-series` | 周固定系列课列表（已排课 / 已完成 / 暂停） |
| `/progress-dashboard` | 学员学习进度看板（按学科展示掌握度与薄弱点） |
| `/student-profile` | 学生档案（年级、薄弱学科、学习目标、风格偏好、活跃度、流失风险） |
| `/parent-center` | 家长中心：续费提醒、流失预警、推荐汇总 |
| `/knowledge` | 学科知识库管理（增删改查 + 关键词检索） |
| `/user_behavior_analysis` | 学员行为分析看板 |

## 演示流程（端到端）

启动服务后，可以按下面的剧本向聊天接口（`POST /chat/stream`）依次输入，观察任务分类 → Agent 路由 → 业务执行 → 数据库落库的完整链路：

1. **咨询** — 输入"初二数学函数基础不好怎么办？" → TaskClassificationAgent 路由到 ConsultantAgent → RAG 检索 → 流式输出学习建议
2. **试讲** — 输入"想给孩子约一次数学试讲" → TrialAgent 多轮追问学科/年级/时段 → 创建 `LessonTypeEnum.TRIAL` 课时
3. **老师筛选** — 输入"有没有耐心点的女老师" → RecommendationService 按 `gender / style / subject` 打分返回 Top-N
4. **正式课排课** — 输入"每周三晚上 7 点" → LessonAgent → ScheduleGenerator 展开为系列课 → ConflictDetector 校验
5. **课后反馈** — 调用 `POST /api/trial/{lesson_id}/feedback` 提交 5 星 + would_continue=true → ConversionLogic 推荐套餐
6. **续费/流失预警** — 后台 `RenewalDetector` + `ChurnPredictor` 异步检测，写回 `UserRecommendation` 表

> 完整的 API 列表（按模块分组）见 `docs/TUTOR_DESIGN.md §10`，本文不重复列全部 endpoint。

**一键端到端演示**：启动服务后，在 `.venv` 中运行

```powershell
python scripts/demo_e2e.py http://127.0.0.1:8001
```

脚本会按 1)看老师 → 2)看套餐 → 3)建学员 → 4)预约试讲 → 5)生成周固定系列课 → 6)提交反馈 → 7)查行为分析 → 8)写课后进度 顺序跑一遍，输出每步 HTTP 状态，验证家长视角完整链路。

## 后续规划

### 更强的 Agent 自主能力

- 增加 Agent 自我反思机制，让系统能够评估回答质量和预约成功率。
- 引入更完整的多轮推理链，提升复杂预约和冲突处理能力，例如多个孩子共用同一个老师的排课优化。
- 根据真实家长反馈优化老师推荐策略。

### 更完整的多 Agent 协作

- 增加 Agent-to-Agent 通信机制，减少所有任务都依赖主分类器转发的问题。
- 把学员行为 Agent 的后台分析能力做得更稳定，支持定时任务和主动触达，例如“本周学习总结”。
- 把预约、推荐、咨询之间的上下文记忆打通得更自然，例如预约过程中家长提到“孩子最近数学吃力”，系统可以自动关联到薄弱知识点分析。

### 生产化能力

- 增加家长登录、权限控制和数据隔离。
- 增加更完整的异常处理和边界场景覆盖，例如老师临时请假、学员临时取消等。
- 优化向量检索性能、缓存策略和响应速度。
- 支持 Docker 部署、云数据库和更标准的日志监控。

## 项目价值

这个项目把多 Agent、RAG、学员行为分析、老师排课预约和外部工具接入放在同一个真实业务场景中验证。它既是一个家教机构智能教务前台原型，也可以作为学习 AI Agent 工程化、分层架构、RAG 系统和业务自动化的综合实践项目。对于想理解“AI Agent 怎么落地到具体行业”的同学来说，家教场景是一个非常合适的入门案例——既不像电商客服那么卷，也不像医疗金融那样对准确性要求过高，同时又能体现多角色协作、个性化推荐和行为分析的价值。



------------------------------------------------------------
这个项目是我最早开始自学大模型做的项目。上面的内容是当时（2025年7月左右完成的）按摩门店场景的总结。
这次改写成家教版本，主要是把业务背景、术语和示例替换成了 K12 一对一上门家教的场景，整体的技术架构、Agent 设计和工程结构都保持不变。
其实我在计划做更加复杂、符合时代趋势的 Agent，预计在 9 月份完成，会分享在笔记中。
但是很多朋友问我这个项目能不能参考，所以我索性总结开源了。值得一提的是，上面的内容是 25 年写的，当时如何配置环境，都是传统方法，写在 README 中，自己配置。但是在 2026 年，不管是 vibe coding 的方法，还是环境配置的方法，都有了极大改进。

我的项目配置的方法，现在都推荐使用 SKILL，所以这里有 setup-environment skill，一键配置。
我也总结了更多的这个项目，放在 2026 年，如何包装，如何使用配套资源，面试真题，视频讲解，使用建议在我的大模型笔记中。
👉 **请关注小红书：[不转到大模型不改名]（id:4740535877) 获取以上所有资源。**
👉 b站：骑猪撞宝马71
