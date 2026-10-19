# 家教老师智能预约系统 - 详细设计方案

> 基于 `smart-appointment-ai-agent` 改造而来
> 业务模式：C 端个人家长 | 排课：单次 + 周固定双模式 | 目标：先出方案确认

---

## 1. 业务定位与场景

### 1.1 一句话定义
一个面向**家长**的 AI 智能家教预约平台，帮家长高效完成「找老师 → 试讲 → 转正式课 → 长期排课 → 进度跟踪」全流程。

### 1.2 目标用户
| 角色                | 描述   | 主要诉求              |
| ----------------- | ---- | ----------------- |
| **家长**（下单人 + 付款方） | 孩子家长 | 找靠谱老师、灵活排课、看到学习效果 |
| **学生**（上课人）       | 中小学生 | 学科辅导、提分、兴趣培养      |

### 1.3 核心业务流程
```text
家长咨询 ──→ 了解课程/老师 ──→ 预约试讲 ──→ 试讲反馈 ──→ 转为正式课
                                              │
                                              ├─ 单次课：每次单独预约
                                              └─ 周固定课：买 N 周套餐，每周固定时段

正式课排期 ──→ 课前提醒 ──→ 上课 ──→ 课后小结 + 作业 ──→ 进度跟踪 ──→ 续费提醒
```

### 1.4 与原项目的关键差异
| 维度   | 原（按摩门店）        | 新（家教预约）                |
| ---- | -------------- | ---------------------- |
| 服务者  | 技师             | 老师                     |
| 服务单元 | 项目（泰式推拿 60min） | 课时（数学辅导 60min）         |
| 时间粒度 | 整点时段           | 45/60/90 分钟，跨整点        |
| 客户结构 | 单一成人           | 家长 + 学生（双人画像）          |
| 复购模式 | 自然复购           | **试讲 → 转化** + **长期套餐** |
| 个性化  | 技师偏好           | 学科薄弱 + 教学风格 + 提分目标     |
| 特有业务 | 临时改约           | **试讲、调课、停课、续费、进度跟踪**   |
| 外部信息 | 天气             | **考试节点、寒暑假、学期时间**      |

---

## 2. 整体架构

完全沿用原项目的**五层架构 + 调用方向约束**，不做架构级改动，只做业务级替换和增强。

```text
┌────────────────────────────────────────────────┐
│  Web & Application Layer                       │
│  - 家长中心 Web（聊天 + 排课日历 + 进度看板）      │
│  - app.py 入口                                  │
└────────────────────┬───────────────────────────┘
                     ↓
┌────────────────────────────────────────────────┐
│  API Layer                                     │
│  - 任务路由 / 流式聊天 / 试讲 / 排课 / 调课       │
│  - 老师管理 / 知识库 / 进度 / 套餐               │
└────────────────────┬───────────────────────────┘
                     ↓
┌────────────────────────────────────────────────┐
│  Agents Layer（多 Agent 协作）                   │
│  Task ──→ Consultation / Trial Lesson /         │
│          Lesson Booking / Behavior              │
└────────────────────┬───────────────────────────┘
                     ↓
┌────────────────────────────────────────────────┐
│  Services Layer                                │
│  老师 / 排课 / 知识库 / 推荐 / 进度 / 套餐        │
└────────────────────┬───────────────────────────┘
                     ↓
┌────────────────────────────────────────────────┐
│  DB Layer                                      │
│  Teacher / Lesson / Curriculum / Student /      │
│  Progress / KnowledgeBase / Behavior            │
└────────────────────────────────────────────────┘
```

### 2.1 Agent 拓扑（扩展后）

```text
用户输入
   │
   ▼
┌──────────────────────────┐
│  Task Classification     │  主路由
│  Agent                   │
└──────────┬───────────────┘
           │
   ┌───────┼─────────┬─────────────────┐
   ▼       ▼         ▼                 ▼
┌──────┐┌──────┐┌──────────┐┌──────────────────┐
│咨询  ││试讲  ││正式课    ││用户行为          │
│Agent ││Agent ││Agent    ││Agent             │
│      ││      ││(含周固定) ││                  │
└──────┘└──────┘└──────────┘└──────────────────┘
   │       │         │              │
   ▼       ▼         ▼              ▼
 RAG     试讲流程   排课/调课      行为记录+画像
 知识库   转化逻辑   冲突检测      偏好分析
```

> 关键变化：把原 Appointment Agent 拆成 **Trial Agent**（试讲转化）和 **Lesson Agent**（正式课管理），因为试讲的业务逻辑和长期排课差异较大。

---

## 3. 数据模型设计（SQLAlchemy）

### 3.1 模型对照表

| 原模型             | 新模型                                 | 主要变化                   |
| --------------- | ----------------------------------- | ---------------------- |
| `Technician`    | `Teacher`                           | 字段重写：学科、年级、教学风格、学历、单价  |
| -               | `Subject`                           | **新增**：学科字典（数学/英语/物理等） |
| -               | `GradeLevel`                        | **新增**：年级字典（初一/高一/高三）  |
| `Service`       | `CoursePackage`                     | 改名为套餐：N 课时 / N 周       |
| `Appointment`   | `Lesson`                            | 改：增加试讲/正式课类型、时长、学科     |
| `KnowledgeBase` | `KnowledgeBase`                     | 保留，分类调整                |
| `UserBehavior`  | `StudentProfile` + `ParentBehavior` | **拆分**：家长行为 + 学生画像     |
| -               | `LearningGoal`                      | **新增**：学习目标（提分/补差/兴趣）  |
| -               | `ProgressRecord`                    | **新增**：每次课后的进度         |
| -               | `LessonSeries`                      | **新增**：周固定排课的「系列」概念    |

### 3.2 核心模型字段（关键字段，非完整）

```python
# agents/services/teacher_service.py 中的领域模型

class Subject(str, Enum):
    MATH = "数学"
    ENGLISH = "英语"
    PHYSICS = "物理"
    CHEMISTRY = "化学"
    CHINESE = "语文"
    # ...

class GradeLevel(str, Enum):
    PRIMARY_LOW = "小学低年级"
    PRIMARY_HIGH = "小学高年级"
    JUNIOR_HIGH = "初中"
    SENIOR_HIGH = "高中"
    # ...
```

```python
# db/models.py - 新增/改造

class Teacher(Base):
    __tablename__ = "teachers"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    avatar_url = Column(String)

    # 学科与年级（多对多关联）
    subjects = relationship("Subject", secondary=teacher_subjects)
    grade_levels = relationship("GradeLevel", secondary=teacher_grade_levels)

    # 教学属性
    teaching_style = Column(String)  # "严格 / 温和 / 启发式"
    education_bg = Column(String)    # "北师大数学系硕士"
    years_experience = Column(Integer)
    hourly_rate = Column(Float)      # 课时单价
    rating = Column(Float, default=5.0)

    # 可用时段（复杂结构，存 JSON）
    availability = Column(JSON)
    # 格式：[{"day": "monday", "slots": ["19:00-20:00", "20:00-21:00"]}, ...]

    # 状态
    is_active = Column(Boolean, default=True)
    trial_available = Column(Boolean, default=True)

    # 元数据
    bio = Column(Text)              # 老师简介
    highlights = Column(JSON)       # 标签：["耐心", "提分快", "押题准"]


class StudentProfile(Base):
    """学生画像（拆分自原 UserBehavior）"""
    __tablename__ = "student_profiles"

    id = Column(Integer, primary_key=True)
    parent_id = Column(Integer, ForeignKey("parents.id"))
    name = Column(String)
    grade_level = Column(Enum(GradeLevel))
    weak_subjects = Column(JSON)     # ["数学", "物理"]
    learning_goals = Column(JSON)    # ["高考数学 120+", "培养兴趣"]
    preferred_style = Column(String) # "启发式"


class Parent(Base):
    """家长账户"""
    __tablename__ = "parents"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    phone = Column(String, unique=True)
    children = relationship("StudentProfile", back_populates="parent")


class LearningGoal(Base):
    """学习目标"""
    __tablename__ = "learning_goals"

    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey("student_profiles.id"))
    subject = Column(Enum(Subject))
    target_score = Column(Float, nullable=True)  # 如高考目标分
    deadline = Column(Date, nullable=True)       # 考试日期
    description = Column(Text)


class CoursePackage(Base):
    """课时套餐"""
    __tablename__ = "course_packages"

    id = Column(Integer, primary_key=True)
    name = Column(String)            # "高一数学 12 周套餐"
    teacher_id = Column(Integer, ForeignKey("teachers.id"))
    subject = Column(Enum(Subject))
    grade_level = Column(Enum(GradeLevel))

    total_lessons = Column(Integer)  # 总课时数
    duration_minutes = Column(Integer)  # 单次时长 45/60/90
    price_total = Column(Float)
    price_per_lesson = Column(Float)

    # 周固定排课配置
    weekly_schedule = Column(JSON)
    # 格式：[{"day_of_week": 3, "start_time": "19:00"}, ...]

    validity_days = Column(Integer)  # 有效期


class Lesson(Base):
    """具体的一节课（替代 Appointment）"""
    __tablename__ = "lessons"

    id = Column(Integer, primary_key=True)
    teacher_id = Column(Integer, ForeignKey("teachers.id"))
    student_id = Column(Integer, ForeignKey("student_profiles.id"))
    package_id = Column(Integer, ForeignKey("course_packages.id"), nullable=True)

    # 课程信息
    lesson_type = Column(Enum(LessonType))  # TRIAL / REGULAR / MAKEUP
    subject = Column(Enum(Subject))
    scheduled_at = Column(DateTime)         # 上课时间
    duration_minutes = Column(Integer)

    # 系列归属（周固定课）
    series_id = Column(Integer, ForeignKey("lesson_series.id"), nullable=True)
    series_index = Column(Integer, nullable=True)  # 第几次课

    # 状态
    status = Column(Enum(LessonStatus))  # PENDING / CONFIRMED / COMPLETED / CANCELLED

    # 课后记录
    progress_record_id = Column(Integer, ForeignKey("progress_records.id"), nullable=True)
    parent_feedback = Column(Text, nullable=True)  # 家长反馈


class LessonSeries(Base):
    """周固定排课的系列"""
    __tablename__ = "lesson_series"

    id = Column(Integer, primary_key=True)
    teacher_id = Column(Integer)
    student_id = Column(Integer)
    package_id = Column(Integer)

    start_date = Column(Date)
    end_date = Column(Date)
    weekly_pattern = Column(JSON)  # [{"day_of_week": 3, "time": "19:00"}]

    total_lessons = Column(Integer)
    completed_lessons = Column(Integer, default=0)


class ProgressRecord(Base):
    """课后进度记录（AI 自动生成 + 老师人工补充）"""
    __tablename__ = "progress_records"

    id = Column(Integer, primary_key=True)
    lesson_id = Column(Integer, ForeignKey("lessons.id"))
    student_id = Column(Integer, ForeignKey("student_profiles.id"))

    content_covered = Column(Text)   # 本次课内容
    mastery_score = Column(Float)    # 掌握度评分 0-10
    homework = Column(Text)          # 作业
    weak_points = Column(JSON)       # 本次暴露的薄弱点
    ai_summary = Column(Text)        # AI 生成的课堂小结
    teacher_comment = Column(Text)   # 老师补充
    parent_comment = Column(Text)    # 家长反馈


class KnowledgeBase(Base):
    """保留，沿用，分类调整"""
    __tablename__ = "knowledge_base"

    id = Column(Integer, primary_key=True)
    content = Column(Text)
    category = Column(String)
    # 新分类：course_intro / subject_method / teacher_intro /
    #         exam_strategy / parent_guide / faq
    keywords = Column(JSON)
    embedding = Column(JSON)  # 由 services/text_embedding 管理
```

---

## 4. 知识库结构

完全沿用 FAISS + Embedding 缓存机制，**只调整内容分类和初始数据**。

### 4.1 新分类
```text
knowledge_base
├── course_intro/         课程介绍（套餐内容、上课流程）
├── subject_method/       学科方法（数学思维、英语听力训练方法）
├── teacher_intro/        老师介绍（教学理念、擅长方向）
├── exam_strategy/        考试策略（中高考复习、时间分配）
├── parent_guide/         家长指南（如何陪写作业、怎么沟通）
└── faq/                  高频问答
    ├── 退课政策
    ├── 调课规则
    ├── 试讲说明
    └── 发票问题
```

### 4.2 初始数据示例
- 《高中数学 12 周提分方案》
- 《北师大张老师：擅长函数与几何，5 年教龄》
- 《中考英语听力 30 天冲刺方法》
- 《家长如何帮孩子养成预习习惯》
- 《试讲不满意可以全额退款吗？》

---

## 5. Agent 详细设计

### 5.1 Task Classification Agent（基本不变）
**变化点**：意图枚举扩展

```python
class IntentType(str, Enum):
    CONSULTATION = "consultation"           # 知识咨询
    TRIAL_BOOKING = "trial_booking"         # 试讲预约
    LESSON_BOOKING = "lesson_booking"       # 正式课预约
    LESSON_RESCHEDULE = "reschedule"        # 调课
    LESSON_CANCEL = "cancel"                # 退课
    PACKAGE_INQUIRY = "package"             # 套餐咨询
    PROGRESS_QUERY = "progress"             # 进度查询
    BEHAVIOR_ANALYSIS = "behavior"          # 行为分析（触发）
    UNRELATED = "unrelated"                # 无关问题
```

### 5.2 Consultation Agent（基本不变）
**变化点**：Prompt 中替换业务术语，知识库分类调整。

```text
旧 Prompt 风格：
"你是按摩门店的智能顾问，可以介绍服务项目、技师专长、注意事项..."

新 Prompt 风格：
"你是 K12 家教平台的智能顾问，可以介绍课程体系、老师特点、
 学习方法、考试策略... 涉及具体课时价格、剩余课时，请引导家长查看套餐页。"
```

### 5.3 Trial Lesson Agent（新增，从原 Appointment Agent 拆分）

**职责**：试讲全流程
```text
用户表达试讲意图
   ↓
解析：学科、年级、目标时间、孩子情况
   ↓
老师匹配（按学科 + 年级 + 可试讲）
   ↓
发送试讲邀请（带家长背景）
   ↓
试讲确认 → 写 Lesson(type=TRIAL)
   ↓
试讲当天提醒
   ↓
试讲反馈收集
   ↓
【关键】转化逻辑：家长是否要转为正式课？
   ├─ 是 → 推荐套餐 → 转 LessonSeries
   └─ 否 → 标记流失原因，更新画像
```

**新增模块**：
- `agents/trial/trial_processor.py`
- `agents/trial/conversion_logic.py`  ← 试讲转化策略
- `agents/trial/trial_feedback.py`

### 5.4 Lesson Booking Agent（原 Appointment Agent 改造）

**核心职责扩展**：
1. **单次课**：保持原 Appointment Agent 逻辑
2. **周固定课**（**新增**）：
   - 解析「每周三周五 19:00，连排 12 周」
   - 生成 LessonSeries + 12 条 Lesson 记录
   - 冲突检测：学生已有课、老师已有课、考试日

**新增模块**：
```text
agents/lesson/
├── input_parser.py            # 改造：识别「周固定」「连排 N 周」
├── teacher_finder.py          # 改造：学科 + 年级硬匹配
├── lesson_processor.py        # 主流程
├── schedule_generator.py      # 【新】生成系列排课
├── conflict_detector.py       # 【新】冲突检测
├── reschedule_handler.py      # 【新】调课处理（带系列联动）
├── cancel_handler.py          # 【新】退课（剩余课时退费计算）
└── message_builder.py         # 改造：试讲邀请、课前提醒、课后反馈
```

**冲突检测维度**：
```python
class ConflictType(str, Enum):
    STUDENT_BUSY = "学生已有课程"      # 学生已有其他课
    TEACHER_BUSY = "老师已被预约"      # 老师已被预约
    TEACHER_OFF = "老师不可用"          # 不在 availability 内
    EXAM_PERIOD = "考试日前后禁课"     # 考试日前 3 天禁排新课
    BLACKOUT_DATE = "平台黑名单日"      # 法定节假日等
    SERIES_OVERFLOW = "系列课时溢出"    # 超出套餐总课时
```

### 5.5 User Behavior Agent（改造）
**变化点**：画像字段调整 + 新增「续费信号检测」

```python
class StudentProfile:
    weak_subjects: list[str]
    learning_goals: list[str]
    preferred_style: str
    grade_level: GradeLevel
    # 新增
    engagement_score: float       # 活跃度
    churn_risk: float             # 流失风险
    upsell_signal: str            # 续费信号：strong / weak / none
```

**新增模块**：
- `agents/user_behavior/renewal_detector.py`  ← 续费提醒
- `agents/user_behavior/churn_predictor.py`   ← 流失预警

---

## 6. Services 层设计

| Service | 状态 | 主要职责 |
|---|---|---|
| `teacher_service.py` | 🔄 改造 | 老师 CRUD + 可用时段管理 |
| `lesson_service.py` | 🆕 新增 | 课时管理 + 系列管理 + 调退课 |
| `package_service.py` | 🆕 新增 | 套餐 CRUD + 剩余课时计算 + 续费 |
| `knowledge_service.py` | 🔄 改造 | 知识库 + FAISS（分类调整） |
| `recommendation_service.py` | 🔄 改造 | 推荐算法升级 |
| `student_profile_service.py` | 🆕 新增 | 学生画像 + 目标 + 进度 |
| `text_embedding.py` | ✅ 保留 | Embedding + 缓存 |
| `payment_service.py` | 🆕 预留 | 支付（暂用 mock） |

### 6.1 推荐算法升级
```python
# services/recommendation_service.py

def score_teacher_for_student(teacher: Teacher, student: StudentProfile) -> float:
    score = 0.0

    # 学科硬匹配（不匹配直接 0 分）
    if student.weak_subject not in teacher.subjects:
        return 0.0

    # 年级匹配
    if student.grade_level in teacher.grade_levels:
        score += 30

    # 教学风格匹配
    style_match = {
        "启发式": ["耐心", "思维引导"],
        "严格": ["严厉", "高强度"],
        "温和": ["亲和", "鼓励式"],
    }
    for tag in teacher.highlights:
        if tag in style_match.get(student.preferred_style, []):
            score += 20
            break

    # 历史行为加权
    if teacher.id in student.previous_teachers:
        score += 25  # 老学员优先

    # 评分加成
    score += teacher.rating * 5

    # 价格适配（避免推荐家长负担不起的）
    if teacher.hourly_rate > student.parent.budget_max:
        score *= 0.5

    return score
```

---

## 7. API 接口设计

### 7.1 路由清单

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/api/task/classify` | 任务分类（主入口） |
| POST | `/api/chat/stream` | 流式聊天（SSE） |
| POST | `/api/trial/book` | 预约试讲 |
| GET | `/api/trial/{id}/feedback` | 试讲反馈 |
| POST | `/api/trial/{id}/convert` | 试讲转正式课 |
| POST | `/api/lessons` | 创建单次课 |
| POST | `/api/lesson-series` | 创建周固定系列 |
| GET | `/api/lesson-series/{id}` | 查看系列 |
| POST | `/api/lessons/{id}/reschedule` | 调课请求（默认待老师确认，课时不变；`immediate=true` 立即改期，单次 + 联动） |
| POST | `/api/lessons/{id}/cancel` | 退课 |
| GET | `/api/teachers` | 老师列表 + 推荐 |
| GET | `/api/teachers/{id}` | 老师详情 |
| GET | `/api/packages` | 套餐列表 |
| POST | `/api/packages/{id}/purchase` | 购买套餐 |
| GET | `/api/progress/{student_id}` | 学生进度 |
| POST | `/api/progress` | 上报进度 |
| GET | `/api/knowledge/search` | 知识库检索 |
| POST | `/api/knowledge` | 知识库管理 |
| POST | `/api/behavior/analyze` | 行为分析 |
| GET | `/api/behavior/{student_id}` | 学生画像 |

### 7.2 关键接口契约示例

```json
// POST /api/lesson-series 请求
{
  "teacher_id": 12,
  "student_id": 5,
  "package_id": 3,
  "weekly_pattern": [
    {"day_of_week": 3, "start_time": "19:00"},
    {"day_of_week": 5, "start_time": "19:00"}
  ],
  "start_date": "2026-09-01",
  "total_weeks": 12
}

// 响应
{
  "series_id": 88,
  "lessons_created": 24,
  "conflicts": [],          // 空表示全部成功
  "warnings": [
    "第 8 周周三与期中考试冲突，建议避开"
  ]
}
```

---

## 8. Web 页面设计

### 8.1 页面结构

```text
web/templates/
├── index.html                  首页聊天（沿用）
├── knowledge_management.html   知识库管理（沿用）
├── teacher_list.html           【改】老师列表（原 technician.html）
├── teacher_detail.html         【新】老师详情 + 预约试讲
├── trial_booking.html          【新】试讲预约流程
├── schedule_calendar.html      【新】排课日历（核心新增）
├── lesson_series.html          【新】我的系列课
├── progress_dashboard.html     【新】学习进度看板
├── parent_center.html          【新】家长中心（汇总）
├── packages.html               【新】套餐选购
└── student_profile.html        【新】学生档案管理
```

### 8.2 核心页面：排课日历
- 周视图 / 月视图切换
- 已排课程清晰标注（单次 vs 系列）
- 一键添加「周固定时段」
- 冲突时段标红
- 与试讲、套餐购买打通

---

## 9. 关键业务逻辑详解

### 9.1 试讲转化流程
```text
1. 家长输入「想给孩子找个数学老师」→ Consultation Agent
2. 推荐 3 位老师 → 家长点击「约试讲」→ Trial Agent
3. 收集：学科、年级、孩子情况、目标时间
4. 老师匹配 → 输出 3 个可试讲时段
5. 家长选定 → 写 Lesson(type=TRIAL, status=PENDING)
6. 老师确认 → status=CONFIRMED → 推送通知
7. 试讲当天：课前 1 小时提醒
8. 试讲后：邀请家长评分 + 是否转为正式课
9. 转化为正式课：
   - 推荐套餐（基于孩子年级 + 学科 + 家长预算）
   - 家长选定 → 生成 LessonSeries + N 条 Lesson
```

### 9.2 周固定排课生成算法
```python
# agents/lesson/schedule_generator.py

def generate_series_lessons(
    start_date: date,
    total_weeks: int,
    weekly_pattern: list,  # [{day_of_week: 3, time: "19:00"}, ...]
    teacher_id: int,
    student_id: int,
) -> list[Lesson]:
    lessons = []
    current_date = start_date

    for week in range(total_weeks):
        for pattern in weekly_pattern:
            lesson_date = current_date + timedelta(days=...)
            lesson_time = combine(lesson_date, pattern["time"])

            # 跳过已过去的日期
            if lesson_time < datetime.now():
                continue

            lessons.append(Lesson(
                teacher_id=teacher_id,
                student_id=student_id,
                scheduled_at=lesson_time,
                duration_minutes=60,
                lesson_type=LessonType.REGULAR,
                series_index=week * len(weekly_pattern) + ...,
            ))

        current_date += timedelta(days=7)

    return lessons
```

### 9.3 调课的级联处理
```text
场景：家长调「本周三的课」到「本周日」

1. 单次课调课：直接修改 Lesson.scheduled_at
2. 系列课调课：
   ├─ 只调一次：仅改当前 Lesson，series 不变
   └─ 调整后续所有周：级联更新 LessonSeries 之后所有 Lesson
3. 冲突检测：周日是否与其他课/老师可用时段冲突
4. 通知老师
```

### 9.4 续费提醒策略
```text
触发条件：
- 套餐剩余课时 ≤ 3
- 距离套餐过期 < 14 天

行为：
1. Agent 生成个性化推荐语：
   "您的套餐还剩 2 次课，建议趁孩子学习势头续费 12 周套餐..."
2. 推荐适合的套餐（基于历史进步 + 当前目标）
3. 家长点击 → 进入购买流程
```

### 9.5 进度跟踪与画像更新
```text
每节课结束后：
1. 老师/AI 上报 ProgressRecord（掌握度、作业、薄弱点）
2. 画像自动更新：
   - weak_subjects：薄弱点变化
   - mastery_score：学科掌握度趋势
   - engagement_score：作业完成率
3. 后续推荐加权使用最新画像
```

---

## 10. 实施路线图

### Phase 1：基础迁移（1-2 周）
**目标**：跑通「咨询 → 试讲 → 转为单次正式课」主链路

任务：
1. 新建 `tutor_design.md`（即本文档）✅
2. DB 模型迁移：Teacher / Subject / Lesson / Parent / StudentProfile
3. 替换 `technician_*` → `teacher_*` 模块
4. 改造 Appointment Agent → Trial Agent（先只做单次课）
5. 改造 Consultation Agent Prompt
6. 基础 Web 页面（老师列表 + 试讲预约 + 课程详情）
7. 配置切换：环境变量加 `TUTOR_MODE=true`

**可验收**：家长能咨询数学辅导 → 找老师 → 约试讲 → 转一次正式课。

### Phase 2：核心增强（2-3 周）
**目标**：完成周固定排课、调课、进度跟踪

任务：
1. LessonSeries + 系列生成算法
2. 排课日历 UI
3. 冲突检测（学生/老师/考试日）
4. 调课级联处理
5. ProgressRecord + 课后小结（AI 自动生成）
6. 学生画像 + 续费信号检测
7. 套餐购买流程

**可验收**：家长可买 12 周套餐 → 一次性生成全部排课 → 调课/退课正常 → 课后有进度报告。

### Phase 3：智能化升级（持续）
- 推荐算法 A/B 优化
- 流失预警 + 主动触达
- 学习路径规划（基于考试目标倒推）
- 老师画像与家长评价体系
- 接入支付（MVP 用 mock，后续接微信/支付宝）
- 多家长协作（父母双方共同决策）

---

## 11. 迁移对照表（原 → 新）

### 文件层面
```text
原文件名                        新文件名                        操作
─────────────────────────────────────────────────────────────────
technician.py (api)            teacher.py                      改名 + 改字段
technician_service.py          teacher_service.py              改名 + 改逻辑
technician_repository.py       teacher_repository.py           改名 + 改字段
appointment.py (api)           trial.py + lesson.py            拆分
appointment_service.py         lesson_service.py               改名 + 增强
appointment_agent.py           trial_agent.py + lesson_agent.py 拆分
agents/appointment/*           agents/trial/* + agents/lesson/* 拆分
agents/user_behavior/*         保留 + 新增 renewal_detector
technician.html                teacher_list.html + teacher_detail.html
-                              schedule_calendar.html          新增
-                              progress_dashboard.html         新增
-                              packages.html                   新增
```

### 配置层面
```env
# .env 新增
TUTOR_MODE=true
INDUSTRY=tutor
DEFAULT_SUBJECT=math
```

---

## 12. 风险与决策点

需要在动手前再确认的细节：

1. **家长账户体系**：是否需要登录注册？MVP 阶段可以先用手机号 + 验证码简化版？
2. **支付**：MVP 用 mock 占位（标记状态），还是先接微信支付沙箱？
3. **老师数据来源**：MVP 用种子数据（10-20 位虚拟老师），还是爬真实数据？
4. **多学科支持**：MVP 先聚焦「数学 + 英语」两个学科，还是全学科铺开？
5. **AI 生成课后小结**：依赖大模型，是否要在每节课后都自动生成（成本考虑）？

---

## 附录：改造后的目录结构（目标态）

```text
smart-tutor-ai-agent/        （可以是新目录，或在本目录内加 -tutor 后缀）
├── app.py
├── agents/
│   ├── task_classification/
│   ├── consultation/                  # 沿用，改 Prompt
│   ├── trial/                         # 新增
│   ├── lesson/                        # 改造 + 新增 schedule 系列
│   └── user_behavior/                 # 沿用 + 加 renewal_detector
├── api/
│   ├── core/
│   ├── task.py
│   ├── chat_handler.py
│   ├── consultation.py
│   ├── trial.py                       # 新增
│   ├── lesson.py                      # 改名自 appointment.py
│   ├── teacher.py                     # 改名自 technician.py
│   ├── package.py                     # 新增
│   ├── progress.py                    # 新增
│   ├── knowledge.py
│   └── user_behavior_analysis.py
├── services/
│   ├── teacher_service.py
│   ├── lesson_service.py
│   ├── package_service.py             # 新增
│   ├── knowledge_service.py
│   ├── recommendation_service.py
│   ├── student_profile_service.py     # 新增
│   └── text_embedding.py
├── db/
│   ├── models.py                      # 全部字段更新
│   ├── repositories/
│   │   ├── teacher_repository.py
│   │   ├── lesson_repository.py
│   │   ├── package_repository.py      # 新增
│   │   ├── progress_repository.py     # 新增
│   │   └── knowledge_repository.py
│   └── ...
├── config/
├── web/templates/                     # 替换为家教场景页面
└── tests/
```

---

**确认这份方案 OK 的话，我就开始 Phase 1 的代码改造。**
**如果某些模块的设计不符合你的预期（比如数据模型、Agent 拆分、套餐模型等），现在调整成本最低。**