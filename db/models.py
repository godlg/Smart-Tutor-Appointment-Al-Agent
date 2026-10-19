"""
数据模型定义

包含两类模型：
- 原有按摩门店业务模型（保留）
- 家教场景业务模型（新增）
"""

from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey, Text, JSON,
    Boolean, Float, Date, Enum as SQLEnum, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
import enum

Base = declarative_base()


# =====================================================================
# 枚举定义（领域常量）
# =====================================================================

class SubjectEnum(str, enum.Enum):
    """学科"""
    MATH = "数学"
    ENGLISH = "英语"
    PHYSICS = "物理"
    CHEMISTRY = "化学"
    CHINESE = "语文"
    BIOLOGY = "生物"
    HISTORY = "历史"
    GEOGRAPHY = "地理"
    POLITICS = "政治"


class GradeLevelEnum(str, enum.Enum):
    """年级段"""
    PRIMARY_LOW = "小学低年级"   # 1-3 年级
    PRIMARY_HIGH = "小学高年级"  # 4-6 年级
    JUNIOR = "初中"
    SENIOR = "高中"


class TeachingStyleEnum(str, enum.Enum):
    """教学风格"""
    STRICT = "严格"
    GENTLE = "温和"
    INSPIRING = "启发式"
    PRESSURE = "高压"


class LessonTypeEnum(str, enum.Enum):
    """课时类型"""
    TRIAL = "trial"          # 试讲
    REGULAR = "regular"      # 正式单次课
    SERIES = "series"        # 系列课
    MAKEUP = "makeup"        # 补课


class LessonStatusEnum(str, enum.Enum):
    """课时状态"""
    PENDING = "pending"          # 待确认
    CONFIRMED = "confirmed"      # 已确认
    COMPLETED = "completed"      # 已完成
    CANCELLED = "cancelled"      # 已取消
    RESCHEDULED = "rescheduled"  # 已调课


class KnowledgeCategoryEnum(str, enum.Enum):
    """知识库分类"""
    COURSE_INTRO = "course_intro"
    SUBJECT_METHOD = "subject_method"
    TEACHER_INTRO = "teacher_intro"
    EXAM_STRATEGY = "exam_strategy"
    PARENT_GUIDE = "parent_guide"
    FAQ = "faq"


# =====================================================================
# 原有按摩门店业务模型（保留，向后兼容）
# =====================================================================

class KnowledgeDocument(Base):
    __tablename__ = 'knowledge_documents'
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    category = Column(String, nullable=False)
    keywords = Column(JSON, nullable=True)
    embedding = Column(JSON, nullable=True)
    # 元数据标签(P1-12: 为 hybrid 检索前置硬约束):
    # 学科 / 年级段 / 文档类型,均 nullable 以兼容跨学科/跨年级/通用型内容
    # (如 parent_guide、faq 全学科适用,不强制学科)
    subject = Column(String, nullable=True)           # SubjectEnum 值
    grade_level = Column(String, nullable=True)       # GradeLevelEnum 值
    doc_type = Column(String, nullable=True)          # KnowledgeCategoryEnum 值
    # P1-13: 上传溯源列(均 nullable,老行保持 NULL)
    # - source_file: 原始文件名,用于审计与 "来自哪个文件" 展示
    # - chunk_index: 文件内 0-based 位置,与 source_hash 联合唯一定位一个 chunk
    # - source_hash: 文件 SHA256 hex,作为去重键(同文件二次上传命中跳过)
    # - chunk_count: 文件总块数,所有 chunk 行值相同(便于单查询回答 "该文件几个块")
    source_file = Column(String, nullable=True)
    chunk_index = Column(Integer, nullable=True)
    source_hash = Column(String, nullable=True)
    chunk_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Integer, default=1)


class UserBehavior(Base):
    __tablename__ = 'user_behaviors'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=False, default='default_user')
    action_type = Column(String, nullable=False)
    action_data = Column(JSON, nullable=True)
    teacher_id = Column(Integer, ForeignKey('teachers.id'), nullable=True)
    session_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    teacher = relationship("Teacher")


class UserPreference(Base):
    __tablename__ = 'user_preferences'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=False, default='default_user')
    preference_type = Column(String, nullable=False)
    preference_value = Column(String, nullable=False)
    confidence_score = Column(Integer, default=1)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UserRecommendation(Base):
    __tablename__ = 'user_recommendations'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=False, default='default_user')
    recommendation_type = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    teacher_id = Column(Integer, ForeignKey('teachers.id'), nullable=True)
    is_sent = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    sent_at = Column(DateTime, nullable=True)
    teacher = relationship("Teacher")


class LongTermMemory(Base):
    """长期记忆（内容型,分层记忆增强）

    承载咨询摘要 / 预约事件 / 偏好等结构化记忆,每条带 embedding + importance + 时间戳,
    供 MemoryContext 按「0.6*向量相似度 + 0.3*时间衰减 + 0.1*重要度」召回 Top-5 注入。
    memory_type: summary(滚动摘要) | appointment(预约事件) | preference(偏好)
    """
    __tablename__ = 'long_term_memories'
    __table_args__ = (UniqueConstraint(
        'session_id', 'start_seq', 'end_seq', name='ux_long_term_memories_span'
    ),)

    student_id = Column(String, nullable=True, index=True)
    start_seq = Column(Integer, nullable=True)
    end_seq = Column(Integer, nullable=True)
    id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=False, default='default_user')
    memory_type = Column(String, nullable=False)      # summary | appointment | preference
    content = Column(Text, nullable=False)
    embedding = Column(JSON, nullable=True)           # list[float],可空(召回时按相似度过滤)
    importance = Column(Float, default=0.5)
    session_id = Column(String, nullable=True)        # 滚动摘要溯源到会话
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed_at = Column(DateTime, default=datetime.utcnow)


# =====================================================================
# AutoDream 后台任务（用户行为增量回放与长期偏好更新）
# =====================================================================

class AutoDreamCheckpoint(Base):
    """AutoDream 增量回放检查点（每 user 一条）"""
    __tablename__ = 'autodream_checkpoints'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=False, unique=True, index=True)
    last_event_id = Column(Integer, default=0)      # 已回放的最大 user_behaviors.id 水位
    last_run_at = Column(DateTime, nullable=True)   # 上次成功回放时间（触发条件 24h 依据）
    total_replays = Column(Integer, default=0)      # 累计回放次数


class AutoDreamLock(Base):
    """AutoDream 任务锁（同一 user 并发/重复调度只跑一次）"""
    __tablename__ = 'autodream_locks'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=False, unique=True, index=True)
    locked_at = Column(DateTime, default=datetime.utcnow)
    lock_token = Column(String, nullable=True)      # 持有者标识（进程/线程）
    expires_at = Column(DateTime, nullable=True)    # TTL，超时视为失效锁


class AutoDreamChangeLog(Base):
    """AutoDream 变更日志（每次偏好写回都留痕）"""
    __tablename__ = 'autodream_change_logs'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    event_id = Column(Integer, nullable=False)      # 触发的 user_behaviors.id
    action = Column(String, nullable=False)         # merge / new / downgrade
    preference_type = Column(String, nullable=False)
    preference_value = Column(String, nullable=False)
    detail = Column(JSON, nullable=True)            # 相似度 / 变更前后置信度等
    created_at = Column(DateTime, default=datetime.utcnow)


# =====================================================================
# 家教场景业务模型（新增）
# =====================================================================

class Parent(Base):
    """家长账户"""
    __tablename__ = 'parents'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    phone = Column(String, unique=True)
    budget_max = Column(Float, nullable=True)  # 单课时预算上限
    preferred_contact = Column(String, default="wechat")
    created_at = Column(DateTime, default=datetime.utcnow)

    children = relationship("StudentProfile", back_populates="parent", cascade="all, delete-orphan")


class StudentProfile(Base):
    """学生画像"""
    __tablename__ = 'student_profiles'

    id = Column(Integer, primary_key=True)
    parent_id = Column(Integer, ForeignKey('parents.id'))
    name = Column(String, nullable=False)
    grade_level = Column(String)        # GradeLevelEnum
    weak_subjects = Column(JSON)        # 薄弱学科 list[str]
    learning_goals = Column(JSON)       # 学习目标 list[str]
    preferred_style = Column(String)    # TeachingStyleEnum
    school = Column(String, nullable=True)
    notes = Column(Text, nullable=True)

    engagement_score = Column(Float, default=0.5)   # 活跃度 0-1
    churn_risk = Column(Float, default=0.0)         # 流失风险 0-1
    upsell_signal = Column(String, default="none")  # 续费信号 strong/weak/none

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    parent = relationship("Parent", back_populates="children")


class Teacher(Base):
    """老师"""
    __tablename__ = 'teachers'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    avatar_url = Column(String, nullable=True)
    gender = Column(String, nullable=True)

    # 教学属性（用 JSON 存储多值，简单直接）
    subjects = Column(JSON)              # 学科列表 SubjectEnum list[str]
    grade_levels = Column(JSON)          # 年级段 list[str]
    teaching_style = Column(String)      # TeachingStyleEnum
    education_bg = Column(String)        # 学历背景
    years_experience = Column(Integer, default=0)
    hourly_rate = Column(Float, default=200.0)   # 课时单价
    rating = Column(Float, default=5.0)

    # 可用时段：[{day: "monday", slots: ["19:00-20:00"]}]
    availability = Column(JSON)

    # 状态
    is_active = Column(Boolean, default=True)
    trial_available = Column(Boolean, default=True)

    # 简介
    bio = Column(Text, nullable=True)
    highlights = Column(JSON)            # 标签 ["耐心", "提分快", "押题准"]

    created_at = Column(DateTime, default=datetime.utcnow)


class CoursePackage(Base):
    """课时套餐"""
    __tablename__ = 'course_packages'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    teacher_id = Column(Integer, ForeignKey('teachers.id'))

    subject = Column(String)             # SubjectEnum
    grade_level = Column(String)        # GradeLevelEnum
    total_lessons = Column(Integer)
    duration_minutes = Column(Integer, default=60)
    price_total = Column(Float)
    price_per_lesson = Column(Float)

    # 周固定配置：[{"day_of_week": 3, "start_time": "19:00"}]
    weekly_schedule = Column(JSON)
    validity_days = Column(Integer, default=180)

    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class LessonSeries(Base):
    """周固定排课系列"""
    __tablename__ = 'lesson_series'

    id = Column(Integer, primary_key=True)
    teacher_id = Column(Integer, ForeignKey('teachers.id'))
    student_id = Column(Integer, ForeignKey('student_profiles.id'))
    package_id = Column(Integer, ForeignKey('course_packages.id'), nullable=True)

    start_date = Column(Date)
    end_date = Column(Date)
    weekly_pattern = Column(JSON)         # [{"day_of_week": 3, "time": "19:00"}]

    total_lessons = Column(Integer)
    completed_lessons = Column(Integer, default=0)
    cancelled_lessons = Column(Integer, default=0)

    status = Column(String, default="active")  # active / paused / ended
    created_at = Column(DateTime, default=datetime.utcnow)


class Lesson(Base):
    """具体一节课（替代 Appointment）"""
    __tablename__ = 'lessons'

    id = Column(Integer, primary_key=True)
    teacher_id = Column(Integer, ForeignKey('teachers.id'))
    student_id = Column(Integer, ForeignKey('student_profiles.id'))
    package_id = Column(Integer, ForeignKey('course_packages.id'), nullable=True)
    series_id = Column(Integer, ForeignKey('lesson_series.id'), nullable=True)

    # 课程属性
    lesson_type = Column(String)        # LessonTypeEnum
    subject = Column(String)             # SubjectEnum
    scheduled_at = Column(DateTime, nullable=False)
    duration_minutes = Column(Integer, default=60)

    # 状态
    status = Column(String)              # LessonStatusEnum

    # 系列内序号
    series_index = Column(Integer, nullable=True)

    # 课后记录
    progress_record_id = Column(Integer, nullable=True)  # 软引用 ProgressRecord.id
    parent_feedback = Column(Text, nullable=True)

    # 价格（用于历史查询）
    price = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ProgressRecord(Base):
    """课后进度记录"""
    __tablename__ = 'progress_records'

    id = Column(Integer, primary_key=True)
    lesson_id = Column(Integer, ForeignKey('lessons.id'))
    student_id = Column(Integer, ForeignKey('student_profiles.id'))
    teacher_id = Column(Integer, ForeignKey('teachers.id'))

    content_covered = Column(Text)        # 本次课内容
    mastery_score = Column(Float)         # 掌握度 0-10
    homework = Column(Text)               # 作业
    weak_points = Column(JSON)            # 暴露的薄弱点 list[str]
    ai_summary = Column(Text, nullable=True)   # AI 生成的课堂小结
    teacher_comment = Column(Text, nullable=True)
    parent_comment = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)


class TrialFeedback(Base):
    """试讲反馈"""
    __tablename__ = 'trial_feedbacks'

    id = Column(Integer, primary_key=True)
    lesson_id = Column(Integer, ForeignKey('lessons.id'))
    parent_id = Column(Integer, ForeignKey('parents.id'))

    rating = Column(Integer)              # 1-5
    would_continue = Column(Boolean)      # 是否继续
    feedback_text = Column(Text, nullable=True)
    recommended_subjects = Column(JSON)   # 推荐补充的学科

    created_at = Column(DateTime, default=datetime.utcnow)


class LearningGoal(Base):
    """学习目标"""
    __tablename__ = 'learning_goals'

    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey('student_profiles.id'))
    subject = Column(String)
    target_score = Column(Float, nullable=True)
    deadline = Column(Date, nullable=True)
    description = Column(Text)
    status = Column(String, default="active")  # active / achieved / abandoned

    created_at = Column(DateTime, default=datetime.utcnow)


# =====================================================================
# 人工接管（Human Handoff）—— 极端 case 兜底
# =====================================================================

class HumanHandoff(Base):
    """人工接管记录

    同一会话多次被分类器/下游 Agent "踢回"(unrelated)后自动转人工,
    或家长主动要求转人工时立即转人工;转人工后会话落库,客服可拉取/标记处理。
    trigger_type: keyword(主动关键词) | kickback_threshold(踢回达阈值) | manual(客服手动)
    status: pending / resolved / closed
    """
    __tablename__ = 'human_handoffs'

    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False, index=True)
    trigger_type = Column(String, nullable=False)
    trigger_reason = Column(String, nullable=True)
    unrelated_count = Column(Integer, default=0)
    status = Column(String, default='pending')
    handled_by = Column(String, nullable=True)
    resolve_note = Column(Text, nullable=True)
    # 触发时快照的会话历史(JSON 字符串 `[{"role","content"}, ...]`,可空)
    transcript = Column(Text, nullable=True)
    # 认领此工单的客服标识,未认领为空
    agent_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)


class HumanHandoffMessage(Base):
    """人工接管工单消息

    转人工后家长消息落库(role='parent'),客服回复也落库(role='agent'),
    供客服侧查看完整对话、家长侧轮询拉取客服回复。
    """
    __tablename__ = 'human_handoff_messages'

    id = Column(Integer, primary_key=True)
    handoff_id = Column(Integer, ForeignKey('human_handoffs.id'), nullable=False, index=True)
    session_id = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)          # parent | agent
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# =====================================================================
# 记忆可靠性 Phase 1：匿名身份 / 会话归属 / 顺序消息 / 滚动摘要 / 检查点
#
# 设计边界(勿夸大):无登录系统。user_id 是服务端签发的不可预测匿名身份,
# 不能证明真实家长身份,也不能证明 student 真实归属。student_id 仅作该身份下的
# 子 namespace,是否可用由服务端 claims 表校验。
# =====================================================================

class IdentityRecord(Base):
    """服务端签发的匿名身份(用于校验 cookie 里的 user_id 是否为本服务签发)。"""
    __tablename__ = 'identities'

    id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class IdentityStudent(Base):
    """匿名身份 → 学员 namespace 的认领关系。

    student_id 全局唯一:一个学员只能被一个身份认领,他人再认领则冲突(409),
    防止用任意 student_id 读取别人的学员业务数据。
    """
    __tablename__ = 'identity_students'

    id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    student_id = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatSession(Base):
    """会话 → 归属身份(ownership)。

    客户端首次带一个新 session_id 出现时绑定为当前匿名身份;此后他人用同一
    session_id 请求则拒绝(403),不允许抢占/重新绑定。
    """
    __tablename__ = 'chat_sessions'

    student_id = Column(String, nullable=True)
    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False, unique=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SessionMessage(Base):
    """会话顺序消息(统一历史持久化)。

    按 (session_id, seq) 唯一保障顺序与幂等:重复写入同一 seq 不产生重复消息。
    role: human | ai;status: complete | partial(取消/异常时只保存已产生的可用内容)。
    """
    __tablename__ = 'session_messages'
    __table_args__ = (UniqueConstraint(
        'session_id', 'seq', name='ux_session_messages_session_seq'
    ),)

    student_id = Column(String, nullable=True)
    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    seq = Column(Integer, nullable=False)            # 会话内顺序,1-based
    role = Column(String, nullable=False)            # human | ai
    content = Column(Text, nullable=False)
    status = Column(String, default='complete')      # complete | partial
    created_at = Column(DateTime, default=datetime.utcnow)


class SessionSummary(Base):
    """滚动摘要水位记录 + 摘要正文。

    以 (session_id, start_seq, end_seq) 唯一保障幂等:同一段只归档一次;
    摘要与水位在事务内原子提交,重启不重复摘要。
    """
    __tablename__ = 'session_summaries'
    __table_args__ = (UniqueConstraint(
        'session_id', 'start_seq', 'end_seq', name='ux_session_summaries_span'
    ),)

    student_id = Column(String, nullable=True)
    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    start_seq = Column(Integer, nullable=False)
    end_seq = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class SessionCheckpoint(Base):
    """业务工作状态检查点(FlowState + 待确认字段/待执行命令)。

    pending_ops 里是不能可靠跨进程恢复的时间敏感待执行操作;恢复时安全失效
    (清空 + 追问重确认),不得自动执行预约/支付/改课。
    """
    __tablename__ = 'session_checkpoints'

    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False, unique=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    state = Column(String, nullable=True)            # FlowState value
    payload = Column(Text, nullable=True)            # JSON:可安全恢复的业务字段
    pending_ops = Column(Text, nullable=True)        # JSON:时间敏感待执行命令
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# =====================================================================
# 机构侧控制台账号与鉴权基座(控制台分面 A)
#
# 边界(勿夸大):这里只为「机构内部后台」建立账号/session 地基,
# 与匿名家长身份(identities/identity_students)是两套独立体系,互不代表对方。
# 建表不写迁移脚本:靠 SessionManager.__init__ 的 Base.metadata.create_all
# (含在老库上补建新表)。
# =====================================================================

class ConsoleUser(Base):
    """机构侧员工账号表(admin 教务主管 / support 客服 / academic 教研 / teacher 老师)。

    命名债:表名 console_users 是历史命名,实际承载的是「机构侧员工账号」四种角色;
    老师账号的 teacher_id 指向 teachers.id,员工账号为 NULL。暂不重命名。
    """
    __tablename__ = 'console_users'

    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)   # pbkdf2_sha256$iters$salt_hex$digest_hex
    role = Column(String(32), nullable=False)             # admin | support | academic | teacher
    display_name = Column(String(64), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    teacher_id = Column(Integer, ForeignKey('teachers.id'), nullable=True, index=True)  # 仅 teacher 账号非空
    created_at = Column(DateTime, default=datetime.now)


class ConsoleSession(Base):
    """控制台登录会话(服务端签发 token,过期即失效并被清理)。"""
    __tablename__ = 'console_sessions'

    token = Column(String(64), primary_key=True)
    user_id = Column(Integer, ForeignKey('console_users.id'), nullable=False, index=True)
    role = Column(String(32), nullable=False)             # 签发时快照,避免角色变更后旧会话静默提权
    created_at = Column(DateTime, default=datetime.now)
    expires_at = Column(DateTime, nullable=False)