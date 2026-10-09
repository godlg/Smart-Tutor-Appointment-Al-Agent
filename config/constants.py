"""
全局常量（家教场景）
"""

from enum import Enum


# =========================================
# 业务枚举
# =========================================

class IntentType(str, Enum):
    """用户意图分类"""
    CONSULTATION = "consultation"        # 知识咨询
    TRIAL_BOOKING = "trial_booking"      # 试讲预约
    LESSON_BOOKING = "lesson_booking"    # 正式课预约
    LESSON_RESCHEDULE = "reschedule"     # 调课
    LESSON_CANCEL = "cancel"             # 退课
    PACKAGE_INQUIRY = "package"          # 套餐咨询
    PROGRESS_QUERY = "progress"          # 进度查询
    RECOMMENDATION = "recommendation"    # 老师推荐
    UNRELATED = "unrelated"             # 无关问题


class StateEnum(str, Enum):
    """会话流转状态（家教场景全状态机）

    用于在 TaskClassificationAgent / TrialAgent / LessonAgent / ConsultantAgent
    之间共享当前所处的会话阶段。
    """
    CLASSIFY = "classify"          # 任务分类中（主路由入口）
    CONSULTATION = "consultation"  # 知识咨询流程中
    RECOMMENDATION_CONFIRM = "recommendation_confirm"  # 推荐确认态（家长已拿到 Top-N,等待选老师/确认）
    TRIAL = "trial"                # 试讲流程中
    LESSON = "lesson"              # 正式课流程中
    UNRELATED = "unrelated"        # 无关问题兜底


# 兼容旧名：state_manager.py 中仍以 FlowState 暴露
FlowState = StateEnum


class SubjectEnum(str, Enum):
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


class GradeLevelEnum(str, Enum):
    """年级段"""
    PRIMARY_LOW = "小学低年级"
    PRIMARY_HIGH = "小学高年级"
    JUNIOR = "初中"
    SENIOR = "高中"


class TeachingStyleEnum(str, Enum):
    """教学风格"""
    STRICT = "严格"
    GENTLE = "温和"
    INSPIRING = "启发式"
    PRESSURE = "高压"


class LessonTypeEnum(str, Enum):
    """课时类型"""
    TRIAL = "trial"
    REGULAR = "regular"
    SERIES = "series"
    MAKEUP = "makeup"


class LessonStatusEnum(str, Enum):
    """课时状态"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    RESCHEDULED = "rescheduled"


class KnowledgeCategoryEnum(str, Enum):
    """知识库分类"""
    COURSE_INTRO = "course_intro"
    SUBJECT_METHOD = "subject_method"
    TEACHER_INTRO = "teacher_intro"
    EXAM_STRATEGY = "exam_strategy"
    PARENT_GUIDE = "parent_guide"
    FAQ = "faq"


# =========================================
# 周映射
# =========================================

WEEKDAY_MAP = {
    0: "周一", 1: "周二", 2: "周三", 3: "周四",
    4: "周五", 5: "周六", 6: "周日",
}

WEEKDAY_EN_MAP = {
    0: "monday", 1: "tuesday", 2: "wednesday", 3: "thursday",
    4: "friday", 5: "saturday", 6: "sunday",
}

WEEKDAY_REVERSE = {v: k for k, v in WEEKDAY_EN_MAP.items()}