"""Lesson Agent 子模块

按 TUTOR_DESIGN.md §5.4 / §9 实现：
- input_parser: 解析用户需求（含周固定识别）
- teacher_finder: 老师匹配
- lesson_processor: 主流程
- schedule_generator: 周固定系列生成（Phase 2）
- conflict_detector: 冲突检测（Phase 2）
- reschedule_handler: 调课级联（Phase 2）
- cancel_handler: 退课 + 退费（Phase 2）
- change_processor: 对话入口的调课/退课（自然语言 → lesson_id → Handler）
- message_builder: 消息构造
"""

from .lesson_agent import LessonAgent
from .lesson_processor import (
    LessonProcessor, LessonInputParser, LessonTeacherFinder, LessonMessageBuilder,
)
from .schedule_generator import ScheduleGenerator
from .conflict_detector import (
    ConflictDetector, Conflict, ConflictType, ConflictSeverity,
)
from .reschedule_handler import RescheduleHandler, RescheduleMode
from .cancel_handler import CancelHandler
from .change_processor import (
    LessonChangeProcessor, LessonChangeParser, LessonChangeMessageBuilder,
)

__all__ = [
    # 主类
    "LessonAgent",
    "LessonProcessor",
    "LessonInputParser",
    "LessonTeacherFinder",
    "LessonMessageBuilder",
    # Phase 2 新增
    "ScheduleGenerator",
    "ConflictDetector",
    "Conflict",
    "ConflictType",
    "ConflictSeverity",
    "RescheduleHandler",
    "RescheduleMode",
    "CancelHandler",
    # 对话入口的调课/退课
    "LessonChangeProcessor",
    "LessonChangeParser",
    "LessonChangeMessageBuilder",
]