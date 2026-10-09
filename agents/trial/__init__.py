"""Trial Agent 子模块

按 TUTOR_DESIGN.md §5.3 / §9.1 实现：
- trial_agent: 主控制器
- trial_processor: 试讲流程（含 parser/teacher_finder/message_builder）
- conversion_logic: 试讲转化（Phase 2）
- trial_feedback: 试讲反馈（Phase 2）
"""

from .trial_agent import TrialAgent
from .trial_processor import (
    TrialProcessor, TrialInputParser, TrialTeacherFinder, TrialMessageBuilder,
)
from .conversion_logic import (
    ConversionLogic, ConversionStrategy, ConversionOutcome,
)
from .trial_feedback import TrialFeedbackCollector

__all__ = [
    # 主类
    "TrialAgent",
    "TrialProcessor",
    "TrialInputParser",
    "TrialTeacherFinder",
    "TrialMessageBuilder",
    # Phase 2 新增
    "ConversionLogic",
    "ConversionStrategy",
    "ConversionOutcome",
    "TrialFeedbackCollector",
]