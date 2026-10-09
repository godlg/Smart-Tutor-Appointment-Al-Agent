"""
Agents 模块入口（家教场景）

- TaskClassificationAgent: 主路由
- ConsultantAgent: 知识咨询 Agent
- TrialAgent: 试讲 Agent
- LessonAgent: 正式课 Agent
- UserBehaviorAgent: 家长行为 Agent
"""

from .task_classification_agent import TaskClassificationAgent
from .consultant_agent import ConsultantAgent
from .user_behavior_agent import UserBehaviorAgent

# 家教场景核心 Agent
from .trial import TrialAgent, TrialFeedbackCollector, ConversionLogic
from .lesson import (
    LessonAgent, ScheduleGenerator, ConflictDetector,
    RescheduleHandler, CancelHandler,
)
from .user_behavior import RenewalDetector, ChurnPredictor

__all__ = [
    # 主路由 + 通用
    "TaskClassificationAgent",
    "ConsultantAgent",
    "UserBehaviorAgent",
    # 家教核心
    "TrialAgent",
    "TrialFeedbackCollector",
    "ConversionLogic",
    "LessonAgent",
    "ScheduleGenerator",
    "ConflictDetector",
    "RescheduleHandler",
    "CancelHandler",
    "RenewalDetector",
    "ChurnPredictor",
]