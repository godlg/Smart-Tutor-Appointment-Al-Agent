"""
UserBehavior Module（家教场景增强版）

按 TUTOR_DESIGN.md §5.5 / §9 实现：
- BehaviorRecorder: 行为记录器
- PatternAnalyzer: 行为模式分析
- PreferenceManager: 偏好管理
- renewal_detector: 续费信号检测（Phase 2 新增）
- churn_predictor: 流失预警（Phase 2 新增）
"""

from .behavior_recorder import BehaviorRecorder
from .pattern_analyzer import PatternAnalyzer
from .preference_manager import PreferenceManager
from .renewal_detector import RenewalDetector
from .churn_predictor import ChurnPredictor

__all__ = [
    # 原有
    "BehaviorRecorder",
    "PatternAnalyzer",
    "PreferenceManager",
    # Phase 2 新增
    "RenewalDetector",
    "ChurnPredictor",
]