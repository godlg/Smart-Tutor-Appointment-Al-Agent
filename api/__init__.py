"""
API 模块入口（家教场景）
"""

from .task import router as task_router
from .knowledge import router as knowledge_router
from .user_behavior_analysis import router as user_behavior_analysis_router
from .user_behavior_analysis import router_underscore as user_behavior_analysis_underscore_router

# 家教场景
from .teacher import router as teacher_router
from .lesson import router as lesson_router
from .lesson import series_router as lesson_series_router
from .package import router as package_router
from .student import router as student_router
from .progress import router as progress_router
from .trial import router as trial_router
from .agentic import router as agentic_router
from .handoff import router as handoff_router


def _filter_routers(routers):
    return [r for r in routers if r is not None]


api_routers = _filter_routers([
    # 基础
    knowledge_router,
    user_behavior_analysis_router,
    user_behavior_analysis_underscore_router,
    # 家教核心
    task_router,
    teacher_router,
    lesson_router,
    lesson_series_router,
    package_router,
    student_router,
    progress_router,
    trial_router,
    agentic_router,
    handoff_router,
])