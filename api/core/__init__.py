"""
API 核心组件初始化（家教场景）

- response_models：通用响应与任务分类模型
- exceptions：通用异常处理
- endpoint：API 端点装饰器(消除 try/except 模板)
"""
from .response_models import (
    BaseResponse,
    DataResponse,
    TaskClassificationRequest,
    TaskClassificationResponse,
)
from .exceptions import (
    general_exception_handler,
)
from .endpoint import api_endpoint

__all__ = [
    # 响应模型
    "BaseResponse",
    "DataResponse",
    "TaskClassificationRequest",
    "TaskClassificationResponse",
    # 异常处理器
    "general_exception_handler",
    # 装饰器
    "api_endpoint",
]