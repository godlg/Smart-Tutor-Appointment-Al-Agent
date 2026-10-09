"""
API 响应模型（家教场景）

只保留当前真正在用的：
- BaseResponse / DataResponse：通用响应壳
- TaskClassificationRequest / Response：任务分类接口
"""

from pydantic import BaseModel
from typing import Any, Dict, Optional
from datetime import datetime
from config.time_config import time_config


class BaseResponse(BaseModel):
    """基础响应模型"""
    message: str
    timestamp: datetime = time_config.now()


class DataResponse(BaseResponse):
    """数据响应模型（直接返回任意 data）"""
    data: Any


# =========================================
# 任务分类
# =========================================

class TaskClassificationRequest(BaseModel):
    """任务分类请求

    与 api/task.py 中 `request.message` 的访问保持一致。
    """
    message: str
    context: Optional[Dict[str, Any]] = None


class TaskClassificationResponse(BaseModel):
    """任务分类响应"""
    message: str
    category: str
    confidence: float = 1.0
    reasoning: Optional[str] = None
    timestamp: datetime = time_config.now()
