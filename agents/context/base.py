"""
上下文抽象基类

定义 BaseContext 接口:每个上下文都需给出唯一 name、组装优先级 priority,
并把自身渲染成可供注入 LLM 的文本片段 build()。
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 仅类型标注使用,避免运行时循环导入
    from .manager import ContextManager


class BaseContext(ABC):
    """上下文基类

    子类实现 build(),把该上下文渲染成适合注入 LLM 的文本片段。
    name 需唯一(用于用量报告与日志),priority 决定组装顺序。
    """

    name: str = "base"
    priority: int = 0

    @abstractmethod
    def build(self, manager: "ContextManager") -> str:
        """返回注入文本。

        Args:
            manager: ContextManager 实例(build 时由它驱动),
                     便于上下文读取共享状态(如 session_id / user_id)。

        Returns:
            该上下文渲染出的字符串;返回空串时,ContextManager 会跳过它。
        """
        raise NotImplementedError