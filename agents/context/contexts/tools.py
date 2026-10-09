"""
工具 / 能力上下文

把 Service 层能力注册成纯文本描述清单(不做 function calling),
让 LLM 知道系统能做什么,便于依能力引导家长,而非凭空捏造。
数据来源: services/ 层各服务的对外能力。
"""

from ..base import BaseContext
from ..types import PRIORITY_TOOLS

# Service 能力清单(能力名 -> 一句话用途),素材取自 services/ 层
SERVICE_CAPABILITIES = [
    ("knowledge.search", "知识库 RAG 检索: 检索课程介绍 / 学习方法 / 老师特点 / 考试策略等资料"),
    ("recommendation.recommend_teachers", "老师推荐: 按学科 / 年级 / 偏好给学生匹配、打分并推荐老师"),
    ("lesson.create_lesson", "排课: 创建正式课 / 试讲课课时"),
    ("lesson.create_lesson_series", "批量排课: 创建每周固定的系列课"),
    ("lesson.reschedule", "调课: 调整已有时课的时间"),
    ("lesson.cancel", "退课: 取消课时并按退费规则计算退款"),
]

# 无注册能力时的兜底文案(不应出现,防御性占位)
_NO_CAPABILITIES = "暂无可用能力"


class ToolContext(BaseContext):
    """工具 / 能力上下文:列出 Service 能力描述清单(priority=30)。"""

    name = "tools"
    priority = PRIORITY_TOOLS

    def build(self, manager) -> str:
        """把能力清单渲染成「能力名: 用途」的多行文本。"""
        if not SERVICE_CAPABILITIES:
            return _NO_CAPABILITIES
        lines = ["【可用能力】"]
        for capability, desc in SERVICE_CAPABILITIES:
            lines.append(f"- {capability}: {desc}")
        return "\n".join(lines)