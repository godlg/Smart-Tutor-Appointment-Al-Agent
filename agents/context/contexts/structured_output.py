"""
结构化输出上下文

收敛 JSON 输出约束(参照 TrialInputParser 的字段说明风格),注入通用约束:
只出纯 JSON、字段命名、枚举取值、缺失兜底等。
本次只做通用约束注入,不强求把具体 parser 迁移过来。
"""

from ..base import BaseContext
from ..types import PRIORITY_STRUCTURED_OUTPUT

# 通用结构化输出约束文案
STRUCTURED_OUTPUT_TEXT = (
    "【输出约束】\n"
    "当你需要以结构化数据作答时,请严格遵循以下规则:\n"
    "- 只输出纯 JSON,不要任何 markdown 代码块标记(如 ```json ```)或其它文字\n"
    "- 字段名使用英文 snake_case,与上文约定的字段保持一致\n"
    "- 枚举字段只能取约定枚举值,不可杜撰\n"
    "- 缺失的信息填写「未知」/空列表,不要编造\n"
    "- 布尔字段只能是 true 或 false,不能是字符串\n"
)


class StructuredOutputContext(BaseContext):
    """结构化输出上下文:注入通用 JSON 约束(priority=40)。"""

    name = "structured_output"
    priority = PRIORITY_STRUCTURED_OUTPUT

    def build(self, manager) -> str:
        """返回固定的结构化输出约束文案。"""
        return STRUCTURED_OUTPUT_TEXT