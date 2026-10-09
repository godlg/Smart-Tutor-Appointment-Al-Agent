"""
上下文类型定义

集中定义上下文构建过程中共享的优先级常量、DTO 与 token 估算工具。
本模块属于 Agent 层,不依赖任何 services / db,可被测试直接导入。
"""

from dataclasses import dataclass

# ----------------------------------------------------------------------
# 组装优先级(越小越靠前)
# ----------------------------------------------------------------------
PRIORITY_SYSTEM_PROMPT = 0
PRIORITY_CONVERSATION = 10
PRIORITY_MEMORY = 20
PRIORITY_TOOLS = 30
PRIORITY_STRUCTURED_OUTPUT = 40

# 用户画像默认用户 id(与应用内 behavior 表默认使用者一致)
DEFAULT_USER_ID = "default_user"


@dataclass(frozen=True)
class ContextBuildResult:
    """单个上下文 build 的结果。

    frozen:组装完成后不可变,避免下游误改注入文本或优先级。
    """

    name: str
    text: str
    priority: int


class ContextBudgetExceeded(Exception):
    """必需上下文在给定预算内无法容纳(明确的、可处理的拒绝语义)。

    抛出而不是静默超限:调用方可据此要求用户缩短输入,或调大预算。
    绝不能通过删除当前问题/系统提示/结构化约束来让断言变绿。
    """

    def __init__(self, message: str, used: int = 0, budget: int = 0):
        super().__init__(message)
        self.message = message
        self.used = used
        self.budget = budget


def estimate_token_count(text: str) -> int:
    """近似 token 估算(中文保守)。

    口径说明(非计费数):按 UTF-8 字节数估算(常用汉字每字 3)。
    这是无需模型/网络的保守 byte-level 上界,刻意高估中文、罕见字和标点,
    不是特定 provider 的 tokenizer/计费数。消息封装与特殊 token 不在纯文本
    口径内;生产可注入匹配模型的 token_counter。
    """
    return len(text.encode("utf-8")) if text else 0


# 兼容别名:spec 里也叫 token_counter 口径
count_tokens = estimate_token_count


@dataclass
class ContextBudget:
    """硬上下文预算:max_tokens 是最终 prompt 的 token 上限(含输出预留)。

    - max_tokens: 最终注入文本的 token 上限(按 estimate_token_count 口径)
    - output_reserve: 预留给模型输出的 token(从 max_tokens 中先扣掉)
    - keep_recent_rounds: 压缩/裁剪时至少保留的最近对话轮数(每轮 2 条)
    """

    max_tokens: int
    output_reserve: int = 0
    keep_recent_rounds: int = 8

    @property
    def prompt_budget(self) -> int:
        """可用于 prompt 的 token 上限(扣除输出预留)。"""
        return max(0, self.max_tokens - self.output_reserve)
