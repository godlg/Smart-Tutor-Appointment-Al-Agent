"""
任务分类器（家教场景）
"""

import logging
import re

from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from observability import trace_span
from ..context.types import ContextBudgetExceeded

logger = logging.getLogger(__name__)


class ClassificationUnavailable(RuntimeError):
    """LLM 分类服务不可用（超时 / 限流 / 网络 / 接口错误），区别于「用户输入无关」。"""


# 合法意图类别(全量 10 类:原 9 类 + recommendation_confirm 确认)
_VALID_CATEGORIES = {
    "consultation", "trial", "lesson", "package",
    "progress", "reschedule", "cancel", "recommendation",
    "recommendation_confirm", "unrelated",
}


class TaskClassifier:
    """任务分类器（家教模式）

    context_manager: 可空依赖注入。为 None 时沿用旧 PromptTemplate 链(向后兼容);
    注入后改由 ContextManager.build 组装输入,分类归一逻辑保持不变(10 类不变)。
    """

    def __init__(self, llm: BaseChatModel, context_manager=None, session_id=None):
        self.llm = llm
        self.context_manager = context_manager
        self.session_id = session_id
        self._initialize_prompt()
        self.chain = self.prompt | self.llm

    def _initialize_prompt(self):
        self.prompt = PromptTemplate(
            input_variables=["task"],
            template=(
                "你是 K12 家教平台智能前台，负责判断家长咨询属于哪种意图。\n"
                "可选类别：\n"
                "1. consultation - 知识咨询（询问学习方法、课程介绍、老师特点、家长指南、考试策略等，**不含老师推荐请求**）\n"
                "2. trial - 试讲预约（家长想体验/试听某位老师）\n"
                "3. lesson - 正式课预约（直接预约上课，可能含周固定排课）\n"
                "4. package - 套餐咨询（询问价格、套餐内容、续费）\n"
                "5. progress - 进度查询（询问孩子学习情况、上课记录）\n"
                "6. reschedule - 调课（调整已有课时时间）\n"
                "7. cancel - 退课（取消已预约的课时）\n"
                "8. recommendation - 老师推荐（家长想推荐/匹配/找一位老师）\n"
                "9. recommendation_confirm - 对推荐结果的确认或选老师（家长在推荐结果后回复'是/好/序号/老师姓名'，如'看高远不错'、'就选3号'）\n"
                "10. unrelated - 无关问题（闲聊、其它话题）\n\n"
                "请只输出上述英文类别之一。\n\n"
                "举例：\n"
                "- '我家孩子数学不好怎么办' → consultation\n"
                "- '想让孩子试试张老师的课' → trial\n"
                "- '给我孩子预约明天下午3点的数学课' → lesson\n"
                "- '套餐多少钱' → package\n"
                "- '孩子最近学得怎么样' → progress\n"
                "- '把周三的课调到周五' → reschedule\n"
                "- '帮我取消明天的课' → cancel\n"
                "- '请给我推荐一位老师' → recommendation\n"
                "- '我看高远不错' → recommendation_confirm\n"
                "- '今天就选3号老师吧' → recommendation_confirm\n"
                "- '今天天气怎么样' → unrelated\n\n"
                "注意:家长无论回复『试讲/试听』还是『正式课』,都分别按第 2、3 分类处理(试讲→trial,正式课→lesson)。\n\n"
                "本次输入：{task}"
            ),
        )

    async def classify_task(self, task: str) -> str:
        with trace_span("agent_route", task=task[:30]) as span:
            try:
                # getattr 兜底:测试常以 __new__ 构造(不跑 __init__),避免 AttributeError
                if getattr(self, "context_manager", None) is not None:
                    # 注入统一上下文管理器:由它组装 system/conversation/structured 等
                    prompt = self.context_manager.build(
                        task=task, session_id=getattr(self, "session_id", None),
                        extra_contexts={"分类指令": self.prompt.format(task="")},
                    )
                    category_msg = await self.llm.ainvoke(
                        [{"role": "user", "content": prompt}]
                    )
                else:
                    # 向后兼容:未注入时沿用旧 PromptTemplate 链
                    category_msg = await self.chain.ainvoke({"task": task})
                category = self._extract_category(category_msg.content or "")
                span.attrs["intent"] = category
                span.attrs["agent"] = self._agent_for_category(category)
                return category
            except ContextBudgetExceeded:
                span.attrs["error"] = True
                raise
            except Exception as e:
                logger.warning("任务分类失败（LLM 不可用）: %s", e)
                span.attrs["error"] = True
                raise ClassificationUnavailable(str(e)) from e

    @staticmethod
    def _agent_for_category(category: str) -> str:
        """把意图类别映射到业务 Agent(仅观测用,不改路由逻辑)。"""
        mapping = {
            "consultation": "consultant",
            "package": "consultant",
            "progress": "consultant",
            "trial": "trial",
            "lesson": "lesson",
            "reschedule": "lesson",
            "cancel": "lesson",
            "recommendation": "recommendation",
            "recommendation_confirm": "recommendation",
            "unrelated": "unrelated",
        }
        return mapping.get(category, "unknown")

    @staticmethod
    def _extract_category(raw_content: str) -> str:
        """把 LLM 原始输出归一化到合法 10 类意图之一。

        1) 精确匹配:剥离引号/中英文标点/空白/编号外壳,再比对合法类别。
           修复:LLM 输出 "reschedule。" / "\"cancel\"" / "6." 等带尾巴时,
           旧实现直接归一成 unrelated 导致随机拒答。
        2) 子串匹配兜底:LLM 输出 "reschedule - 调课(调整已有课时时间)" 之类,
           在原文里找合法类别词。
        """
        raw = raw_content.strip().lower()

        cleaned = re.sub(
            r"[\s\"'“”‘’`。，.！!？?；;、：:()\[\]\-–—]+", "", raw
        )
        cleaned = re.sub(r"^\d+\.?", "", cleaned)
        if cleaned in _VALID_CATEGORIES:
            return cleaned

        for c in sorted(_VALID_CATEGORIES):
            if c in raw:
                return c

        return "unrelated"

    def get_category_description(self, category: str) -> str:
        descriptions = {
            "consultation": "知识咨询 - 家长咨询学习方法、课程介绍、考试策略",
            "trial": "试讲预约 - 家长想体验老师",
            "lesson": "正式课预约 - 直接预约上课",
            "package": "套餐咨询 - 询问价格、套餐、续费",
            "progress": "进度查询 - 查询学习情况",
            "reschedule": "调课请求",
            "cancel": "退课请求",
            "recommendation": "老师推荐 - 家长想匹配/找一位老师",
            "recommendation_confirm": "推荐确认 - 家长对推荐结果选老师/确认",
            "unrelated": "无关问题",
        }
        return descriptions.get(category, "未知任务类型")