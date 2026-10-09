"""
提示词构建器（家教场景）
"""

from typing import List, Dict, Any

# 系统提示词已收敛到统一上下文(agents.context)单一声明,此处委托,避免两处漂移
from agents.context.contexts.system_prompt import SYSTEM_PROMPT_TEXT


class PromptBuilder:
    """提示词构建器"""

    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT_TEXT
        self.classification_prompt_template = self._create_classification_prompt_template()

    def _create_classification_prompt_template(self) -> str:
        return (
            "你是一个分类器，判断用户输入是否是 K12 家教相关的咨询问题。\n"
            "咨询类问题包括：学习方法、课程介绍、老师特点、考试策略、家长指南、家校沟通、套餐价格等。\n"
            "非咨询类问题包括：试讲预约（想体验/试听某老师）、正式课预约、"
            "调课、退课、套餐购买、天气、新闻等无关话题。\n"
            "如果是咨询类问题，回答'YES'。否则回答'NO'。只回答YES或NO。\n\n"
            "用户输入：{user_input}"
        )

    def build_consultation_prompt(self, user_input: str, knowledge_docs: List[Dict[str, Any]]) -> str:
        context = self._build_knowledge_context(knowledge_docs)
        return f"{self.system_prompt}\n\n{context}\n用户问题：{user_input}\n\n请回答用户的问题。"

    def build_classification_prompt(self, user_input: str) -> str:
        return self.classification_prompt_template.format(user_input=user_input)

    def build_knowledge_context(self, knowledge_docs: List[Dict[str, Any]]) -> str:
        """把检索出的知识文档渲染成上下文文本(对外暴露,供上下文管理复用)。"""
        return self._build_knowledge_context(knowledge_docs)

    def _build_knowledge_context(self, knowledge_docs: List[Dict[str, Any]]) -> str:
        if not knowledge_docs:
            return "没有找到直接相关的知识库信息，请基于教育学常识回答，并标明「仅供参考」。"

        context = "\n以下是相关的知识库信息：\n"
        for i, doc in enumerate(knowledge_docs, 1):
            context += f"{i}. {doc['content']}\n"
        context += "\n请基于以上信息回答用户问题。如果知识库信息不足以回答问题，请基于教育学常识补充回答。\n"
        return context