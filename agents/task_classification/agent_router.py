"""
智能体路由器（家教场景）
"""

from contextlib import aclosing
from typing import Any, AsyncGenerator
from .state_manager import StateManager


# 业务分流行文案（首轮分类命中业务意图时 yield）：单一真源，图节点复用 route_thought
# 而不是复制字符串，避免两条链路文案漂移。
THOUGHT_PREFIX = "[THOUGHT][归类机器人] "
ROUTE_THOUGHTS = {
    "trial": "这是一个试讲预约请求，转给试讲顾问处理。",
    "lesson": "这是一个正式课预约，转给课程助手处理。",
    "reschedule": "这是一个调课请求，转给课程助手处理。",
    "cancel": "这是一个退课请求，转给课程助手处理。",
    "recommendation": "这是一个老师推荐请求，转给推荐顾问处理。",
    "consultation": "这是一个咨询问题，转给咨询机器人处理。",
}


def route_thought(intent: str) -> str:
    """返回某业务意图的首轮分流行完整文案（含前缀）。"""
    return f"{THOUGHT_PREFIX}{ROUTE_THOUGHTS[intent]}"


class AgentRouter:
    """智能体路由器"""

    def __init__(
        self,
        appointment_agent: Any = None,
        consultant_agent: Any = None,
        state_manager: StateManager = None,
        trial_agent: Any = None,
        lesson_agent: Any = None,
        recommendation_agent: Any = None,
    ):
        self.appointment_agent = appointment_agent
        self.consultant_agent = consultant_agent
        self.trial_agent = trial_agent
        self.lesson_agent = lesson_agent
        self.recommendation_agent = recommendation_agent
        self.state_manager = state_manager

        self._setup_agent_states()

    def _setup_agent_states(self):
        for agent in [self.consultant_agent, self.trial_agent, self.lesson_agent, self.recommendation_agent]:
            if agent and hasattr(agent, "set_shared_state") and self.state_manager:
                agent.set_shared_state(self.state_manager.state)

    async def route_to_trial(self, task: str) -> AsyncGenerator[str, None]:
        if not self.trial_agent:
            yield "[ERROR]试讲服务暂时不可用"
            return
        yield route_thought("trial")
        try:
            async for token in self.trial_agent.run_stream(user_input=task):
                yield token
        except Exception as e:
            yield f"[ERROR]试讲处理失败: {str(e)}"

    async def route_to_lesson(self, task: str) -> AsyncGenerator[str, None]:
        if not self.lesson_agent:
            yield "[ERROR]正式课服务暂时不可用"
            return
        yield route_thought("lesson")
        try:
            async for token in self.lesson_agent.run_stream(user_input=task):
                yield token
        except Exception as e:
            yield f"[ERROR]正式课处理失败: {str(e)}"

    async def route_to_reschedule(self, task: str) -> AsyncGenerator[str, None]:
        """调课请求 → LessonAgent（复用 RescheduleHandler）"""
        async for token in self._route_lesson_change(
            task, intent="reschedule", error_prefix="调课处理失败",
        ):
            yield token

    async def route_to_cancel(self, task: str) -> AsyncGenerator[str, None]:
        """退课请求 → LessonAgent（复用 CancelHandler）"""
        async for token in self._route_lesson_change(
            task, intent="cancel", error_prefix="退课处理失败",
        ):
            yield token

    async def _route_lesson_change(
        self, task: str, intent: str, error_prefix: str
    ) -> AsyncGenerator[str, None]:
        """调课/退课共用的转发逻辑，仅意图与提示文案不同"""
        if not self.lesson_agent:
            yield "[ERROR]正式课服务暂时不可用"
            return
        yield route_thought(intent)
        try:
            async for token in self.lesson_agent.run_stream(
                user_input=task, intent=intent
            ):
                yield token
        except Exception as e:
            yield f"[ERROR]{error_prefix}: {str(e)}"

    async def route_to_recommendation(self, task: str) -> AsyncGenerator[str, None]:
        """老师推荐请求 → RecommendationAgent"""
        if not self.recommendation_agent:
            yield "[ERROR]老师推荐服务暂时不可用"
            return
        yield route_thought("recommendation")
        try:
            async for token in self.recommendation_agent.run_stream(user_input=task):
                yield token
        except Exception as e:
            yield f"[ERROR]老师推荐处理失败: {str(e)}"

    async def route_to_recommendation_confirm(self, task: str) -> AsyncGenerator[str, None]:
        """推荐确认态续轮 → RecommendationAgent（不带首轮分流文案,家长已经在确认流程里）

        与 route_to_recommendation 的区别:不 yield "[THOUGHT][归类机器人]" 那句,
        避免家长看到『又来一个老师推荐请求』。调用方已保证当前 state=RECOMMENDATION_CONFIRM。
        """
        if not self.recommendation_agent:
            yield "[ERROR]老师推荐服务暂时不可用"
            return
        try:
            async for token in self.recommendation_agent.run_stream(user_input=task):
                yield token
        except Exception as e:
            yield f"[ERROR]老师推荐处理失败: {str(e)}"

    async def route_to_trial_confirm(self, task: str) -> AsyncGenerator[str, None]:
        """试讲预约确认态续轮 → TrialAgent（不带首轮分流文案,家长已经在确认流程里）

        TrialAgent 入口优先判断 trial_history.pending_trial + 用户输入是『确认/是/好』:
        - 命中 → 直接创建 Lesson(TRIAL),不走 LLM 分类
        - 不命中(用户想改时间等)→ 走原 TrialInputParser 解析

        调用方已保证当前 state=TRIAL_CONFIRM。
        与 route_to_trial 的区别:不 yield "[THOUGHT][归类机器人]" 那句,
        避免家长看到『又来一个试讲预约请求』错乱文案。
        """
        if not self.trial_agent:
            yield "[ERROR]试讲服务暂时不可用"
            return
        try:
            async for token in self.trial_agent.run_stream(user_input=task):
                yield token
        except Exception as e:
            yield f"[ERROR]试讲处理失败: {str(e)}"

    async def route_to_trial_convert(self, task: str) -> AsyncGenerator[str, None]:
        """试讲转化态续轮 → TrialAgent（不带首轮分流文案,家长已在转化流程里）

        TrialAgent 入口优先判断 trial_history.awaiting_convert:
        - 命中考藏(继续/满意)→ 推荐套餐
        - 命中套餐选择 → 创建正式系列课
        - 命中流失否定 → 引导兜底

        调用方已保证当前 state=TRIAL_CONVERT。
        与 route_to_trial 的区别:不 yield "[THOUGHT][归类机器人]" 那句,
        避免家长看到『又来一个试讲预约请求』错乱文案(与 route_to_trial_confirm 同型)。
        """
        if not self.trial_agent:
            yield "[ERROR]试讲服务暂时不可用"
            return
        try:
            async for token in self.trial_agent.run_stream(user_input=task):
                yield token
        except Exception as e:
            yield f"[ERROR]试讲处理失败: {str(e)}"

    async def route_to_consultation(self, task: str) -> AsyncGenerator[str, None]:
        if not self.consultant_agent:
            yield "[ERROR]咨询服务暂时不可用"
            return
        yield route_thought("consultation")
        try:
            async with self.consultant_agent as agent:
                async with aclosing(agent.consult_stream(task)) as stream:
                    async for token in stream:
                        yield token
        except Exception as e:
            yield f"[ERROR]咨询处理失败: {str(e)}"

    async def handle_unsupported_task(self, category: str) -> AsyncGenerator[str, None]:
        reply = (
            "这个问题暂时超出我的能力范围。我可以帮您：\n"
            "- 课程/学习方法咨询\n"
            "- 老师推荐与试讲预约\n"
            "- 正式课预约\n"
            "- 调课与退课\n"
            "- 套餐咨询\n"
            "- 进度查询"
        )
        yield "[REPLY][归类机器人]"
        for char in reply:
            yield char

    async def route_by_state(self, task: str) -> AsyncGenerator[str, None]:
        """根据当前状态路由（用于多轮对话）"""
        # 简化：根据状态直接走相应 agent
        # 推荐确认态优先:避免与 CONSULTATION/TRIAL/CLASSIFY 误命中
        if self.recommendation_agent and self.state_manager.is_in_recommendation_confirm_flow():
            try:
                async for token in self.recommendation_agent.run_stream(user_input=task):
                    yield token
            except Exception as e:
                yield f"[ERROR]老师推荐处理失败: {str(e)}"
        elif self.consultant_agent and self.state_manager.is_in_consultation_flow():
            async with self.consultant_agent as agent:
                async with aclosing(agent.consult_stream(task)) as stream:
                    async for token in stream:
                        yield token
        elif self.trial_agent and self.state_manager.is_in_trial_flow():
            # 试讲续轮：直接回 TrialAgent，不再走 route_to_trial（避免首轮分流文案）
            try:
                async for token in self.trial_agent.run_stream(user_input=task):
                    yield token
            except Exception as e:
                yield f"[ERROR]试讲处理失败: {str(e)}"
        elif self.trial_agent and self.state_manager.is_in_trial_convert_flow():
            # 试讲转化续轮：直接回 TrialAgent（awaiting_convert 分支），不再走首轮分流文案
            try:
                async for token in self.trial_agent.run_stream(user_input=task):
                    yield token
            except Exception as e:
                yield f"[ERROR]试讲处理失败: {str(e)}"
        elif self.lesson_agent and self.state_manager.is_in_lesson_flow():
            # 正式课续轮：直接回 LessonAgent，不再走 route_to_lesson（避免首轮分流文案）
            try:
                async for token in self.lesson_agent.run_stream(user_input=task):
                    yield token
            except Exception as e:
                yield f"[ERROR]正式课处理失败: {str(e)}"
        else:
            # 重置后重新分类
            self.state_manager.reset()
            async for token in self.handle_unsupported_task(""):
                yield token

    def get_available_services(self) -> list:
        services = []
        if self.consultant_agent:
            services.append("咨询服务")
        if self.trial_agent:
            services.append("试讲预约")
        if self.lesson_agent:
            services.append("正式课预约")
            services.append("调课/退课")
        if self.recommendation_agent:
            services.append("老师推荐")
        return services