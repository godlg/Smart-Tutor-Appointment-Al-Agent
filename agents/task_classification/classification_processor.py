"""
分类流程处理器（家教场景）
"""

from contextlib import aclosing
from typing import AsyncGenerator
from config.identity import get_current_identity
from ..context.types import ContextBudgetExceeded
from .task_classifier import TaskClassifier, ClassificationUnavailable
from .state_manager import StateManager
from .agent_router import AgentRouter
from .unrelated_handler import UnrelatedHandler
from ..availability.availability_handler import (
    AvailabilityHandler,
    is_availability_query,
)


_BUSINESS_CATEGORIES = frozenset({"trial", "lesson", "reschedule", "cancel", "progress"})
_BUSINESS_STATES = frozenset({"trial", "trial_confirm", "trial_convert", "lesson"})
_VERIFICATION_REPLY = (
    "[REPLY]预约、调退课或查询学员进度需要先验证家长身份及学员归属。"
    "当前匿名会话不能访问学员业务数据；您可以继续咨询课程或匹配公开老师。"
)
_BUSY_REPLY = "[REPLY]系统繁忙，暂时无法理解您的问题，请稍后重试。"
# 异常兜底文案（单一真源）：图路径导入复用，避免复制漂移。
_ERROR_BUDGET_REPLY = "[ERROR]上下文超过安全预算，请缩短输入后重试。"
_ERROR_GENERIC_TEMPLATE = "[ERROR]处理任务时发生错误: {error}"


class ClassificationProcessor:
    """分类流程处理器"""

    def __init__(
        self,
        task_classifier: TaskClassifier,
        state_manager: StateManager,
        agent_router: AgentRouter,
        unrelated_handler: UnrelatedHandler,
        availability_handler: AvailabilityHandler = None,
        handoff_service=None,
        session_id: str = None,
        session_chat_history=None,
    ):
        self.task_classifier = task_classifier
        self.state_manager = state_manager
        self.agent_router = agent_router
        self.unrelated_handler = unrelated_handler
        # 全局档期查询 handler(可注入替身);None 时惰性建实例
        self.availability_handler = availability_handler or AvailabilityHandler()
        # 人工接管:多次踢回达阈值后自动转人工(可空,向后兼容)
        self.handoff_service = handoff_service
        self.session_id = session_id or "default"
        # 跨 Agent 共享的会话历史(可空);阈值转人工时快照进 DB
        self.session_chat_history = session_chat_history

    def _snapshot_transcript(self):
        """安全地快照会话历史(未注入 / 无 messages 时返回 None)。"""
        if not self.handoff_service:
            return None
        history = getattr(self, "session_chat_history", None)
        messages = getattr(history, "messages", None)
        return self.handoff_service.snapshot_transcript(messages)

    def _build_transcript(self):
        """会话历史快照 + 最近用户输入(无关轮兜底);未注入 handoff 时 None。"""
        if not self.handoff_service:
            return None
        history = getattr(self, "session_chat_history", None)
        messages = getattr(history, "messages", None)
        return self.handoff_service.build_transcript(self.session_id, messages)

    def _requires_business_verification(self, category=None) -> bool:
        """匿名身份及student namespace不授权访问业务学员，legacy内部保持兼容。"""
        identity = get_current_identity()
        if identity is None or identity.legacy:
            return False
        return (category in _BUSINESS_CATEGORIES
                or self.state_manager.get_current_state() in _BUSINESS_STATES)

    async def process_task_stream(self, task: str) -> AsyncGenerator[str, None]:
        try:
            if self._requires_business_verification():
                yield _VERIFICATION_REPLY
                return
            # 推荐确认态优先:跳过分类器,直接转给 RecommendationAgent
            # 理由:家长已拿到 Top-N,在『确认/选老师/不』阶段再走 LLM 分类会把它
            # 误判成 recommendation_confirm/recommendation,出现『又来一次老师推荐请求』
            # 的体验混乱(语义错乱,功能无错但不必要)。
            if self.state_manager.is_in_recommendation_confirm_flow() and self.agent_router.recommendation_agent:
                async for token in self.agent_router.route_to_recommendation_confirm(task):
                    yield token
                return
            # 试讲预约确认态优先:跳过分类器,直接转给 TrialAgent
            # 理由:TrialAgent 已 yield 『请回复确认』引导消息,家长回『确认/是/好』时
            # 若再走 LLM 分类会被判成 recommendation/trial/无关等任意类别,把预约
            # 流程从 TrialAgent 侧带走(典型:把『确认』+ 上轮『老师』误判为新推荐)。
            # TrialAgent 入口会自行判断 pending_trial + 简单确认 → 直接创建 Lesson(TRIAL)。
            if self.state_manager.is_in_trial_confirm_flow() and self.agent_router.trial_agent:
                async for token in self.agent_router.route_to_trial_confirm(task):
                    yield token
                return
            # 试讲转化态优先:跳过分类器,直接转给 TrialAgent 的 awaiting_convert 分支。
            # 理由:家长已表达『试讲满意』后回『继续/选套餐/流失』,再走 LLM 分类会
            # 被判成 unrelated/consultation 等任意类别,把转化流程带走。
            if self.state_manager.is_in_trial_convert_flow() and self.agent_router.trial_agent:
                async for token in self.agent_router.route_to_trial_convert(task):
                    yield token
                return
            if self.state_manager.should_classify():
                # 全局档期查询白名单:任意可分类态(CONSULTATION 流外,如新会话/冲突 abort 后
                # 重置回 CLASSIFY)下问『老师什么时候有空』,先用规则关键词拦截,临时应答可用时段。
                # 不新增意图类、不改 FlowState,下一轮仍按原状态机走;也不消耗 LLM 分类调用。
                # TRIAL / *_CONFIRM 态由上方 confirm 早返回短路,仍走对应 Agent 自身的档期查询。
                if is_availability_query(task):
                    async for token in self.availability_handler.handle(task):
                        yield token
                    return
                try:
                    category = await self.task_classifier.classify_task(task)
                except ClassificationUnavailable:
                    # LLM 分类服务不可用:优雅降级,不踢回/不转人工/不改状态,家长稍后可重试
                    yield _BUSY_REPLY
                    return
                if self._requires_business_verification(category):
                    yield _VERIFICATION_REPLY
                    return

                if category == "trial" and self.agent_router.trial_agent:
                    self.state_manager.transition_to_trial()
                    async for token in self.agent_router.route_to_trial(task):
                        yield token
                elif category == "lesson" and self.agent_router.lesson_agent:
                    self.state_manager.transition_to_lesson()
                    async for token in self.agent_router.route_to_lesson(task):
                        yield token
                elif category == "reschedule" and self.agent_router.lesson_agent:
                    async for token in self.agent_router.route_to_reschedule(task):
                        yield token
                elif category == "cancel" and self.agent_router.lesson_agent:
                    async for token in self.agent_router.route_to_cancel(task):
                        yield token
                # 推荐确认(recommendation_confirm)复用同一位推荐 agent:run_stream 内按
                # session 状态判断是否处于确认流程,避免新增路由方法(方案 6B,更易测)。
                elif category in ("recommendation", "recommendation_confirm") and self.agent_router.recommendation_agent:
                    async for token in self.agent_router.route_to_recommendation(task):
                        yield token
                elif category in ("consultation", "package", "progress") and self.agent_router.consultant_agent:
                    async with aclosing(self.agent_router.route_to_consultation(task)) as stream:
                        async for token in stream:
                            yield token
                else:
                    # 踢回计数:同会话累计踢回达阈值 → 自动转人工,不再反复输出能力列表
                    if self.handoff_service and self.handoff_service.record_kickback(self.session_id):
                        count = self.handoff_service.get_count(self.session_id)
                        self.handoff_service.trigger_handoff(
                            self.session_id, "kickback_threshold",
                            reason=f"累计踢回 {count} 次",
                            unrelated_count=count,
                            transcript=self._build_transcript(),
                        )
                        yield "[REPLY][HANDOFF]"
                        for char in self.handoff_service.handoff_reply():
                            yield char
                        return
                    async for token in self.agent_router.handle_unsupported_task(category):
                        yield token
            else:
                async with aclosing(self.agent_router.route_by_state(task)) as stream:
                    async for token in stream:
                        yield token
        except ContextBudgetExceeded:
            yield _ERROR_BUDGET_REPLY
        except Exception as e:
            yield _ERROR_GENERIC_TEMPLATE.format(error=e)
            self.state_manager.force_reset()

    async def process_task_sync(self, task: str) -> str:
        try:
            if self._requires_business_verification():
                return _VERIFICATION_REPLY
            # 推荐确认态优先:跳过分类器,直接收 RecommendationAgent 的输出
            if self.state_manager.is_in_recommendation_confirm_flow() and self.agent_router.recommendation_agent:
                return await self._collect_stream(
                    self.agent_router.route_to_recommendation_confirm(task)
                )
            # 试讲预约确认态优先:跳过分类器,直接收 TrialAgent 的输出
            if self.state_manager.is_in_trial_confirm_flow() and self.agent_router.trial_agent:
                return await self._collect_stream(
                    self.agent_router.route_to_trial_confirm(task)
                )
            # 试讲转化态优先:跳过分类器,直接收 TrialAgent 的 awaiting_convert 分支输出
            if self.state_manager.is_in_trial_convert_flow() and self.agent_router.trial_agent:
                return await self._collect_stream(
                    self.agent_router.route_to_trial_convert(task)
                )
            if self.state_manager.should_classify():
                # 全局档期查询白名单(与流式版同):规则关键词拦截,临时应答可用时段。
                if is_availability_query(task):
                    return await self._collect_stream(self.availability_handler.handle(task))
                try:
                    category = await self.task_classifier.classify_task(task)
                except ClassificationUnavailable:
                    # LLM 分类服务不可用:优雅降级,不踢回/不转人工/不改状态,家长稍后可重试
                    return _BUSY_REPLY
                if self._requires_business_verification(category):
                    return _VERIFICATION_REPLY

                # 同步版必须把下游 agent 的整个异步 token 流拼成一条完整字符串返回。
                # 旧实现 `async for token in ...: return token` 只返回首个 token，
                # yield 第一个之后 generator 就被 close，下游 agent 的 run_stream
                # 实际上还没被真正迭代，副作用（状态写入、调课/退课执行等）会丢失。
                if category == "trial" and self.agent_router.trial_agent:
                    self.state_manager.transition_to_trial()
                    return await self._collect_stream(self.agent_router.route_to_trial(task))
                elif category == "lesson" and self.agent_router.lesson_agent:
                    self.state_manager.transition_to_lesson()
                    return await self._collect_stream(self.agent_router.route_to_lesson(task))
                elif category == "reschedule" and self.agent_router.lesson_agent:
                    return await self._collect_stream(self.agent_router.route_to_reschedule(task))
                elif category == "cancel" and self.agent_router.lesson_agent:
                    return await self._collect_stream(self.agent_router.route_to_cancel(task))
                elif category in ("recommendation", "recommendation_confirm") and self.agent_router.recommendation_agent:
                    return await self._collect_stream(self.agent_router.route_to_recommendation(task))
                elif category in ("consultation", "package", "progress") and self.agent_router.consultant_agent:
                    return await self._collect_stream(self.agent_router.route_to_consultation(task))
                else:
                    # 踢回计数:同会话累计踢回达阈值 → 自动转人工(同步版返回字符串)
                    if self.handoff_service and self.handoff_service.record_kickback(self.session_id):
                        count = self.handoff_service.get_count(self.session_id)
                        self.handoff_service.trigger_handoff(
                            self.session_id, "kickback_threshold",
                            reason=f"累计踢回 {count} 次",
                            unrelated_count=count,
                            transcript=self._build_transcript(),
                        )
                        return "[REPLY][HANDOFF]" + self.handoff_service.handoff_reply()
                    return "暂不支持该类型任务"
            return await self._continue_conversation(task)
        except ContextBudgetExceeded:
            return "[ERROR]上下文超过安全预算，请缩短输入后重试。"
        except Exception as e:
            self.state_manager.force_reset()
            return f"处理任务时发生错误: {str(e)}"

    @staticmethod
    async def _collect_stream(stream) -> str:
        """把异步 generator 的所有 token 拼成完整字符串返回"""
        parts = []
        async for token in stream:
            parts.append(token)
        return "".join(parts)

    async def _continue_conversation(self, task: str) -> str:
        if self.state_manager.is_in_consultation_flow() and self.agent_router.consultant_agent:
            async with self.agent_router.consultant_agent as agent:
                return await agent.consult(task)
        if self.state_manager.is_in_trial_flow() and self.agent_router.trial_agent:
            # 试讲续轮：直接拼 TrialAgent 流(不带首轮分流文案)
            return await self._collect_stream(
                self.agent_router.trial_agent.run_stream(user_input=task)
            )
        if self.state_manager.is_in_lesson_flow() and self.agent_router.lesson_agent:
            # 正式课续轮：直接拼 LessonAgent 流(不带首轮分流文案)
            return await self._collect_stream(
                self.agent_router.lesson_agent.run_stream(user_input=task)
            )
        return "暂不支持该类型任务"

    def get_current_state_info(self) -> dict:
        return {
            "current_state": self.state_manager.get_current_state(),
            "state_description": self.state_manager.get_state_description(),
            "available_services": self.agent_router.get_available_services(),
            "can_classify": self.state_manager.should_classify(),
        }

    def reset_conversation(self) -> None:
        self.state_manager.force_reset()
        self.unrelated_handler.reset_reply_rotation()