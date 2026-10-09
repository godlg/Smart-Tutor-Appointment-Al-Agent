"""
状态管理器
"""

from enum import Enum
from typing import Optional


class FlowState(str, Enum):
    """会话流转状态"""
    CLASSIFY = "classify"        # 任务分类中
    CONSULTATION = "consultation"  # 咨询中
    RECOMMENDATION_CONFIRM = "recommendation_confirm"  # 推荐确认态（家长已拿到 Top-N,等待选老师/确认）
    TRIAL = "trial"              # 试讲流程中
    TRIAL_CONFIRM = "trial_confirm"  # 试讲预约确认态（已 yield direct trial 引导消息,等待『确认』）
    TRIAL_CONVERT = "trial_convert"  # 试讲转化态（家长已表达『试讲满意』,等待继续/选套餐/流失）
    LESSON = "lesson"            # 正式课流程中


class SharedState:
    """简单的共享状态容器"""

    def __init__(self):
        self.value: FlowState = FlowState.CLASSIFY


class StateManager:
    """状态管理器"""

    def __init__(self, shared_state: Optional[SharedState] = None):
        self.state = shared_state or SharedState()

    def get_current_state(self) -> FlowState:
        return self.state.value

    def set_state(self, new_state: FlowState) -> None:
        self.state.value = new_state

    def reset_to_classify(self) -> None:
        self.set_state(FlowState.CLASSIFY)

    def should_classify(self) -> bool:
        return self.get_current_state() == FlowState.CLASSIFY

    def is_in_consultation_flow(self) -> bool:
        return self.get_current_state() == FlowState.CONSULTATION

    def transition_to_consultation(self) -> None:
        self.set_state(FlowState.CONSULTATION)

    def is_in_recommendation_confirm_flow(self) -> bool:
        """是否处于"推荐确认态":已向家长输出 Top-N,等待选老师/确认『是/不』"""
        return self.get_current_state() == FlowState.RECOMMENDATION_CONFIRM

    def transition_to_recommendation_confirm(self) -> None:
        """进入推荐确认态:下一轮家长回复(姓名/ID/是/不)由主路由跳过分类器直接转给推荐 agent"""
        self.set_state(FlowState.RECOMMENDATION_CONFIRM)

    def is_in_trial_flow(self) -> bool:
        return self.get_current_state() == FlowState.TRIAL

    def transition_to_trial(self) -> None:
        self.set_state(FlowState.TRIAL)

    def is_in_trial_confirm_flow(self) -> bool:
        """是否处于『试讲预约确认态』：TrialAgent 已 yield direct trial 引导消息,
        等待家长回『确认』真正创建 Lesson(TRIAL)。

        与 is_in_trial_flow 的区别:
        - TRIAL:试讲流程中,但 trial_history 信息还在补全(学科/年级/时间)
        - TRIAL_CONFIRM:信息已齐,仅等家长回『确认』做最终落库
        """
        return self.get_current_state() == FlowState.TRIAL_CONFIRM

    def transition_to_trial_confirm(self) -> None:
        """进入试讲预约确认态:下一轮家长回复(确认/是/好)由主路由跳过分类器
        直接转给 TrialAgent 的 pending_trial 分支,真正创建 Lesson。"""
        self.set_state(FlowState.TRIAL_CONFIRM)

    def is_in_trial_convert_flow(self) -> bool:
        """是否处于『试讲转化态』：家长已表达试讲满意,等待继续/选套餐/流失。

        与 TRIAL_CONFIRM 的区别:
        - TRIAL_CONFIRM:试讲未真正创建 Lesson,仅等『确认』落库
        - TRIAL_CONVERT:试讲 Lesson 已创建,家长在『转正式课』引导中
        """
        return self.get_current_state() == FlowState.TRIAL_CONVERT

    def transition_to_trial_convert(self) -> None:
        """进入试讲转化态:下一轮家长回复(继续/选套餐/流失)由主路由跳过分类器
        直接转给 TrialAgent 的 awaiting_convert 分支。"""
        self.set_state(FlowState.TRIAL_CONVERT)

    def is_in_lesson_flow(self) -> bool:
        return self.get_current_state() == FlowState.LESSON

    def transition_to_lesson(self) -> None:
        self.set_state(FlowState.LESSON)

    def get_state_description(self) -> str:
        state = self.get_current_state()
        descriptions = {
            FlowState.CLASSIFY: "任务分类状态",
            FlowState.CONSULTATION: "咨询流程状态",
            FlowState.RECOMMENDATION_CONFIRM: "推荐确认态",
            FlowState.TRIAL: "试讲流程状态",
            FlowState.TRIAL_CONFIRM: "试讲预约确认态",
            FlowState.TRIAL_CONVERT: "试讲转化态",
            FlowState.LESSON: "正式课流程状态",
        }
        return descriptions.get(state, "未知状态")

    def force_reset(self) -> None:
        self.reset_to_classify()