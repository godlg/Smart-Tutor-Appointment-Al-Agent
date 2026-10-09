"""
全局档期查询处理器（家教场景）

新会话 / 任意"可分类"状态下,家长直接问"老师什么时候有空/查一下某老师档期/什么时间段没空"
时,在主分类器运行之前用规则关键词拦截(不新增 LLM 调用、不改 FlowState),返回该老师
最近可预约时段列表并引导选择。

与试讲流程内档期查询(agents/trial/trial_processor.py 的 _handle_availability_query)为
**独立实现**:本模块是全局入口,不依赖 trial_history / confirmed_teacher / 试讲态,
纯查询;它从输入本身提取老师名,而非复用试讲会话上下文。

依赖方向:agents/availability ——> services(TeacherService / LessonService) + config/time_config,
与其它 agent 模块平级,不反向依赖 trial / task_classification,避免跨 Agent 耦合。
"""

import re
from datetime import timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional

from config.time_config import time_config

# 档期查询触发关键词(与试讲流程内 _AVAILABILITY_KEYWORDS 同款集合,规则检测)
_AVAILABILITY_KEYWORDS = (
    "有空", "没空", "空闲", "档期", "什么时候方便", "时间安排", "什么时间",
)

# 老师姓名候选长度(中文名通常 2-4 字)
_TEACHER_NAME_PATTERN = re.compile(r"([一-龥]{2,4})\s*老师")

# 部分田地:未来 N 天 × 每天 9:00-21:00 逐小时整点,收集最多 N 个可用时段
_LOOKAHEAD_DAYS = 3
_DAY_START_HOUR = 9
_DAY_END_HOUR = 22  # range 上限:21 点算最后一个
_MAX_SLOTS = 5
_SLOT_DURATION = timedelta(minutes=30)  # 试讲默认时长,冲突判定区间

_WEEKDAY_LABELS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def is_availability_query(user_input: str) -> bool:
    """判定家长输入是否为『档期查询』意图(老师什么时间段有空/没空)。

    纯关键字命中,无 LLM 调用。命中后在分类入口被全局拦截,临时应答可用时段。
    """
    s = (user_input or "").strip()
    if not s:
        return False
    return any(k in s for k in _AVAILABILITY_KEYWORDS)


class AvailabilityHandler:
    """全局档期查询 handler(无状态,每次独立查询)"""

    def __init__(self, teacher_service: Any = None, lesson_service: Any = None):
        """可注入 teacher_service / lesson_service,便于测试传替身;

        为 None 时在 handle() 内按需惰性创建(避免构造即连 DB)。
        """
        self.teacher_service = teacher_service
        self.lesson_service = lesson_service

    async def handle(self, user_input: str) -> AsyncGenerator[str, None]:
        """解析老师名 → 遍历未来可预约时段 → yield 引导/时段消息。"""
        from services.teacher_service import TeacherService
        from services.lesson_service import LessonService

        teacher_svc = self.teacher_service or TeacherService()
        teacher = self._extract_teacher(teacher_svc, user_input)
        teacher_name = teacher.get("name", "老师") if teacher else "老师"

        if teacher is None:
            yield (
                "[REPLY][归类机器人]\n顾问：请告诉我要查询哪位老师，"
                "回复老师姓名即可。\n"
            )
            return

        lesson_svc = self.lesson_service or LessonService()
        lesson_repo = lesson_svc.lesson_repo

        today = time_config.today()
        # ⚠️ 关键:time_config.today() 返回带 Beijing tzinfo 的 aware datetime,而
        # Lesson.scheduled_at 走 naive datetime + sqlite 不存 tzinfo;
        # check_teacher_conflict 内部 `l.scheduled_at < end_time` 会在 naive vs aware 之间
        # 抛 "can't compare offset-naive and offset-aware datetimes"。
        # 与试讲态 _create_pending_trial_lesson 一致:统一下沉为 naive(参考该处 line 242-243)。
        if today.tzinfo is not None:
            today = today.replace(tzinfo=None)

        slots: List[str] = []
        for d in range(_LOOKAHEAD_DAYS):
            day = today + timedelta(days=d)
            for hour in range(_DAY_START_HOUR, _DAY_END_HOUR):
                slot = day.replace(hour=hour, minute=0, second=0, microsecond=0)
                end = slot + _SLOT_DURATION
                if lesson_repo.check_teacher_conflict(teacher["id"], slot, end):
                    continue
                slots.append(f"{_WEEKDAY_LABELS[day.weekday()]} {hour}:00")
                if len(slots) >= _MAX_SLOTS:
                    break
            if len(slots) >= _MAX_SLOTS:
                break

        if not slots:
            yield (
                f"[REPLY][归类机器人]\n顾问：抱歉，{teacher_name}老师最近三天暂时没有空闲时段，"
                f"请稍后重试或换个时间。\n"
            )
            return

        shown = "、".join(slots)
        yield (
            f"[REPLY][归类机器人]\n顾问：{teacher_name}老师最近可预约:{shown}…"
            f"请告诉我要哪个时间。\n"
        )

    def _extract_teacher(self, teacher_svc: Any, user_input: str) -> Optional[Dict[str, Any]]:
        """从输入提取老师名。

        ① 姓名子串优先:遍历老师表,任一名字出现在输入里即命中。子串匹配天然不受
           前缀干扰(『查一下高远老师的档期』→『高远』),参考 _match_teacher 的姓名
           子串思路,这也是 spec 要求的首选。
        ② 'XX老师' 模式兜底:覆盖名字只在数据库、不在常量表的情况。因 `X老师` 前可被
           前缀词污染(如『查一下高远老师』会抓成『查一下高远』),若整段取不到,再试
           紧贴『老师』的最后 2/3 字(种子老师名均为 2 字)。
        """
        text = (user_input or "").strip()
        if not text:
            return None

        # ① 姓名子串匹配(唯一正确性来源,先跑)
        for t in teacher_svc.get_all_teachers() or []:
            name = t.get("name", "")
            if name and name in text:
                return t

        # ② 'XX老师' 模式兜底
        m = _TEACHER_NAME_PATTERN.search(text)
        if m:
            raw = m.group(1)
            t = teacher_svc.get_teacher_by_name(raw)
            if t:
                return t
            # 前缀干扰兜底:取紧贴『老师』前的 2/3 字再试
            for cand in (raw[-2:], raw[-3:]):
                t = teacher_svc.get_teacher_by_name(cand)
                if t:
                    return t
            return None
        return None