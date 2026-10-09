"""
试讲 Agent 子模块

职责：
1. 解析试讲需求（学科/年级/时间/孩子情况）
2. 老师匹配（按学科 + 年级 + 可试讲）
3. 试讲邀请消息构建
4. 试讲反馈处理
5. 试讲转化逻辑（推荐正式课）
"""

import os
import re
import json
import asyncio
from typing import Dict, Any, AsyncGenerator, Optional, List, Iterable, Union
from datetime import datetime, timedelta
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage


def extract_confirmed_teacher_id(chat_history) -> Optional[int]:
    """从 chat_history 最近一条 AI 消息里抽 teacher_id。

    支持两种格式:
      - "老师ID: 1"  / "老师ID:1"
      - "(ID 1)" / "ID 1"

    只看最近一条 AI 消息(逐条找到首条 ai 就 break),避免老消息污染。

    chat_history 支持两种入参(测试友好):
      - List[Dict]: 每条 {"role": ..., "content": ...}
      - LangChain InMemoryChatMessageHistory / 任何有 .messages 属性的对象: 自动读 .messages
      - LangChain BaseMessage 列表: 用 type=='ai' 判定角色

    Returns:
        teacher_id (int) 或 None(没找到 / chat_history 为空)
    """
    if not chat_history:
        return None

    # 兼容:InMemoryChatMessageHistory(直接迭代会 yield (name, value) tuple,
    # 不是消息),所以优先读 .messages 属性
    if hasattr(chat_history, "messages") and not isinstance(chat_history, dict):
        msg_source = list(chat_history.messages)
    elif isinstance(chat_history, dict):
        # 不应该传 dict 进来,但防御一下
        msg_source = []
    else:
        msg_source = list(chat_history)

    # 把对象归一成 dict-style(role, content)
    def _normalize(m):
        if isinstance(m, dict):
            return m.get("role"), m.get("content", "")
        # langchain Message (HumanMessage/AIMessage/...): 用 type 字段
        msg_type = getattr(m, "type", None)
        # langchain 约定:ai = assistant
        role = "assistant" if msg_type == "ai" else msg_type
        return role, getattr(m, "content", "") or ""

    # 倒序遍历,找第一条 AI 消息(并保证 content 非空,避免被空元数据消息拦截)
    for raw in reversed(msg_source):
        role, content = _normalize(raw)
        if role != "assistant":
            continue
        # 优先匹配"老师ID: N"
        m = re.search(r"老师ID\s*[:：]\s*(\d+)", content)
        if m:
            return int(m.group(1))
        # 兜底匹配"(ID N)" / "ID N"
        m = re.search(r"\(?ID\s+(\d+)\)?", content)
        if m:
            return int(m.group(1))
        # 只看最近一条 AI 消息,避免老消息污染
        break
    return None


def _coerce_bool(value) -> bool:
    """把 JSON 里的布尔字段归一到真布尔:字符串"false"/"true"与真布尔统一。

    根因:LLM(deepseek-chat)字段提取常输出 `"unrelated": "false"`(带引号字符串),
    Python 里非空字符串恒为真 bool("false")==True,会把"无关"误判成真、踢回主路由
    形成死循环,或把"信息不全"误判为"信息齐全"跳过追问。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


# 简单确认关键字集合(纯字符串判定,无 LLM 调用)
# 跟 recommendation_confirm 的『是/好/确认』同型,但额外覆盖『确认…』起始词
# 典型误用:『确认时间/确认安排』也是对预约的肯定回复(只是补一句时间),
# 也认作简单确认(让 TrialAgent 走 pending_trial 落库分支);
# 明确疑问句『确认吗?/要确认吗?』不认,让家长明确回复。
_SIMPLE_CONFIRM_EXACT = frozenset({
    "是", "好", "可以", "行", "确认",
    "ok", "OK", "Ok",
    "yes", "Yes", "YES",
    "y", "Y",
    "好呀", "好的", "行吧", "可以呀", "可以啊",
    "ok啦", "ok的", "okay", "OKAY", "Okay",
    "好的呢", "好的吧", "是的", "对的", "没问题", "可以的",
    "可以吧", "行吧",
})


# 档期查询触发关键词(规则检测,不新增 LLM 调用)
_AVAILABILITY_KEYWORDS = (
    "有空", "没空", "空闲", "档期", "什么时候方便", "时间安排", "什么时间",
)


def _is_availability_query(user_input: str) -> bool:
    """判定家长输入是否为『档期查询』意图(老师什么时间段有空/没空)。

    纯关键字命中,无 LLM 调用。命中后走试讲流程内档期查询分支,
    返回老师最近可预约时段并引导选择。
    """
    s = (user_input or "").strip()
    if not s:
        return False
    return any(k in s for k in _AVAILABILITY_KEYWORDS)


def _fallback_start_time_from_input(data: Dict[str, Any], user_input: Optional[str]) -> bool:
    """B 兜底:LLM 字段提取没抽到 start_time 时,用 parse_datetime 解析家长原始输入。

    背景:试讲流程把 start_time 交给 LLM 提取,但『这个星期周五晚上』『周五晚上』这类
    相对表达 deepseek 常直接返回「未知」,导致流程永远卡在『信息不完整,缺少:start_time』。
    parse_datetime 已增强支持这些相对表达(见 config/time_config.py _parse_relative_datetime),
    故这里在字段层做确定性兜底:LLM 没给时间时,直接把家长原文喂给 parse_datetime,
    把换算后的绝对时间(YYYY-MM-DD HH:MM)写回 data["start_time"],喂给状态机。

    Args:
        data: TrialInputParser.parse_data 的输出(原地补写 start_time)
        user_input: 家长本轮原始输入

    Returns:
        True: 兜底生效,data["start_time"] 已被补成绝对时间
        False: 已有时或解析失败(调用方保持原样)
    """
    user_in = (user_input or "").strip()
    if not user_in:
        return False
    from config.time_config import time_config

    raw = (data.get("start_time") or "").strip()
    # 仅当 raw 能 parse 成功时才不兜底;否则(空 / "未知" / 残缺值如"2026-09-18 未知")
    # 一律用家长原文再兜一次,避免被"非空但不可解析"的残缺值骗过、漏掉原文兜底。
    if raw:
        try:
            dt0 = time_config.parse_datetime(raw)
        except Exception:
            dt0 = None
        if dt0 is not None:
            return False
    try:
        dt = time_config.parse_datetime(user_in)
    except Exception:
        return False
    if dt is None:
        return False
    data["start_time"] = dt.strftime("%Y-%m-%d %H:%M")
    return True


def _is_simple_confirm(user_input: str) -> bool:
    """判定用户输入是否为『对预约的简单确认』。纯关键字,无 LLM 调用。

    与 recommendation_confirm.md 里的正负向判断同型(同样的简单肯定回复集合),
    但额外支持『确认…』起始词(『确认时间/确认安排』也是对预约的肯定)。

    Args:
        user_input: 家长本轮输入

    Returns:
        True: 简单确认 → 走 pending_trial 落库分支
        False: 其他意图(疑问/修改时间/选老师) → 走原 TrialInputParser
    """
    s = (user_input or "").strip()
    if not s:
        return False
    # 负向:疑问句结尾(如『确认吗?/要确认吗?/是吗?』)—不认作确认
    if s.endswith("?") or s.endswith("？"):
        return False
    # 负向:否定前缀(如『不是/不要/不要确认』) — 也不认
    neg_prefixes = ("不是", "不要", "不确认", "不是确认", "别确认", "先不")
    if any(s.startswith(p) for p in neg_prefixes):
        return False
    # 正向精确匹配
    if s in _SIMPLE_CONFIRM_EXACT:
        return True
    # 正向起始词匹配:『确认…』
    if s.startswith("确认"):
        return True
    return False


def _extract_date_part(start_time_str: str) -> str:
    """从残缺的 start_time 字符串(如 "2026-09-25 未知")里提取日期部分。

    背景:LLM 字段提取偶发把日期带出、小时漏成「未知」,导致时间解析失败。
    追问时带上这个日期作为上下文,让家长只需补时间(如「晚上7点」),
    再拼回 `{date} {时间}` 一起解析。

    Args:
        start_time_str: 残缺时间串,形如 "2026-09-25 未知" / "未知" / ""。

    Returns:
        日期串 "YYYY-MM-DD";无可提取的合法日期返回 ""。
    """
    if not start_time_str:
        return ""
    m = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", start_time_str)
    return m.group(1) if m else ""


# 时间追问时的否定关键词(纯关键字,无 LLM 调用):命中表示家长放弃/改约,
# 应清掉追问标志走正常解析,而不是把「不约了」当时间硬解析。
_TIME_NEGATION_KEYWORDS = (
    "不约了", "不约", "不要", "取消", "算了", "算了吧",
    "换老师", "换时间", "改天", "别了", "不上了",
)


def _is_time_negation(user_input: str) -> bool:
    """判定家长在时间追问后的输入是否为『否定/放弃/改约』意图。

    纯关键字命中,无 LLM 调用。命中后 TrialAgent 会清掉 pending_time_query
    标志,走正常 TrialInputParser 流程(让家长重新表达),而不是把它当作
    「几点」硬解析失败后兜底「请换个说法」。

    Args:
        user_input: 家长本轮输入。

    Returns:
        True: 含放弃/改约否定词 → 清追问标志走正常流程
        False: 其他输入(当作补充具体时间处理)
    """
    s = (user_input or "").strip()
    if not s:
        return False
    return any(k in s for k in _TIME_NEGATION_KEYWORDS)


# ========================== 试讲转化意图(试讲满意 → 转正式课) ==========================

# 转化意图正向关键词(纯关键字,无 LLM 调用):命中表示家长表达
# 『试讲满意,想转正式课/继续上』。注意:含"不/没"的短语("试讲不错""讲得不错")
# 也是正项,因此 _is_convert_intent 必须先判正向、再判否定,顺序不能反。
_CONVERT_INTENT_POSITIVES = (
    "试讲觉得不错", "试讲不错", "试讲满意", "试讲挺好", "试讲感觉不错",
    "讲得不错", "很满意", "非常满意",
    "要这位老师", "就要他", "就要她", "就他了", "就她吧", "喜欢这位老师",
    "继续上", "想继续",
    "转正式课", "转成正式", "报正式课", "长期学", "报长期",
    "跟他上", "跟他学", "跟她上", "跟她学",
)

# 转化意图否定词(先行拦截):命中表示不满意/不报,一律不算转化意图。
# 故意不放裸"不/没/美"——"试讲不错""讲得不错"都含"不"但属正项(上面正项已先命中)。
# 只放不会与正项冲突的明确否定义短语。
_CONVERT_INTENT_NEGATIVES = (
    "不满意", "不太满意", "感觉一般", "一般", "不行", "不要", "不报", "不上了",
    "先不", "再想想", "算了", "婉拒",
)

# 转化续轮(awaiting_convert 分支)的流失否定词:命中表示家长放弃/再考虑,
# 引导流失兜底并清转化态。
_DECLINE_CONVERT_KEYWORDS = (
    "不要", "不上了", "先不", "算了", "再考虑", "先看看", "不报", "不了吧",
    "不用了", "婉拒", "先等等", "先不报",
)

# 转化续轮的正向表达词:除 _is_convert_intent 外,含"继续/愿意/可以/满意"等
# 正向表达也视为 would_continue=True,走推荐套餐分支。
_CONTINUE_POSITIVE_KEYWORDS = (
    "继续", "想上", "愿意", "可以", "不错", "满意", "很好", "ok", "OK", "好",
)


def _is_convert_intent(user_input: str) -> bool:
    """判定家长输入是否为『试讲满意,想转正式课/继续上』的转化意图。

    纯关键字命中,无 LLM 调用。用于两处:
    1. TrialAgent pending_trial 分支:_is_simple_confirm 未命中后,若命中本函数
       则走 _handle_convert_intent(先建试讲课 + 引导转化),不再走 TrialInputParser
       (避免 LLM 把『试讲觉得不错,就要他俩』抽成 unrelated,报『没理解』)。
    2. TrialAgent awaiting_convert 分支:命中(或含_继续正向词)视为 would_continue=True,
       推荐正式课套餐。

    正向短语(含"不""没"的"试讲不错""讲得不错"等)优先命中 → 返回 True;
    否定短语(不满意/试讲一般/不要等)命中 → 返回 False;无正无负 → False。

    Args:
        user_input: 家长本轮输入。

    Returns:
        True: 表达『试讲满意,想转正式课』的正向转化意图
        False: 否定/无关/其他意图
    """
    s = (user_input or "").strip()
    if not s:
        return False
    # 正向优先:含"不/没"的正项短语(试讲不错/讲得不错)也是正项
    if any(k in s for k in _CONVERT_INTENT_POSITIVES):
        return True
    # 否定拦截:明确的否定短语 → 不算转化意图
    if any(k in s for k in _CONVERT_INTENT_NEGATIVES):
        return False
    return False


def _is_positive_continue(user_input: str) -> bool:
    """判定家长在转化续轮是否表达『要继续上』的正向意思(would_continue=True)。

    在 _is_convert_intent 未命中时补充覆盖"继续/想上/愿意/可以/不错/满意"等。
    纯关键字,无 LLM 调用。仅用于 awaiting_convert 分支的正向推荐分支。
    """
    s = (user_input or "").strip()
    if not s:
        return False
    return any(k in s for k in _CONTINUE_POSITIVE_KEYWORDS)


def _is_declined_convert_intent(user_input: str) -> bool:
    """判定家长在转化续轮是否表达『放弃/再考虑』流失意向。

    命中后引领流失兜底(『好的,有需要随时找我』)并清 awaiting_convert 状态。
    纯关键字,无 LLM 调用。仅用于 awaiting_convert 分支。
    """
    s = (user_input or "").strip()
    if not s:
        return False
    return any(k in s for k in _DECLINE_CONVERT_KEYWORDS)


_CHINESE_DIGITS = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _chinese_digit_to_int(token: str) -> int:
    """把中文数字 token(一~十、十二、二十 等)转为 int;已是数字串则原样 int。

    用于套餐选择(『套餐一』『第一个』『第十二个』)的序号解析。简化实现:
    只支持 0-10 个位、以及含"十"的十位组合(十、十二、二十三 等);不支持大写中文数字。
    """
    if not token:
        return 0
    if token.isdigit():
        return int(token)
    if "十" in token:
        if token == "十":
            return 10
        parts = token.split("十")
        tens = _CHINESE_DIGITS.get(parts[0], 1) if parts[0] else 1
        ones = _CHINESE_DIGITS.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return tens * 10 + ones
    return _CHINESE_DIGITS.get(token, 0)


def _create_pending_trial_lesson(processor, teacher_id: int, time_iso: str,
                                 duration_minutes: int = 30, subject: str = None):
    """真正创建 Lesson(TRIAL 类型)。

    走 LessonService(已有完整流程:teacher/student 冲突检测 + lesson_repo.add_lesson
    + 默认学员兜底)。这是 trial_processor 内部 helper,通过 processor 参数访问
    student_service / lesson_service(避免 trial_processor 顶层 import 服务层,
    保持 5 层单向依赖)。

    Args:
        processor: TrialProcessor 实例(挂 student_service / lesson_service)
        teacher_id: 老师 ID
        time_iso: ISO 格式的预约时间(由 trial_history.pending_trial.time_iso 提供)
        duration_minutes: 试讲时长(默认 30)
        subject: 学科(可选,None 时不强制)

    Returns:
        (success: bool, lesson_id_or_error: int | str)
        - success=True 时 lesson_id_or_error 是 Lesson.id
        - success=False 时 lesson_id_or_error 是错误文案
    """
    from datetime import datetime as _dt
    from services.lesson_service import LessonService

    try:
        time_dt = _dt.fromisoformat(time_iso)
    except (TypeError, ValueError):
        return False, f"无法解析预约时间: {time_iso!r}"

    # ⚠️ 关键:time_dt 来自 time_config.parse_datetime(返回带 Beijing tzinfo 的 tz-aware datetime),
    # 但 db/models.py 里 Lesson.scheduled_at 走的是 naive datetime(default=datetime.utcnow)
    # + sqlite 不存储 tzinfo。LessonRepository.check_teacher_conflict 内部
    # `if l.scheduled_at < end_time` 会在 naive vs aware 之间抛
    # `can't compare offset-naive and offset-aware datetimes`。
    # 这里统一 strip tzinfo(跟现有 _create_trial 行为一致——它也存在同型隐患,
    # 但 _create_trial 没被真触发是因为 trial_processor 走 direct trial 后等家长确认
    # → 本任务就是补这个路径)。
    if time_dt.tzinfo is not None:
        time_dt = time_dt.replace(tzinfo=None)

    # 默认学员兜底(试讲流程里没有显式选学员)
    student = processor.student_service.initialize_default_student()
    if not student or "id" not in student:
        return False, "默认学员创建失败"

    # 直接用 lesson_service.create_lesson(它内部已含 conflict check)
    # lesson_type="trial" 走 LessonTypeEnum.TRIAL
    success, lesson_id_or_err = processor.lesson_service.create_lesson(
        teacher_id=teacher_id,
        student_id=student["id"],
        scheduled_at=time_dt,
        lesson_type="trial",
        subject=subject,
        duration_minutes=duration_minutes,
        status="confirmed",  # 试讲直接确认(免费,不像正式课需要家长付费后再确认)
        price=0.0,           # 试讲免费
    )
    return success, lesson_id_or_err


class TrialInputParser:
    """试讲需求解析器

    context_manager: 可空依赖注入。为 None 时沿用旧 PromptTemplate 链(向后兼容,
    状态机的「字段提取→硬规则校验」链路原样保留);注入后把提取指令与历史交给
    上下文管理器组装大输入,字段与 JSON 约束不变。
    """


    def __init__(self, llm: BaseChatModel, context_manager=None, session_id=None,
                 session_chat_history=None):
        self.llm = llm
        self.context_manager = context_manager
        self.session_id = session_id
        self.session_chat_history = session_chat_history
        from config.time_config import time_config
        self.current_date = time_config.current_date_str()
        self.current_datetime = time_config.current_datetime_str()

        # 提取指令(不含历史/用户输入,注入路径用它作为额外上下文)
        self.extraction_instruction = (
            "你是一个 K12 家教平台的智能试讲顾问，负责帮家长预约免费试讲。\n"
            f"当前日期是{self.current_date}，当前北京时间是{self.current_datetime}。\n"
            "请提取以下信息为纯 JSON（不要任何 markdown 标记）：\n"
            "{{\n"
            '  "subject": "学科（数学/英语/物理/化学/语文/生物/历史/地理/政治/未知）",\n'
            '  "grade_level": "年级段（小学低年级/小学高年级/初中/高中/未知）",\n'
            '  "start_time": "试讲时间，YYYY-MM-DD HH:MM 格式，如果没说日期默认今天，时间未说则为未知。'
            '相对时间必须按当前日期换算成绝对时间输出(YYYY-MM-DD HH:MM)，'
            '如今天=今天日期、明天=明天日期、这个星期X/本周X=本周的星期X、下周一=下周周一、'
            '周五晚上7点=本周五 19:00。只输出换算后的绝对时间，不要输出相对描述",\n'
            '  "duration_minutes": "试讲时长分钟数，默认30",\n'
            '  "student_situation": "孩子情况（年级、学校、薄弱点等简要描述）",\n'
            '  "preferred_teacher_name": "指定老师姓名（无则未知）",\n'
            '  "preferred_style": "教学风格偏好（严格/温和/启发式/高压/未知）",\n'
            '  "confirmation": "对推荐老师的确认回复（是/好/不/不要/未知）",\n'
            '  "info_complete": "true/false，subject+grade_level+start_time 都不为未知时为 true",\n'
            '  "unrelated": "是否与试讲无关",\n'
            '  "missing_info": "缺失字段列表"\n'
            "}}\n"
            "再次强调：只输出纯 JSON，不要任何代码块标记或其他文字。"
        )

        self.prompt = PromptTemplate(
            input_variables=["history", "context", "user_input"],
            template=(
                "你是一个 K12 家教平台的智能试讲顾问，负责帮家长预约免费试讲。\n"
                f"当前日期是{self.current_date}，当前北京时间是{self.current_datetime}。\n"
                "{context}"   # 上游已确认的上下文(学科/年级/老师),为"未知"时为空串
                "已知信息：{history}\n"
                "用户输入：{user_input}\n\n"
                "请提取以下信息为纯 JSON（不要任何 markdown 标记）：\n"
                "{{\n"
                '  "subject": "学科（数学/英语/物理/化学/语文/生物/历史/地理/政治/未知）",\n'
                '  "grade_level": "年级段（小学低年级/小学高年级/初中/高中/未知）",\n'
                '  "start_time": "试讲时间，YYYY-MM-DD HH:MM 格式，如果没说日期默认今天，时间未说则为未知",\n'
                '  "duration_minutes": "试讲时长分钟数，默认30",\n'
                '  "student_situation": "孩子情况（年级、学校、薄弱点等简要描述）",\n'
                '  "preferred_teacher_name": "指定老师姓名（无则未知）",\n'
                '  "preferred_style": "教学风格偏好（严格/温和/启发式/高压/未知）",\n'
                '  "confirmation": "对推荐老师的确认回复（是/好/不/不要/未知）",\n'
                '  "info_complete": "true/false，subject+grade_level+start_time 都不为未知时为 true",\n'
                '  "unrelated": "是否与试讲无关",\n'
                '  "missing_info": "缺失字段列表"\n'
                "}}\n"
                "再次强调：只输出纯 JSON，不要任何代码块标记或其他文字。"
            ),
        )
        self.chain = self.prompt | self.llm

    def parse_stream(self, user_input: str, chat_history: InMemoryChatMessageHistory, context: str = ""):
        is_managed = getattr(chat_history, "entry_managed", False)
        if not is_managed:
            chat_history.add_message(HumanMessage(content=user_input))
        history_str = "\n".join(
            [f"用户：{m.content}" if m.type == "human" else f"顾问：{m.content}"
             for m in chat_history.messages]
        )

        if self.context_manager is not None:
            # 正常入口由conversation读取唯一历史；局部提取JSON仅作诊断，不再重复注入。
            extras = {
                "上游已确认上下文(学科/年级/老师)": context,
                "提取指令": self.extraction_instruction.format(),
            }
            if self.session_chat_history is None and not is_managed:
                extras = {"history": history_str, **extras}
            prompt = self.context_manager.build(
                task=user_input, session_id=self.session_id, extra_contexts=extras,
            )
            response_stream = self.llm.stream([{"role": "user", "content": prompt}])
        else:
            # 向后兼容:沿用旧 PromptTemplate 链,状态机字段提取链路保持不变(新增 context 透传)
            response_stream = self.chain.stream(
                {"history": history_str, "user_input": user_input, "context": context}
            )

        ai_content = ""

        for chunk in response_stream:
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            ai_content += token
            yield token
        if not is_managed:
            chat_history.add_message(AIMessage(content=ai_content))
        return ai_content

    def parse_data(self, ai_content: str) -> Dict[str, Any]:
        try:
            data = json.loads(ai_content)
        except json.JSONDecodeError:
            return {
                "subject": "未知",
                "grade_level": "未知",
                "start_time": "未知",
                "duration_minutes": "30",
                "student_situation": "",
                "preferred_teacher_name": "未知",
                "preferred_style": "未知",
                "confirmation": "未知",
                "info_complete": False,
                "unrelated": False,
                "missing_info": ["所有信息"],
            }
        # LLM 常把布尔字段输出成字符串("true"/"false"),归一化为真布尔,
        # 避免 bool("false")==True 把"无关"误判成正(死循环根因)。
        for key in ("unrelated", "info_complete"):
            if key in data:
                data[key] = _coerce_bool(data[key])
        return data


class TrialTeacherFinder:
    """试讲老师匹配"""

    def __init__(self):
        pass

    def find_recommended_teachers(
        self,
        subject: str, 
        grade_level: str,
        preferred_teacher_name: str = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """匹配老师"""
        from services.recommendation_service import RecommendationService
        from services.teacher_service import TeacherService

        teacher_service = TeacherService()
        recommendation_service = RecommendationService()

        # 1. 如果指定了老师，优先返回该老师
        if preferred_teacher_name and preferred_teacher_name != "未知":
            teacher = teacher_service.get_teacher_by_name(preferred_teacher_name)
            if teacher:
                return [{"teacher": teacher, "score": 100, "reason": "您指定的老师"}]
            # 指定老师不存在，按学科+年级继续

        # 2. 学科 + 年级筛选
        candidates = teacher_service.find_by_subject_and_grade(subject, grade_level)
        if not candidates:
            candidates = teacher_service.find_by_subject(subject)
        if not candidates:
            candidates = teacher_service.get_available_teachers()

        # 3. 用推荐算法打分
        scored = []
        for t in candidates:
            score = recommendation_service.score_teacher_for_student(
                {"subjects": t.get("subjects"), "grade_levels": t.get("grade_levels"),
                 "teaching_style": t.get("teaching_style"), "highlights": t.get("highlights"),
                 "rating": t.get("rating"), "years_experience": t.get("years_experience")},
                {"grade_level": grade_level, "preferred_style": "未知"},
                subject=subject,
            )
            if score > 0:
                scored.append({"teacher": t, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:top_k]

        for item in top:
            item["reason"] = recommendation_service._build_reason(
                item["teacher"], {"grade_level": grade_level}, subject
            )
        return top

    def check_teacher_availability(
        self, teacher_id: int, start_time: datetime, duration_minutes: int
    ) -> bool:
        """检查老师 + 学生档期"""
        from services.lesson_service import LessonService
        lesson_service = LessonService()

        end_time = start_time + timedelta(minutes=duration_minutes)
        if lesson_service.lesson_repo.check_teacher_conflict(teacher_id, start_time, end_time):
            return False
        return True


class TrialMessageBuilder:
    """试讲消息构造"""

    def __init__(self):
        self.missing_info_prompts = {
            "subject": "请问孩子需要辅导哪个学科呢？比如数学、英语、物理？",
            "grade_level": "孩子目前是哪个年级段呢？（小学/初中/高中）",
            "start_time": "请问您希望什么时候试讲？比如本周三晚上 7 点？",
            "duration_minutes": "试讲一般 30 分钟，默认就好，对吗？",
        }

    def create_recommendation_message(
        self,
        subject: str,
        grade_level: str,
        start_time: str,
        recommendations: List[Dict[str, Any]],
    ) -> str:
        """创建老师推荐消息"""
        if not recommendations:
            return f"\n顾问：抱歉，目前没有找到适合{grade_level}{subject}的老师。我们会持续更新老师库，您可以换个学科或年级看看。\n"

        lines = [f"\n顾问：根据您说的{grade_level}{subject}辅导需求，为您推荐以下老师：\n"]
        for idx, r in enumerate(recommendations, 1):
            t = r["teacher"]
            lines.append(
                f"{idx}. **{t['name']}**（{t.get('gender', '')}）\n"
                f"   - 教学经验：{t.get('years_experience', 0)} 年\n"
                f"   - 教学风格：{t.get('teaching_style', '')}\n"
                f"   - 单价：{t.get('hourly_rate', 0)} 元/小时\n"
                f"   - 特点：{'、'.join(t.get('highlights', [])[:3])}\n"
                f"   - 简介：{t.get('bio', '')[:60]}...\n"
            )
        lines.append(f"如需预约 {start_time} 的试讲，请告诉我您倾向哪位老师（回复姓名或序号即可）。\n")
        return "\n".join(lines)

    def create_trial_success_message(
        self,
        teacher: Dict[str, Any],
        scheduled_at: datetime,
        student_situation: str = "",
    ) -> str:
        """创建试讲预约成功消息"""
        time_str = scheduled_at.strftime("%Y-%m-%d %H:%M")
        return (
            f"\n顾问：已为您预约 {time_str} 的试讲课！\n"
            f"老师：**{teacher['name']}**（{teacher.get('gender', '')}）\n"
            f"教学风格：{teacher.get('teaching_style', '')}\n"
            f"单价：{teacher.get('hourly_rate', 0)} 元/小时\n\n"
            f"📌 **温馨提示**：\n"
            f"- 试讲时长约 30 分钟，免费的\n"
            f"- 如果对老师满意，可以直接在聊天中回复「转正式课」购买套餐\n"
            f"- 孩子情况已同步给老师：{student_situation or '无'}\n\n"
            f"期待孩子的学习进步！\n"
        )

    def create_missing_info_questions(self, missing_info: List[str]) -> str:
        questions = [self.missing_info_prompts.get(field, f"请补充{field}信息") for field in missing_info]
        return "\n顾问：" + " ".join(questions) + "\n"

    def create_unrelated_message(self) -> str:
        return "\n顾问：这个问题和试讲无关，您是想咨询课程、预约试讲，还是了解套餐呢？\n"

    def create_parse_error_message(self) -> str:
        return "\n顾问：抱歉没理解您的话，请重新描述一下需求，比如「我想给孩子找个数学老师」。\n"

    def create_trial_feedback_request(self, teacher_name: str) -> str:
        return (
            f"\n顾问：{teacher_name} 老师的试讲已经结束！\n"
            f"请问您对这位老师满意吗？是否希望转为正式课？\n"
            f"回复「满意」或「不满意」即可，满意的话我会推荐适合的套餐。\n"
        )

    def create_direct_trial_message(self, teacher: Dict[str, Any], time_str: str) -> str:
        """当上游已确认老师时,直接根据 chat_history 里的『老师ID: N』yield 试讲预约引导消息。

        保持二次确认环节(不直接创建 Lesson),避免误操作落库。
        学科/年级从老师字段里取(subject/grade_level 是列表,join 后展示)。
        """
        subjects = teacher.get("subjects") or []
        grades = teacher.get("grade_levels") or []
        subject_str = "、".join(subjects) if subjects else "未指定学科"
        grade_str = "、".join(grades) if grades else "未指定年级"
        return (
            f"\n顾问：✅ 已为您预约 {teacher['name']} 老师"
            f"({subject_str}, {grade_str} 段)"
            f"于 {time_str} 的试讲。"
            f"确认请回复『确认』,修改时间请直接告诉我。\n"
        )


class TrialProcessor:
    """试讲流程主处理器"""

    def __init__(self, llm: BaseChatModel, lesson_service=None, context_manager=None,
                 session_id=None, session_chat_history=None):
        self.llm = llm
        # 可空统一上下文管理器,透传给输入解析器
        self.input_parser = TrialInputParser(
            llm, context_manager=context_manager, session_id=session_id,
            session_chat_history=session_chat_history,
        )
        self.teacher_finder = TrialTeacherFinder()
        self.message_builder = TrialMessageBuilder()

        from services.lesson_service import LessonService
        from services.student_profile_service import StudentProfileService
        self.lesson_service = lesson_service or LessonService()
        self.student_service = StudentProfileService()

    def update_history_from_data(self, trial_history: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """更新试讲历史，判断信息是否齐全"""
        # 检查确认回复
        if trial_history.get("awaiting_confirmation"):
            return self._handle_confirmation(trial_history, data)

        # 更新字段
        for key in ["subject", "grade_level", "student_situation", "preferred_teacher_name", "preferred_style"]:
            if data.get(key) and data[key] != "未知":
                trial_history[key] = data[key]

        if data.get("start_time") and data["start_time"] != "未知":
            trial_history["start_time"] = data["start_time"]

        if data.get("duration_minutes") and data["duration_minutes"] != "未知":
            trial_history["duration_minutes"] = data["duration_minutes"]

        # 判断信息完整性
        required = ["subject", "grade_level", "start_time"]
        has_all = all(
            trial_history.get(f) and trial_history[f] != "未知"
            for f in required
        )
        return has_all

    def _handle_confirmation(self, trial_history: Dict[str, Any], data: Dict[str, Any]) -> bool:
        confirmation = data.get("confirmation", "").lower()
        positive = any(p in confirmation for p in ["是", "好", "可以", "行", "yes", "ok"])
        negative = any(p in confirmation for p in ["不", "不要", "no", "换"])

        if positive and not negative:
            trial_history["confirmed_teacher"] = trial_history.get("recommended_teacher")
            trial_history["awaiting_confirmation"] = False
            return True
        elif negative:
            trial_history["recommendation_declined"] = True
            trial_history["awaiting_confirmation"] = False
            return True
        return False

    async def handle_unrelated_request(self, user_input: str, unrelated_callback, state) -> AsyncGenerator[str, None]:
        yield "[REPLY][试讲顾问]和试讲无关，已交给主路由处理\n"
        if unrelated_callback:
            try:
                result = await unrelated_callback(user_input)
                if hasattr(result, '__aiter__'):
                    async for token in result:
                        yield token
                else:
                    yield result
            except Exception as e:
                yield self.message_builder.create_unrelated_message()
        else:
            yield self.message_builder.create_unrelated_message()

    async def handle_complete_trial(
        self, trial_history: Dict[str, Any], session_id: str,
        chat_history: Optional[Iterable[Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """处理试讲信息完整的情况"""

        # 拒绝推荐
        if trial_history.get("recommendation_declined"):
            yield f"[REPLY][试讲顾问]\n{self.message_builder.create_unrelated_message()}"
            trial_history.pop("recommendation_declined", None)
            trial_history.pop("recommended_teacher", None)
            return

        # 确认了推荐老师
        if trial_history.get("confirmed_teacher"):
            teacher = trial_history["confirmed_teacher"]
            reply = await self._create_trial(teacher, trial_history, session_id)
            yield f"[REPLY][试讲顾问]{reply}"
            trial_history.pop("confirmed_teacher", None)
            trial_history.pop("recommended_teacher", None)
            trial_history.pop("awaiting_confirmation", None)
            return

        # 还在等待确认
        if trial_history.get("awaiting_confirmation"):
            yield f"[REPLY][试讲顾问]\n请回复「是」或「不」明确您的选择\n"
            return

        # 老师匹配
        subject = trial_history.get("subject")
        grade_level = trial_history.get("grade_level")
        preferred_name = trial_history.get("preferred_teacher_name")
        start_time_str = trial_history.get("start_time")
        duration = int(trial_history.get("duration_minutes", "30"))

        if not start_time_str or start_time_str == "未知":
            # 完全没有时间信息 → 追问具体几点,并设追问标志,让下一轮
            # TrialAgent 入口接住(见 trial_agent._handle_time_followup)。
            trial_history["pending_time_query"] = True
            trial_history["pending_date_hint"] = ""
            yield "[REPLY][试讲顾问]请问您希望几点开始试讲？例如回复：晚上7点\n"
            return

        # 解析时间
        from config.time_config import time_config
        start_time = time_config.parse_datetime(start_time_str)
        if not start_time:
            # 时间残缺(如 "2026-09-25 未知",日期对、小时漏)→ 追问具体时间,
            # 并把已解析出的日期作为上下文拼给家长。若 date_hint 失败(空),
            # 则不加日期,只追问几点。
            date_hint = _extract_date_part(start_time_str)
            trial_history["pending_time_query"] = True
            trial_history["pending_date_hint"] = date_hint
            hint = f"已安排在 {date_hint}" if date_hint else ""
            yield f"[REPLY][试讲顾问]{hint}，请问具体几点开始试讲？例如回复：晚上7点\n"
            return

        # ✅ Bug 修复:上游(推荐 agent)已确认老师时,直接走 chat_history 里的『老师ID: N』
        # 不再调 RecommendationService.score_teacher_for_student,也避免重列 Top-3 让家长再选。
        # 优先级:① trial_history["teacher_id"] (TrialAgent._extract_confirmed_context 已回填)
        #         ② chat_history 最近一条 AI 消息里的『老师ID: N』(本任务新加的兜底)
        # 若 teacher_id 无效(老师被删/ID 错位),降级走 preferred_name 路径,
        # preferred_name 也无 → 再走『列 3 个老师』fallback,保证冷启动不崩。
        confirmed_id = trial_history.get("teacher_id")
        if not confirmed_id and chat_history is not None:
            confirmed_id = extract_confirmed_teacher_id(chat_history)
        if confirmed_id:
            from services.teacher_service import TeacherService
            teacher = TeacherService().get_teacher_by_id(confirmed_id)
            if teacher:
                time_str = start_time.strftime("%Y-%m-%d %H:%M")
                # ⚠️ 冲突预检(Bug 修复):家长给时间后【立即】检查老师该时段是否已冲突。
                # 冲突检测原本只在『确认』落库(_create_pending_trial_lesson)时才触发,
                # 造成:预约消息成功发出 → 家长确认 → 落库失败(自相矛盾)。
                # 现在发【成功消息前】先预检:冲突 → 当场让家长换时间,不承诺成功、
                # 不写 pending_trial、不切 TRIAL_CONFIRM(状态仍留在试讲流程)。
                # tz 坑:start_time 来自 time_config.parse_datetime(带 Beijing tzinfo),
                # 而 lesson_repo.check_teacher_conflict 内部与 naive 比较会抛
                # `can't compare offset-naive and offset-aware`,参照 _create_pending_trial_lesson
                # 先 strip tzinfo,保证与 lesson_repo 比较一致。
                pre_check_start = start_time
                if pre_check_start.tzinfo is not None:
                    pre_check_start = pre_check_start.replace(tzinfo=None)
                if not self.teacher_finder.check_teacher_availability(
                    teacher["id"], pre_check_start, duration
                ):
                    yield (
                        f"[REPLY][试讲顾问]\n顾问：抱歉，{teacher['name']}老师在该时段已有课程，请重新选择时间\n"
                    )
                    return
                yield f"[REPLY][试讲顾问]{self.message_builder.create_direct_trial_message(teacher, time_str)}"
                # 记录给下一轮『确认』用
                trial_history["confirmed_teacher_id"] = teacher["id"]
                trial_history["confirmed_teacher"] = teacher
                trial_history["start_time_parsed"] = start_time
                trial_history["duration_minutes_int"] = duration
                # 新增:存 pending_trial 状态,让 TrialAgent 入口
                # 见到家长下轮『确认/是/好』时真正创建 Lesson(TRIAL),
                # 不再依赖 LLM 分类器(避免『确认』被误判成 recommendation)。
                # subject/grade_level 优先从 trial_history 取(已回填),
                # 缺则从 teacher.subjects/grade_levels 兜底(teacher 必有这些字段)。
                import time as _time
                subjects = teacher.get("subjects") or []
                grades = teacher.get("grade_levels") or []
                pending_subject = (
                    trial_history.get("subject")
                    or (subjects[0] if subjects else None)
                )
                pending_grade = (
                    trial_history.get("grade_level")
                    or (grades[0] if grades else None)
                )
                trial_history["pending_trial"] = {
                    "teacher_id": teacher["id"],
                    "teacher_name": teacher.get("name", ""),
                    "subject": pending_subject,
                    "grade_level": pending_grade,
                    "time_str": time_str,
                    "time_iso": start_time.isoformat(),
                    "duration_minutes": duration,
                    "created_at": _time.time(),
                }
                return
            # teacher_id 无效 → 静默降级到下面的 preferred_name 路径

        # 匹配老师
        # 上游(推荐 agent)已确认老师时:直接用 preferred_name 查 teacher_service,不再
        # 走『学科+年级』全量打分(避免浪费 LLM 调用,也避免列 3 个候选让家长再选——这是用户报的 bug)。
        teacher = None
        if preferred_name and preferred_name != "未知":
            from services.teacher_service import TeacherService
            teacher = TeacherService().get_teacher_by_name(preferred_name)
        if teacher is not None:
            recommendations = [{"teacher": teacher, "score": 100, "reason": "您指定的老师"}]
            trial_history["recommended_teacher"] = teacher
            trial_history["recommendations"] = recommendations
            trial_history["start_time_parsed"] = start_time
            trial_history["duration_minutes_int"] = duration
            reply = await self._create_trial(teacher, trial_history, session_id)
            yield f"[REPLY][试讲顾问]{reply}"
            return

        recommendations = self.teacher_finder.find_recommended_teachers(
            subject=subject, grade_level=grade_level,
            preferred_teacher_name=preferred_name, top_k=3,
        )

        if not recommendations:
            yield f"[REPLY][试讲顾问]{self.message_builder.create_recommendation_message(subject, grade_level, start_time_str, [])}"
            return

        # 推荐第一位 + 展示候选
        trial_history["recommended_teacher"] = recommendations[0]["teacher"]
        trial_history["recommendations"] = recommendations
        trial_history["start_time_parsed"] = start_time
        trial_history["duration_minutes_int"] = duration
        trial_history["awaiting_confirmation"] = True

        msg = self.message_builder.create_recommendation_message(
            subject, grade_level, start_time_str, recommendations
        )
        yield f"[REPLY][试讲顾问]{msg}"

    async def _create_trial(
        self,
        teacher: Dict[str, Any],
        trial_history: Dict[str, Any],
        session_id: str,
    ) -> str:
        """创建试讲 Lesson"""
        start_time = trial_history["start_time_parsed"]
        duration = trial_history["duration_minutes_int"]

        # 冲突检测
        if not self.teacher_finder.check_teacher_availability(teacher["id"], start_time, duration):
            return f"\n顾问：抱歉，{teacher['name']}老师在该时段已有课程，请重新选择时间\n"

        # 获取默认学生
        student = self.student_service.initialize_default_student()

        # 创建 Lesson (type=trial)
        success, result = self.lesson_service.create_lesson(
            teacher_id=teacher["id"],
            student_id=student["id"],
            scheduled_at=start_time,
            lesson_type="trial",
            subject=trial_history.get("subject"),
            duration_minutes=duration,
            status="confirmed",
            price=0.0,  # 试讲免费
        )

        if success:
            trial_history["created_lesson_id"] = result
            return self.message_builder.create_trial_success_message(
                teacher, start_time, trial_history.get("student_situation", "")
            )
        else:
            return f"\n顾问：试讲预约失败：{result}\n"

    async def _handle_availability_query(
        self, user_input: str, trial_history: Dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """试讲流程内档期查询:返回老师最近可预约时段并引导选择。

        规则检测(_is_availability_query 命中)后进入,不新增 LLM 调用。
        取当前已确认/已选老师;家长输入里提到老师名时按名解析。
        遍历未来 3 天、每天 9:00-21:00 每小时整点,用 lesson_repo.check_teacher_conflict
        找可用时段(参考 user_behavior_agent 既有遍历模式)。
        """
        from services.teacher_service import TeacherService
        from config.time_config import time_config

        teacher = None
        teacher_id = trial_history.get("confirmed_teacher_id") or trial_history.get("teacher_id")
        if teacher_id is None:
            ct = trial_history.get("confirmed_teacher")
            if isinstance(ct, dict) and ct.get("id"):
                teacher_id = ct["id"]
        if teacher_id:
            teacher = TeacherService().get_teacher_by_id(teacher_id)
        # 兜底:家长输入里提到的『XX 老师』
        if teacher is None:
            m = re.search(r"([一-龥]{2,4})\s*老师", user_input or "")
            if m:
                teacher = TeacherService().get_teacher_by_name(m.group(1))

        teacher_name = teacher.get("name", "老师") if teacher else "老师"
        if teacher is None:
            yield "[REPLY][试讲顾问]\n顾问：请问您想为哪位老师查询档期？回复老师姓名即可。\n"
            return

        lesson_repo = self.lesson_service.lesson_repo
        today = time_config.today()
        if today.tzinfo is not None:
            today = today.replace(tzinfo=None)
        weekday_labels = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        slots: List[str] = []
        for d in range(3):
            day = today + timedelta(days=d)
            for hour in range(9, 22):
                slot = day.replace(hour=hour, minute=0, second=0, microsecond=0)
                end = slot + timedelta(minutes=30)
                if lesson_repo.check_teacher_conflict(teacher["id"], slot, end):
                    continue
                slots.append(f"{weekday_labels[day.weekday()]} {hour}:00")
                if len(slots) >= 5:
                    break
            if len(slots) >= 5:
                break

        if not slots:
            yield (
                f"[REPLY][试讲顾问]\n顾问：抱歉，{teacher_name}老师最近三天暂时没有空闲时段，"
                f"请稍后重试或换个时间。\n"
            )
            return
        shown = "、".join(slots)
        yield (
            f"[REPLY][试讲顾问]\n顾问：{teacher_name}老师最近可预约:{shown}…"
            f"请告诉我要哪个时间。\n"
        )

    async def handle_incomplete_info(
        self, data: Dict[str, Any], trial_history: Dict[str, Any], user_input: str = None,
    ) -> AsyncGenerator[str, None]:
        """处理信息不完整"""
        # 档期查询规则检测(无新 LLM)
        if user_input and _is_availability_query(user_input):
            async for token in self._handle_availability_query(user_input, trial_history):
                yield token
            return
        missing = []
        for field in ["subject", "grade_level", "start_time"]:
            if not trial_history.get(field) or trial_history[field] == "未知":
                missing.append(field)

        reply = self.message_builder.create_missing_info_questions(missing)
        yield f"[THOUGHT][试讲顾问]信息不完整，缺少：{', '.join(missing)}"
        yield f"[REPLY][试讲顾问]{reply}"