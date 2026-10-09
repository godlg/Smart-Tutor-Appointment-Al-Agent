"""
时间配置模块

统一管理系统时间，确保所有组件使用相同的时间基准
"""

import re
from datetime import datetime, timezone, timedelta
from typing import Optional


# 中文星期 → Python weekday()(周一=0 ... 周日=6)
_WEEKDAY_MAP = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}

# 时段词 → 未给具体小时时的默认小时
_PERIOD_DEFAULTS = {
    "早上": 8,
    "清晨": 8,
    "上午": 9,
    "中午": 12,
    "下午": 14,
    "晚上": 19,
    "夜间": 22,
}
# 下午/晚上的小时需 +12 偏移
_PM_PERIODS = ("下午", "晚上", "夜间")


class TimeConfig:
    """时间配置类"""
    
    # 设置北京时区 (UTC+8)
    BEIJING_TZ = timezone(timedelta(hours=8))
    
    @classmethod
    def now(cls) -> datetime:
        """获取当前北京时间"""
        # 先获取UTC时间，然后转换为北京时间
        utc_now = datetime.now(timezone.utc)
        beijing_now = utc_now.astimezone(cls.BEIJING_TZ)
        return beijing_now
    
    @classmethod
    def today(cls) -> datetime:
        """获取今天的日期（北京时间）"""
        return cls.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    @classmethod
    def current_date_str(cls, format_str: str = "%Y年%m月%d日") -> str:
        """获取当前日期字符串（北京时间）"""
        return cls.now().strftime(format_str)
    
    @classmethod
    def current_datetime_str(cls, format_str: str = "%Y-%m-%d %H:%M") -> str:
        """获取当前日期时间字符串（北京时间）"""
        return cls.now().strftime(format_str)
    
    @classmethod
    def parse_datetime(cls, date_str: str, format_str: str = "%Y-%m-%d %H:%M") -> Optional[datetime]:
        """解析日期时间字符串为北京时间的datetime对象"""
        try:
            dt = datetime.strptime(date_str, format_str)
            # 如果没有时区信息，假设是北京时间
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=cls.BEIJING_TZ)
            return dt
        except ValueError:
            pass  # 落入相对时间兜底解析
        # 兜底:相对时间(这个星期X/周五/明天晚上等),按当前日期换算成绝对时间
        dt = cls._parse_relative_datetime(date_str, cls.today())
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=cls.BEIJING_TZ)
        return dt

    @classmethod
    def _parse_relative_datetime(cls, date_str: str, base_date: datetime) -> Optional[datetime]:
        """把相对时间表达解析成以 base_date 为基准的绝对 datetime(naive 或与基准同 tz)。

        支持输入:
          - 今天/明天/后天 + 时段
          - 这个星期/这周/本周 + 星期X/周X + 上午|下午|晚上|早上
          - 星期X/周X/礼拜X + 时段(取最近的未来;过去的星期几顺延到下周)
          - N点 / N点M分(可带时段词,如「晚上 7 点」「下午 3 点」)
          - 绝对日期前缀 + 时段(如「2026-09-25 晚上7点」:日期带出来、
            只补时间。试讲时间追问后拼接 date_hint 用)
        时段默认小时:晚上→19、下午→14、上午→9、早上→8。

        base_date 可注入(测试友好,避免依赖真实当前日期);实际调用由
        parse_datetime 传 cls.today()。
        """
        if not isinstance(date_str, str) or not date_str.strip():
            return None
        s = date_str.strip()

        # ---- 日期部分 ----
        target_date = None
        # 绝对日期前缀(YYYY-MM-DD):如 "2026-09-25 晚上7点",取绝对日期再补时间。
        # 放 今天/明天 之前,因为这种输入通常不携带相对日词。
        m_abs = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
        if m_abs:
            try:
                target_date = datetime(
                    int(m_abs.group(1)), int(m_abs.group(2)), int(m_abs.group(3))
                )
            except ValueError:
                target_date = None
        if target_date is None:
            if re.search(r"今天|今日|今儿", s):
                target_date = base_date
            elif re.search(r"明天|明日|明儿", s):
                target_date = base_date + timedelta(days=1)
            elif re.search(r"后天", s):
                target_date = base_date + timedelta(days=2)

        if target_date is None:
            m = re.search(
                r"(这个星期|这星期|本星期|本周|这周|下星期|下个星期|下周|下礼拜|礼拜|星期|周|周末)?\s*([一二三四五六日天])",
                s,
            )
            if m and m.group(2):
                prefix = m.group(1)
                wd = _WEEKDAY_MAP[m.group(2)]
                this_monday = base_date - timedelta(days=base_date.weekday())
                if prefix in ("这个星期", "这星期", "本星期", "本周", "这周"):
                    target_date = this_monday + timedelta(days=wd)
                elif prefix in ("下星期", "下个星期", "下周", "下礼拜"):
                    target_date = this_monday + timedelta(days=7 + wd)
                else:
                    # 裸星期X/周X:取最近的未来(含今天)
                    delta = (wd - base_date.weekday()) % 7
                    target_date = base_date + timedelta(days=delta)

        if target_date is None:
            return None

        # ---- 时间部分 ----
        hour = None
        minute = 0
        hour_explicit = False
        mh = re.search(r"(\d{1,2})\s*点\s*(\d{1,2})\s*分", s)
        if mh:
            hour, minute = int(mh.group(1)), int(mh.group(2))
            hour_explicit = True
        else:
            mh2 = re.search(r"(\d{1,2})\s*点", s)
            if mh2:
                hour = int(mh2.group(1))
                hour_explicit = True

        period_name = None
        for p in _PERIOD_DEFAULTS:
            if p in s:
                period_name = p
                break
        if hour is None and period_name is not None:
            hour = _PERIOD_DEFAULTS[period_name]
            hour_explicit = False

        if hour is None:
            return None

        # 下午/晚上且小时是显式小数字 → +12(如「晚上 7 点」=19 点)
        if hour_explicit and hour < 13 and period_name in _PM_PERIODS:
            hour += 12
        if not (0 <= hour <= 23):
            return None

        return target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)

    @classmethod
    def format_datetime(cls, dt: datetime, format_str: str = "%Y-%m-%d %H:%M") -> str:
        """格式化datetime对象为字符串"""
        if dt.tzinfo is None:
            # 如果没有时区信息，假设是北京时间
            dt = dt.replace(tzinfo=cls.BEIJING_TZ)
        elif dt.tzinfo != cls.BEIJING_TZ:
            # 如果是其他时区，转换为北京时间
            dt = dt.astimezone(cls.BEIJING_TZ)
        
        return dt.strftime(format_str)
    
    @classmethod
    def get_business_hours(cls) -> tuple:
        """获取营业时间范围"""
        return (12, 22)  # 12:00 - 22:00
    
    @classmethod
    def is_business_time(cls, dt: Optional[datetime] = None) -> bool:
        """检查是否在营业时间内"""
        if dt is None:
            dt = cls.now()
        
        hour = dt.hour
        start_hour, end_hour = cls.get_business_hours()
        return start_hour <= hour < end_hour


# 创建全局实例
time_config = TimeConfig()
