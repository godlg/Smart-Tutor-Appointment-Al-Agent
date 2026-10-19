"""控制台鉴权配置(机构后台分面 A)

本模块是**叶子依赖**:config 包不依赖 agents / services / db,任何层都可安全导入。

边界(勿夸大):这里只是机构后台账号体系的常量,与匿名家长身份
(`config/identity.py`)完全独立——控制台登录不代表任何家长身份,
反之匿名家长 cookie 也不能用来访问控制台端点。
"""
from __future__ import annotations

import os

# 控制台会话 cookie 名(与家长侧的 tutor_uid 区分开,避免互相污染)
CONSOLE_COOKIE_NAME = "tutor_console"

# 会话有效期(小时),默认 12 小时;到期由 resolve 主动删除该行
CONSOLE_SESSION_TTL_HOURS = int(os.getenv("CONSOLE_SESSION_TTL_HOURS", "12"))
