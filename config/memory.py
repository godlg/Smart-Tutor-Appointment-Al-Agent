"""记忆可靠性默认配置。不读取.env,可通过进程环境覆盖。"""
import os


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


CONTEXT_MAX_TOKENS = _positive_int("CONTEXT_MAX_TOKENS", 8192)
CONTEXT_OUTPUT_TOKENS = _positive_int("CONTEXT_OUTPUT_TOKENS", 1024)
CONTEXT_KEEP_RECENT_ROUNDS = _positive_int("CONTEXT_KEEP_RECENT_ROUNDS", 4)
CONTEXT_ROLLUP_ROUNDS = _positive_int("CONTEXT_ROLLUP_ROUNDS", 4)
CONTEXT_SUMMARY_MAX_CHARS = _positive_int("CONTEXT_SUMMARY_MAX_CHARS", 600)
SESSION_CACHE_TTL_SECONDS = _positive_int("SESSION_CACHE_TTL_SECONDS", 1800)
SESSION_CACHE_MAX_ENTRIES = _positive_int("SESSION_CACHE_MAX_ENTRIES", 128)
