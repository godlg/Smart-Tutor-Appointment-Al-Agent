"""Phase 1 记忆可靠性 —— 唯一历史 / 写入 / 并发 / 持久化恢复（Slice 2+3）

覆盖:
- 主控入口统一记录 human + ai;[THOUGHT] 路由调试输出按角色过滤后落库
- 取消/异常:human 保留,ai 只保存已产生的可用内容(partial),不伪造完整问答
- 同 session 输入按 async 锁串行;不同 session 可并行
- 缓存 drop 清理 history(drop 后持久记录仍在,可再恢复)
- 持久化:顺序消息按 (session_id,seq) 幂等;重建服务/独立进程后恢复;重复 restore 不重复

不触网:mock get_task_agent 返回假 agent。DB 用临时文件,不碰生产库。
"""
import asyncio
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run(coro):
    return asyncio.run(coro)


async def _collect(stream):
    out = []
    async for tok in stream:
        out.append(tok)
    return "".join(out)


# ======================================================================
# 1. [THOUGHT] 过滤
# ======================================================================

class TestThoughtFiltering:

    def test_strips_thought_lines_keeps_reply(self):
        from services.conversation_service import strip_thought_output
        raw = (
            "[THOUGHT][归类机器人] 这是一个咨询问题。\n"
            "[REPLY][咨询机器人]您好，关于英语学习...\n"
            "[THOUGHT][咨询机器人] 内部调试\n"
        )
        cleaned = strip_thought_output(raw)
        assert "[THOUGHT]" not in cleaned
        assert "您好，关于英语学习" in cleaned

    def test_keeps_all_when_no_thought(self):
        from services.conversation_service import strip_thought_output
        raw = "[REPLY][试讲顾问]已为您预约。"
        assert strip_thought_output(raw) == raw


# ======================================================================
# 2. 主控入口统一写入
# ======================================================================

@pytest.fixture
def chat_env(tmp_path, monkeypatch):
    """搭好 chat_handler 的会话服务与假 agent,返回 (module, identity, service)。"""
    from api import chat_handler
    from services.conversation_service import ConversationService
    from config.identity import Identity

    svc = ConversationService(db_path=f"sqlite:///{tmp_path / 'chat.db'}")
    monkeypatch.setattr(chat_handler, "_conversation_service", svc, raising=False)

    identity = Identity(user_id="anon_test", session_id="sess_chat")

    class _FakeAgent:
        def __init__(self, tokens, fail_after=None):
            self.tokens = tokens
            self.fail_after = fail_after

        async def classify_task_stream(self, task):
            for i, tok in enumerate(self.tokens):
                if self.fail_after is not None and i >= self.fail_after:
                    raise RuntimeError("boom")
                yield tok

    monkeypatch.setattr(chat_handler, "get_task_agent",
                        lambda session_id=None, identity=None: _FakeAgent(
                            ["[THOUGHT][归类机器人] 路由中\n", "[REPLY][咨询机器人]你好呀"]))
    yield chat_handler, identity, svc
    svc.close()


class TestUnifiedWrites:

    def test_records_human_and_ai(self, chat_env):
        mod, ident, svc = chat_env
        out = _run(_collect(mod.ProcessUserInput_stream(
            "孩子英语怎么学", session_id="sess_chat", identity=ident)))
        assert "你好呀" in out
        msgs = svc.conversation_repo.get_messages("sess_chat")
        assert [m["role"] for m in msgs] == ["human", "ai"]
        assert msgs[0]["content"] == "孩子英语怎么学"
        # [THOUGHT] 调试输出不落库
        assert "[THOUGHT]" not in msgs[1]["content"]
        assert "你好呀" in msgs[1]["content"]

    def test_partial_on_exception_human_kept(self, chat_env, monkeypatch):
        mod, ident, svc = chat_env
        monkeypatch.setattr(mod, "get_task_agent",
                            lambda session_id=None, identity=None: _make_failing_agent())
        with pytest.raises(RuntimeError):
            _run(_collect(mod.ProcessUserInput_stream(
                "预约试讲", session_id="sess_chat", identity=ident)))
        msgs = svc.conversation_repo.get_messages("sess_chat")
        # human 一定保留;ai 只有部分内容,且标记 partial(不伪造完整回答)
        roles = [m["role"] for m in msgs]
        assert "human" in roles
        ai = [m for m in msgs if m["role"] == "ai"]
        if ai:
            assert ai[-1]["status"] == "partial"

    def test_same_session_serialized(self, chat_env, monkeypatch):
        mod, ident, svc = chat_env
        events = []

        def make_slow_agent():
            class _Slow:
                async def classify_task_stream(self, task):
                    events.append("start")
                    await asyncio.sleep(0.05)
                    events.append("end")
                    yield "[REPLY]ok"
            return _Slow()

        monkeypatch.setattr(mod, "get_task_agent",
                            lambda session_id=None, identity=None: make_slow_agent())

        async def two_calls():
            a = _collect(mod.ProcessUserInput_stream("q1", session_id="sess_chat", identity=ident))
            b = _collect(mod.ProcessUserInput_stream("q2", session_id="sess_chat", identity=ident))
            await asyncio.gather(a, b)

        _run(two_calls())
        # 串行:一定 start,end,start,end,不会交叠
        assert events == ["start", "end", "start", "end"]

    def test_different_sessions_parallel(self, chat_env, monkeypatch):
        mod, ident, svc = chat_env
        from config.identity import Identity
        events = []

        def make_slow_agent(tag):
            class _Slow:
                async def classify_task_stream(self, task):
                    events.append(f"{tag}-start")
                    await asyncio.sleep(0.05)
                    events.append(f"{tag}-end")
                    yield "[REPLY]ok"
            return _Slow()

        monkeypatch.setattr(mod, "get_task_agent",
                            lambda session_id=None, identity=None: make_slow_agent(session_id))

        ida = Identity(user_id="anon_test", session_id="sess_a")
        idb = Identity(user_id="anon_test", session_id="sess_b")

        async def two_calls():
            a = _collect(mod.ProcessUserInput_stream("q1", session_id="sess_a", identity=ida))
            b = _collect(mod.ProcessUserInput_stream("q2", session_id="sess_b", identity=idb))
            await asyncio.gather(a, b)

        _run(two_calls())
        # 并行:两个 start 在前,说明未被同一把锁串行(不同 session 各用各的锁)
        assert events.index("sess_a-start") < events.index("sess_a-end")
        assert events.index("sess_b-start") < events.index("sess_b-end")
        assert events[:2] == ["sess_a-start", "sess_b-start"] or \
               events[:2] == ["sess_b-start", "sess_a-start"]


def _make_failing_agent():
    class _Fail:
        async def classify_task_stream(self, task):
            yield "[REPLY]开始回答"
            raise RuntimeError("boom")
    return _Fail()


# ======================================================================
# 3. 持久化 + 恢复
# ======================================================================

class TestPersistenceRecovery:

    def test_ordered_messages_by_seq(self, tmp_path):
        from services.conversation_service import ConversationService
        from config.identity import Identity
        svc = ConversationService(db_path=f"sqlite:///{tmp_path / 'c.db'}")
        ident = Identity(user_id="u1", session_id="s1")
        svc.record_human(ident, "第一句")
        svc.record_ai(ident, "第一次回答")
        svc.record_human(ident, "重复的话")
        svc.record_human(ident, "重复的话")  # 用户确实重复说了,不能被去重误删
        msgs = svc.conversation_repo.get_messages("s1")
        assert [m["seq"] for m in msgs] == [1, 2, 3, 4]
        assert [m["content"] for m in msgs][2:] == ["重复的话", "重复的话"]
        svc.close()

    def test_restore_after_rebuild_no_duplicates(self, tmp_path):
        from services.conversation_service import ConversationService
        from config.identity import Identity
        db = f"sqlite:///{tmp_path / 'c.db'}"
        svc1 = ConversationService(db_path=db)
        ident = Identity(user_id="u1", session_id="s1")
        svc1.record_human(ident, "hi")
        svc1.record_ai(ident, "hello")
        svc1.close()

        svc2 = ConversationService(db_path=db)
        try:
            hist = svc2.restore_history(Identity(user_id="u1", session_id="s1"))
            assert len(hist.messages) == 2
            hist2 = svc2.restore_history(Identity(user_id="u1", session_id="s1"))
            assert len(hist2.messages) == 2  # 重复恢复不重复
            assert svc2.message_count(Identity(user_id="u1", session_id="s1")) == 2
        finally:
            svc2.close()

    def test_restore_in_fresh_process(self, tmp_path):
        """独立子进程恢复:证明持久化不依赖进程内内存(不仅同一个实例)。"""
        from services.conversation_service import ConversationService
        from config.identity import Identity
        db_file = tmp_path / "proc.db"
        db = f"sqlite:///{db_file}"
        svc = ConversationService(db_path=db)
        ident = Identity(user_id="u1", session_id="s1")
        svc.record_human(ident, "跨进程第一句")
        svc.record_ai(ident, "跨进程回答")
        svc.close()

        script = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, r"{PROJECT_ROOT}")
            from services.conversation_service import ConversationService
            from config.identity import Identity
            svc = ConversationService(db_path="sqlite:///{db_file.as_posix()}")
            h = svc.restore_history(Identity(user_id="u1", session_id="s1"))
            print(len(h.messages))
            print(h.messages[0].content)
            svc.close()
        """)
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, encoding="utf-8",
        )
        assert proc.returncode == 0, proc.stderr
        lines = proc.stdout.strip().splitlines()
        assert lines[0] == "2"
        assert lines[1] == "跨进程第一句"

    def test_drop_history_keeps_persisted(self, tmp_path):
        from services.conversation_service import ConversationService
        from config.identity import Identity
        svc = ConversationService(db_path=f"sqlite:///{tmp_path / 'c.db'}")
        ident = Identity(user_id="u1", session_id="s1")
        svc.record_human(ident, "hi")
        svc.drop_history("s1")
        # 热层已清,但持久记录仍在,可再恢复
        h = svc.restore_history(ident)
        assert len(h.messages) == 1
        svc.close()
