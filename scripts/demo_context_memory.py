"""分层记忆增强演示脚本

流程:
1. 造几条长期记忆(预约事件 / 偏好 / 摘要,带 embedding + importance)
2. MemoryContext 带 task 时向量召回 Top-5 相关记忆注入
3. ContextManager 窗口占用 ≥60% 时触发滚动摘要,并持久化回 long_term_memories

运行:
    python scripts/demo_context_memory.py
"""
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.context.contexts.conversation import ConversationContext
from agents.context.contexts.memory import MemoryContext
from agents.context.manager import ContextManager
from services.long_term_memory_service import LongTermMemoryService
from langchain_core.messages import HumanMessage, AIMessage


def _fake_embed(text: str):
    """确定性词袋语义向量(不触网):相同/相近汉字越多 cos_sim 越高。

    sha256 会让不同文本向量完全无关(cos_sim≈0),召回永远为空;
    词袋按汉字位置累加 + 归一化,让 query 与相关记忆有可比的相似度。
    """
    vec = [0.0] * 16
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            vec[ord(ch) % 16] += 1.0
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec] if norm else vec


def main():
    # 内存库长期记忆服务 + 全程用语义 fake embedding(不触网)
    from services import text_embedding

    text_embedding.embed_input = _fake_embed  # retrieve_top_k 内部延迟 from-import 会读到
    svc = LongTermMemoryService(db_path="sqlite:///:memory:")

    # 造几条长期记忆(不同类型 / 重要度 / 时间)
    svc.add_memory(
        "default_user", "preference",
        "家长明确希望每周三晚 7 点给孩子补英语,时长 60 分钟。",
        embedding=_fake_embed("周三晚英语补课 60 分钟"), importance=0.9,
    )
    svc.add_memory(
        "default_user", "appointment",
        "已预约本周六上午 10:00 张老师英语试讲。",
        embedding=_fake_embed("周六 10 点英语试讲 张老师"), importance=0.7,
    )
    svc.add_memory(
        "default_user", "preference",
        "孩子初二,数学是薄弱学科,喜欢启发式教学风格的老师。",
        embedding=_fake_embed("初二数学薄弱 启发式老师"), importance=0.6,
    )
    svc.add_memory(
        "default_user", "summary",
        "早期对话摘要: 用户咨询过家教收费与退费政策。",
        embedding=_fake_embed("收费 退费 政策"), importance=0.4,
    )

    # 1) 向量召回 Top-5 注入
    mem_ctx = MemoryContext(
        analyzer=None, user_id="default_user", memory_service=svc,
    )
    text = mem_ctx.build(_Manager(task="想给孩子安排英语补课"))
    print("===== MemoryContext 召回注入 =====")
    print(text)
    assert "【相关记忆】" in text, "应召回相关记忆"
    assert "每周三晚" in text, "最相关记忆应被召回"

    # 2) 60% 窗口滚动摘要:长会话 + 小 max_tokens
    conv = ConversationContext()
    session_id = "demo-sess"
    history = conv.get_history(session_id)
    for i in range(30):
        history.add_message(HumanMessage(content=f"用户问题{i}"))
        history.add_message(AIMessage(content=f"顾问回答{i}"))

    def persister(user_id, summary_text, session_id=None):
        svc.save_rollup_summary(
            user_id=user_id, summary_text=summary_text, session_id=session_id,
        )

    mgr = ContextManager(
        [],
        rollup_enabled=True, rollup_window_ratio=0.6,
        rollup_rounds=4, rollup_persister=persister,
    )
    mgr.session_id = session_id
    mgr.contexts = [conv, mem_ctx]
    out = mgr.build(task="继续安排补课", max_tokens=100)
    print("\n===== 滚动摘要后输出(截断) =====")
    print(out[:200] + "...")
    assert mgr.last_rollup_triggered, "窗口占用≥60%应触发滚动摘要"
    assert "[更早的对话已滚动摘要存档]" in out, "最旧部分应折叠为占位"

    # 3) 摘要已持久化回长期记忆表
    summaries = svc.get_user_memories("default_user", memory_type="summary")
    print(f"\n===== 长期记忆表 summary 记录 {len(summaries)} 条 =====")
    for s in summaries:
        print(f"  [{s['session_id']}] {s['content'][:40]}...")
    assert summaries, "滚动摘要应写入长期记忆表"

    print("\n演示通过: Top-5 召回注入 + 60% 滚动摘要 + 持久化 全部 OK")


class _Manager:
    def __init__(self, task):
        self.task = task
        self.session_id = "demo-sess"


if __name__ == "__main__":
    main()
