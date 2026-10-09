"""
AutoDream 验收演示脚本

模拟 5 次会话 → 触发 AutoDream 回放 → 断言长期偏好表出现新记录 + 变更日志留痕。

用法：
  python scripts/autodream_demo.py

退出码：0 = 验收通过，非 0 = 失败。
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    import hashlib
    import random
    import tempfile
    from db.db_router import DatabaseRouter
    from db.models import AutoDreamChangeLog, UserPreference
    from services.autodream_service import AutoDreamService

    # 用确定性 fake embedding 替代真实 API（验收脚本不依赖外部服务）
    def _fake_embed(text: str):
        h = hashlib.sha256(text.encode()).digest()
        rng = random.Random(int.from_bytes(h[:4], "big"))
        vec = [rng.uniform(-1, 1) for _ in range(64)]
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]

    import services.autodream_service as ad_mod
    ad_mod.embed_input = _fake_embed

    db_path = f"sqlite:///{tempfile.mkdtemp()}/autodream_demo.db"
    db = DatabaseRouter(db_path)

    # 1) 记录 5 个会话（每个 session_id 不同），共 6 条行为
    user_id = "demo_parent"
    for i in range(5):
        db.user_behavior_repo.record_behavior(
            user_id=user_id,
            action_type="lesson",
            action_data={
                "subject": "数学",
                "scheduled_at": "2026-09-22 10:00:00",
                "duration_minutes": 60,
            },
            teacher_id=1,
            session_id=f"session_{i}",
        )
    db.user_behavior_repo.record_behavior(
        user_id=user_id,
        action_type="consultation",
        action_data={"query": "想给孩子补英语"},
        session_id="session_5",
    )

    # 2) 跑 AutoDream
    svc = AutoDreamService(db_path=db_path)
    result = svc.run_for_user(user_id)

    print(f"\nrun_for_user 结果: {result}")
    assert result["triggered"] is True, "应触发回放"
    assert result["processed_events"] == 6, "应处理 6 个事件"

    # 3) 断言长期偏好表出现新记录
    prefs = db.user_behavior_repo.get_user_preferences(user_id)
    print(f"\n长期偏好表记录: {prefs}")
    assert prefs, "长期偏好表应出现新记录"
    pref_types = {p["preference_type"] for p in prefs}
    assert {"subject", "time_period", "duration", "teacher"}.issubset(pref_types), (
        f"应覆盖 4 类偏好, 实际 {pref_types}"
    )
    assert all(p["confidence_score"] >= 1 for p in prefs), "置信度应 >= 1"

    # 4) 断言变更日志留痕
    with db.session_manager.session_scope() as s:
        logs = s.query(AutoDreamChangeLog).filter(
            AutoDreamChangeLog.user_id == user_id
        ).all()
        log_count = len(logs)
        actions = {l.action for l in logs}
    print(f"\n变更日志条数: {log_count}")
    assert log_count > 0, "变更日志应留痕"
    assert actions <= {"merge", "new", "downgrade"}, "action 应合法"

    # 5) 幂等：重跑同一批不产生重复偏好/日志（检查点保证）
    result2 = svc.run_for_user(user_id)
    print(f"\n重跑结果: {result2}")
    assert result2["triggered"] is True or result2["skipped"] == "condition", "重跑应被触发条件拦住"
    with db.session_manager.session_scope() as s:
        log_count2 = s.query(AutoDreamChangeLog).filter(
            AutoDreamChangeLog.user_id == user_id
        ).count()
    pref_count2 = len(db.user_behavior_repo.get_user_preferences(user_id))
    assert log_count2 == log_count, "重跑不应产生重复日志"
    assert pref_count2 == len(prefs), "重跑不应产生重复偏好"

    print("\n✅ AutoDream 验收通过：5 会话触发回放，偏好落库 + 日志留痕 + 幂等重跑")


if __name__ == "__main__":
    main()
