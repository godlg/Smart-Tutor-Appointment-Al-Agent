"""
端到端演示脚本（家长视角）

不依赖 LLM，直接通过 HTTP 调 FastAPI 接口走通：
  1. 看老师库
  2. 看套餐
  3. 建一个学员档案
  4. 预约一次试讲
  5. 生成一份周固定系列课
  6. 模拟一次课后反馈
  7. 看行为分析数据
  8. 写课后进度记录

用法:
  .\\.venv\\Scripts\\Activate.ps1
  python scripts\\demo_e2e.py [base_url]
  默认 base_url = http://127.0.0.1:8001

依赖: 标准库 urllib + json（无需 requests）

实现要点:
- 脚本会自动把项目根目录（与 app.py 同级）加入 sys.path，
  无需硬编码绝对路径，复制到其他机器也能直接跑。
- Step 5（生成系列课）直接调 LessonService.create_lesson_series，
  需要 faiss / langchain 等重依赖。若不在 .venv 中运行，该步会被跳过。
"""
import os, sys, json, urllib.request, urllib.error, traceback
from pathlib import Path
from datetime import datetime, timedelta

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"
# 从脚本自身位置向上找到 app.py 同级目录，加入 sys.path。
# 这样脚本可被复制/移动到任意位置，仍能 import 项目模块。
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _SCRIPT_DIR.parent  # scripts/ 的父目录 = 项目根
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))


def _venv_ready():
    """演示脚本 Step 5 调 LessonService 需要 faiss / langchain。
    检测到缺失则跳过 Step 5，HTTP 段不受影响。"""
    try:
        import faiss  # noqa
        import langchain  # noqa
        return True
    except ImportError as e:
        print(f"[warn] 缺依赖 {e.name}。Step 5 会被跳过，其余步骤不受影响。")
        print("        启用 venv 后重跑: .\\.venv\\Scripts\\Activate.ps1 ; "
              f"python {_SCRIPT_DIR.relative_to(_PROJ_ROOT)}\\demo_e2e.py")
        return False


def http(method, path, body=None, expect=(200, 201)):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            payload = r.read().decode()
            try:
                return r.status, json.loads(payload) if payload else {}
            except json.JSONDecodeError:
                return r.status, {"raw": payload}
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        return e.code, {"error": body}
    except Exception as e:
        return -1, {"error": f"{type(e).__name__}: {e}"}


def step(title):
    print(f"\n=== {title} ===")


def show(label, status, payload, ok_status, max_chars=400):
    mark = "OK" if status in ok_status else "FAIL"
    snippet = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + f"... (truncated, total {len(snippet)} chars)"
    print(f"  [{mark}] {label}  HTTP {status}\n        {snippet}")


def main():
    print(f"BASE = {BASE}")
    venv_ok = _venv_ready()
    if not venv_ok:
        print("继续跑 HTTP 部分（Step 1/2/3/4/6/7/8），Step 5 会被跳过。\n")

    step("1. 看老师库（按学科筛选 数学）")
    status, body = http("GET", "/api/teachers/")
    show("GET /api/teachers/", status, body, (200,))
    teacher_id = (body[0].get("id") if isinstance(body, list) and body
                  else body.get("teachers", [{}])[0].get("id") if isinstance(body, dict) else None)

    step("2. 看套餐列表")
    status, body = http("GET", "/api/packages/")
    show("GET /api/packages/", status, body, (200,))
    packages = body if isinstance(body, list) else body.get("packages", [])
    package_id = packages[0].get("id") if packages else None

    step("3. 建一个学员档案（初二 数学薄弱）")
    student_body = {
        "name": "演示-小明",
        "grade_level": "初中",
        "weak_subjects": ["数学"],
        "learning_goals": ["函数基础打牢"],
        "preferred_style": "温和",
        "school": "演示中学",
    }
    status, body = http("POST", "/api/students/", student_body)
    show("POST /api/students/", status, body, (200, 201))
    student_id = body.get("id") or body.get("student_id")

    step("4. 预约一次试讲")
    if teacher_id:
        trial_body = {
            "teacher_id": teacher_id,
            "student_id": student_id,
            "subject": "数学",
            "scheduled_at": (datetime.now() + timedelta(days=3)).replace(hour=20, minute=0, second=0, microsecond=0).isoformat(),
            "duration_minutes": 30,
        }
        status, body = http("POST", "/api/trial/book", trial_body)
        show("POST /api/trial/book", status, body, (200, 201))
        trial_lesson_id = body.get("lesson_id") or body.get("id")
    else:
        print("  跳过（无可用老师）")
        trial_lesson_id = None

    step("5. 生成一份周固定系列课（每周4 周 × 1次/周）")
    if not venv_ok:
        print("  跳过（需在 .venv 中运行）")
    elif teacher_id and student_id and package_id:
        # 注: series_router 只有 GET，没有 POST。创建系列课走 LessonService。
        # 演示数据库可能已堆积多条试讲/课，导致硬编码时段被占满。
        # 改进: 先从老师 availability 中挑一个数学老师 + 一个可用时段，
        # 再让 ScheduleGenerator 排课；连续失败时再换老师/时段。
        try:
            from services.lesson_service import LessonService
            from services.teacher_service import TeacherService
            ls = LessonService()
            ts = TeacherService()
            # ScheduleGenerator 约定 start_date 必须是周一（weekly_pattern 的
            # day_of_week 是其周内偏移），否则排出的日期会整体错位到 start_date 那一天。
            start = (datetime.now() + timedelta(days=1)).date()
            start = start + timedelta(days=(7 - start.weekday()) % 7)

            # 收集候选：(teacher_id, weekday_index, time_slot)
            day_map = {"monday": 0, "tuesday": 1, "wednesday": 2,
                       "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
            candidates = []
            for t in ts.get_all_teachers():
                if "数学" not in (t.get("subjects") or []):
                    continue
                for slot in (t.get("availability") or []):
                    d = slot.get("day")
                    if d not in day_map:
                        continue
                    for s in slot.get("slots", []):
                        # s = "19:00-20:00", 提取起始时间
                        start_time = s.split("-")[0]
                        candidates.append((t["id"], day_map[d], start_time))

            # 按"明天+0天"最早匹配优先：直接遍历即可
            success_info = None
            used = None
            for tid, dow, tstr in candidates:
                ok, info = ls.create_lesson_series(
                    teacher_id=tid, student_id=student_id, package_id=package_id,
                    start_date=start, total_weeks=4,
                    weekly_pattern=[{"day_of_week": dow, "time": tstr}],
                )
                lessons_created = (info or {}).get("lessons_created", 0) if isinstance(info, dict) else 0
                if ok and lessons_created > 0:
                    success_info = info
                    used = (tid, dow, tstr)
                    break

            if success_info:
                tid, dow, tstr = used
                print(f"  [OK] LessonService.create_lesson_series "
                      f"(teacher_id={tid}, day_of_week={dow}, time={tstr}) -> {success_info}")
            else:
                print(f"  [FAIL] 所有 {len(candidates)} 个候选 (老师,时段) 都无法排课")
        except Exception as e:
            print(f"  [FAIL] 调 LessonService 失败: {type(e).__name__}: {e}")
            traceback.print_exc()
    else:
        print("  跳过（缺少前置数据）")

    step("6. 提交试讲反馈（5 星 + 继续上课）")
    if trial_lesson_id:
        fb_body = {
            "lesson_id": trial_lesson_id,  # 该接口的 schema 强制要求 body 带 lesson_id
            "rating": 5,
            "would_continue": True,
            "feedback_text": "孩子说老师讲得很清楚",
            "recommended_subjects": ["物理"],
        }
        status, body = http("POST", f"/api/trial/{trial_lesson_id}/feedback", fb_body)
        show(f"POST /api/trial/{trial_lesson_id}/feedback", status, body, (200, 201))
    else:
        print("  跳过（无试讲 lesson_id）")

    step("7. 看行为分析数据（家长视角）")
    status, body = http("GET", "/api/user-behavior/analysis")
    show("GET /api/user-behavior/analysis", status, body, (200,))

    step("8. 课后进度记录")
    if teacher_id and student_id:
        prog_body = {
            "lesson_id": 1,  # 演示用，真实应该用真实 lesson_id
            "student_id": student_id,
            "teacher_id": teacher_id,
            "content_covered": "一元一次方程复习",
            "mastery_score": 7.5,
            "homework": "课本 P12 第 1-5 题",
            "weak_points": ["移项符号"],
        }
        status, body = http("POST", "/api/progress/", prog_body)
        show("POST /api/progress/", status, body, (200, 201))

    print("\nDONE. 如果有 FAIL，请检查：")
    print("  - 服务是否启动:", BASE)
    print("  - .env 是否配置好 LLM（不影响本脚本，仅对话流需要）")


if __name__ == "__main__":
    main()