"""
推荐服务（升级版：支持家教场景的智能推荐）

旧逻辑保留（按摩技师推荐），新增家教推荐算法
"""

import logging
from typing import List, Dict, Any, Optional
from db.db_router import DatabaseRouter

logger = logging.getLogger(__name__)


class RecommendationService:
    """推荐调度服务（支持双模式）"""

    def __init__(self):
        self._behavior_agent = None
        self.is_running = False
        self.scheduler_thread = None

    @property
    def behavior_agent(self):
        if self._behavior_agent is None:
            from services.user_behavior_service import UserBehaviorService
            self._behavior_agent = UserBehaviorService()
        return self._behavior_agent

    # ------------------- 家教推荐算法 -------------------

    def score_teacher_for_student(
        self,
        teacher: Dict[str, Any],
        student: Dict[str, Any],
        subject: str = None,
    ) -> float:
        """
        为学生打分老师

        评分维度:
        - 学科硬匹配:仅当调用方显式传 subject 时生效,0 分过滤;
                      若 subject 由 student.weak_subjects 推导出来,则改为软匹配(不命中仅小幅减分),
                      否则一旦家长填了"数学薄弱"就永远推不出英语老师,推荐退化为空。
        - 年级匹配 +30
        - 教学风格匹配 +20
        - 评分加成 +0~50
        - 价格适配 ×0.5~1.0(MVP 暂未接 parent.budget_max,留 TODO)
        """
        score = 0.0

        # 学科:区分"显式指定"和"从学生弱项推导"
        weak_subs = student.get("weak_subjects") or []
        if subject:
            target_subject = subject
            subject_source = "explicit"
        elif weak_subs:
            target_subject = weak_subs[0]
            subject_source = "weak_subject"
        else:
            target_subject = None
            subject_source = None

        teacher_subjects = teacher.get("subjects") or []
        if target_subject:
            if subject_source == "explicit" and target_subject not in teacher_subjects:
                # 调用方明确要这个学科 → 老师不会这门 → 0 分
                return 0.0
            elif target_subject in teacher_subjects:
                score += 20  # 软匹配:命中加 20
            # 弱项学科软匹配未命中不直接归零,让其他维度(年级/风格/评分)继续起作用

        # 年级匹配
        if student.get("grade_level") in (teacher.get("grade_levels") or []):
            score += 30

        # 教学风格匹配
        style_keywords = {
            "启发式": ["耐心", "思路清晰", "会建模", "善于总结", "思维引导"],
            "温和":   ["耐心", "亲和", "激发兴趣", "会讲故事"],
            "严格":   ["考点清晰", "系统化", "高强度", "押题准"],
            "高压":   ["冲刺型", "押题准", "高强度"],
        }

        preferred = student.get("preferred_style", "")
        for tag in (teacher.get("highlights") or []):
            if tag in style_keywords.get(preferred, []):
                score += 20
                break

        # 评分加成(0~50)
        # 4.0→0, 5.0→25, 6.0→50;rating 缺省按 4.5 算(中性)
        rating = teacher.get("rating")
        if rating is not None:
            score += (float(rating) - 4.0) * 25

        # 价格适配 TODO: 需要 parent.budget_max, 当前 student 没接 parent 信息

        return score

    def recommend_teachers(
        self,
        student: Dict[str, Any],
        subject: str = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        推荐 top_k 个老师

        Returns:
            [{"teacher": ..., "score": ..., "reason": ...}, ...]
        """
        db = DatabaseRouter()
        all_teachers = db.teachers.get_all_teachers()

        scored = []
        for t in all_teachers:
            score = self.score_teacher_for_student(t, student, subject)
            if score > 0:
                scored.append({"teacher": t, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:top_k]

        # 生成推荐理由
        for item in top:
            item["reason"] = self._build_reason(item["teacher"], student, subject)

        return top

    def _build_reason(self, teacher: Dict, student: Dict, subject: str = None) -> str:
        parts = []
        if subject and subject in (teacher.get("subjects") or []):
            parts.append(f"擅长{subject}")

        if student.get("grade_level") in (teacher.get("grade_levels") or []):
            parts.append(f"覆盖{student['grade_level']}")

        if teacher.get("highlights"):
            parts.append("标签：" + "、".join(teacher["highlights"][:3]))

        if teacher.get("years_experience", 0) >= 5:
            parts.append(f"{teacher['years_experience']}年经验")

        return "，".join(parts) if parts else "综合推荐"

    # ------------------- 定时任务:落库 UserRecommendation,供家长中心拉取 -------------------

    # 每个学生每天最多落库 N 条 tutor_recommendation(避免定时器狂写)
    DAILY_RECOMMENDATION_LIMIT_PER_STUDENT = 1

    def _has_recent_recommendation(self, repo, user_id: str, since_hours: int = 20) -> bool:
        """近 since_hours 小时内是否已写过 tutor_recommendation(限频)。"""
        from datetime import datetime, timedelta
        try:
            pending = repo.get_pending_recommendations(user_id)
        except Exception:
            return False
        cutoff = datetime.utcnow() - timedelta(hours=since_hours)
        for r in pending:
            t = r.get("created_at")
            if not t:
                continue
            if isinstance(t, str):
                try:
                    t = datetime.fromisoformat(t)
                except Exception:
                    continue
            if t >= cutoff and r.get("recommendation_type") == "tutor_recommendation":
                return True
        return False

    def generate_recommendations_job(self):
        """
        定时推荐任务:遍历所有学生,生成 Top-1 老师推荐,**写入 user_recommendations 表**。

        返回值保留原有结构(为了不破坏老调用方),但真正生效的是 DB 中的新行,
        家长中心(/parent-center)或 /api/user-behavior/pending 端点可以直接拉取。
        """
        try:
            logger.info("开始执行定时推荐生成任务...")
            db = DatabaseRouter()
            students = db.students.get_all_students()
            persisted = 0
            skipped = 0

            for student in students:
                user_id = f"student_{student['id']}"  # 行为表统一用 student_<id>
                if self._has_recent_recommendation(db.user_behavior, user_id):
                    skipped += 1
                    continue

                recs = self.recommend_teachers(student, top_k=1)
                if not recs:
                    continue
                r = recs[0]
                content = f"为{student['name']}推荐老师:{r['teacher']['name']}({r['reason']})"
                try:
                    db.user_behavior.create_recommendation(
                        user_id=user_id,
                        recommendation_type="tutor_recommendation",
                        content=content,
                        teacher_id=r["teacher"]["id"],
                    )
                    persisted += 1
                except Exception as e:
                    logger.warning(f"落库推荐失败 student_id={student['id']}: {e}")

            logger.info(
                f"推荐任务完成: 落库 {persisted} 条, 因限频跳过 {skipped} 条"
            )
            return {"persisted": persisted, "skipped": skipped, "total_students": len(students)}
        except Exception as e:
            logger.error(f"定时推荐生成任务失败: {e}")
            return None

    def start_scheduler(self) -> bool:
        if self.is_running:
            return False
        try:
            import schedule
            import threading
            import time

            schedule.every().day.at("09:00").do(self.generate_recommendations_job)
            schedule.every().day.at("14:00").do(self.generate_recommendations_job)
            schedule.every().day.at("19:00").do(self.generate_recommendations_job)

            self.is_running = True

            def run_scheduler():
                logger.info("推荐调度器已启动")
                while self.is_running:
                    schedule.run_pending()
                    time.sleep(60)
                logger.info("推荐调度器已停止")

            self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
            self.scheduler_thread.start()
            return True

        except Exception as e:
            logger.error(f"启动推荐调度器失败: {str(e)}")
            return False

    def stop_scheduler(self) -> bool:
        try:
            self.is_running = False
            import schedule
            schedule.clear()
            return True
        except Exception as e:
            logger.error(f"停止推荐调度器失败: {str(e)}")
            return False

    def run_immediate_check(self):
        return self.generate_recommendations_job()

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running,
            "thread_alive": self.scheduler_thread.is_alive() if self.scheduler_thread else False,
            "total_jobs": 3,
        }