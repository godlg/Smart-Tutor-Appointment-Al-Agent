"""调课请求服务(C2a:调课需老师确认)

职责与边界:
- 只负责「请求 → 老师确认 → 应用」的状态机与守卫。
- 真正的改期(冲突检测 + 级联)全部复用 agents/lesson/reschedule_handler.py 的 RescheduleHandler,
  本服务绝不重写一套调课逻辑。
- requested_by 不做鉴权(与 api/lesson.py 其它 C 端端点一致,目前都无鉴权)。
- 返回值直接带 datetime,由 FastAPI 序列化为 ISO 字符串。
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from agents.lesson.reschedule_handler import RescheduleHandler, RescheduleMode
from db.base.session_manager import DEFAULT_DB_PATH, SessionManager
from db.models import Lesson, LessonChangeRequest
from db.repositories.lesson_repository import LessonRepository

logger = logging.getLogger(__name__)

VALID_MODES = (RescheduleMode.ONLY_THIS, RescheduleMode.THIS_AND_FOLLOWING)
CLOSED_LESSON_STATUSES = ("completed", "cancelled")


class LessonChangeError(Exception):
    """调课请求业务错误(带 HTTP 语义状态码)。"""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _request_to_dict(req: LessonChangeRequest) -> Dict[str, Any]:
    return {
        "id": req.id,
        "lesson_id": req.lesson_id,
        "series_id": req.series_id,
        "requested_by": req.requested_by,
        "mode": req.mode,
        "old_scheduled_at": req.old_scheduled_at,
        "new_scheduled_at": req.new_scheduled_at,
        "status": req.status,
        "reason": req.reason,
        "decided_by_teacher_id": req.decided_by_teacher_id,
        "created_at": req.created_at,
        "decided_at": req.decided_at,
    }


class LessonChangeService:
    """调课请求的创建 / 查询 / 老师确认或拒绝。"""

    def __init__(self, db_path: Optional[str] = None):
        self.session_manager = SessionManager(db_path or DEFAULT_DB_PATH)
        self.lesson_repo = LessonRepository(self.session_manager)

    def close(self):
        self.session_manager.close()

    # ------------------- 创建 -------------------

    def create_request(
        self,
        lesson_id: int,
        new_time: datetime,
        mode: str = RescheduleMode.ONLY_THIS,
        requested_by: str = "parent",
    ) -> Dict[str, Any]:
        if mode not in VALID_MODES:
            raise LessonChangeError(f"不支持的调课模式: {mode}", 422)

        lesson = self.lesson_repo.get_lesson_by_id(lesson_id)
        if lesson is None:
            raise LessonChangeError("课时不存在", 404)
        if lesson["status"] in CLOSED_LESSON_STATUSES:
            raise LessonChangeError(f"课时状态为 {lesson['status']}，不可调课", 409)

        # 幂等:已有 pending 请求则直接返回它(不新建、不报错)
        existing = self._pending_for_lesson(lesson_id)
        if existing is not None:
            return {"created": False, "request": existing}

        # 落库前先校验新时间可排(只校验目标课时;级联完整校验在 approve 时由 RescheduleHandler 做)
        self._assert_slot_free(lesson, new_time)

        with self.session_manager.session_scope() as session:
            # 二次检查:并发请求可能在上面的校验窗口内已插入 pending
            row = session.query(LessonChangeRequest).filter(
                LessonChangeRequest.lesson_id == lesson_id,
                LessonChangeRequest.status == "pending",
            ).first()
            if row is not None:
                return {"created": False, "request": _request_to_dict(row)}

            row = LessonChangeRequest(
                lesson_id=lesson_id,
                series_id=lesson.get("series_id"),
                requested_by=requested_by,
                mode=mode,
                old_scheduled_at=lesson["scheduled_at"],
                new_scheduled_at=new_time,
                status="pending",
            )
            session.add(row)
            session.flush()
            request = _request_to_dict(row)

        logger.info(
            "调课请求已创建",
            extra={"event": "lesson_change_requested", "lesson_id": lesson_id,
                   "request_id": request["id"], "requested_by": requested_by},
        )
        return {"created": True, "request": request}

    def _pending_for_lesson(self, lesson_id: int) -> Optional[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            row = session.query(LessonChangeRequest).filter(
                LessonChangeRequest.lesson_id == lesson_id,
                LessonChangeRequest.status == "pending",
            ).first()
            return _request_to_dict(row) if row else None

    def _assert_slot_free(self, lesson: Dict[str, Any], new_time: datetime) -> None:
        end_time = new_time + timedelta(minutes=lesson["duration_minutes"])
        lesson_id = lesson["id"]
        if self.lesson_repo.check_teacher_conflict(
            lesson["teacher_id"], new_time, end_time, exclude_lesson_id=lesson_id
        ):
            raise LessonChangeError("新时间与老师已有课程冲突", 409)
        if self.lesson_repo.check_student_conflict(
            lesson["student_id"], new_time, end_time, exclude_lesson_id=lesson_id
        ):
            raise LessonChangeError("新时间与学生已有课程冲突", 409)

    # ------------------- 查询 -------------------

    def list_for_lesson(self, lesson_id: int) -> List[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            rows = session.query(LessonChangeRequest).filter(
                LessonChangeRequest.lesson_id == lesson_id,
            ).order_by(LessonChangeRequest.created_at.desc(), LessonChangeRequest.id.desc()).all()
            return [_request_to_dict(r) for r in rows]

    def list_for_teacher(self, teacher_id: int, status: Optional[str] = "pending") -> List[Dict[str, Any]]:
        """只返回该老师名下课时的请求(join lessons.teacher_id)。"""
        with self.session_manager.session_scope() as session:
            query = session.query(LessonChangeRequest, Lesson).join(
                Lesson, LessonChangeRequest.lesson_id == Lesson.id,
            ).filter(Lesson.teacher_id == teacher_id)
            if status is not None:
                query = query.filter(LessonChangeRequest.status == status)
            rows = query.order_by(
                LessonChangeRequest.created_at.desc(), LessonChangeRequest.id.desc(),
            ).all()
            return [
                {
                    **_request_to_dict(req),
                    "lesson": {
                        "id": lesson.id,
                        "scheduled_at": lesson.scheduled_at,
                        "subject": lesson.subject,
                        "student_id": lesson.student_id,
                        "duration_minutes": lesson.duration_minutes,
                    },
                }
                for req, lesson in rows
            ]

    # ------------------- 老师决策 -------------------

    def approve(self, request_id: int, teacher_id: int) -> Dict[str, Any]:
        req = self._load_owned(request_id, teacher_id)
        if req["status"] != "pending":
            raise LessonChangeError(f"请求状态为 {req['status']}，仅待确认请求可操作", 409)

        # 改期只走 RescheduleHandler(冲突检测 + 级联全部在那里)
        ok, message = RescheduleHandler().reschedule(
            lesson_id=req["lesson_id"],
            new_time=req["new_scheduled_at"],
            mode=req["mode"],
        )
        if not ok:
            logger.warning(
                "调课确认失败(请求保持 pending)",
                extra={"event": "lesson_change_approve_failed", "teacher_id": teacher_id,
                       "request_id": request_id, "lesson_id": req["lesson_id"], "reason": message},
            )
            raise LessonChangeError(message, 409)

        decided = self._decide(request_id, teacher_id, "approved", reason=None)
        logger.info(
            "调课请求已确认",
            extra={"event": "lesson_change_approved", "teacher_id": teacher_id,
                   "request_id": request_id, "lesson_id": req["lesson_id"]},
        )
        return decided

    def reject(self, request_id: int, teacher_id: int, reason: Optional[str] = None) -> Dict[str, Any]:
        req = self._load_owned(request_id, teacher_id)
        if req["status"] != "pending":
            raise LessonChangeError(f"请求状态为 {req['status']}，仅待确认请求可操作", 409)

        decided = self._decide(request_id, teacher_id, "rejected", reason=reason)
        logger.info(
            "调课请求已拒绝",
            extra={"event": "lesson_change_rejected", "teacher_id": teacher_id,
                   "request_id": request_id, "lesson_id": req["lesson_id"]},
        )
        return decided

    def _load_owned(self, request_id: int, teacher_id: int) -> Dict[str, Any]:
        """取请求并校验归属;不存在或课时不属于该老师一律 404。"""
        with self.session_manager.session_scope() as session:
            row = session.query(LessonChangeRequest, Lesson).join(
                Lesson, LessonChangeRequest.lesson_id == Lesson.id,
            ).filter(LessonChangeRequest.id == request_id).first()
            if row is None or row[1].teacher_id != teacher_id:
                raise LessonChangeError("调课请求不存在", 404)
            return _request_to_dict(row[0])

    def _decide(self, request_id: int, teacher_id: int, status: str,
                reason: Optional[str]) -> Dict[str, Any]:
        with self.session_manager.session_scope() as session:
            row = session.query(LessonChangeRequest).filter(
                LessonChangeRequest.id == request_id,
            ).first()
            if row is None:
                raise LessonChangeError("调课请求不存在", 404)
            if row.status != "pending":
                raise LessonChangeError(f"请求状态为 {row.status}，仅待确认请求可操作", 409)
            row.status = status
            row.reason = reason
            row.decided_by_teacher_id = teacher_id
            row.decided_at = datetime.now()
            return _request_to_dict(row)
