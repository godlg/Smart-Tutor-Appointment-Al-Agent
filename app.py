"""
FastAPI 应用程序（家教智能预约系统）
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期: 启动 -> yield -> 关闭

    启动: 知识库 / 老师 / 学生 / 套餐种子数据, 启动推荐调度器
    关闭: 停掉调度器线程, 让线程自然退出 (避免 uvicorn 退出会杀后台线程的脏状态)
    """
    # --- 启动 ---
    await initialize_system()

    # 把 scheduler 引用挂到 app.state, 关闭时能拿到
    from services.recommendation_service import RecommendationService
    app.state.recommendation_service = RecommendationService()

    # AutoDream 后台任务（行为增量回放 + 长期偏好更新）
    from services.autodream_service import AutoDreamService
    app.state.autodream_service = AutoDreamService()
    if app.state.autodream_service.start_scheduler():
        logger.info("✅ AutoDream 调度服务启动成功")
    else:
        logger.warning("⚠️ AutoDream 调度服务启动失败")

    yield

    # --- 关闭 ---
    rec_svc = app.state.recommendation_service
    if rec_svc and getattr(rec_svc, "is_running", False):
        logger.info("正在停止推荐调度服务...")
        try:
            if hasattr(rec_svc, "stop_scheduler"):
                rec_svc.stop_scheduler()
            elif hasattr(rec_svc, "scheduler_thread"):
                rec_svc.scheduler_thread = None
            rec_svc.is_running = False
        except Exception as e:
            logger.warning(f"关闭推荐调度服务时出错: {e}")
        logger.info("推荐调度服务已停止")

    ad_svc = getattr(app.state, "autodream_service", None)
    if ad_svc and getattr(ad_svc, "is_running", False):
        logger.info("正在停止 AutoDream 调度服务...")
        try:
            if hasattr(ad_svc, "stop_scheduler"):
                ad_svc.stop_scheduler()
            elif hasattr(ad_svc, "scheduler_thread"):
                ad_svc.scheduler_thread = None
            ad_svc.is_running = False
        except Exception as e:
            logger.warning(f"关闭 AutoDream 调度服务时出错: {e}")
        logger.info("AutoDream 调度服务已停止")


async def initialize_system():
    """系统启动时自动初始化"""
    try:
        logger.info("🚀 正在初始化家教智能预约系统...")

        # 知识库初始化
        logger.info("📚 初始化知识库服务...")
        from services.knowledge_service import KnowledgeService
        knowledge_service = KnowledgeService()
        await knowledge_service.initialize()

        # 老师 + 学生 + 套餐默认数据
        logger.info("👨‍🏫 初始化老师数据...")
        from services.teacher_service import TeacherService
        TeacherService().initialize_default_teachers()

        logger.info("🎒 初始化默认学生...")
        from services.student_profile_service import StudentProfileService
        StudentProfileService().initialize_default_student()

        logger.info("📦 初始化套餐数据...")
        from services.package_service import PackageService
        PackageService().initialize_default_packages()

        # 推荐服务
        logger.info("🎯 启动推荐调度服务...")
        from services.recommendation_service import RecommendationService
        recommendation_service = RecommendationService()
        if recommendation_service.start_scheduler():
            logger.info("✅ 推荐调度服务启动成功")
        else:
            logger.warning("⚠️ 推荐调度服务启动失败")

        logger.info("✅ 家教智能预约系统初始化完成！")

    except Exception as e:
        logger.error(f"❌ 系统初始化失败: {e}")
        raise


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例"""

    app = FastAPI(
        title="家教智能预约系统",
        description="面向家长的 K12 家教智能预约平台 - 试讲预约 / 套餐管理 / 周固定排课 / 调课退课 / 学习进度跟踪",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 补4: Prometheus 指标中间件 (记录每个 HTTP 请求的总数 + 延迟)
    from observability import PrometheusMiddleware
    app.add_middleware(PrometheusMiddleware)

    # 补4(Logs): 统一 JSON 结构化日志 + request_id 中间件(X-Request-ID 请求关联)
    from observability import RequestIdMiddleware, setup_logging
    setup_logging()  # env LOG_LEVEL 控制级别;JSON 行输出;幂等
    app.add_middleware(RequestIdMiddleware)

    from api import api_routers
    from api.core.exceptions import general_exception_handler

    # 通用兜底 handler: 任何未捕获异常都打 traceback + 返回 500
    app.add_exception_handler(Exception, general_exception_handler)

    for r in api_routers:
        app.include_router(r)

    # 补4: Prometheus /metrics 端点 (Prometheus 抓取用)
    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    from web import router as web_router
    app.include_router(web_router)
    app.mount("/static", StaticFiles(directory="web/static"), name="static")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)