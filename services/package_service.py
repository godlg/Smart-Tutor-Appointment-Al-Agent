"""
套餐服务层
"""

import logging
from typing import List, Dict, Any, Optional
from db.db_router import DatabaseRouter

logger = logging.getLogger(__name__)


class PackageService:
    """套餐业务服务"""

    def __init__(self):
        self.db = DatabaseRouter()

    @property
    def package_repo(self):
        return self.db.packages

    def add_package(self, **kwargs) -> int:
        return self.package_repo.add_package(**kwargs)

    def get_package(self, package_id: int) -> Optional[Dict[str, Any]]:
        return self.package_repo.get_package_by_id(package_id)

    def get_all_packages(
        self, subject: str = None, grade_level: str = None,
        teacher_id: int = None,
    ) -> List[Dict[str, Any]]:
        return self.package_repo.get_all_packages(
            only_active=True,
            subject=subject, grade_level=grade_level, teacher_id=teacher_id,
        )

    def update_package(self, package_id: int, **kwargs) -> bool:
        return self.package_repo.update_package(package_id, **kwargs)

    def delete_package(self, package_id: int) -> bool:
        return self.package_repo.delete_package(package_id)

    # ------------------- 种子数据 -------------------

    DEFAULT_PACKAGES = [
        {
            "name": "高一数学 12 周提分套餐",
            "teacher_id": 1,  # 张明
            "subject": "数学",
            "grade_level": "高中",
            "total_lessons": 24,  # 12 周 × 2 次/周
            "duration_minutes": 60,
            "price_total": 7200.0,
            "price_per_lesson": 300.0,
            "weekly_schedule": [
                {"day_of_week": 0, "start_time": "19:00"},  # 周一
                {"day_of_week": 2, "start_time": "19:00"},  # 周三
            ],
            "validity_days": 120,
            "description": "针对高一数学，函数+几何双模块，配套作业批改",
        },
        {
            "name": "高三物理冲刺 8 周套餐",
            "teacher_id": 3,  # 王强
            "subject": "物理",
            "grade_level": "高中",
            "total_lessons": 16,  # 8 周 × 2 次/周
            "duration_minutes": 90,
            "price_total": 6400.0,
            "price_per_lesson": 400.0,
            "weekly_schedule": [
                {"day_of_week": 4, "start_time": "19:00"},
                {"day_of_week": 5, "start_time": "15:00"},
            ],
            "validity_days": 90,
            "description": "力学+电磁学冲刺，针对高考物理压轴题",
        },
        {
            "name": "初中英语兴趣启蒙套餐",
            "teacher_id": 2,  # 李静
            "subject": "英语",
            "grade_level": "初中",
            "total_lessons": 24,
            "duration_minutes": 60,
            "price_total": 5760.0,
            "price_per_lesson": 240.0,
            "weekly_schedule": [
                {"day_of_week": 1, "start_time": "19:00"},
                {"day_of_week": 3, "start_time": "19:00"},
            ],
            "validity_days": 120,
            "description": "口语+语法双线并进，提升孩子开口说英语的信心",
        },
        {
            "name": "初中数学 10 周提分套餐",
            "teacher_id": 6,  # 周敏（数学，覆盖小学高年级、初中）
            "subject": "数学",
            "grade_level": "初中",
            "total_lessons": 20,  # 10 周 × 2 次/周
            "duration_minutes": 60,
            "price_total": 5000.0,
            "price_per_lesson": 250.0,
            "weekly_schedule": [
                {"day_of_week": 2, "start_time": "19:00"},  # 周三
                {"day_of_week": 5, "start_time": "19:00"},  # 周六
            ],
            "validity_days": 100,
            "description": "初二数学函数+几何双模块，配套作业批改与错题复盘",
        },
    ]

    def initialize_default_packages(self) -> bool:
        """幂等补插默认套餐。

        库为非空时不再整体跳过,而是按 name 检查缺失项并补插:
        - 全新库:逐条插入所有 DEFAULT_PACKAGES。
        - 已有部分套餐:只补 DEFAULT_PACKAGES 中缺失的名字,不重复插入已有项。
        这样已初始化的库(如旧版仅 3 个套餐)重启后也能补上新增的「初中数学」。
        """
        try:
            existing = self.get_all_packages()
            existing_names = {p["name"] for p in existing}

            if existing_names == {data["name"] for data in self.DEFAULT_PACKAGES}:
                logger.info(f"已存在全部 {len(existing)} 个默认套餐，跳过初始化")
                return True

            missing = [
                data for data in self.DEFAULT_PACKAGES
                if data["name"] not in existing_names
            ]
            if not missing:
                return True

            logger.info(f"开始补充默认套餐: {len(missing)} 个（已有 {len(existing)} 个）")
            for data in missing:
                try:
                    self.package_repo.add_package(**data)
                    logger.debug(f"添加套餐: {data['name']}")
                except Exception as e:
                    logger.error(f"添加套餐失败: {e}")
                    return False

            final_count = len(self.get_all_packages())
            logger.info(f"套餐初始化完成，共 {final_count} 个")
            return True

        except Exception as e:
            logger.error(f"套餐初始化失败: {e}")
            return False