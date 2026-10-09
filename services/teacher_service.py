"""
老师服务层

职责：
1. 老师 CRUD 业务封装
2. 老师筛选 + 推荐
3. 默认老师数据初始化
"""

import logging
from typing import List, Dict, Any, Optional
from db.db_router import DatabaseRouter

logger = logging.getLogger(__name__)


class TeacherService:
    """老师业务服务"""

    def __init__(self):
        self.db = DatabaseRouter()

    @property
    def teacher_repo(self):
        return self.db.teachers

    # ------------------- 初始化种子数据 -------------------

    DEFAULT_TEACHERS = [
        {
            "name": "张明",
            "gender": "男",
            "subjects": ["数学", "物理"],
            "grade_levels": ["初中", "高中"],
            "teaching_style": "严格",
            "education_bg": "北师大数学系硕士",
            "years_experience": 8,
            "hourly_rate": 350.0,
            "rating": 4.9,
            "highlights": ["押题准", "提分快", "逻辑清晰"],
            "bio": "8 年高中数学一线教学经验，擅长函数与几何模块，2024 届所带学生高考数学平均提分 28 分。",
            "availability": [
                {"day": "monday", "slots": ["19:00-20:00", "20:00-21:00"]},
                {"day": "wednesday", "slots": ["19:00-20:00", "20:00-21:00"]},
                {"day": "saturday", "slots": ["09:00-10:00", "10:00-11:00", "14:00-15:00"]},
            ],
        },
        {
            "name": "李静",
            "gender": "女",
            "subjects": ["英语"],
            "grade_levels": ["小学高年级", "初中", "高中"],
            "teaching_style": "温和",
            "education_bg": "北京外国语大学英语系本科",
            "years_experience": 6,
            "hourly_rate": 300.0,
            "rating": 4.8,
            "highlights": ["耐心", "发音标准", "擅长口语"],
            "bio": "6 年英语教学，注重培养兴趣，擅长用场景对话让孩子敢开口说英语。",
            "availability": [
                {"day": "tuesday", "slots": ["19:00-20:00", "20:00-21:00"]},
                {"day": "thursday", "slots": ["19:00-20:00", "20:00-21:00"]},
                {"day": "sunday", "slots": ["10:00-11:00", "14:00-15:00"]},
            ],
        },
        {
            "name": "王强",
            "gender": "男",
            "subjects": ["物理"],
            "grade_levels": ["高中"],
            "teaching_style": "启发式",
            "education_bg": "清华大学物理系博士",
            "years_experience": 5,
            "hourly_rate": 400.0,
            "rating": 5.0,
            "highlights": ["思路清晰", "会建模", "耐心引导"],
            "bio": "清华物理博士，善于把抽象物理问题转化为生活场景，擅长力学和电磁学。",
            "availability": [
                {"day": "monday", "slots": ["20:00-21:00"]},
                {"day": "friday", "slots": ["19:00-20:00", "20:00-21:00"]},
                {"day": "saturday", "slots": ["15:00-16:00", "16:00-17:00"]},
            ],
        },
        {
            "name": "陈雪",
            "gender": "女",
            "subjects": ["语文", "英语"],
            "grade_levels": ["小学低年级", "小学高年级"],
            "teaching_style": "温和",
            "education_bg": "华东师范大学中文系硕士",
            "years_experience": 4,
            "hourly_rate": 260.0,
            "rating": 4.7,
            "highlights": ["亲和", "会讲故事", "激发兴趣"],
            "bio": "擅长把语文变成故事课，让孩子在故事里理解字词和文章。",
            "availability": [
                {"day": "wednesday", "slots": ["19:00-20:00"]},
                {"day": "saturday", "slots": ["10:00-11:00", "14:00-15:00"]},
                {"day": "sunday", "slots": ["10:00-11:00", "15:00-16:00"]},
            ],
        },
        {
            "name": "刘洋",
            "gender": "男",
            "subjects": ["化学", "生物"],
            "grade_levels": ["高中"],
            "teaching_style": "严格",
            "education_bg": "中国科学技术大学化学系硕士",
            "years_experience": 7,
            "hourly_rate": 330.0,
            "rating": 4.8,
            "highlights": ["考点清晰", "实验丰富", "系统化教学"],
            "bio": "7 年高三把关经验，所带班级一本率 90%+。",
            "availability": [
                {"day": "tuesday", "slots": ["19:00-20:00", "20:00-21:00"]},
                {"day": "thursday", "slots": ["20:00-21:00"]},
                {"day": "sunday", "slots": ["14:00-15:00", "15:00-16:00"]},
            ],
        },
        {
            "name": "周敏",
            "gender": "女",
            "subjects": ["数学"],
            "grade_levels": ["小学高年级", "初中"],
            "teaching_style": "启发式",
            "education_bg": "北京师范大学数学系硕士",
            "years_experience": 5,
            "hourly_rate": 280.0,
            "rating": 4.9,
            "highlights": ["思维引导", "重视基础", "善于总结"],
            "bio": "5 年初中数学教学，注重思维训练而非死记硬背。",
            "availability": [
                {"day": "monday", "slots": ["19:00-20:00"]},
                {"day": "wednesday", "slots": ["19:00-20:00", "20:00-21:00"]},
                {"day": "friday", "slots": ["19:00-20:00"]},
            ],
        },
        {
            "name": "高远",
            "gender": "男",
            "subjects": ["数学", "物理", "化学"],
            "grade_levels": ["高中"],
            "teaching_style": "高压",
            "education_bg": "北京大学物理系本科",
            "years_experience": 10,
            "hourly_rate": 450.0,
            "rating": 4.9,
            "highlights": ["冲刺型", "押题准", "高强度"],
            "bio": "10 年高考冲刺经验，专攻理综，2023 届 3 位学生考上清北。",
            "availability": [
                {"day": "tuesday", "slots": ["20:00-21:00"]},
                {"day": "thursday", "slots": ["19:00-20:00"]},
                {"day": "saturday", "slots": ["09:00-10:00", "10:00-11:00"]},
            ],
        },
        {
            "name": "苏雨",
            "gender": "女",
            "subjects": ["英语", "语文"],
            "grade_levels": ["小学低年级"],
            "teaching_style": "温和",
            "education_bg": "北京语言大学汉语国际教育硕士",
            "years_experience": 3,
            "hourly_rate": 240.0,
            "rating": 4.8,
            "highlights": ["启蒙教育", "活泼有趣", "懂孩子"],
            "bio": "专为 1-3 年级孩子设计课程，把英语和语文融入游戏和绘本。",
            "availability": [
                {"day": "saturday", "slots": ["10:00-11:00", "14:00-15:00", "15:00-16:00"]},
                {"day": "sunday", "slots": ["10:00-11:00", "14:00-15:00", "15:00-16:00"]},
            ],
        },
    ]

    def initialize_default_teachers(self) -> bool:
        """初始化默认老师数据"""
        try:
            existing = self.teacher_repo.get_all_teachers()
            if existing:
                logger.info(f"数据库中已有 {len(existing)} 位老师，跳过初始化")
                return True

            logger.info("数据库中无老师数据，开始初始化默认老师")
            for data in self.DEFAULT_TEACHERS:
                try:
                    self.teacher_repo.add_teacher(**data)
                    logger.debug(f"添加老师: {data['name']}")
                except Exception as e:
                    logger.error(f"添加老师 {data['name']} 失败: {e}")
                    return False

            final_count = len(self.teacher_repo.get_all_teachers())
            logger.info(f"老师初始化完成，共 {final_count} 位")
            return True

        except Exception as e:
            logger.error(f"老师初始化失败: {e}")
            return False

    # ------------------- 业务查询 -------------------

    def get_all_teachers(self) -> List[Dict[str, Any]]:
        return self.teacher_repo.get_all_teachers()

    def get_teacher_by_id(self, teacher_id: int) -> Optional[Dict[str, Any]]:
        return self.teacher_repo.get_teacher_by_id(teacher_id)

    def get_teacher_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        return self.teacher_repo.get_teacher_by_name(name)

    def get_available_teachers(self) -> List[Dict[str, Any]]:
        return self.teacher_repo.get_available_teachers()

    def get_all_subjects(self) -> List[str]:
        return self.teacher_repo.get_all_subjects()

    def get_all_grade_levels(self) -> List[str]:
        return self.teacher_repo.get_all_grade_levels()

    def find_by_subject(self, subject: str) -> List[Dict[str, Any]]:
        return self.teacher_repo.find_teachers_by_subject(subject)

    def find_by_subject_and_grade(self, subject: str, grade_level: str) -> List[Dict[str, Any]]:
        return self.teacher_repo.find_teachers_by_subject_and_grade(subject, grade_level)

    def get_teacher_count(self) -> int:
        return len(self.get_all_teachers())

    def get_teacher_availability(self, teacher_id: int) -> List[Dict[str, Any]]:
        return self.teacher_repo.get_teacher_availability(teacher_id)

    def is_teacher_available_at(self, teacher_id: int, weekday: str, time_slot: str) -> bool:
        return self.teacher_repo.is_teacher_available_at(teacher_id, weekday, time_slot)

    def add_teacher(self, **kwargs) -> int:
        return self.teacher_repo.add_teacher(**kwargs)

    def update_teacher(self, teacher_id: int, **kwargs) -> bool:
        return self.teacher_repo.update_teacher(teacher_id, **kwargs)

    def delete_teacher(self, teacher_id: int) -> bool:
        return self.teacher_repo.delete_teacher(teacher_id)