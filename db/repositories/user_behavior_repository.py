from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from ..base.interfaces import BaseUserBehaviorRepository
from ..base.session_manager import SessionManager
from ..models import UserBehavior, UserPreference, UserRecommendation, Teacher


class UserBehaviorRepository(BaseUserBehaviorRepository):
    """
    用户行为数据访问对象
    
    职责：
    1. 用户行为记录和查询
    2. 用户偏好管理
    3. 推荐系统数据支持
    4. 用户统计信息生成
    """
    
    def __init__(self, session_manager: SessionManager):
        """
        初始化用户行为数据仓库
        
        Args:
            session_manager: 会话管理器
        """
        self.session_manager = session_manager

    def record_behavior(self, user_id: str, action_type: str, action_data: Optional[Dict[str, Any]] = None,
                       teacher_id: Optional[int] = None, session_id: Optional[str] = None) -> int:
        """
        记录用户行为

        Args:
            user_id: 用户ID（家长）
            action_type: 行为类型
            action_data: 行为数据
            teacher_id: 老师ID
            session_id: 会话ID

        Returns:
            新创建的行为记录ID
        """
        with self.session_manager.session_scope() as session:
            behavior = UserBehavior(
                user_id=user_id,
                action_type=action_type,
                action_data=action_data,
                teacher_id=teacher_id,
                session_id=session_id
            )
            session.add(behavior)
            session.flush()
            return behavior.id

    def get_user_behaviors(self, user_id: str, action_type: Optional[str] = None, 
                          days_back: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取用户行为历史
        
        Args:
            user_id: 用户ID
            action_type: 行为类型过滤
            days_back: 查询多少天内的记录
            
        Returns:
            用户行为列表
        """
        with self.session_manager.session_scope() as session:
            query = session.query(UserBehavior).filter(UserBehavior.user_id == user_id)
            
            if action_type:
                query = query.filter(UserBehavior.action_type == action_type)
            
            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query = query.filter(UserBehavior.created_at >= cutoff_date)
            
            behaviors = query.order_by(UserBehavior.created_at.desc()).all()
            
            return [self._behavior_to_dict(behavior) for behavior in behaviors]

    def get_recent_behaviors(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取用户最近的行为记录
        
        Args:
            user_id: 用户ID
            limit: 返回记录数量限制
            
        Returns:
            最近的行为记录列表
        """
        with self.session_manager.session_scope() as session:
            behaviors = session.query(UserBehavior).filter(
                UserBehavior.user_id == user_id
            ).order_by(UserBehavior.created_at.desc()).limit(limit).all()
            
            return [self._behavior_to_dict(behavior) for behavior in behaviors]

    def update_user_preference(self, user_id: str, preference_type: str, preference_value: str) -> bool:
        """
        更新用户偏好
        
        Args:
            user_id: 用户ID
            preference_type: 偏好类型
            preference_value: 偏好值
            
        Returns:
            更新是否成功
        """
        with self.session_manager.session_scope() as session:
            # 查找现有偏好
            existing = session.query(UserPreference).filter(
                UserPreference.user_id == user_id,
                UserPreference.preference_type == preference_type,
                UserPreference.preference_value == preference_value
            ).first()
            
            if existing:
                # 增加置信度
                existing.confidence_score += 1
                existing.last_updated = datetime.utcnow()
            else:
                # 创建新偏好
                preference = UserPreference(
                    user_id=user_id,
                    preference_type=preference_type,
                    preference_value=preference_value,
                    confidence_score=1
                )
                session.add(preference)
            
            return True

    def get_user_preferences(self, user_id: str, preference_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取用户偏好
        
        Args:
            user_id: 用户ID
            preference_type: 偏好类型过滤
            
        Returns:
            用户偏好列表
        """
        with self.session_manager.session_scope() as session:
            query = session.query(UserPreference).filter(UserPreference.user_id == user_id)
            
            if preference_type:
                query = query.filter(UserPreference.preference_type == preference_type)
            
            preferences = query.order_by(UserPreference.confidence_score.desc()).all()
            
            return [self._preference_to_dict(preference) for preference in preferences]

    def get_top_preferences(self, user_id: str, preference_type: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        获取用户特定类型的高置信度偏好
        
        Args:
            user_id: 用户ID
            preference_type: 偏好类型
            limit: 返回数量限制
            
        Returns:
            高置信度偏好列表
        """
        with self.session_manager.session_scope() as session:
            preferences = session.query(UserPreference).filter(
                UserPreference.user_id == user_id,
                UserPreference.preference_type == preference_type
            ).order_by(UserPreference.confidence_score.desc()).limit(limit).all()
            
            return [self._preference_to_dict(preference) for preference in preferences]

    def create_recommendation(self, user_id: str, recommendation_type: str, content: str,
                            teacher_id: Optional[int] = None) -> int:
        """
        创建推荐

        Args:
            user_id: 用户ID
            recommendation_type: 推荐类型
            content: 推荐内容
            teacher_id: 相关老师ID

        Returns:
            新创建的推荐ID
        """
        with self.session_manager.session_scope() as session:
            recommendation = UserRecommendation(
                user_id=user_id,
                recommendation_type=recommendation_type,
                content=content,
                teacher_id=teacher_id
            )
            session.add(recommendation)
            session.flush()
            return recommendation.id

    def get_pending_recommendations(self, user_id: str) -> List[Dict[str, Any]]:
        """
        获取待发送的推荐
        
        Args:
            user_id: 用户ID
            
        Returns:
            待发送推荐列表
        """
        with self.session_manager.session_scope() as session:
            recommendations = session.query(UserRecommendation).filter(
                UserRecommendation.user_id == user_id,
                UserRecommendation.is_sent == 0
            ).order_by(UserRecommendation.created_at.desc()).all()
            
            return [self._recommendation_to_dict(recommendation) for recommendation in recommendations]

    def mark_recommendation_sent(self, recommendation_id: int) -> bool:
        """
        标记推荐为已发送
        
        Args:
            recommendation_id: 推荐ID
            
        Returns:
            标记是否成功
        """
        with self.session_manager.session_scope() as session:
            recommendation = session.query(UserRecommendation).filter(
                UserRecommendation.id == recommendation_id
            ).first()
            
            if recommendation:
                recommendation.is_sent = 1
                recommendation.sent_at = datetime.utcnow()
                return True
            return False

    def get_user_statistics(self, user_id: str, days_back: int = 30) -> Dict[str, Any]:
        """
        获取用户统计信息
        
        Args:
            user_id: 用户ID
            days_back: 统计天数
            
        Returns:
            用户统计信息字典
        """
        with self.session_manager.session_scope() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            # 总行为数
            total_behaviors = session.query(UserBehavior).filter(
                UserBehavior.user_id == user_id,
                UserBehavior.created_at >= cutoff_date
            ).count()
            
            # 预约次数
            appointment_count = session.query(UserBehavior).filter(
                UserBehavior.user_id == user_id,
                UserBehavior.action_type == 'appointment',
                UserBehavior.created_at >= cutoff_date
            ).count()
            
            # 咨询次数
            consultation_count = session.query(UserBehavior).filter(
                UserBehavior.user_id == user_id,
                UserBehavior.action_type == 'consultation',
                UserBehavior.created_at >= cutoff_date
            ).count()
            
            # 最喜欢的老师
            from sqlalchemy import func
            favorite_teacher = session.query(
                UserBehavior.teacher_id,
                Teacher.name,
                func.count(UserBehavior.teacher_id).label('count')
            ).join(Teacher).filter(
                UserBehavior.user_id == user_id,
                UserBehavior.action_type == 'lesson',
                UserBehavior.created_at >= cutoff_date
            ).group_by(UserBehavior.teacher_id, Teacher.name).order_by(
                func.count(UserBehavior.teacher_id).desc()
            ).first()

            # 最后一次访问
            last_visit = session.query(UserBehavior).filter(
                UserBehavior.user_id == user_id,
                UserBehavior.action_type == 'lesson'
            ).order_by(UserBehavior.created_at.desc()).first()

            return {
                'total_behaviors': total_behaviors,
                'lesson_count': appointment_count,
                'consultation_count': consultation_count,
                'favorite_teacher_id': favorite_teacher[0] if favorite_teacher else None,
                'favorite_teacher_name': favorite_teacher[1] if favorite_teacher else None,
                'favorite_teacher_visits': favorite_teacher[2] if favorite_teacher else 0,
                'last_visit_date': last_visit.created_at if last_visit else None,
                'days_since_last_visit': (datetime.utcnow() - last_visit.created_at).days if last_visit else None,
                'period_days': days_back
            }

    def get_teacher_popularity(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        获取老师受欢迎程度统计

        Args:
            days_back: 统计天数

        Returns:
            老师受欢迎程度列表
        """
        with self.session_manager.session_scope() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)

            from sqlalchemy import func

            popularity = session.query(
                UserBehavior.teacher_id,
                Teacher.name,
                func.count(UserBehavior.teacher_id).label('lesson_count'),
                func.count(func.distinct(UserBehavior.user_id)).label('unique_users')
            ).join(Teacher).filter(
                UserBehavior.action_type == 'lesson',
                UserBehavior.created_at >= cutoff_date
            ).group_by(UserBehavior.teacher_id, Teacher.name).order_by(
                func.count(UserBehavior.teacher_id).desc()
            ).all()

            return [
                {
                    'teacher_id': p[0],
                    'teacher_name': p[1],
                    'lesson_count': p[2],
                    'unique_users': p[3],
                }
                for p in popularity
            ]

    def _behavior_to_dict(self, behavior: UserBehavior) -> Dict[str, Any]:
        """将行为对象转换为字典"""
        return {
            'id': behavior.id,
            'user_id': behavior.user_id,
            'action_type': behavior.action_type,
            'action_data': behavior.action_data,
            'teacher_id': behavior.teacher_id,
            'teacher_name': behavior.teacher.name if behavior.teacher else None,
            'session_id': behavior.session_id,
            'created_at': behavior.created_at
        }

    def _preference_to_dict(self, preference: UserPreference) -> Dict[str, Any]:
        """将偏好对象转换为字典"""
        return {
            'id': preference.id,
            'user_id': preference.user_id,
            'preference_type': preference.preference_type,
            'preference_value': preference.preference_value,
            'confidence_score': preference.confidence_score,
            'last_updated': preference.last_updated
        }

    def _recommendation_to_dict(self, recommendation: UserRecommendation) -> Dict[str, Any]:
        """将推荐对象转换为字典"""
        return {
            'id': recommendation.id,
            'user_id': recommendation.user_id,
            'recommendation_type': recommendation.recommendation_type,
            'content': recommendation.content,
            'teacher_id': recommendation.teacher_id,
            'teacher_name': recommendation.teacher.name if recommendation.teacher else None,
            'is_sent': bool(recommendation.is_sent),
            'created_at': recommendation.created_at,
            'sent_at': recommendation.sent_at
        }
