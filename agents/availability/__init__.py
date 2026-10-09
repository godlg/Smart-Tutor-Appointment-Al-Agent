"""全局档期查询子包"""

from .availability_handler import AvailabilityHandler, is_availability_query

__all__ = ["AvailabilityHandler", "is_availability_query"]