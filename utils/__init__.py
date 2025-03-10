"""
工具函数模块包
"""

# 确保日志功能可用
from utils.logger import get_logger, setup_logger

# 初始化日志
logger = get_logger(__name__)
logger.debug("工具模块已加载")