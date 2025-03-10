"""
技术指标模块包
"""

from indicators.base_indicator import BaseIndicator

# 初始化日志
from utils.logger import get_logger
logger = get_logger(__name__)

logger.debug("指标模块已加载")