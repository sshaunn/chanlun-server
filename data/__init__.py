"""
数据获取模块包

提供股票数据的获取和缓存功能。
"""

from data.stock_fetcher import StockFetcher
from data.cache_manager import CacheManager

# 初始化日志
from utils.logger import get_logger
logger = get_logger(__name__)

logger.debug("数据模块已加载")

# 导出便捷函数
def get_stock_fetcher():
    """获取股票数据获取器实例"""
    return StockFetcher()

def get_cache_manager():
    """获取缓存管理器实例"""
    from config import get_config
    cache_dir = get_config('cache.dir', './stock_cache')
    expiry_days = get_config('cache.expiry_days', 1)
    return CacheManager(cache_dir=cache_dir, expiry_days=expiry_days)