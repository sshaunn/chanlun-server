"""
服务模块包

提供各种业务逻辑服务。
"""

# 初始化日志
from utils.logger import get_logger
logger = get_logger(__name__)

logger.debug("服务模块已加载")

# 导出便捷函数
def get_screening_service():
    """获取股票筛选服务实例"""
    from services.screen_service import ScreeningService
    return ScreeningService()

# 预留其他服务获取函数
def get_analyzer_service():
    """获取分析服务实例(预留)"""
    # from services.analyzer_service import AnalyzerService
    # return AnalyzerService()
    logger.warning("分析服务尚未实现")
    return None

def get_trading_service():
    """获取交易服务实例(预留)"""
    # from services.trading_service import TradingService
    # return TradingService()
    logger.warning("交易服务尚未实现")
    return None