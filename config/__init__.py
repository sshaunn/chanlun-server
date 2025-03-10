"""
配置模块包

提供应用配置加载和管理功能。
"""

from config.config_loader import ConfigLoader

# 创建一个全局配置加载器实例，便于其他模块使用
config_loader = ConfigLoader()

# 初始化日志
from utils.logger import get_logger

logger = get_logger(__name__)

logger.debug("配置模块已加载")


# 导出便捷函数
def get_config(key=None, default=None):
    """
    获取配置值的便捷函数

    参数:
    key (str): 配置键名，使用点分隔路径，如'server.port'
    default: 默认值，当配置不存在时返回

    返回:
    配置值或整个配置字典(如果key为None)
    """
    if key is None:
        return config_loader.get_all()
    return config_loader.get(key, default)


def get_strategy_params(strategy_type='rsi', strategy_name='default'):
    """
    获取策略参数的便捷函数

    参数:
    strategy_type (str): 策略类型，如'rsi', 'macd'等
    strategy_name (str): 策略名称，如'default', 'aggressive'等

    返回:
    dict: 策略参数字典
    """
    return config_loader.get_strategy_params(strategy_type, strategy_name)


def reload_config(env=None):
    """
    重新加载配置

    参数:
    env (str): 环境名称，如'local', 'prod'等
    """
    global config_loader
    config_loader = ConfigLoader(env)
    logger.info(f"配置已重新加载，当前环境: {config_loader.env}")
    return config_loader