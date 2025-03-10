"""
策略模块包

提供各种股票筛选策略的实现。
"""

from strategies.base_strategy import BaseStrategy

# 先注册基础策略
registered_strategies = {}

# 初始化日志
from utils.logger import get_logger

logger = get_logger(__name__)

logger.debug("策略模块已加载")


def register_strategy(strategy_class):
    """
    注册策略类

    参数:
    strategy_class: 策略类，必须是BaseStrategy的子类

    返回:
    strategy_class: 原始策略类(用于装饰器模式)
    """
    if not issubclass(strategy_class, BaseStrategy):
        raise TypeError(f"{strategy_class.__name__} 必须是 BaseStrategy 的子类")

    strategy_name = strategy_class.__name__
    registered_strategies[strategy_name] = strategy_class
    logger.debug(f"已注册策略: {strategy_name}")

    return strategy_class


def get_strategy(strategy_name, **kwargs):
    """
    获取策略实例

    参数:
    strategy_name (str): 策略名称
    **kwargs: 传递给策略构造函数的参数

    返回:
    BaseStrategy: 策略实例

    异常:
    ValueError: 如果策略不存在
    """
    if strategy_name not in registered_strategies:
        # 尝试动态导入
        try:
            # 修复策略名称到模块名称的映射
            # 例如: "RSIStrategy" -> "rsi_strategy"
            if strategy_name.lower().endswith('strategy'):
                # 将驼峰命名转换为下划线命名
                # 例如: RSIStrategy -> rsi_strategy
                import re
                module_name = re.sub(r'(?<!^)(?=[A-Z])', '_', strategy_name).lower()
            else:
                module_name = strategy_name.lower() + "_strategy"

            logger.debug(f"尝试从 strategies.{module_name} 导入 {strategy_name}")
            module = __import__(f"strategies.{module_name}", fromlist=[strategy_name])

            # 检查模块中是否有指定的策略类
            if hasattr(module, strategy_name):
                strategy_class = getattr(module, strategy_name)
                # 注册找到的策略
                register_strategy(strategy_class)
            else:
                # 如果没有找到指定名称的类，尝试查找模块中的所有策略类
                for name in dir(module):
                    obj = getattr(module, name)
                    if isinstance(obj, type) and issubclass(obj, BaseStrategy) and obj != BaseStrategy:
                        register_strategy(obj)
                        logger.debug(f"从 {module_name} 自动注册了策略: {name}")

                # 再次检查策略是否已注册
                if strategy_name not in registered_strategies:
                    raise AttributeError(f"在 {module_name} 中找不到 {strategy_name}")

        except (ImportError, AttributeError) as e:
            logger.error(f"无法加载策略 {strategy_name}: {e}")

            # 特殊处理：尝试查找类似名称的策略
            if strategy_name == "RsiStrategy" and "RSIStrategy" in registered_strategies:
                logger.warning(f"使用 RSIStrategy 代替 RsiStrategy")
                return registered_strategies["RSIStrategy"](**kwargs)

            raise ValueError(f"未知策略: {strategy_name}")

    # 创建并返回策略实例
    return registered_strategies[strategy_name](**kwargs)


def list_strategies():
    """
    列出所有已注册的策略

    返回:
    list: 策略名称列表
    """
    return list(registered_strategies.keys())


# 直接导入并注册常用策略
try:
    from strategies.rsi_strategy import RSIStrategy

    register_strategy(RSIStrategy)
    logger.info(f"已注册RSI策略: {RSIStrategy.__name__}")
except ImportError as e:
    logger.warning(f"无法加载RSI策略: {e}")

# 其他策略将在后续导入