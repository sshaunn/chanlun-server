"""
股票筛选服务模块

集成各种策略，提供股票筛选服务。
"""

from typing import Dict, List, Any, Optional
import pandas as pd

from config import get_config, get_strategy_params
from utils.logger import get_logger
from data.stock_fetcher import StockFetcher
from strategies import get_strategy  # 正确导入get_strategy函数
from indicators.rsi import RSIIndicator
from indicators.base_indicator import BaseIndicator

logger = get_logger(__name__)


class ScreeningService:
    """股票筛选服务类，整合数据获取和策略应用"""

    def __init__(self):
        """初始化筛选服务"""
        # 创建数据获取器
        self.stock_fetcher = StockFetcher()

        # 初始化指标列表
        self.indicators = []

        # 添加默认指标
        self._register_default_indicators()

        # 最近一次筛选结果
        self.last_results = {}

    def _register_default_indicators(self):
        """注册默认的技术指标"""
        # 添加RSI指标
        rsi_periods = get_config('strategy.rsi_periods', [14])
        self.add_indicator(RSIIndicator(periods=rsi_periods, include_stochastic=True))

        # 后续可以添加其他指标
        # TODO: 添加MACD, EMA, KDJ, BOLL等指标

    def add_indicator(self, indicator: BaseIndicator):
        """
        添加技术指标

        参数:
        indicator (BaseIndicator): 指标实例
        """
        if not isinstance(indicator, BaseIndicator):
            raise TypeError("指标必须是BaseIndicator的子类")

        self.indicators.append(indicator)
        logger.debug(f"添加指标: {indicator.name}")

    def run_screening(self, strategy_type='rsi', strategy_name='default',
                     max_stocks=None, days_window=5, show_progress=True):
        """
        运行股票筛选

        参数:
        strategy_type (str): 策略类型，如'rsi'
        strategy_name (str): 策略名称，如'default'
        max_stocks (int): 最大处理股票数量，None表示处理所有
        days_window (int): 查找信号的天数窗口
        show_progress (bool): 是否显示进度条

        返回:
        List[Dict]: 筛选结果列表
        """
        logger.info(f"开始运行{strategy_type}策略({strategy_name})筛选...")

        # 获取策略参数
        strategy_params = get_strategy_params(strategy_type, strategy_name)

        # 创建策略实例 - 修复策略类名称映射
        try:
            # 将策略类型转换为策略类名称
            if strategy_type.lower() == 'rsi':
                strategy_class_name = "RSIStrategy"  # 使用正确的类名
            else:
                strategy_class_name = f"{strategy_type.capitalize()}Strategy"

            from strategies import get_strategy  # 如果上方导入出错，在这里再次导入
            strategy = get_strategy(strategy_class_name)
            strategy.set_parameters(strategy_params)
        except (ValueError, ImportError) as e:
            logger.error(f"创建策略实例失败: {e}")
            # 尝试直接导入RSI策略
            try:
                from strategies.rsi_strategy import RSIStrategy
                strategy = RSIStrategy()
                strategy.set_parameters(strategy_params)
                logger.info("使用直接导入的RSIStrategy")
            except ImportError:
                logger.error("无法导入RSI策略，放弃筛选")
                return []

        # 获取所有股票数据
        if not self.stock_fetcher.all_stocks:
            self.stock_fetcher.get_all_stocks()

        # 处理股票数据
        stock_data = self.stock_fetcher.parallel_process_stocks(
            indicators=self.indicators,
            max_stocks=max_stocks,
            show_progress=show_progress
        )

        # 应用策略筛选
        filtered_codes = strategy.screen(stock_data, days_window=days_window)

        # 获取详细结果
        details = strategy.get_details()
        results = details.get('screening_results', {})

        # 格式化结果列表
        formatted_results = []
        for code in filtered_codes:
            result = results.get(code, {})

            # 获取股票名称
            name = result.get('name', '未知')
            if name == '未知' and code in self.stock_fetcher.all_stocks['code'].values:
                name_index = self.stock_fetcher.all_stocks.index[
                    self.stock_fetcher.all_stocks['code'] == code
                ][0]
                name = self.stock_fetcher.all_stocks.iloc[name_index]['name']

            # 添加到结果列表
            formatted_results.append({
                'code': code,
                'name': name,
                **{k: v for k, v in result.items() if k not in ['code', 'name']}
            })

        logger.info(f"筛选完成，找到 {len(formatted_results)} 只符合条件的股票")

        # 保存最近一次结果
        self.last_results = {
            'strategy_type': strategy_type,
            'strategy_name': strategy_name,
            'parameters': strategy_params,
            'results': formatted_results
        }

        return formatted_results

    def get_last_results(self) -> Dict[str, Any]:
        """获取最近一次筛选结果"""
        return self.last_results

    def get_available_strategies(self) -> List[str]:
        """获取可用策略列表"""
        try:
            from strategies import list_strategies
            return list_strategies()
        except ImportError:
            logger.warning("无法导入list_strategies函数")
            # 返回硬编码的策略列表
            return ["RSIStrategy"]