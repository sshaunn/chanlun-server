"""
RSI策略模块

实现基于RSI指标的股票筛选策略，包括：
- 超买/超卖条件筛选
- RSI背离检测
- 趋势确认过滤
- 可自定义参数的策略变种
"""

import pandas as pd
import traceback
from typing import Dict, List, Any, Optional, Tuple

from strategies.base_strategy import BaseStrategy
from indicators.rsi import RSIIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class RSIStrategy(BaseStrategy):
    """RSI策略实现类，继承BaseStrategy"""

    def __init__(self, name: str = "RSI策略", description: str = "基于RSI指标的股票筛选策略"):
        """
        初始化RSI策略

        参数:
        name (str): 策略名称
        description (str): 策略描述
        """
        super().__init__(name, description)

        # 设置默认参数
        self.parameters = {
            "rsi_period": 14,
            "overbought_threshold": 70,
            "oversold_threshold": 30,
            "mode": "both",  # 'overbought', 'oversold', 'both'
            "require_trend_confirm": False,
            "min_strength": 0,
            "rsi_threshold_buffer": 5,
            "trend_days": 3,
            "consecutive_days": 2,
            "extreme_rsi_only": False,
            "use_adaptive_threshold": False,
            "detect_divergence": True,
            "volume_confirm": False,
            "include_rsi_crossing": True
        }

        # 存储筛选结果详情
        self.screening_details = {}

        # 创建RSI指标计算器
        self.rsi_indicator = RSIIndicator(
            periods=[self.parameters["rsi_period"]],
            include_stochastic=True
        )

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        设置策略参数

        参数:
        parameters (Dict[str, Any]): 参数字典
        """
        if not parameters:
            return

        # 更新参数
        self.parameters.update(parameters)

        # 更新RSI指标计算器的周期
        if "rsi_period" in parameters and hasattr(self, 'rsi_indicator'):
            self.rsi_indicator = RSIIndicator(
                periods=[self.parameters["rsi_period"]],
                include_stochastic=True
            )

        logger.info(f"RSI策略参数已更新: {self.parameters}")

    def screen(self, data: Dict[str, pd.DataFrame], **kwargs) -> List[str]:
        """
        基于RSI策略对股票池进行筛选

        参数:
        data (Dict[str, pd.DataFrame]): 股票数据，键为股票代码，值为股票数据
        kwargs: 可选参数，包括 days_window

        返回:
        List[str]: 筛选出的股票代码列表
        """
        # 获取参数
        days_window = kwargs.get('days_window', 5)
        rsi_period = self.parameters["rsi_period"]
        rsi_key = f'rsi{rsi_period}'

        filtered = []
        self.screening_details = {}

        # 计算有效阈值
        effective_overbought = self.parameters["overbought_threshold"] - self.parameters["rsi_threshold_buffer"] \
            if not self.parameters["extreme_rsi_only"] else self.parameters["overbought_threshold"]
        effective_oversold = self.parameters["oversold_threshold"] + self.parameters["rsi_threshold_buffer"] \
            if not self.parameters["extreme_rsi_only"] else self.parameters["oversold_threshold"]

        logger.info(f"开始RSI策略筛选，期间={rsi_period}，超买阈值={effective_overbought}，超卖阈值={effective_oversold}")

        # 遍历所有股票数据
        for code, stock_df in data.items():
            # 检查数据是否足够
            if stock_df is None or len(stock_df) <= days_window + self.parameters["trend_days"]:
                continue

            try:
                # 确保有RSI数据
                if rsi_key not in stock_df.columns:
                    # 计算RSI指标
                    stock_df = self.rsi_indicator.calculate(stock_df)

                    # 再次检查RSI列是否存在
                    if rsi_key not in stock_df.columns:
                        logger.warning(f"无法为股票 {code} 计算RSI指标")
                        continue

                # 获取最近数据
                recent_data = stock_df.iloc[-days_window - self.parameters["trend_days"]:].copy()

                if len(recent_data) < self.parameters["trend_days"] + 2:
                    continue

                # 获取收盘价列和成交量列
                close_col = next((col for col in ['收盘', 'close', 'Close'] if col in recent_data.columns), None)
                volume_col = next((col for col in ['成交量', 'volume', 'Volume'] if col in recent_data.columns), None)

                if not close_col:
                    logger.warning(f"股票 {code} 数据缺少收盘价列")
                    continue

                # 获取最近的RSI值
                latest_rsi_values = recent_data[rsi_key].iloc[-self.parameters["trend_days"]:].values

                # 符合条件的原因
                reason = ""
                selected = False

                # 超买条件检查
                if self.parameters["mode"] in ['overbought', 'both']:
                    selected, reason = self._check_overbought_conditions(
                        recent_data, rsi_key, latest_rsi_values,
                        effective_overbought, close_col, volume_col, reason
                    )

                # 超卖条件检查（如果尚未选中）
                if not selected and self.parameters["mode"] in ['oversold', 'both']:
                    selected, reason = self._check_oversold_conditions(
                        recent_data, rsi_key, latest_rsi_values,
                        effective_oversold, close_col, volume_col, reason
                    )

                # 背离检查（如果尚未选中且启用了背离检测）
                if not selected and self.parameters["detect_divergence"]:
                    selected, reason = self._check_divergence(
                        recent_data, close_col, rsi_key, reason
                    )

                # 斐波那契水平检查（如果尚未选中）
                if not selected and len(recent_data) >= 5:
                    selected, reason = self._check_fibonacci_levels(
                        recent_data, rsi_key, reason
                    )

                # 保存结果
                if selected:
                    filtered.append(code)

                    # 存储详细信息
                    self.screening_details[code] = {
                        'code': code,
                        'rsi_value': recent_data[rsi_key].iloc[-1],
                        'prev_rsi': recent_data[rsi_key].iloc[-2],
                        'reason': reason,
                        'last_price': recent_data[close_col].iloc[-1],
                        'price_change': (recent_data[close_col].iloc[-1] / recent_data[close_col].iloc[-2] - 1) * 100,
                        'effective_thresholds': {
                            'overbought': effective_overbought,
                            'oversold': effective_oversold
                        }
                    }

                    # 尝试添加股票名称（如果可用）
                    if '股票名称' in stock_df.columns:
                        name_val = stock_df['股票名称'].iloc[0]
                        if isinstance(name_val, str):
                            self.screening_details[code]['name'] = name_val

            except Exception as e:
                logger.error(f"处理股票 {code} 时出错: {e}")
                logger.debug(traceback.format_exc())
                continue

        logger.info(f"RSI策略筛选完成，共找到 {len(filtered)} 只符合条件的股票")
        return filtered

    def get_details(self) -> Dict[str, Any]:
        """
        获取策略详细信息和最近一次筛选结果

        返回:
        Dict[str, Any]: 策略详情
        """
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
            'screening_results': self.screening_details
        }

    def _check_overbought_conditions(self, data: pd.DataFrame, rsi_key: str,
                                     latest_rsi_values: List[float],
                                     threshold: float, close_col: str,
                                     volume_col: Optional[str], reason: str) -> Tuple[bool, str]:
        """检查超买条件"""
        # 检查RSI是否处于或接近超买区域
        rsi_in_overbought_zone = data[rsi_key].iloc[-1] >= threshold

        # 检查RSI是否从高位开始回落
        rsi_falling_from_high = (data[rsi_key].iloc[-1] < data[rsi_key].iloc[-2]) and \
                                any(val >= threshold for val in latest_rsi_values)

        # 检查RSI是否穿越超买阈值（从上方向下）
        rsi_crossing_down = self.parameters["include_rsi_crossing"] and \
                            data[rsi_key].iloc[-2] >= threshold > data[rsi_key].iloc[-1]

        # 检查是否连续满足超买条件
        consecutive_overbought = sum(
            1 for val in latest_rsi_values if val >= threshold) >= self.parameters["consecutive_days"]

        # 组合条件判断超买
        if (rsi_in_overbought_zone or rsi_falling_from_high or rsi_crossing_down) and \
                (not self.parameters["extreme_rsi_only"] or consecutive_overbought):

            # 判断是否需要趋势确认
            price_confirm = True
            if self.parameters["require_trend_confirm"]:
                # 价格开始下跌
                price_confirm = data[close_col].iloc[-1] < data[close_col].iloc[-2]

            # 判断是否需要成交量确认
            volume_confirm_condition = True
            if self.parameters["volume_confirm"] and volume_col is not None:
                # 成交量放大确认卖出信号
                volume_confirm_condition = data[volume_col].iloc[-1] > data[volume_col].iloc[-2] * 1.1

            if price_confirm and volume_confirm_condition:
                local_reason = ""

                if rsi_in_overbought_zone:
                    local_reason += f"RSI({data[rsi_key].iloc[-1]:.2f})处于超买区域; "
                if rsi_falling_from_high:
                    local_reason += f"RSI从高位({max(latest_rsi_values):.2f})开始回落; "
                if rsi_crossing_down:
                    local_reason += f"RSI向下穿越超买阈值({threshold}); "
                if consecutive_overbought:
                    local_reason += f"RSI连续{self.parameters['consecutive_days']}天处于超买区域; "

                return True, reason + local_reason

        return False, reason

    def _check_oversold_conditions(self, data: pd.DataFrame, rsi_key: str,
                                   latest_rsi_values: List[float],
                                   threshold: float, close_col: str,
                                   volume_col: Optional[str], reason: str) -> Tuple[bool, str]:
        """检查超卖条件"""
        # 检查RSI是否处于或接近超卖区域
        rsi_in_oversold_zone = data[rsi_key].iloc[-1] <= threshold

        # 检查RSI是否从低位开始回升
        rsi_rising_from_low = (data[rsi_key].iloc[-1] > data[rsi_key].iloc[-2]) and \
                              any(val <= threshold for val in latest_rsi_values)

        # 检查RSI是否穿越超卖阈值（从下方向上）
        rsi_crossing_up = self.parameters["include_rsi_crossing"] and \
                          data[rsi_key].iloc[-2] <= threshold < data[rsi_key].iloc[-1]

        # 检查是否连续满足超卖条件
        consecutive_oversold = sum(
            1 for val in latest_rsi_values if val <= threshold) >= self.parameters["consecutive_days"]

        # 组合条件判断超卖
        if (rsi_in_oversold_zone or rsi_rising_from_low or rsi_crossing_up) and \
                (not self.parameters["extreme_rsi_only"] or consecutive_oversold):

            # 判断是否需要趋势确认
            price_confirm = True
            if self.parameters["require_trend_confirm"]:
                # 价格开始上涨
                price_confirm = data[close_col].iloc[-1] > data[close_col].iloc[-2]

            # 判断是否需要成交量确认
            volume_confirm_condition = True
            if self.parameters["volume_confirm"] and volume_col is not None:
                # 成交量放大确认买入信号
                volume_confirm_condition = data[volume_col].iloc[-1] > data[volume_col].iloc[-2] * 1.1

            if price_confirm and volume_confirm_condition:
                local_reason = ""

                if rsi_in_oversold_zone:
                    local_reason += f"RSI({data[rsi_key].iloc[-1]:.2f})处于超卖区域; "
                if rsi_rising_from_low:
                    local_reason += f"RSI从低位({min(latest_rsi_values):.2f})开始回升; "
                if rsi_crossing_up:
                    local_reason += f"RSI向上穿越超卖阈值({threshold}); "
                if consecutive_oversold:
                    local_reason += f"RSI连续{self.parameters['consecutive_days']}天处于超卖区域; "

                return True, reason + local_reason

        return False, reason

    def _check_divergence(self, data: pd.DataFrame, price_col: str,
                          rsi_key: str, reason: str) -> Tuple[bool, str]:
        """检查RSI背离"""
        # 如果数据中已经有背离结果，直接使用
        if 'rsi_divergence' in data.columns and data['rsi_divergence'].iloc[-1] is not None:
            last_divergence = data['rsi_divergence'].iloc[-1]

            # 如果是字典类型的背离数据
            if isinstance(last_divergence, dict):
                # 检查是否有足够强度的背离
                if last_divergence.get('strength', 0) >= self.parameters["min_strength"]:
                    if self.parameters["mode"] in ['oversold', 'both'] and last_divergence.get('bullish', False):
                        return True, reason + f"检测到看涨背离(强度:{last_divergence['strength']:.2f}); "
                    elif self.parameters["mode"] in ['overbought', 'both'] and last_divergence.get('bearish', False):
                        return True, reason + f"检测到看跌背离(强度:{last_divergence['strength']:.2f}); "
        else:
            # 没有预计算的背离数据，尝试计算
            window_size = min(len(data), 20)  # 使用最多20天的窗口

            if window_size < 5:  # 至少需要5天数据才能检测背离
                return False, reason

            prices = data[price_col].iloc[-window_size:]
            rsi_values = data[rsi_key].iloc[-window_size:]

            # 使用RSIIndicator的静态方法计算背离
            divergence = RSIIndicator.detect_rsi_divergence(prices, rsi_values, window=window_size)

            # 检查背离强度
            if divergence.get('strength', 0) >= self.parameters["min_strength"]:
                if self.parameters["mode"] in ['oversold', 'both'] and divergence.get('bullish', False):
                    return True, reason + f"检测到看涨背离(强度:{divergence['strength']:.2f}); "
                elif self.parameters["mode"] in ['overbought', 'both'] and divergence.get('bearish', False):
                    return True, reason + f"检测到看跌背离(强度:{divergence['strength']:.2f}); "

        return False, reason

    def _check_fibonacci_levels(self, data: pd.DataFrame, rsi_key: str, reason: str) -> Tuple[bool, str]:
        """检查RSI是否处于斐波那契回调位置"""
        # 获取RSI值
        rsi_values = data[rsi_key].values
        rsi_high = max(rsi_values[:-1])  # 不包括最新值
        rsi_low = min(rsi_values[:-1])  # 不包括最新值
        rsi_range = rsi_high - rsi_low

        if rsi_range < 5:  # 如果范围太小，不做分析
            return False, reason

        # 常见的斐波那契回调位置
        fib_levels = [0.382, 0.5, 0.618]
        current_rsi = data[rsi_key].iloc[-1]

        for level in fib_levels:
            if self.parameters["mode"] in ['overbought', 'both']:
                # 检查RSI是否回调到高点的斐波那契水平
                target = rsi_high - rsi_range * level
                if abs(current_rsi - target) < 3:  # 允许3个点的误差
                    return True, reason + f"RSI回调至高点的{level * 100:.1f}%位置; "

            if self.parameters["mode"] in ['oversold', 'both']:
                # 检查RSI是否反弹到低点的斐波那契水平
                target = rsi_low + rsi_range * level
                if abs(current_rsi - target) < 3:  # 允许3个点的误差
                    return True, reason + f"RSI反弹至低点的{level * 100:.1f}%位置; "

        return False, reason