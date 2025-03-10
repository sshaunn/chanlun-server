import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class RSIIndicator(BaseIndicator):
    """
    相对强弱指标(RSI)计算和分析类

    RSI是一种动量指标，测量价格变动的速度和变化，主要用于识别超买或超卖条件。
    该类提供了多种RSI计算方法和分析功能。
    """

    def __init__(self,
                 periods: List[int] = [14],
                 smoothing_type: str = 'simple',
                 include_stochastic: bool = False,
                 stoch_period: int = 14):
        """
        初始化RSI指标计算器

        参数:
        periods (List[int]): RSI计算周期列表，默认[14]
        smoothing_type (str): 平滑方法，可选'simple'（简单平均）或'exponential'（指数平滑）
        include_stochastic (bool): 是否计算随机RSI
        stoch_period (int): 随机RSI的周期
        """
        super().__init__(name="RSI")
        self.periods = periods
        self.smoothing_type = smoothing_type
        self.include_stochastic = include_stochastic
        self.stoch_period = stoch_period

    def calculate(self, data: pd.DataFrame, price_col: str = '收盘') -> pd.DataFrame:
        """
        计算数据中的RSI指标

        参数:
        data (pd.DataFrame): 股票数据，必须包含价格列
        price_col (str): 价格列名

        返回:
        pd.DataFrame: 添加了RSI列的数据
        """
        if data is None or data.empty:
            logger.warning("输入数据为空，无法计算RSI")
            return data

        if price_col not in data.columns:
            logger.error(f"数据中缺少 {price_col} 列，无法计算RSI")
            return data

        # 创建结果的副本，避免修改原始数据
        result = data.copy()

        try:
            # 为每个周期计算RSI
            for period in self.periods:
                # 计算传统RSI
                rsi_col = f'rsi{period}'
                result[rsi_col] = self.calculate_traditional_rsi(
                    result[price_col],
                    period=period,
                    smoothing_type=self.smoothing_type
                )

                # 可选：计算随机RSI
                if self.include_stochastic:
                    stoch_rsi_col = f'stoch_rsi{period}'
                    result[stoch_rsi_col] = self.calculate_stochastic_rsi(
                        result[price_col],
                        rsi_period=period,
                        stoch_period=self.stoch_period
                    )

                # 可选：计算Cutler的RSI变种
                if period <= 30:  # 仅为合理的周期计算Cutler RSI
                    cutler_rsi_col = f'cutler_rsi{period}'
                    result[cutler_rsi_col] = self.calculate_cutler_rsi(
                        result[price_col],
                        period=period
                    )

                # 检测背离
                if len(result) > max(period * 2, 30):
                    # 初始化背离列为None
                    result['rsi_divergence'] = None

                    # 每10行计算一次背离，减少计算量
                    for i in range(period * 2, len(result), 10):
                        end_idx = i
                        start_idx = max(0, end_idx - 20)  # 使用20天窗口

                        if end_idx < len(result):
                            price_window = result[price_col].iloc[start_idx:end_idx + 1]
                            rsi_window = result[rsi_col].iloc[start_idx:end_idx + 1]

                            divergence = self.detect_rsi_divergence(
                                price_window,
                                rsi_window,
                                window=min(20, len(price_window))
                            )

                            # 修复部分 - 使用at而不是loc
                            try:
                                # 获取对应的索引值
                                idx_value = result.index[end_idx]
                                # 使用at方法设置值，这对于复杂对象更安全
                                result.at[idx_value, 'rsi_divergence'] = divergence
                            except Exception as e:
                                logger.warning(f"设置RSI背离值时出错: {e}")
                                # 尝试备用方法
                                try:
                                    result.iloc[end_idx, result.columns.get_loc('rsi_divergence')] = str(divergence)
                                except:
                                    pass

            return result

        except Exception as e:
            logger.error(f"计算RSI时出错: {e}", exc_info=True)
            return data

    @staticmethod
    def calculate_traditional_rsi(prices: pd.Series, period: int = 14, smoothing_type: str = 'simple') -> pd.Series:
        """
        计算传统RSI指标

        参数:
        prices (pd.Series): 价格序列
        period (int): RSI计算周期，默认14
        smoothing_type (str): 平滑方法，可选'simple'（简单平均）或'exponential'（指数平滑）

        返回:
        pd.Series: RSI值序列
        """
        if len(prices) < period + 1:
            return pd.Series(np.nan, index=prices.index)

        # 计算价格变化
        deltas = prices.diff().dropna()

        # 分离上涨和下跌
        gains = deltas.clip(lower=0)
        losses = -deltas.clip(upper=0)

        # 根据平滑方法计算平均涨跌幅
        if smoothing_type == 'simple':
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()
        elif smoothing_type == 'exponential':
            avg_gains = gains.ewm(com=period - 1, min_periods=period).mean()
            avg_losses = losses.ewm(com=period - 1, min_periods=period).mean()
        else:
            raise ValueError(f"不支持的平滑方法: {smoothing_type}")

        # 避免除零错误
        avg_losses = avg_losses.replace(0, 1e-10)

        # 计算相对强度和RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_cutler_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        计算Cutler's RSI，这是一个不同于标准RSI的变种

        参数:
        prices (pd.Series): 价格序列
        period (int): 计算周期

        返回:
        pd.Series: Cutler's RSI值序列
        """
        # 使用Rolling百分比变化而非价格差异
        up_days = prices.rolling(period).apply(lambda x: sum(1 for i in range(1, len(x)) if x[i] > x[i - 1]), raw=True)
        rsi = 100 * up_days / period
        return rsi

    @staticmethod
    def calculate_stochastic_rsi(prices: pd.Series, rsi_period: int = 14, stoch_period: int = 14) -> pd.Series:
        """
        计算随机RSI (StochRSI)

        参数:
        prices (pd.Series): 价格序列
        rsi_period (int): RSI计算周期
        stoch_period (int): StochRSI周期

        返回:
        pd.Series: 随机RSI值序列
        """
        # 计算传统RSI
        rsi = RSIIndicator.calculate_traditional_rsi(prices, rsi_period)

        # 计算RSI的最高和最低值
        rsi_min = rsi.rolling(window=stoch_period).min()
        rsi_max = rsi.rolling(window=stoch_period).max()

        # 计算随机RSI，避免除零错误
        denominator = rsi_max - rsi_min
        denominator = denominator.replace(0, 1e-10)  # 避免除零错误

        stoch_rsi = 100 * (rsi - rsi_min) / denominator
        return stoch_rsi

    @staticmethod
    def detect_rsi_divergence(prices: pd.Series, rsi: pd.Series, window: int = 10) -> Dict:
        """
        检测RSI背离

        参数:
        prices (pd.Series): 价格序列
        rsi (pd.Series): RSI序列
        window (int): 检测窗口大小

        返回:
        dict: 包含背离类型和强度
        """
        # 默认返回值
        default_result = {"bullish": False, "bearish": False, "strength": 0}

        # 数据验证
        if prices is None or rsi is None:
            return default_result

        # 确保数据长度足够
        if len(prices) < max(window, 5) or len(rsi) < max(window, 5):
            return default_result

        try:
            # 获取最近的数据
            recent_prices = prices[-window:].reset_index(drop=True)
            recent_rsi = rsi[-window:].reset_index(drop=True)

            # 确保没有缺失值
            if recent_prices.isna().any() or recent_rsi.isna().any():
                # 清理缺失值
                valid_indices = ~(recent_prices.isna() | recent_rsi.isna())
                recent_prices = recent_prices[valid_indices]
                recent_rsi = recent_rsi[valid_indices]

                # 再次检查数据长度
                if len(recent_prices) < 5 or len(recent_rsi) < 5:
                    return default_result

            # 计算数据点的简单趋势（采用线性回归而非简单比较）
            try:
                # 使用线性回归来确定趋势
                x = np.arange(len(recent_prices))
                price_slope = np.polyfit(x, recent_prices, 1)[0]
                rsi_slope = np.polyfit(x, recent_rsi, 1)[0]

                price_trend = price_slope > 0
                rsi_trend = rsi_slope > 0
            except Exception:
                # 回退到简单比较
                price_trend = recent_prices.iloc[-1] > recent_prices.iloc[0]
                rsi_trend = recent_rsi.iloc[-1] > recent_rsi.iloc[0]

            # 使用更安全的相关性计算，避免除零错误
            try:
                # 确保有足够的变异用于计算相关性
                if recent_prices.std() > 1e-10 and recent_rsi.std() > 1e-10:
                    correlation = np.corrcoef(recent_prices, recent_rsi)[0, 1]
                    if np.isnan(correlation):
                        correlation = 0
                else:
                    correlation = 0
            except Exception:
                # 如果相关性计算失败，使用默认值
                correlation = 0

            # 计算背离强度
            strength = abs(correlation)

            # 判断背离类型
            bullish_divergence = not price_trend and rsi_trend  # 价格下降但RSI上升
            bearish_divergence = price_trend and not rsi_trend  # 价格上升但RSI下降

            # 返回结果
            return {
                "bullish": bullish_divergence,
                "bearish": bearish_divergence,
                "strength": strength
            }
        except Exception as e:
            logger.error(f"计算RSI背离时出错: {e}", exc_info=True)
            return default_result

    @staticmethod
    def analyze_rsi_conditions(rsi: pd.Series, overbought: float = 70, oversold: float = 30,
                               neutral_zone: Optional[Tuple[float, float]] = None) -> Dict:
        """
        分析RSI的超买超卖状态

        参数:
        rsi (pd.Series): RSI序列
        overbought (float): 超买阈值
        oversold (float): 超卖阈值
        neutral_zone (tuple): 中性区间，如(40, 60)

        返回:
        dict: RSI分析结果
        """
        if neutral_zone is None:
            neutral_zone = (oversold + 10, overbought - 10)

        # 检查RSI序列是否为空
        if rsi is None or len(rsi) == 0 or rsi.isna().all():
            return {
                "current_value": None,
                "status": "数据不足",
                "trend": "数据不足",
                "reversal_signal": "数据不足",
                "is_actionable": False
            }

        latest_rsi = rsi.iloc[-1]

        # 基本状态判断
        is_overbought = latest_rsi >= overbought
        is_oversold = latest_rsi <= oversold
        is_neutral = neutral_zone[0] <= latest_rsi <= neutral_zone[1]

        # RSI趋势分析（最近3天）
        if len(rsi) >= 3:
            recent_trend = "上升" if rsi.iloc[-1] > rsi.iloc[-3] else "下降" if rsi.iloc[-1] < rsi.iloc[-3] else "盘整"
        else:
            recent_trend = "数据不足"

        # 检测超买超卖区域的反转信号
        reversal_signal = False
        if len(rsi) >= 3:
            if is_overbought and rsi.iloc[-1] < rsi.iloc[-2]:
                reversal_signal = "顶部反转可能"
            elif is_oversold and rsi.iloc[-1] > rsi.iloc[-2]:
                reversal_signal = "底部反转可能"

        return {
            "current_value": latest_rsi,
            "status": "超买" if is_overbought else "超卖" if is_oversold else "中性",
            "trend": recent_trend,
            "reversal_signal": reversal_signal if reversal_signal else "无明显反转信号",
            "is_actionable": is_overbought or is_oversold  # 是否需要采取行动
        }