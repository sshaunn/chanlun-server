"""
股票数据获取模块

使用akshare获取A股数据，并支持缓存功能提高性能。
"""

import os
import json
import traceback
import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config import get_config
from utils.logger import get_logger
from data.cache_manager import CacheManager

logger = get_logger(__name__)


class StockFetcher:
    """股票数据获取类，负责获取和缓存A股数据"""

    def __init__(self, cache_manager=None):
        """
        初始化股票数据获取器

        参数:
        cache_manager: 缓存管理器实例，如果为None则创建新实例
        """
        # 获取配置
        self.cache_dir = get_config('cache.dir', './stock_cache')
        self.cache_expiry_days = get_config('cache.expiry_days', 1)
        self.max_workers = get_config('data_fetch.max_workers', 5)
        self.retry_times = get_config('data_fetch.retry_times', 3)
        self.akshare_version = get_config('data_fetch.akshare_version', '1.16.35')

        # 验证akshare版本
        self._check_akshare_version()

        # 初始化缓存管理器
        if cache_manager is None:
            self.cache_manager = CacheManager(
                cache_dir=self.cache_dir,
                expiry_days=self.cache_expiry_days
            )
        else:
            self.cache_manager = cache_manager

        # 初始化其他属性
        self.all_stocks = None
        self.stock_data = {}

    def _check_akshare_version(self):
        """检查akshare版本是否符合要求"""
        try:
            current_version = ak.__version__
            required_version = self.akshare_version

            if current_version != required_version:
                logger.warning(f"当前akshare版本({current_version})与配置的版本({required_version})不匹配")
                logger.warning("这可能导致数据获取问题，建议安装指定版本: pip install akshare=={required_version}")
        except Exception as e:
            logger.error(f"检查akshare版本失败: {e}")

    @lru_cache(maxsize=1)
    def get_all_stocks(self):
        """获取所有A股的股票代码和名称，使用缓存减少API调用"""
        cache_key = "all_stocks"

        # 尝试从缓存加载
        all_stocks = self.cache_manager.load_dataframe(cache_key)
        if all_stocks is not None:
            self.all_stocks = all_stocks
            logger.info(f"从缓存加载了 {len(all_stocks)} 只股票")
            return all_stocks

        logger.info("正在获取所有A股股票列表...")

        try:
            for attempt in range(self.retry_times):
                try:
                    stock_info = ak.stock_info_a_code_name()
                    break
                except Exception as e:
                    if attempt < self.retry_times - 1:
                        logger.warning(f"获取股票列表重试 ({attempt + 1}/{self.retry_times}): {e}")
                        continue
                    else:
                        raise e

            # 只保留主板、中小板和创业板，去掉北交所等
            stock_info = stock_info[stock_info['code'].apply(lambda x: x.startswith(('60', '00', '30')))]
            self.all_stocks = stock_info

            # 保存到缓存
            self.cache_manager.save_dataframe(cache_key, stock_info)

            logger.info(f"共获取到 {len(self.all_stocks)} 只股票")
            return self.all_stocks

        except Exception as e:
            logger.error(f"获取A股列表出错: {e}")
            logger.debug(traceback.format_exc())

            # 如果获取失败但缓存存在，尝试使用过期缓存
            expired_cache = self.cache_manager.load_dataframe(cache_key, ignore_expiry=True)
            if expired_cache is not None:
                logger.warning("使用过期缓存...")
                self.all_stocks = expired_cache
                return expired_cache

            return pd.DataFrame()

    def get_stock_data(self, code, name=None, period="daily",
                       start_date=None, end_date=None, use_cache=True,
                       adjust="qfq"):
        """
        获取单个股票的历史数据，支持缓存

        参数:
        code (str): 股票代码
        name (str): 股票名称，可选
        period (str): 周期，如daily/weekly/monthly
        start_date (str): 开始日期，格式YYYYMMDD
        end_date (str): 结束日期，格式YYYYMMDD
        use_cache (bool): 是否使用缓存
        adjust (str): 复权方式，qfq(前复权)、hfq(后复权)、none(不复权)

        返回:
        pd.DataFrame: 股票历史数据
        """
        # 设置默认日期
        if start_date is None:
            start_date = get_config('stock.default_start_date',
                                    (datetime.now() - timedelta(days=365)).strftime('%Y%m%d'))
        if end_date is None:
            end_date = get_config('stock.default_end_date', datetime.now().strftime('%Y%m%d'))

        # 缓存键
        cache_key = f"{code}_{period}_{start_date}_{end_date}_{adjust}"

        # 从缓存加载数据
        if use_cache:
            stock_data = self.cache_manager.load_dataframe(cache_key)
            if stock_data is not None and len(stock_data) >= 30:  # 确保有足够数据
                logger.debug(f"使用缓存的股票数据: {code}")
                return stock_data

        logger.debug(f"开始获取股票 {code} 的历史数据")

        try:
            # 定义可能的股票代码格式
            code_formats = []
            code_formats.append(code)  # 原始代码

            # 根据代码前缀添加格式化代码
            if code.startswith('6'):
                code_formats.extend([f"sh{code}", f"SH{code}", f"1.{code}"])
            else:
                code_formats.extend([f"sz{code}", f"SZ{code}", f"0.{code}"])

            code_formats.append(code.lstrip('0'))  # 去掉前导零

            # 尝试所有代码格式和数据获取方法
            stock_data = None
            successful_format = None

            # 方法1: stock_zh_a_hist
            for code_format in code_formats:
                try:
                    stock_data = ak.stock_zh_a_hist(
                        symbol=code_format,
                        period=period,
                        start_date=start_date,
                        end_date=end_date,
                        adjust=adjust
                    )

                    if stock_data is not None and not stock_data.empty:
                        successful_format = f"stock_zh_a_hist with {code_format}"
                        break
                except Exception as e:
                    logger.debug(f"使用stock_zh_a_hist尝试代码 {code_format} 失败: {e}")
                    continue

            # 方法2: 东方财富数据源
            if stock_data is None or stock_data.empty:
                try:
                    logger.debug(f"尝试使用东方财富数据源获取 {code} 数据")
                    stock_data = ak.stock_zh_a_hist_163(
                        symbol=code,
                        start_date=start_date,
                        end_date=end_date,
                        adjust=adjust
                    )

                    if stock_data is not None and not stock_data.empty:
                        successful_format = "stock_zh_a_hist_163"
                except Exception as e:
                    logger.debug(f"使用stock_zh_a_hist_163获取 {code} 失败: {e}")

            # 方法3: 同花顺数据源
            if stock_data is None or stock_data.empty:
                try:
                    logger.debug(f"尝试使用同花顺数据源获取 {code} 数据")
                    # 对于沪市股票，需要添加前缀
                    ths_code = code
                    if code.startswith('6'):
                        ths_code = f"sh{code}"
                    elif code.startswith(('0', '3')):
                        ths_code = f"sz{code}"

                    stock_data = ak.stock_zh_a_hist_min_em(
                        symbol=ths_code,
                        period='daily',  # 使用日线数据
                        start_date=start_date,
                        end_date=end_date,
                        adjust=adjust
                    )

                    if stock_data is not None and not stock_data.empty:
                        successful_format = "stock_zh_a_hist_min_em"
                except Exception as e:
                    logger.debug(f"使用stock_zh_a_hist_min_em获取 {code} 失败: {e}")

            # 方法4: 实时行情数据
            if stock_data is None or stock_data.empty:
                try:
                    logger.debug(f"尝试获取 {code} 实时行情数据")
                    real_time_data = ak.stock_zh_a_spot_em()

                    # 筛选出目标股票
                    for col in ['代码', '股票代码', 'code']:
                        if col in real_time_data.columns:
                            stock_real_time = real_time_data[real_time_data[col] == code]
                            if not stock_real_time.empty:
                                break
                    else:
                        stock_real_time = pd.DataFrame()

                    if not stock_real_time.empty:
                        # 创建一个最小的历史数据框架
                        today = datetime.now().strftime('%Y-%m-%d')
                        columns_mapping = {
                            '开盘': 'open', '最新价': 'close', '最高': 'high', '最低': 'low',
                            '成交量': 'volume', '成交额': 'amount'
                        }

                        data_dict = {'日期': [today], '股票代码': [code], '股票名称': [name or code]}

                        for zh_col, en_col in columns_mapping.items():
                            if zh_col in stock_real_time.columns:
                                data_dict[en_col] = [float(stock_real_time[zh_col].iloc[0])]
                            else:
                                data_dict[en_col] = [np.nan]

                        stock_data = pd.DataFrame(data_dict)
                        stock_data['日期'] = pd.to_datetime(stock_data['日期'])
                        stock_data.set_index('日期', inplace=True)
                        successful_format = "real_time_data"

                        logger.warning(f"注意: 仅获取到 {code} 的实时数据，不足以进行技术分析")
                except Exception as e:
                    logger.debug(f"获取 {code} 实时行情失败: {e}")

            # 如果所有方法都失败
            if stock_data is None or stock_data.empty:
                logger.warning(f"无法获取股票 {code} 的数据，已尝试所有可能的方法")
                return None

            # 处理成功获取的数据
            logger.debug(f"成功使用{successful_format}获取 {code} 数据")

            # 标准化数据
            stock_data = self._standardize_stock_data(stock_data, code, name)

            # 检查数据是否足够分析
            if len(stock_data) < 30:
                logger.warning(f"股票 {code} 数据不足30条 (实际: {len(stock_data)}条)，可能无法进行有效分析")
                if len(stock_data) < 5:  # 如果数据极少，返回None
                    return None

            # 如果启用缓存，保存数据
            if use_cache and not stock_data.empty:
                self.cache_manager.save_dataframe(cache_key, stock_data)

            return stock_data

        except Exception as e:
            logger.error(f"获取 {code} 数据时发生未捕获的错误: {e}")
            logger.debug(traceback.format_exc())
            return None

    def _standardize_stock_data(self, data, code, name=None):
        """标准化股票数据，确保列名一致"""
        if data is None or data.empty:
            return data

        # 确保索引是日期类型
        if not isinstance(data.index, pd.DatetimeIndex):
            # 查找可能的日期列
            date_columns = ['日期', 'date', '时间', 'trade_date', 'datetime', 'Date', '日线日期']
            date_column = None

            for col in date_columns:
                if col in data.columns:
                    date_column = col
                    break

            # 如果找到日期列，设置为索引
            if date_column:
                try:
                    data[date_column] = pd.to_datetime(data[date_column])
                    data.set_index(date_column, inplace=True)
                except Exception as e:
                    logger.warning(f"设置日期索引失败: {e}")

        # 标准化列名 - 中文到英文
        column_mapping_zh_en = {
            '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low',
            '成交量': 'volume', '成交额': 'amount', '换手率': 'turnover'
        }

        # 标准化列名 - 英文到中文
        column_mapping_en_zh = {
            'open': '开盘', 'close': '收盘', 'high': '最高', 'low': '最低',
            'volume': '成交量', 'amount': '成交额', 'turnover': '换手率'
        }

        # 检查原始数据使用的是哪种命名方式
        if any(col in column_mapping_zh_en for col in data.columns):
            # 使用中文列名，标准化为英文
            for zh, en in column_mapping_zh_en.items():
                if zh in data.columns and en not in data.columns:
                    data[en] = data[zh]
        else:
            # 使用英文列名，标准化为中文
            for en, zh in column_mapping_en_zh.items():
                if en in data.columns and zh not in data.columns:
                    data[zh] = data[en]

        # 添加股票代码和名称
        data['股票代码'] = code
        data['股票名称'] = name or code

        return data

    def process_stock(self, code, name=None, indicators=None):
        """
        处理单个股票的完整流程，包括获取数据和计算指标

        参数:
        code (str): 股票代码
        name (str): 股票名称，可选
        indicators (list): 需要计算的指标列表

        返回:
        pd.DataFrame: 处理后的股票数据
        """
        # 获取股票数据
        data = self.get_stock_data(code, name)
        if data is None:
            return None

        # 计算指标
        if indicators:
            for indicator in indicators:
                data = indicator.calculate(data)

        # 保存处理后的数据
        self.stock_data[code] = data
        return data

    def parallel_process_stocks(self, indicators=None, max_stocks=None, show_progress=True):
        """
        并行处理多只股票

        参数:
        indicators (list): 需要计算的指标列表
        max_stocks (int): 最大处理股票数量，用于测试
        show_progress (bool): 是否显示进度条

        返回:
        dict: 处理后的股票数据字典
        """
        if self.all_stocks is None:
            self.get_all_stocks()

        # 可以限制处理的股票数量，用于测试
        stocks_to_process = self.all_stocks
        if max_stocks:
            stocks_to_process = self.all_stocks.head(max_stocks)

        total_stocks = len(stocks_to_process)
        logger.info(f"开始处理 {total_stocks} 只股票的数据...")

        # 清空旧数据
        self.stock_data = {}

        # 使用线程池并行处理股票数据
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 创建任务列表
            future_to_stock = {}
            for _, row in stocks_to_process.iterrows():
                future = executor.submit(
                    self.process_stock,
                    row['code'],
                    row['name'],
                    indicators
                )
                future_to_stock[future] = row['code']

            # 处理结果
            if show_progress:
                for future in tqdm(as_completed(future_to_stock), total=total_stocks, desc="处理股票数据"):
                    stock_code = future_to_stock[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"处理股票 {stock_code} 时发生错误: {e}")
            else:
                for future in as_completed(future_to_stock):
                    stock_code = future_to_stock[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"处理股票 {stock_code} 时发生错误: {e}")

        logger.info(f"成功处理 {len(self.stock_data)} 只股票的数据")
        return self.stock_data

    def get_stock_data_dict(self):
        """获取处理后的股票数据字典"""
        return self.stock_data