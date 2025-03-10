"""
缓存管理模块

提供数据缓存的保存和加载功能，减少API调用。
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta

from utils.logger import get_logger

logger = get_logger(__name__)


class CacheManager:
    """缓存管理类，处理数据的存储和读取"""

    def __init__(self, cache_dir='./stock_cache', expiry_days=1):
        """
        初始化缓存管理器

        参数:
        cache_dir (str): 缓存目录路径
        expiry_days (int): 缓存过期天数
        """
        self.cache_dir = cache_dir
        self.expiry_days = expiry_days

        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_path(self, key):
        """获取缓存文件路径"""
        # 替换可能导致文件路径问题的字符
        safe_key = key.replace('/', '_').replace('\\', '_').replace(':', '_')
        return os.path.join(self.cache_dir, f"{safe_key}.json")

    def is_cache_valid(self, key, ignore_expiry=False):
        """
        检查缓存是否有效

        参数:
        key (str): 缓存键
        ignore_expiry (bool): 是否忽略过期检查

        返回:
        bool: 缓存是否有效
        """
        cache_path = self.get_cache_path(key)

        if not os.path.exists(cache_path):
            return False

        if ignore_expiry:
            return True

        # 检查文件修改时间
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return datetime.now() - file_time < timedelta(days=self.expiry_days)

    def save_dataframe(self, key, df):
        """
        将DataFrame保存到缓存

        参数:
        key (str): 缓存键
        df (pd.DataFrame): 要保存的DataFrame

        返回:
        bool: 是否成功保存
        """
        if df is None or df.empty:
            logger.warning(f"尝试缓存空数据: {key}")
            return False

        try:
            cache_path = self.get_cache_path(key)

            # 准备用于JSON的数据
            json_data = df.reset_index()

            # 处理日期列
            for col in json_data.columns:
                if pd.api.types.is_datetime64_any_dtype(json_data[col]):
                    json_data[col] = json_data[col].astype(str)

            # 保存到JSON文件
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(json_data.to_dict('records'), f, ensure_ascii=False)

            logger.debug(f"缓存数据已保存: {key}")
            return True
        except Exception as e:
            logger.error(f"保存缓存失败 {key}: {e}")
            return False

    def load_dataframe(self, key, date_columns=None, ignore_expiry=False):
        """
        从缓存加载DataFrame

        参数:
        key (str): 缓存键
        date_columns (list): 需要转换为日期类型的列名列表
        ignore_expiry (bool): 是否忽略过期检查

        返回:
        pd.DataFrame: 加载的DataFrame，如果缓存无效则返回None
        """
        if not self.is_cache_valid(key, ignore_expiry):
            return None

        try:
            cache_path = self.get_cache_path(key)

            with open(cache_path, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)

            if not data_dict:
                return None

            df = pd.DataFrame(data_dict)

            # 处理日期列
            if date_columns is None:
                date_columns = ['日期', 'date', '时间', 'trade_date', 'datetime', 'Date']

            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df.set_index(col, inplace=True)
                    break

            logger.debug(f"从缓存加载数据: {key}")
            return df
        except Exception as e:
            logger.error(f"读取缓存失败 {key}: {e}")
            return None

    def clear_cache(self, key=None):
        """
        清除缓存文件

        参数:
        key (str): 要清除的缓存键，None表示清除所有缓存
        """
        if key:
            cache_path = self.get_cache_path(key)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                logger.debug(f"已清除缓存: {key}")
        else:
            # 清除所有缓存
            for file in os.listdir(self.cache_dir):
                if file.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, file))
            logger.debug("已清除所有缓存")

    def clear_expired_cache(self):
        """清除所有过期的缓存文件"""
        count = 0
        for file in os.listdir(self.cache_dir):
            if not file.endswith('.json'):
                continue

            file_path = os.path.join(self.cache_dir, file)
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))

            if datetime.now() - file_time >= timedelta(days=self.expiry_days):
                os.remove(file_path)
                count += 1

        logger.debug(f"已清除 {count} 个过期缓存文件")