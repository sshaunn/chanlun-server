"""
指标基类模块

定义了所有技术指标类必须实现的接口，提供基础功能和通用方法。
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Dict, Optional


class BaseIndicator(ABC):
    """所有技术指标的基类，定义通用接口"""

    def __init__(self, name: str, description: str = ""):
        """
        初始化指标类

        参数:
        name (str): 指标名称
        description (str): 指标描述
        """
        self.name = name
        self.description = description

    @abstractmethod
    def calculate(self, data: pd.DataFrame, price_col: str = '收盘') -> pd.DataFrame:
        """
        计算指标值

        参数:
        data (pd.DataFrame): 输入数据，通常是K线数据
        price_col (str): 价格列名称

        返回:
        pd.DataFrame: 添加了指标列的DataFrame
        """
        pass

    def validate_data(self, data: pd.DataFrame, required_columns: Optional[list] = None) -> bool:
        """
        验证输入数据是否符合要求

        参数:
        data (pd.DataFrame): 输入数据
        required_columns (list): 必需的列名列表

        返回:
        bool: 数据是否有效
        """
        if data is None or data.empty:
            return False

        if required_columns:
            return all(col in data.columns for col in required_columns)

        return True

    def __str__(self) -> str:
        """返回指标的字符串表示"""
        return f"{self.name}: {self.description}"

    def get_info(self) -> Dict[str, Any]:
        """
        获取指标信息

        返回:
        Dict[str, Any]: 指标信息字典
        """
        return {
            'name': self.name,
            'description': self.description,
            'type': self.__class__.__name__
        }


# 实现一个简单的空指标，用于测试
class DummyIndicator(BaseIndicator):
    """用于测试的空指标类"""

    def __init__(self):
        """初始化空指标"""
        super().__init__(name="Dummy", description="用于测试的空指标")

    def calculate(self, data: pd.DataFrame, price_col: str = '收盘') -> pd.DataFrame:
        """简单地返回原始数据"""
        if not self.validate_data(data):
            return data

        result = data.copy()
        result['dummy'] = 0  # 添加一个虚拟列
        return result