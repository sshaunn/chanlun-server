from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd


class BaseStrategy(ABC):
    """策略基类，定义了所有策略必须实现的接口"""

    def __init__(self, name: str, description: str = ""):
        """
        初始化策略

        参数:
        name (str): 策略名称
        description (str): 策略描述
        """
        self.name = name
        self.description = description
        self.parameters = {}

    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        设置策略参数

        参数:
        parameters (Dict[str, Any]): 参数字典
        """
        pass

    @abstractmethod
    def screen(self, data: Dict[str, pd.DataFrame], **kwargs) -> List[str]:
        """
        对股票池进行筛选

        参数:
        data (Dict[str, pd.DataFrame]): 股票数据，键为股票代码，值为股票数据

        返回:
        List[str]: 筛选出的股票代码列表
        """
        pass

    @abstractmethod
    def get_details(self) -> Dict[str, Any]:
        """
        获取策略详细信息和最近一次筛选结果

        返回:
        Dict[str, Any]: 策略详情
        """
        pass

    def __str__(self) -> str:
        """返回策略的字符串表示"""
        return f"{self.name}: {self.description}"