"""
API模块包

提供REST API接口，用于与前端或第三方应用交互。
"""

# 初始化日志
from utils.logger import get_logger
logger = get_logger(__name__)

logger.debug("API模块已加载")

# 导出便捷函数
def get_blueprint_names():
    """获取所有API蓝图名称"""
    return ['stock']  # 后续可以添加其他蓝图