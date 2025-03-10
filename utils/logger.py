"""
日志工具模块

提供统一的日志配置和获取方法，确保整个应用使用一致的日志格式和级别。
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    获取指定名称的日志记录器

    参数:
    name (str): 日志记录器名称，通常是模块名(__name__)
    level (int, optional): 日志级别，如果为None则使用全局配置

    返回:
    logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)

    # 如果已经配置过处理器，直接返回
    if logger.handlers:
        if level is not None:
            logger.setLevel(level)
        return logger

    # 获取全局日志配置
    log_level = level or os.environ.get('LOG_LEVEL', 'INFO')
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)

    logger.setLevel(log_level)

    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(console_handler)

    # 如果指定了日志文件，添加文件处理器
    log_file = os.environ.get('LOG_FILE')
    if log_file:
        try:
            # 确保日志目录存在
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # 创建轮转文件处理器
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(file_handler)
        except Exception as e:
            logger.error(f"无法创建日志文件处理器: {e}")

    return logger


def setup_logger(app=None):
    """
    设置全局日志配置

    参数:
    app: Flask应用实例(可选)
    """
    # 设置根日志记录器的默认级别
    log_level = os.environ.get('LOG_LEVEL', 'INFO')
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)

    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 移除任何现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    root_logger.addHandler(console_handler)

    # 如果指定了日志文件，添加文件处理器
    log_file = os.environ.get('LOG_FILE')
    if log_file:
        try:
            # 确保日志目录存在
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # 创建轮转文件处理器
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"无法创建日志文件处理器: {e}")

    # 降低一些第三方库的日志级别，避免过多日志
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

    # 如果提供了Flask应用，配置Flask日志
    if app:
        app.logger.handlers = root_logger.handlers
        app.logger.setLevel(root_logger.level)


# 当模块被直接执行时，配置根日志
if __name__ == "__main__":
    setup_logger()
    logger = get_logger("logger_test")
    logger.debug("这是一条调试日志")
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告日志")
    logger.error("这是一条错误日志")
    logger.critical("这是一条严重错误日志")