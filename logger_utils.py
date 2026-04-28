#日志系统
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "topo_opt",
    log_dir: str = "./logs",
    level: int = logging.INFO,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    format_string: Optional[str] = None
) -> logging.Logger:
     #参数：
     # name: 日志记录器名称（通常为模块名 __name__）
     # log_dir: 日志文件存储目录
     # level: 记录器总级别
     # console_level: 控制台输出级别
     # file_level: 文件输出级别
     # format_string: 自定义日志格式，默认使用标准格式

    # 返回:配置好的 logging.Logger 实例
    # 默认日志格式：时间戳 | 级别 | 模块名 | 函数名：行号 | 消息
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s | "
            "%(funcName)s:%(lineno)d | %(message)s"
        )

    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{name}_{timestamp}.log"

    # 获取或创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 创建格式化器
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 记录初始化信息
    logger.info(f"日志系统初始化完成 - 文件：{log_file}")
    logger.debug(f"日志级别：记录器={level}, 控制台={console_level}, 文件={file_level}")

    return logger


def get_logger(name: str = "topo_opt") -> logging.Logger:

    #获取已存在的日志记录器

    #参数: name: 日志记录器名称

    #返回:日志记录器实例，如果不存在则返回根记录器
    logger = logging.getLogger(name)
    if not logger.handlers:
        # 如果尚未配置，返回默认配置的记录器
        return setup_logger(name)
    return logger


# 预定义的快捷日志函数（可选使用）
def log_info(message: str, name: str = "topo_opt"):
    # 记录INFO
    get_logger(name).info(message)


def log_debug(message: str, name: str = "topo_opt"):
    # 记录DEBUG
    get_logger(name).debug(message)


def log_warning(message: str, name: str = "topo_opt"):
    # 记录 WARNING
    get_logger(name).warning(message)


def log_error(message: str, name: str = "topo_opt", exc_info: bool = True):
    # 记录 ERROR
    get_logger(name).error(message, exc_info=exc_info)


def log_critical(message: str, name: str = "topo_opt", exc_info: bool = True):
    # 记录 CRITICAL
    get_logger(name).critical(message, exc_info=exc_info)


if __name__ == "__main__":
    # 初始化日志系统
    logger = setup_logger("test_logger")

    # 测试各等级日志
    logger.debug("DEBUG- 详细调试信息")
    logger.info("INFO - 常规运行信息")
    logger.warning("WARNING - 需要注意的情况")
    logger.error(" ERROR - 错误事件")

    # 测试异常日志
    try:
        raise ValueError("测试异常")
    except Exception:
        logger.error("发生异常", exc_info=True)

    print(f"\n日志文件已生成至：./logs/")
