#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
日志管理器模块
Logger Manager Module

提供统一的日志管理、格式化、输出控制等功能。
"""

import os
import sys
import logging
import logging.handlers
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import threading
from datetime import datetime

from ..utils.exceptions import DeepLearningException


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LogFormatter(logging.Formatter):
    """自定义日志格式化器"""
    
    def __init__(self, use_colors: bool = True):
        """
        初始化日志格式化器
        
        Args:
            use_colors: 是否使用颜色
        """
        self.use_colors = use_colors
        
        # 颜色代码
        self.colors = {
            'DEBUG': '\033[36m',      # 青色
            'INFO': '\033[32m',       # 绿色
            'WARNING': '\033[33m',    # 黄色
            'ERROR': '\033[31m',      # 红色
            'CRITICAL': '\033[35m',   # 紫色
            'RESET': '\033[0m'        # 重置
        }
        
        # 基础格式
        self.base_format = "%(asctime)s - %(levelname)s - %(message)s"
        self.detailed_format = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        
        super().__init__(self.base_format, datefmt='%H:%M:%S')
    
    def format(self, record):
        """格式化日志记录"""
        # 选择格式
        if record.levelno >= logging.ERROR:
            self._style._fmt = self.detailed_format
        else:
            self._style._fmt = self.base_format
        
        # 格式化消息
        formatted = super().format(record)
        
        # 添加颜色
        if self.use_colors and sys.stderr.isatty():
            level_name = record.levelname
            if level_name in self.colors:
                color = self.colors[level_name]
                reset = self.colors['RESET']
                formatted = f"{color}{formatted}{reset}"
        
        return formatted


@dataclass
class LoggerConfig:
    """日志配置"""
    name: str = "deep_learning"
    level: LogLevel = LogLevel.INFO
    use_colors: bool = True
    log_to_file: bool = True
    log_file: str = "logs/deep_learning.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    use_console: bool = True


class LoggerManager:
    """日志管理器"""
    
    def __init__(self, config: LoggerConfig = None):
        """
        初始化日志管理器
        
        Args:
            config: 日志配置
        """
        self.config = config or LoggerConfig()
        self.loggers: Dict[str, logging.Logger] = {}
        self._lock = threading.RLock()
        
        # 创建主日志器
        self.main_logger = self._create_logger(self.config.name)
        
        # 设置为全局日志器
        self._setup_global_logger()
    
    def _create_logger(self, name: str) -> logging.Logger:
        """创建日志器"""
        try:
            with self._lock:
                if name in self.loggers:
                    return self.loggers[name]
                
                # 创建日志器
                logger = logging.getLogger(name)
                logger.setLevel(self.config.level.value)
                
                # 清除现有处理器
                logger.handlers.clear()
                
                # 创建格式化器
                formatter = LogFormatter(self.config.use_colors)
                
                # 控制台处理器
                if self.config.use_console:
                    console_handler = logging.StreamHandler(sys.stderr)
                    console_handler.setLevel(self.config.level.value)
                    console_handler.setFormatter(formatter)
                    logger.addHandler(console_handler)
                
                # 文件处理器
                if self.config.log_to_file:
                    # 确保日志目录存在
                    log_file_path = Path(self.config.log_file)
                    log_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 创建轮转文件处理器
                    file_handler = logging.handlers.RotatingFileHandler(
                        self.config.log_file,
                        maxBytes=self.config.max_file_size,
                        backupCount=self.config.backup_count,
                        encoding='utf-8'
                    )
                    file_handler.setLevel(self.config.level.value)
                    
                    # 文件格式化器（不使用颜色）
                    file_formatter = LogFormatter(use_colors=False)
                    file_handler.setFormatter(file_formatter)
                    logger.addHandler(file_handler)
                
                # 防止重复日志
                logger.propagate = False
                
                self.loggers[name] = logger
                return logger
                
        except Exception as e:
            print(f"创建日志器失败: {e}")
            # 返回基础日志器
            return logging.getLogger(name)
    
    def _setup_global_logger(self):
        """设置全局日志器"""
        # 将主要方法绑定到实例
        self.debug = self.main_logger.debug
        self.info = self.main_logger.info
        self.warning = self.main_logger.warning
        self.error = self.main_logger.error
        self.critical = self.main_logger.critical
        self.exception = self.main_logger.exception
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        获取日志器
        
        Args:
            name: 日志器名称
            
        Returns:
            日志器实例
        """
        return self._create_logger(name)
    
    def set_level(self, level: LogLevel):
        """
        设置日志级别
        
        Args:
            level: 日志级别
        """
        try:
            with self._lock:
                self.config.level = level
                
                # 更新所有日志器的级别
                for logger in self.loggers.values():
                    logger.setLevel(level.value)
                    
                    # 更新处理器级别
                    for handler in logger.handlers:
                        handler.setLevel(level.value)
                        
        except Exception as e:
            print(f"设置日志级别失败: {e}")
    
    def get_level(self) -> str:
        """获取当前日志级别"""
        return self.config.level.name
    
    def add_file_handler(self, log_file: str, level: LogLevel = None):
        """
        添加文件处理器
        
        Args:
            log_file: 日志文件路径
            level: 日志级别
        """
        try:
            level = level or self.config.level
            
            # 确保日志目录存在
            log_file_path = Path(log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 创建文件处理器
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(level.value)
            
            # 设置格式化器
            formatter = LogFormatter(use_colors=False)
            file_handler.setFormatter(formatter)
            
            # 添加到主日志器
            self.main_logger.addHandler(file_handler)
            
        except Exception as e:
            print(f"添加文件处理器失败: {e}")
    
    def remove_handlers(self):
        """移除所有处理器"""
        try:
            with self._lock:
                for logger in self.loggers.values():
                    for handler in logger.handlers[:]:
                        logger.removeHandler(handler)
                        handler.close()
                        
        except Exception as e:
            print(f"移除处理器失败: {e}")
    
    def log_system_info(self):
        """记录系统信息"""
        try:
            import platform
            import psutil
            
            self.info("=" * 50)
            self.info("系统信息:")
            self.info(f"操作系统: {platform.system()} {platform.release()}")
            self.info(f"Python版本: {platform.python_version()}")
            self.info(f"CPU核心数: {psutil.cpu_count()}")
            self.info(f"内存总量: {psutil.virtual_memory().total / (1024**3):.1f} GB")
            self.info("=" * 50)
            
        except Exception as e:
            self.error(f"记录系统信息失败: {e}")
    
    def create_context_logger(self, context: str) -> logging.Logger:
        """
        创建上下文日志器
        
        Args:
            context: 上下文名称
            
        Returns:
            上下文日志器
        """
        context_name = f"{self.config.name}.{context}"
        return self.get_logger(context_name)
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """
        记录性能信息
        
        Args:
            operation: 操作名称
            duration: 持续时间（秒）
            **kwargs: 额外信息
        """
        try:
            perf_info = f"性能: {operation} 耗时 {duration:.3f}s"
            
            if kwargs:
                extra_info = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                perf_info += f" ({extra_info})"
            
            self.info(perf_info)
            
        except Exception as e:
            self.error(f"记录性能信息失败: {e}")
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """获取日志统计信息"""
        try:
            stats = {
                "loggers_count": len(self.loggers),
                "current_level": self.config.level.name,
                "log_to_file": self.config.log_to_file,
                "log_file": self.config.log_file if self.config.log_to_file else None,
                "handlers_count": len(self.main_logger.handlers)
            }
            
            # 文件大小信息
            if self.config.log_to_file and Path(self.config.log_file).exists():
                file_size = Path(self.config.log_file).stat().st_size
                stats["log_file_size"] = f"{file_size / 1024:.1f} KB"
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}


# 创建全局日志管理器实例
logger_manager = LoggerManager()


if __name__ == "__main__":
    # 测试日志管理器功能
    print("📝 测试日志管理器功能...")
    
    try:
        # 创建日志管理器
        config = LoggerConfig(
            name="test_logger",
            level=LogLevel.DEBUG,
            log_file="test_logs/test.log"
        )
        
        manager = LoggerManager(config)
        
        # 测试不同级别的日志
        manager.debug("这是调试信息")
        manager.info("这是信息日志")
        manager.warning("这是警告信息")
        manager.error("这是错误信息")
        
        # 测试上下文日志器
        context_logger = manager.create_context_logger("test_context")
        context_logger.info("这是上下文日志")
        
        # 测试性能日志
        manager.log_performance("test_operation", 0.123, items=100, success=True)
        
        # 获取统计信息
        stats = manager.get_log_statistics()
        print(f"✅ 日志统计: {stats}")
        
        print("✅ 日志管理器功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("日志管理器功能测试完成")
