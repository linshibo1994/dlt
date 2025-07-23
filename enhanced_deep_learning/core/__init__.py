"""
核心模块
Core Module

提供核心配置管理、异常处理、依赖注入、日志系统等基础功能。
"""

from .config_manager import (
    ConfigManager, ConfigSection, ConfigValidator,
    config_manager
)
from .exception_handler import (
    ExceptionHandler, ErrorLevel, ErrorContext,
    exception_handler
)
from .dependency_injector import (
    DependencyInjector, ServiceRegistry, ServiceScope,
    dependency_injector
)
from .logger_manager import (
    LoggerManager, LogLevel, LogFormatter,
    logger_manager
)

__all__ = [
    # 配置管理器
    'ConfigManager',
    'ConfigSection',
    'ConfigValidator',
    'config_manager',
    
    # 异常处理器
    'ExceptionHandler',
    'ErrorLevel',
    'ErrorContext',
    'exception_handler',
    
    # 依赖注入器
    'DependencyInjector',
    'ServiceRegistry',
    'ServiceScope',
    'dependency_injector',
    
    # 日志管理器
    'LoggerManager',
    'LogLevel',
    'LogFormatter',
    'logger_manager'
]
