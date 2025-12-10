# -*- coding: utf-8 -*-
"""
核心模块 - 数据管理、缓存管理、日志管理、任务管理
"""

from . import core_modules

# 导出核心类
from .core_modules import (
    DataManager,
    CacheManager,
    LoggerManager,
    TaskManager
)

# smart_cache_system 按需导入以避免循环依赖
