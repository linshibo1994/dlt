#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
核心模块包
Core Modules Package

提供系统核心功能模块的统一导入接口。
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 从core_modules.py导入核心管理器
try:
    import core_modules as cm
    logger_manager = cm.logger_manager
    data_manager = cm.data_manager
    cache_manager = cm.cache_manager
    task_manager = cm.task_manager
except ImportError as e:
    print(f"Warning: Failed to import core modules: {e}")
    logger_manager = None
    data_manager = None
    cache_manager = None
    task_manager = None

# 导入复式预测功能
from .compound_predictor import (
    CompoundPredictorMixin,
    CompoundPredictorBase,
    CompoundConfig,
    CompoundResult
)

__all__ = [
    # 核心管理器
    'logger_manager',
    'data_manager',
    'cache_manager',
    'task_manager',

    # 复式预测
    'CompoundPredictorMixin',
    'CompoundPredictorBase',
    'CompoundConfig',
    'CompoundResult'
]

__version__ = '1.0.0'
__author__ = 'DLT Prediction System'
