"""
数据处理模块
Data Processing Module

提供数据预处理、增强、窗口管理、增量更新和异常检测功能。
"""

from .data_preprocessor import DataPreprocessor
from .data_augmentor import DataAugmentor
from .window_data_manager import WindowDataManager
from .incremental_data_updater import IncrementalDataUpdater
from .anomaly_detector import AnomalyDetector
from .data_manager import DeepLearningDataManager

# 为了向后兼容，添加DataProcessor别名
DataProcessor = DataPreprocessor

__all__ = [
    'DataPreprocessor',
    'DataProcessor',  # 别名
    'DataAugmentor',
    'WindowDataManager',
    'IncrementalDataUpdater',
    'AnomalyDetector',
    'DeepLearningDataManager'
]
