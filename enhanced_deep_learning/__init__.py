#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强深度学习模块
提供高级神经网络架构、集成学习和元学习框架
"""

from .base import BaseDeepPredictor
from .transformer_predictor import TransformerPredictor
from .gan_predictor import GANPredictor
from .ensemble_manager import EnsembleManager
from .meta_learning import MetaLearningOptimizer
from .data_manager import DeepLearningDataManager
from .performance_optimizer import PerformanceOptimizer
from .cli_integration import DeepLearningCommands

__all__ = [
    'BaseDeepPredictor',
    'TransformerPredictor',
    'GANPredictor',
    'EnsembleManager',
    'MetaLearningOptimizer',
    'DeepLearningDataManager',
    'PerformanceOptimizer',
    'DeepLearningCommands'
]