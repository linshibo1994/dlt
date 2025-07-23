#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
深度学习增强模块
Enhanced Deep Learning Module

提供深度学习预测、优化和管理功能的完整解决方案。

项目结构：
- models/: 深度学习模型实现
- data/: 数据处理和管理
- optimization/: 性能优化功能
- learning/: 学习和训练管理
- cli/: 命令行接口
- utils/: 工具和配置
"""

# 模型模块
from .models import (
    BaseDeepPredictor, BaseDeepLearningModel, ModelMetadata,
    ModelRegistry, model_registry,
    LSTMPredictor, TransformerPredictor, GANPredictor, EnsembleManager
)

# 数据处理模块
from .data import (
    DataPreprocessor, DataAugmentor, WindowDataManager,
    IncrementalDataUpdater, AnomalyDetector, DeepLearningDataManager
)

# 优化模块
from .optimization import (
    GPUAccelerator, HardwareAccelerator, AcceleratorType, hardware_accelerator,
    BatchProcessor, ModelOptimizer, ResourceMonitor, resource_monitor
)

# 学习模块
from .learning import (
    WeightOptimizer, RetrainManager, RetrainTrigger, RetrainTask,
    RetrainingManager, PerformanceTracker, DecisionExplainer, MetaLearningOptimizer
)

# CLI模块
from .cli import CommandDefinition, CommandHandler

# 工具模块
from .utils import (
    ConfigManager, DeepLearningException, ModelException, DataException,
    IModel, IDataProcessor, IOptimizer, ILearner, ICommandHandler,
    PredictionUtils, TrainingUtils
)

# 系统集成
from .integration import SystemIntegrator as SystemIntegration

# 版本信息
__version__ = "2.0.0"
__author__ = "Deep Learning Team"

# 导出列表
__all__ = [
    # 模型模块
    'BaseDeepPredictor', 'BaseDeepLearningModel', 'ModelMetadata',
    'ModelRegistry', 'model_registry',
    'LSTMPredictor', 'TransformerPredictor', 'GANPredictor', 'EnsembleManager',

    # 数据处理模块
    'DataPreprocessor', 'DataAugmentor', 'WindowDataManager',
    'IncrementalDataUpdater', 'AnomalyDetector', 'DeepLearningDataManager',

    # 优化模块
    'GPUAccelerator', 'HardwareAccelerator', 'AcceleratorType', 'hardware_accelerator',
    'BatchProcessor', 'ModelOptimizer', 'ResourceMonitor', 'resource_monitor',

    # 学习模块
    'WeightOptimizer', 'RetrainManager', 'RetrainTrigger', 'RetrainTask',
    'RetrainingManager', 'PerformanceTracker', 'DecisionExplainer', 'MetaLearningOptimizer',

    # CLI模块
    'CommandDefinition', 'CommandHandler',

    # 工具模块
    'ConfigManager', 'DeepLearningException', 'ModelException', 'DataException',
    'IModel', 'IDataProcessor', 'IOptimizer', 'ILearner', 'ICommandHandler',
    'PredictionUtils', 'TrainingUtils',

    # 系统集成
    'SystemIntegration',
]