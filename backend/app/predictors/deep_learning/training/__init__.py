#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练模块包
Training Module Package

提供智能训练优化功能。
"""

from .smart_epochs_calculator import (
    SmartEpochsCalculator,
    TrainingConfig,
    EpochsRecommendation,
    TrainingMonitor,
    AdaptiveTrainingAdjuster,
    ModelType,
    PerformanceMode
)

__all__ = [
    'SmartEpochsCalculator',
    'TrainingConfig', 
    'EpochsRecommendation',
    'TrainingMonitor',
    'AdaptiveTrainingAdjuster',
    'ModelType',
    'PerformanceMode'
]
