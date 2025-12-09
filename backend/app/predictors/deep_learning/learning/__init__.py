"""
学习模块
Learning Module

提供权重优化、重训练管理、性能跟踪和决策解释功能。
"""

from .weight_optimizer import WeightOptimizer
from .retrain_manager import RetrainManager, RetrainTrigger, RetrainTask
from .performance_tracker import PerformanceTracker
from .decision_explainer import DecisionExplainer
from .meta_learning import MetaLearningOptimizer

# 为了兼容性，创建RetrainingManager别名
RetrainingManager = RetrainManager

__all__ = [
    'WeightOptimizer',
    'RetrainManager',
    'RetrainTrigger', 
    'RetrainTask',
    'RetrainingManager',
    'PerformanceTracker',
    'DecisionExplainer',
    'MetaLearningOptimizer'
]
