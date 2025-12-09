"""
预测模块
Prediction Module

提供预测引擎、结果处理、置信度计算等功能。
"""

from .prediction_engine import (
    PredictionEngine, PredictionRequest, PredictionResult,
    PredictionStatus, prediction_engine
)
from .result_processor import (
    ResultProcessor, ResultFormat, ResultAnalyzer,
    result_processor
)
from .confidence_calculator import (
    ConfidenceCalculator, ConfidenceMethod, ConfidenceScore,
    confidence_calculator
)
from .prediction_cache import (
    PredictionCache, CacheStrategy, CacheEntry,
    prediction_cache
)

__all__ = [
    # 预测引擎
    'PredictionEngine',
    'PredictionRequest',
    'PredictionResult',
    'PredictionStatus',
    'prediction_engine',
    
    # 结果处理器
    'ResultProcessor',
    'ResultFormat',
    'ResultAnalyzer',
    'result_processor',
    
    # 置信度计算器
    'ConfidenceCalculator',
    'ConfidenceMethod',
    'ConfidenceScore',
    'confidence_calculator',
    
    # 预测缓存
    'PredictionCache',
    'CacheStrategy',
    'CacheEntry',
    'prediction_cache'
]
