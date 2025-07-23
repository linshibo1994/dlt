"""
模型模块
Models Module

提供各种深度学习模型的实现，包括LSTM、Transformer、GAN等。
"""

# 导入基础模型类和配置
from .base_model import BaseModel, ModelConfig, ModelType, ModelStatus, ModelMetrics, TrainingConfig

# 导入深度学习模型
from .deep_learning_models import LSTMPredictor, TransformerPredictor, GANPredictor, EnsembleManager

# 导入模型注册表
from .model_registry import ModelRegistry, model_registry

# 为了兼容性，创建别名
BaseDeepPredictor = BaseModel
BaseDeepLearningModel = BaseModel

class ModelMetadata:
    """模型元数据"""
    def __init__(self, name: str, version: str = "1.0.0", description: str = ""):
        self.name = name
        self.version = version
        self.description = description

# 导出所有模型类
__all__ = [
    'BaseModel', 'ModelConfig', 'ModelType', 'ModelStatus', 'ModelMetrics', 'TrainingConfig',
    'LSTMPredictor', 'TransformerPredictor', 'GANPredictor', 'EnsembleManager',
    'ModelRegistry', 'model_registry',
    'BaseDeepPredictor', 'BaseDeepLearningModel', 'ModelMetadata'
]
