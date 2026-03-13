"""
模型模块
Models Module

提供各种深度学习模型的实现，包括LSTM、Transformer、GAN等。
"""

# 导入基础模型类和配置
from .base_model import BaseModel, ModelConfig, ModelType, ModelStatus, ModelMetrics, TrainingConfig

# 导入深度学习模型 - 使用具体的模块导入
from .lstm_predictor import LSTMPredictor
from .transformer_predictor import TransformerPredictor
from .gan_predictor import GANPredictor
from .ensemble_manager import EnsembleManager

# 导入模型注册表
try:
    from .model_registry import ModelRegistry, model_registry
except ImportError:
    # 如果导入失败，创建一个简单的替代类
    class ModelRegistry:
        def __init__(self):
            self.models = {}

        def register(self, name, model_class, version="1.0.0"):
            self.models[name] = {'class': model_class, 'version': version}

        def get(self, name):
            return self.models.get(name, {}).get('class')

        def list_models(self):
            return list(self.models.keys())

    model_registry = ModelRegistry()

def get_model_registry():
    """获取模型注册表实例"""
    return model_registry

def get_model_registry_class():
    """获取模型注册表类"""
    return ModelRegistry

# 为了兼容性，创建别名
BaseDeepPredictor = BaseModel
BaseDeepLearningModel = BaseModel

# 导入模型元数据
from .metadata import ModelMetadata

# 导出所有模型类
__all__ = [
    'BaseModel', 'ModelConfig', 'ModelType', 'ModelStatus', 'ModelMetrics', 'TrainingConfig',
    'LSTMPredictor', 'TransformerPredictor', 'GANPredictor', 'EnsembleManager',
    'ModelRegistry', 'model_registry', 'get_model_registry', 'get_model_registry_class',
    'BaseDeepPredictor', 'BaseDeepLearningModel', 'ModelMetadata'
]
