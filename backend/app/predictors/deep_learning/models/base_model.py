#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型基类模块
Base Model Module

定义所有深度学习模型的通用接口和基础功能。
"""

import os
import time
import json
import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime

from core_modules import logger_manager
from ..utils.exceptions import ModelException

# 导入ModelMetadata
try:
    from .metadata import ModelMetadata
except ImportError:
    # 如果导入失败，创建一个简单的ModelMetadata类
    class ModelMetadata:
        def __init__(self, name: str, version: str = "1.0.0", description: str = ""):
            self.name = name
            self.version = version
            self.description = description


class ModelType(Enum):
    """模型类型枚举"""
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    GAN = "gan"
    CNN = "cnn"
    RNN = "rnn"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"


class ModelStatus(Enum):
    """模型状态枚举"""
    INITIALIZED = "initialized"
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    PREDICTING = "predicting"
    ERROR = "error"


@dataclass
class ModelConfig:
    """模型配置"""
    model_type: ModelType
    model_name: str
    version: str = "1.0.0"
    description: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping: bool = True
    patience: int = 10
    save_best_only: bool = True
    verbose: int = 1
    callbacks: List[Any] = field(default_factory=list)


@dataclass
class ModelMetrics:
    """模型指标"""
    loss: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingHistory:
    """训练历史"""
    epoch: int
    train_loss: float
    val_loss: float
    train_metrics: ModelMetrics
    val_metrics: ModelMetrics
    timestamp: datetime = field(default_factory=datetime.now)


class BaseModel(ABC):
    """模型基类"""
    
    def __init__(self, config: ModelConfig):
        """
        初始化模型基类
        
        Args:
            config: 模型配置
        """
        self.config = config
        self.status = ModelStatus.INITIALIZED
        self.model = None
        self.training_history = []
        self.metrics = ModelMetrics()
        self.created_time = datetime.now()
        self.last_updated = datetime.now()
        
        logger_manager.info(f"初始化模型: {config.model_name} ({config.model_type.value})")
    
    @abstractmethod
    def build_model(self, input_shape: Tuple[int, ...]) -> Any:
        """
        构建模型
        
        Args:
            input_shape: 输入形状
            
        Returns:
            构建的模型
        """
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              config: Optional[TrainingConfig] = None) -> 'BaseModel':
        """
        训练模型
        
        Args:
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据
            y_val: 验证标签
            config: 训练配置
            
        Returns:
            训练后的模型
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 输入数据
            
        Returns:
            预测结果
        """
        pass
    
    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> ModelMetrics:
        """
        评估模型
        
        Args:
            X_test: 测试数据
            y_test: 测试标签
            
        Returns:
            评估指标
        """
        pass
    
    def save_model(self, file_path: str) -> bool:
        """
        保存模型
        
        Args:
            file_path: 保存路径
            
        Returns:
            是否保存成功
        """
        try:
            # 创建保存目录
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 保存模型数据
            model_data = {
                'config': {
                    'model_type': self.config.model_type.value,
                    'model_name': self.config.model_name,
                    'version': self.config.version,
                    'description': self.config.description,
                    'hyperparameters': self.config.hyperparameters,
                    'metadata': self.config.metadata
                },
                'status': self.status.value,
                'metrics': {
                    'loss': self.metrics.loss,
                    'accuracy': self.metrics.accuracy,
                    'precision': self.metrics.precision,
                    'recall': self.metrics.recall,
                    'f1_score': self.metrics.f1_score,
                    'custom_metrics': self.metrics.custom_metrics
                },
                'training_history': [
                    {
                        'epoch': h.epoch,
                        'train_loss': h.train_loss,
                        'val_loss': h.val_loss,
                        'timestamp': h.timestamp.isoformat()
                    } for h in self.training_history
                ],
                'created_time': self.created_time.isoformat(),
                'last_updated': self.last_updated.isoformat()
            }
            
            # 保存模型文件
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'model_data': model_data,
                    'model': self.model
                }, f)
            
            logger_manager.info(f"模型保存成功: {file_path}")
            return True
            
        except Exception as e:
            logger_manager.error(f"保存模型失败: {e}")
            return False
    
    def load_model(self, file_path: str) -> bool:
        """
        加载模型
        
        Args:
            file_path: 模型文件路径
            
        Returns:
            是否加载成功
        """
        try:
            if not os.path.exists(file_path):
                raise ModelException(f"模型文件不存在: {file_path}")
            
            # 加载模型文件
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            model_data = data['model_data']
            self.model = data['model']
            
            # 恢复配置
            config_data = model_data['config']
            self.config = ModelConfig(
                model_type=ModelType(config_data['model_type']),
                model_name=config_data['model_name'],
                version=config_data['version'],
                description=config_data['description'],
                hyperparameters=config_data['hyperparameters'],
                metadata=config_data['metadata']
            )
            
            # 恢复状态
            self.status = ModelStatus(model_data['status'])
            
            # 恢复指标
            metrics_data = model_data['metrics']
            self.metrics = ModelMetrics(
                loss=metrics_data['loss'],
                accuracy=metrics_data['accuracy'],
                precision=metrics_data['precision'],
                recall=metrics_data['recall'],
                f1_score=metrics_data['f1_score'],
                custom_metrics=metrics_data['custom_metrics']
            )
            
            # 恢复时间信息
            self.created_time = datetime.fromisoformat(model_data['created_time'])
            self.last_updated = datetime.fromisoformat(model_data['last_updated'])
            
            logger_manager.info(f"模型加载成功: {file_path}")
            return True
            
        except Exception as e:
            logger_manager.error(f"加载模型失败: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_name': self.config.model_name,
            'model_type': self.config.model_type.value,
            'version': self.config.version,
            'description': self.config.description,
            'status': self.status.value,
            'metrics': {
                'loss': self.metrics.loss,
                'accuracy': self.metrics.accuracy,
                'precision': self.metrics.precision,
                'recall': self.metrics.recall,
                'f1_score': self.metrics.f1_score
            },
            'hyperparameters': self.config.hyperparameters,
            'created_time': self.created_time.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'training_epochs': len(self.training_history)
        }
    
    def add_training_history(self, history: TrainingHistory):
        """添加训练历史"""
        self.training_history.append(history)
        self.last_updated = datetime.now()
    
    def get_training_history(self) -> List[TrainingHistory]:
        """获取训练历史"""
        return self.training_history.copy()
    
    def reset_model(self):
        """重置模型"""
        try:
            self.model = None
            self.status = ModelStatus.INITIALIZED
            self.training_history.clear()
            self.metrics = ModelMetrics()
            self.last_updated = datetime.now()
            
            logger_manager.info(f"模型重置完成: {self.config.model_name}")
            
        except Exception as e:
            logger_manager.error(f"重置模型失败: {e}")
    
    def clone_model(self, new_name: str) -> 'BaseModel':
        """
        克隆模型
        
        Args:
            new_name: 新模型名称
            
        Returns:
            克隆的模型
        """
        try:
            # 创建新配置
            new_config = ModelConfig(
                model_type=self.config.model_type,
                model_name=new_name,
                version=self.config.version,
                description=f"Cloned from {self.config.model_name}",
                hyperparameters=self.config.hyperparameters.copy(),
                metadata=self.config.metadata.copy()
            )
            
            # 创建新模型实例（需要子类实现）
            cloned_model = self.__class__(new_config)
            
            # 如果原模型已训练，复制模型结构
            if self.model is not None:
                cloned_model.model = self._clone_model_structure()
            
            logger_manager.info(f"模型克隆完成: {new_name}")
            return cloned_model
            
        except Exception as e:
            logger_manager.error(f"克隆模型失败: {e}")
            raise ModelException(f"克隆模型失败: {e}")
    
    def _clone_model_structure(self) -> Any:
        """克隆模型结构（子类可重写）"""
        # 默认实现：返回None，子类应该重写此方法
        return None
    
    def update_config(self, **kwargs):
        """更新配置"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                elif key in ['hyperparameters', 'metadata']:
                    getattr(self.config, key).update(value)
            
            self.last_updated = datetime.now()
            logger_manager.debug(f"模型配置已更新: {self.config.model_name}")
            
        except Exception as e:
            logger_manager.error(f"更新模型配置失败: {e}")
    
    def validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> bool:
        """
        验证输入数据
        
        Args:
            X: 输入数据
            y: 标签数据
            
        Returns:
            是否有效
        """
        try:
            # 基本验证
            if not isinstance(X, np.ndarray):
                raise ModelException("输入数据必须是numpy数组")
            
            if len(X.shape) < 2:
                raise ModelException("输入数据至少需要2维")
            
            if y is not None:
                if not isinstance(y, np.ndarray):
                    raise ModelException("标签数据必须是numpy数组")
                
                if len(X) != len(y):
                    raise ModelException("输入数据和标签数据长度不匹配")
            
            return True
            
        except Exception as e:
            logger_manager.error(f"输入验证失败: {e}")
            return False
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.config.model_name} ({self.config.model_type.value}) - {self.status.value}"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"BaseModel(name='{self.config.model_name}', "
                f"type='{self.config.model_type.value}', "
                f"status='{self.status.value}', "
                f"accuracy={self.metrics.accuracy:.4f})")


if __name__ == "__main__":
    # 测试模型基类功能
    print("🧠 测试模型基类功能...")
    
    try:
        # 创建测试配置
        config = ModelConfig(
            model_type=ModelType.LSTM,
            model_name="test_model",
            version="1.0.0",
            description="测试模型",
            hyperparameters={'units': 64, 'dropout': 0.2}
        )
        
        # 由于BaseModel是抽象类，我们创建一个简单的实现用于测试
        class TestModel(BaseModel):
            def build_model(self, input_shape):
                return "test_model"
            
            def train(self, X_train, y_train, X_val=None, y_val=None, config=None):
                self.status = ModelStatus.TRAINED
                self.metrics.accuracy = 0.85
                return self
            
            def predict(self, X):
                return np.random.random((len(X), 1))
            
            def evaluate(self, X_test, y_test):
                return ModelMetrics(accuracy=0.85, loss=0.15)
        
        # 测试模型创建
        model = TestModel(config)
        print(f"✅ 模型创建成功: {model}")
        
        # 测试模型信息
        info = model.get_model_info()
        print(f"✅ 模型信息获取成功: {info['model_name']}")
        
        # 测试配置更新
        model.update_config(description="更新后的测试模型")
        print("✅ 模型配置更新成功")
        
        # 测试输入验证
        test_X = np.random.random((100, 10))
        test_y = np.random.random((100, 1))
        
        if model.validate_input(test_X, test_y):
            print("✅ 输入验证通过")
        
        print("✅ 模型基类功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("模型基类功能测试完成")
