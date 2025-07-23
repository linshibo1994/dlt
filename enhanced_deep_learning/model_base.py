#!/usr/bin/env python3
"""
模型基类
定义所有深度学习模型的通用接口和共享功能
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import pickle
import joblib

from core_modules import logger_manager, config_manager
from .interfaces import PredictorInterface


class ModelMetadata:
    """模型元数据类"""
    
    def __init__(self, name: str, version: str = "1.0.0", author: str = "System", 
                 description: str = "", dependencies: List[str] = None):
        """
        初始化模型元数据
        
        Args:
            name: 模型名称
            version: 模型版本
            author: 作者信息
            description: 模型描述
            dependencies: 依赖列表
        """
        self.name = name
        self.version = version
        self.author = author
        self.description = description
        self.dependencies = dependencies or []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.performance_metrics = {}
        self.training_history = []
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'version': self.version,
            'author': self.author,
            'description': self.description,
            'dependencies': self.dependencies,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'performance_metrics': self.performance_metrics,
            'training_history': self.training_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """从字典创建"""
        metadata = cls(
            name=data['name'],
            version=data['version'],
            author=data['author'],
            description=data['description'],
            dependencies=data.get('dependencies', [])
        )
        
        if 'created_at' in data:
            metadata.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            metadata.updated_at = datetime.fromisoformat(data['updated_at'])
        
        metadata.performance_metrics = data.get('performance_metrics', {})
        metadata.training_history = data.get('training_history', [])
        
        return metadata


class BaseDeepLearningModel(PredictorInterface, ABC):
    """深度学习模型基类"""
    
    def __init__(self, config: Dict[str, Any] = None, metadata: ModelMetadata = None):
        """
        初始化基础模型
        
        Args:
            config: 模型配置
            metadata: 模型元数据
        """
        self.config = config or {}
        self.metadata = metadata or ModelMetadata(name=self.__class__.__name__)
        
        # 模型状态
        self.is_trained = False
        self.is_loaded = False
        self.training_start_time = None
        self.training_end_time = None
        
        # 性能监控
        self.performance_history = []
        self.resource_usage = {}
        
        # 模型保存路径
        self.model_dir = self.config.get('model_dir', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        logger_manager.info(f"初始化模型: {self.metadata.name}")
    
    @abstractmethod
    def _build_model(self) -> Any:
        """
        构建模型架构
        
        Returns:
            构建的模型对象
        """
        pass
    
    @abstractmethod
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练数据
        
        Args:
            data: 原始数据
            
        Returns:
            (X, y) 训练数据
        """
        pass
    
    @abstractmethod
    def _train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            X: 输入特征
            y: 目标标签
            
        Returns:
            训练结果
        """
        pass
    
    @abstractmethod
    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """
        模型预测
        
        Args:
            X: 输入特征
            
        Returns:
            预测结果
        """
        pass
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            data: 训练数据
            
        Returns:
            训练结果
        """
        try:
            logger_manager.info(f"开始训练模型: {self.metadata.name}")
            self.training_start_time = datetime.now()
            
            # 输入验证
            if not self._validate_input_data(data):
                raise ValueError("输入数据验证失败")
            
            # 准备数据
            X, y = self._prepare_data(data)
            
            # 训练模型
            training_result = self._train_model(X, y)
            
            # 更新状态
            self.is_trained = True
            self.training_end_time = datetime.now()
            
            # 记录训练历史
            training_record = {
                'timestamp': self.training_end_time.isoformat(),
                'duration': (self.training_end_time - self.training_start_time).total_seconds(),
                'data_size': len(data),
                'result': training_result
            }
            
            self.metadata.training_history.append(training_record)
            self.metadata.updated_at = self.training_end_time
            
            logger_manager.info(f"模型训练完成: {self.metadata.name}")
            
            return training_result
            
        except Exception as e:
            logger_manager.error(f"模型训练失败: {e}")
            raise
    
    def predict(self, data: pd.DataFrame, count: int = 1) -> List[Tuple[List[int], List[int]]]:
        """
        预测号码
        
        Args:
            data: 历史数据
            count: 预测数量
            
        Returns:
            预测结果列表
        """
        try:
            if not self.is_trained and not self.is_loaded:
                raise ValueError("模型未训练或加载")
            
            # 输入验证
            if not self._validate_input_data(data):
                raise ValueError("输入数据验证失败")
            
            # 准备数据
            X, _ = self._prepare_data(data)
            
            # 预测
            predictions = []
            for _ in range(count):
                pred_result = self._predict_model(X)
                front_balls, back_balls = self._convert_prediction_to_numbers(pred_result)
                predictions.append((front_balls, back_balls))
            
            logger_manager.info(f"预测完成: {len(predictions)} 注")
            
            return predictions
            
        except Exception as e:
            logger_manager.error(f"预测失败: {e}")
            return []
    
    def evaluate(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            data: 测试数据
            
        Returns:
            评估结果
        """
        try:
            if not self.is_trained and not self.is_loaded:
                raise ValueError("模型未训练或加载")
            
            # 准备数据
            X, y = self._prepare_data(data)
            
            # 预测
            y_pred = self._predict_model(X)
            
            # 计算评估指标
            metrics = self._calculate_metrics(y, y_pred)
            
            # 更新性能指标
            self.metadata.performance_metrics.update(metrics)
            self.performance_history.append({
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            })
            
            logger_manager.info(f"模型评估完成: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger_manager.error(f"模型评估失败: {e}")
            return {}
    
    def save_models(self) -> bool:
        """
        保存模型
        
        Returns:
            是否保存成功
        """
        try:
            model_path = os.path.join(self.model_dir, f"{self.metadata.name}_model.pkl")
            metadata_path = os.path.join(self.model_dir, f"{self.metadata.name}_metadata.json")
            
            # 保存模型
            self._save_model_file(model_path)
            
            # 保存元数据
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger_manager.info(f"模型保存成功: {model_path}")
            
            return True
            
        except Exception as e:
            logger_manager.error(f"模型保存失败: {e}")
            return False
    
    def load_models(self) -> bool:
        """
        加载模型
        
        Returns:
            是否加载成功
        """
        try:
            model_path = os.path.join(self.model_dir, f"{self.metadata.name}_model.pkl")
            metadata_path = os.path.join(self.model_dir, f"{self.metadata.name}_metadata.json")
            
            if not os.path.exists(model_path) or not os.path.exists(metadata_path):
                logger_manager.warning(f"模型文件不存在: {model_path}")
                return False
            
            # 加载模型
            self._load_model_file(model_path)
            
            # 加载元数据
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
                self.metadata = ModelMetadata.from_dict(metadata_dict)
            
            self.is_loaded = True
            logger_manager.info(f"模型加载成功: {model_path}")
            
            return True
            
        except Exception as e:
            logger_manager.error(f"模型加载失败: {e}")
            return False
    
    def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据"""
        if data is None or data.empty:
            return False
        
        required_columns = ['front_balls', 'back_balls']
        for col in required_columns:
            if col not in data.columns:
                return False
        
        return True
    
    def _validate_output(self, output: Any) -> bool:
        """验证输出数据"""
        # 子类可以重写此方法
        return output is not None
    
    def _validate_state(self) -> bool:
        """验证模型状态"""
        # 子类可以重写此方法
        return True
    
    def _convert_prediction_to_numbers(self, prediction: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        将预测结果转换为号码
        
        Args:
            prediction: 预测结果
            
        Returns:
            (前区号码, 后区号码)
        """
        # 默认实现，子类应该重写
        if len(prediction) >= 7:
            front_balls = sorted([max(1, min(35, int(round(x)))) for x in prediction[:5]])
            back_balls = sorted([max(1, min(12, int(round(x)))) for x in prediction[5:7]])
        else:
            front_balls = sorted([max(1, min(35, int(round(x)))) for x in prediction[:5]])
            back_balls = [1, 2]  # 默认值
        
        return front_balls, back_balls
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            评估指标
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        try:
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            
            return {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(np.sqrt(mse))
            }
        except Exception as e:
            logger_manager.error(f"计算指标失败: {e}")
            return {}
    
    @abstractmethod
    def _save_model_file(self, file_path: str):
        """保存模型文件"""
        pass
    
    @abstractmethod
    def _load_model_file(self, file_path: str):
        """加载模型文件"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'metadata': self.metadata.to_dict(),
            'is_trained': self.is_trained,
            'is_loaded': self.is_loaded,
            'performance_history': self.performance_history,
            'resource_usage': self.resource_usage
        }
