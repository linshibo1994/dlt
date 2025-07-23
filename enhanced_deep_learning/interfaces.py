#!/usr/bin/env python3
"""
接口定义
定义深度学习模块的标准接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np


class PredictorInterface(ABC):
    """预测器接口"""
    
    @abstractmethod
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            data: 训练数据
            
        Returns:
            训练结果
        """
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame, count: int = 1) -> List[Tuple[List[int], List[int]]]:
        """
        预测号码
        
        Args:
            data: 历史数据
            count: 预测数量
            
        Returns:
            预测结果列表
        """
        pass
    
    @abstractmethod
    def save_models(self) -> bool:
        """
        保存模型
        
        Returns:
            是否保存成功
        """
        pass
    
    @abstractmethod
    def load_models(self) -> bool:
        """
        加载模型
        
        Returns:
            是否加载成功
        """
        pass
    
    @abstractmethod
    def evaluate(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            data: 测试数据
            
        Returns:
            评估结果
        """
        pass


class AnalyzerInterface(ABC):
    """分析器接口"""
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析数据
        
        Args:
            data: 输入数据
            
        Returns:
            分析结果
        """
        pass


class OptimizerInterface(ABC):
    """优化器接口"""
    
    @abstractmethod
    def optimize(self, objective_function, parameter_space: Dict[str, Any], 
                data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        执行优化
        
        Args:
            objective_function: 目标函数
            parameter_space: 参数空间
            data: 数据
            
        Returns:
            优化结果
        """
        pass


class DataManagerInterface(ABC):
    """数据管理器接口"""
    
    @abstractmethod
    def load_data(self, source: str = None) -> pd.DataFrame:
        """
        加载数据
        
        Args:
            source: 数据源
            
        Returns:
            数据DataFrame
        """
        pass
    
    @abstractmethod
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        预处理数据
        
        Args:
            data: 原始数据
            
        Returns:
            预处理后的数据
        """
        pass
    
    @abstractmethod
    def save_data(self, data: pd.DataFrame, file_path: str) -> bool:
        """
        保存数据
        
        Args:
            data: 数据
            file_path: 文件路径
            
        Returns:
            是否保存成功
        """
        pass


class EnsembleInterface(ABC):
    """集成学习接口"""
    
    @abstractmethod
    def add_model(self, model_name: str, model, weight: float = 1.0) -> bool:
        """
        添加模型
        
        Args:
            model_name: 模型名称
            model: 模型实例
            weight: 模型权重
            
        Returns:
            是否添加成功
        """
        pass
    
    @abstractmethod
    def remove_model(self, model_name: str) -> bool:
        """
        移除模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            是否移除成功
        """
        pass
    
    @abstractmethod
    def predict_ensemble(self, data: pd.DataFrame, count: int = 1) -> List[Tuple[List[int], List[int]]]:
        """
        集成预测
        
        Args:
            data: 输入数据
            count: 预测数量
            
        Returns:
            预测结果
        """
        pass
    
    @abstractmethod
    def update_weights(self, performance_scores: Dict[str, float]) -> bool:
        """
        更新模型权重
        
        Args:
            performance_scores: 性能分数
            
        Returns:
            是否更新成功
        """
        pass


class MonitorInterface(ABC):
    """监控器接口"""
    
    @abstractmethod
    def start_monitoring(self) -> bool:
        """
        开始监控
        
        Returns:
            是否启动成功
        """
        pass
    
    @abstractmethod
    def stop_monitoring(self) -> bool:
        """
        停止监控
        
        Returns:
            是否停止成功
        """
        pass
    
    @abstractmethod
    def get_current_status(self) -> Dict[str, Any]:
        """
        获取当前状态
        
        Returns:
            状态信息
        """
        pass


class ProcessorInterface(ABC):
    """处理器接口"""
    
    @abstractmethod
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        处理数据
        
        Args:
            data: 输入数据
            
        Returns:
            处理后的数据
        """
        pass
    
    @abstractmethod
    def batch_process(self, data_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """
        批量处理数据
        
        Args:
            data_list: 数据列表
            
        Returns:
            处理后的数据列表
        """
        pass


class CacheInterface(ABC):
    """缓存接口"""
    
    @abstractmethod
    def get(self, key: str) -> Any:
        """
        获取缓存
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值
        """
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """
        设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）
            
        Returns:
            是否设置成功
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        删除缓存
        
        Args:
            key: 缓存键
            
        Returns:
            是否删除成功
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """
        清空缓存
        
        Returns:
            是否清空成功
        """
        pass


class ConfigInterface(ABC):
    """配置接口"""
    
    @abstractmethod
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        获取配置
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            配置值
        """
        pass
    
    @abstractmethod
    def set_config(self, key: str, value: Any) -> bool:
        """
        设置配置
        
        Args:
            key: 配置键
            value: 配置值
            
        Returns:
            是否设置成功
        """
        pass
    
    @abstractmethod
    def save_config(self, file_path: str = None) -> bool:
        """
        保存配置
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否保存成功
        """
        pass
    
    @abstractmethod
    def load_config(self, file_path: str) -> bool:
        """
        加载配置
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否加载成功
        """
        pass
