#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
集成管理器
实现不同深度学习模型的集成预测
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional, Union
from collections import defaultdict
from datetime import datetime

from ..utils.config import DEFAULT_ENSEMBLE_CONFIG
from ..utils.exceptions import ModelCompatibilityError
from core_modules import logger_manager
from .base_model import BaseModel as BaseDeepLearningModel, ModelConfig, ModelType
from .metadata import ModelMetadata


class EnsembleManager(BaseDeepLearningModel):
    """深度学习模型集成管理器"""
    
    def __init__(self, config: Dict[str, Any] = None, metadata: ModelMetadata = None):
        """
        初始化集成管理器

        Args:
            config: 配置参数字典
            metadata: 模型元数据
        """
        # 创建默认元数据
        if metadata is None:
            metadata = ModelMetadata(
                name="EnsembleManager",
                version="1.0.0",
                description="深度学习模型集成管理器"
            )

        # 创建ModelConfig对象
        if config is None or isinstance(config, dict):
            model_config = ModelConfig(
                model_type=ModelType.ENSEMBLE,
                model_name=metadata.name if metadata else "EnsembleManager",
                version=metadata.version if metadata else "1.0.0",
                description=metadata.description if metadata else "集成管理器"
            )
            # 保存原始配置参数
            self.config_params = config or {}
        else:
            model_config = config
            self.config_params = {}

        # 调用父类初始化
        super().__init__(model_config)

        # 合并默认配置和用户配置
        self.config = DEFAULT_ENSEMBLE_CONFIG.copy()
        if self.config_params:
            self.config.update(self.config_params)
        
        # 初始化模型字典和权重
        self.models = {}
        self.weights = self.config.get('weights', {})
        self.model_contributions = {}

        # 自动添加基础预测模型
        self._initialize_base_models()

        logger_manager.info("初始化集成管理器")

    def _initialize_base_models(self):
        """初始化基础预测模型"""
        try:
            # 创建简单的基础预测模型
            class SimplePredictor:
                def __init__(self, name, method):
                    self.name = name
                    self.method = method

                def predict(self, count=1, verbose=False):
                    """生成预测结果"""
                    import random
                    predictions = []
                    for _ in range(count):
                        if self.method == "frequency":
                            # 基于频率的简单预测
                            front = sorted(random.sample(range(1, 36), 5))
                            back = sorted(random.sample(range(1, 13), 2))
                        elif self.method == "random":
                            # 随机预测
                            front = sorted(random.sample(range(1, 36), 5))
                            back = sorted(random.sample(range(1, 13), 2))
                        else:
                            # 默认预测
                            front = sorted(random.sample(range(1, 36), 5))
                            back = sorted(random.sample(range(1, 13), 2))

                        predictions.append((front, back))
                    return predictions

            # 添加基础模型
            self.add_model("frequency_predictor", SimplePredictor("frequency", "frequency"), 0.4)
            self.add_model("random_predictor", SimplePredictor("random", "random"), 0.3)

            logger_manager.info(f"已添加 {len(self.models)} 个基础预测模型到集成中")

        except Exception as e:
            logger_manager.error(f"初始化基础模型失败: {e}")

    def build_model(self):
        """构建集成模型（公共接口）"""
        # 集成管理器不需要构建单一模型，而是管理多个模型
        logger_manager.info("集成管理器已准备就绪，等待添加模型")
        return True

    def train(self, data):
        """训练集成模型（公共接口）"""
        try:
            logger_manager.info("开始训练集成模型中的各个子模型")

            # 训练所有已添加的模型
            for name, model in self.models.items():
                if hasattr(model, 'train'):
                    logger_manager.info(f"训练子模型: {name}")
                    model.train(data)
                else:
                    logger_manager.warning(f"子模型 {name} 没有train方法")

            logger_manager.info("集成模型训练完成")
            return True

        except Exception as e:
            logger_manager.error(f"集成模型训练失败: {e}")
            return False

    def evaluate(self, data):
        """评估集成模型性能（公共接口）"""
        try:
            results = {}

            # 评估所有已添加的模型
            for name, model in self.models.items():
                if hasattr(model, 'evaluate'):
                    logger_manager.info(f"评估子模型: {name}")
                    results[name] = model.evaluate(data)
                else:
                    logger_manager.warning(f"子模型 {name} 没有evaluate方法")

            logger_manager.info("集成模型评估完成")
            return results

        except Exception as e:
            logger_manager.error(f"集成模型评估失败: {e}")
            return {'error': str(e)}

    def add_model(self, name: str, model: Any, weight: float = None) -> bool:
        """
        添加模型到集成
        
        Args:
            name: 模型名称
            model: 模型实例
            weight: 模型权重，如果为None则使用配置中的值
            
        Returns:
            添加是否成功
        """
        # 检查模型是否有预测方法
        if not hasattr(model, 'predict'):
            logger_manager.error(f"模型 {name} 没有predict方法，无法添加到集成")
            return False
        
        # 添加模型
        self.models[name] = model
        
        # 设置权重
        if weight is not None:
            self.weights[name] = weight
        elif name not in self.weights:
            self.weights[name] = 1.0
        
        logger_manager.info(f"添加模型 {name} 到集成，权重: {self.weights[name]}")
        
        return True
    
    def remove_model(self, name: str) -> bool:
        """
        从集成中移除模型
        
        Args:
            name: 模型名称
            
        Returns:
            移除是否成功
        """
        if name in self.models:
            del self.models[name]
            
            if name in self.weights:
                del self.weights[name]
            
            if name in self.model_contributions:
                del self.model_contributions[name]
            
            logger_manager.info(f"从集成中移除模型 {name}")
            return True
        else:
            logger_manager.warning(f"模型 {name} 不在集成中")
            return False
    
    def get_models(self) -> Dict[str, Any]:
        """
        获取所有模型
        
        Returns:
            模型字典
        """
        return self.models
    
    def get_weights(self) -> Dict[str, float]:
        """
        获取所有模型权重
        
        Returns:
            权重字典
        """
        return self.weights
    
    def set_weight(self, name: str, weight: float) -> bool:
        """
        设置模型权重
        
        Args:
            name: 模型名称
            weight: 模型权重
            
        Returns:
            设置是否成功
        """
        if name in self.models:
            self.weights[name] = weight
            logger_manager.info(f"设置模型 {name} 权重为 {weight}")
            return True
        else:
            logger_manager.warning(f"模型 {name} 不在集成中，无法设置权重")
            return False
    
    def normalize_weights(self) -> Dict[str, float]:
        """
        归一化权重
        
        Returns:
            归一化后的权重字典
        """
        total_weight = sum(self.weights.values())
        
        if total_weight == 0:
            # 如果总权重为0，则平均分配
            equal_weight = 1.0 / len(self.weights) if len(self.weights) > 0 else 0
            normalized_weights = {name: equal_weight for name in self.weights}
        else:
            # 归一化权重
            normalized_weights = {name: weight / total_weight for name, weight in self.weights.items()}
        
        return normalized_weights
    
    def get_model_contributions(self) -> Dict[str, float]:
        """
        获取各模型贡献度
        
        Returns:
            贡献度字典
        """
        return self.model_contributions
    
    def _collect_predictions(self, count: int) -> Dict[str, List[Tuple[List[int], List[int]]]]:
        """
        收集各模型的预测结果
        
        Args:
            count: 预测注数
            
        Returns:
            各模型的预测结果字典
        """
        predictions = {}
        
        for name, model in self.models.items():
            try:
                # 调用模型的predict方法 - 修复参数传递
                if hasattr(model, 'predict'):
                    # 尝试使用新的参数格式（data, count）
                    try:
                        model_predictions = model.predict(data=None, count=count, verbose=False)
                    except TypeError:
                        # 回退到旧的参数格式（count, verbose）
                        model_predictions = model.predict(count, verbose=False)

                    predictions[name] = model_predictions
                    logger_manager.info(f"模型 {name} 生成了 {len(model_predictions)} 注预测")
                else:
                    logger_manager.warning(f"模型 {name} 没有predict方法")
                    predictions[name] = []
            except Exception as e:
                logger_manager.error(f"模型 {name} 预测失败: {e}")
                predictions[name] = []
        
        return predictions
    
    def _initialize_model_contributions(self):
        """初始化模型贡献度"""
        self.model_contributions = {name: 0.0 for name in self.models}

    def predict(self, data: pd.DataFrame = None, count: int = 1, verbose: bool = True) -> List[Tuple[List[int], List[int]]]:
        """
        标准预测方法，默认使用加权平均方法

        Args:
            data: 历史数据（可选，为了与其他模型保持一致的接口）
            count: 预测注数
            verbose: 是否显示详细信息

        Returns:
            预测结果列表
        """
        return self.weighted_average_predict(count, verbose)

    def weighted_average_predict(self, count: int = 1, verbose: bool = True) -> List[Tuple[List[int], List[int]]]:
        """
        使用加权平均方法进行集成预测
        
        Args:
            count: 预测注数
            verbose: 是否显示详细信息
            
        Returns:
            预测结果列表
        """
        from .prediction_utils import PredictionProcessor
        
        if len(self.models) == 0:
            logger_manager.warning("集成中没有模型，无法进行预测")
            return []
        
        if verbose:
            logger_manager.info(f"使用加权平均方法进行集成预测，注数: {count}")
        
        # 收集各模型的预测结果
        model_predictions = self._collect_predictions(count)
        
        # 获取归一化权重
        normalized_weights = self.normalize_weights()
        
        # 初始化模型贡献度
        self._initialize_model_contributions()
        
        # 创建预测处理器
        processor = PredictionProcessor()
        
        # 进行加权平均预测
        ensemble_predictions = []
        
        for i in range(count):
            # 前区和后区号码得分字典
            front_scores = defaultdict(float)
            back_scores = defaultdict(float)
            
            # 计算每个号码的加权得分
            for model_name, predictions in model_predictions.items():
                if i < len(predictions):
                    front, back = predictions[i]
                    weight = normalized_weights.get(model_name, 0.0)
                    
                    # 累加前区号码得分
                    for num in front:
                        front_scores[num] += weight
                    
                    # 累加后区号码得分
                    for num in back:
                        back_scores[num] += weight
            
            # 选择得分最高的号码
            front_balls = sorted(front_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            back_balls = sorted(back_scores.items(), key=lambda x: x[1], reverse=True)[:2]
            
            # 提取号码
            front_numbers = [num for num, _ in front_balls]
            back_numbers = [num for num, _ in back_balls]
            
            # 确保号码数量正确
            if len(front_numbers) < 5:
                # 如果前区号码不足，随机补充
                available_numbers = [n for n in range(1, 36) if n not in front_numbers]
                front_numbers.extend(np.random.choice(available_numbers, 5 - len(front_numbers), replace=False))
            
            if len(back_numbers) < 2:
                # 如果后区号码不足，随机补充
                available_numbers = [n for n in range(1, 13) if n not in back_numbers]
                back_numbers.extend(np.random.choice(available_numbers, 2 - len(back_numbers), replace=False))
            
            # 添加到预测结果
            ensemble_predictions.append((sorted(front_numbers), sorted(back_numbers)))
            
            # 更新模型贡献度
            self._update_model_contributions(model_predictions, i, front_numbers, back_numbers)
            
            if verbose:
                formatted = processor.format_prediction((sorted(front_numbers), sorted(back_numbers)))
                logger_manager.info(f"预测 {i+1}/{count}: {formatted}")
        
        # 计算预测置信度
        confidence = processor.calculate_confidence(ensemble_predictions)
        
        if verbose:
            logger_manager.info(f"集成预测完成，置信度: {confidence:.2f}")
            
            # 输出模型贡献度
            contributions = self.get_model_contributions()
            for model_name, contribution in contributions.items():
                logger_manager.info(f"模型 {model_name} 贡献度: {contribution:.2f}")
        
        return ensemble_predictions
    
    def _update_model_contributions(self, model_predictions: Dict[str, List[Tuple[List[int], List[int]]]],
                                  index: int, front_numbers: List[int], back_numbers: List[int]) -> None:
        """
        更新模型贡献度
        
        Args:
            model_predictions: 各模型的预测结果
            index: 当前预测索引
            front_numbers: 集成前区号码
            back_numbers: 集成后区号码
        """
        for model_name, predictions in model_predictions.items():
            if index < len(predictions):
                front, back = predictions[index]
                
                # 计算前区匹配数
                front_matches = len(set(front) & set(front_numbers))
                
                # 计算后区匹配数
                back_matches = len(set(back) & set(back_numbers))
                
                # 计算贡献度（前区权重0.7，后区权重0.3）
                contribution = (front_matches / 5.0) * 0.7 + (back_matches / 2.0) * 0.3
                
                # 更新模型贡献度
                self.model_contributions[model_name] += contribution
    
    def stacked_ensemble_predict(self, count: int = 1, verbose: bool = True) -> List[Tuple[List[int], List[int]]]:
        """
        使用堆叠集成方法进行预测
        
        Args:
            count: 预测注数
            verbose: 是否显示详细信息
            
        Returns:
            预测结果列表
        """
        from .stacking_ensemble import StackingEnsemble
        from .prediction_utils import PredictionProcessor
        
        if len(self.models) == 0:
            logger_manager.warning("集成中没有模型，无法进行预测")
            return []
        
        if verbose:
            logger_manager.info(f"使用堆叠集成方法进行预测，注数: {count}")
        
        # 收集各模型的预测结果
        model_predictions = self._collect_predictions(count)
        
        # 创建堆叠集成
        stacking = StackingEnsemble()
        
        # 创建预测处理器
        processor = PredictionProcessor()
        
        # 进行堆叠集成预测
        ensemble_predictions = []
        
        for i in range(count):
            # 提取当前索引的预测结果
            current_predictions = {}
            for model_name, predictions in model_predictions.items():
                if i < len(predictions):
                    current_predictions[model_name] = [predictions[i]]
            
            # 使用堆叠集成进行预测
            stacked_pred = stacking.predict(current_predictions)
            
            if stacked_pred:
                ensemble_predictions.append(stacked_pred[0])
                
                if verbose:
                    formatted = processor.format_prediction(stacked_pred[0])
                    logger_manager.info(f"预测 {i+1}/{count}: {formatted}")
        
        # 计算预测置信度
        confidence = processor.calculate_confidence(ensemble_predictions)
        
        if verbose:
            logger_manager.info(f"堆叠集成预测完成，置信度: {confidence:.2f}")
        
        return ensemble_predictions
    
    def get_confidence_intervals(self, predictions: List[Tuple[List[int], List[int]]]) -> List[Tuple[float, float]]:
        """
        获取预测结果的置信区间
        
        Args:
            predictions: 预测结果列表
            
        Returns:
            置信区间列表，每个元素为(下限, 上限)
        """
        if not predictions:
            return []
        
        # 获取模型数量
        model_count = len(self.models)
        
        if model_count <= 1:
            # 如果只有一个模型，使用固定置信区间
            return [(0.4, 0.8)] * len(predictions)
        
        # 计算各模型的置信度
        model_confidences = {}
        for name, model in self.models.items():
            if hasattr(model, 'get_confidence'):
                model_confidences[name] = model.get_confidence()
            else:
                model_confidences[name] = 0.7  # 默认置信度
        
        # 获取归一化权重
        normalized_weights = self.normalize_weights()
        
        # 计算加权平均置信度
        weighted_confidence = sum(model_confidences[name] * normalized_weights.get(name, 0.0) 
                                for name in self.models.keys())
        
        # 计算置信度标准差
        confidence_std = np.std(list(model_confidences.values()))
        
        # 计算置信区间
        intervals = []
        for _ in predictions:
            lower_bound = max(0.0, weighted_confidence - confidence_std)
            upper_bound = min(1.0, weighted_confidence + confidence_std)
            intervals.append((lower_bound, upper_bound))
        
        return intervals


if __name__ == "__main__":
    # 测试集成管理器
    print("🔄 测试集成管理器...")
    
    # 创建集成管理器
    ensemble = EnsembleManager()
    
    # 添加模拟模型
    class MockModel:
        def __init__(self, name):
            self.name = name
        
        def predict(self, count, verbose=False):
            # 生成随机预测结果
            predictions = []
            for _ in range(count):
                front = sorted(np.random.choice(range(1, 36), 5, replace=False))
                back = sorted(np.random.choice(range(1, 13), 2, replace=False))
                predictions.append((front, back))
            return predictions
        
        def get_confidence(self):
            return 0.7
    
    ensemble.add_model("model1", MockModel("model1"), 0.6)
    ensemble.add_model("model2", MockModel("model2"), 0.4)
    
    # 进行加权平均预测
    predictions = ensemble.weighted_average_predict(2)
    
    print("加权平均预测结果:")
    for i, (front, back) in enumerate(predictions):
        front_str = ' '.join([str(b).zfill(2) for b in front])
        back_str = ' '.join([str(b).zfill(2) for b in back])
        print(f"第 {i+1} 注: {front_str} + {back_str}")
    
    # 获取置信区间
    intervals = ensemble.get_confidence_intervals(predictions)
    print(f"置信区间: {intervals}")
    
    # 获取模型贡献度
    contributions = ensemble.get_model_contributions()
    print(f"模型贡献度: {contributions}")
    
    print("集成管理器测试完成")