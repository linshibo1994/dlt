#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
堆叠集成方法
实现基于元模型的堆叠集成
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from collections import defaultdict
from datetime import datetime

from core_modules import logger_manager


class StackingEnsemble:
    """堆叠集成类"""
    
    def __init__(self, meta_model_type: str = "random_forest"):
        """
        初始化堆叠集成
        
        Args:
            meta_model_type: 元模型类型，支持'random_forest', 'xgboost', 'neural_network'
        """
        self.meta_model_type = meta_model_type
        self.meta_model = None
        self.base_predictions = []
        self.actual_results = []
        self.is_trained = False
        
        logger_manager.info(f"初始化堆叠集成，元模型类型: {meta_model_type}")
    
    def _create_meta_model(self):
        """创建元模型"""
        if self.meta_model_type == "random_forest":
            try:
                from sklearn.ensemble import RandomForestRegressor
                self.meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
                logger_manager.info("创建随机森林元模型")
            except ImportError:
                logger_manager.warning("sklearn未安装，使用简单平均代替")
                self.meta_model_type = "simple_average"
        
        elif self.meta_model_type == "xgboost":
            try:
                import xgboost as xgb
                self.meta_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
                logger_manager.info("创建XGBoost元模型")
            except ImportError:
                logger_manager.warning("xgboost未安装，使用随机森林代替")
                self.meta_model_type = "random_forest"
                self._create_meta_model()
        
        elif self.meta_model_type == "neural_network":
            try:
                from sklearn.neural_network import MLPRegressor
                self.meta_model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
                logger_manager.info("创建神经网络元模型")
            except ImportError:
                logger_manager.warning("sklearn未安装，使用简单平均代替")
                self.meta_model_type = "simple_average"
        
        else:
            logger_manager.info("使用简单平均作为元模型")
            self.meta_model_type = "simple_average"
    
    def add_training_data(self, base_predictions: List[Dict[str, List[Tuple[List[int], List[int]]]]], 
                         actual_results: List[Tuple[List[int], List[int]]]):
        """
        添加训练数据
        
        Args:
            base_predictions: 基础模型的预测结果列表
            actual_results: 实际结果列表
        """
        self.base_predictions.extend(base_predictions)
        self.actual_results.extend(actual_results)
        
        logger_manager.info(f"添加训练数据: {len(base_predictions)} 组预测, {len(actual_results)} 组实际结果")
    
    def _prepare_training_data(self):
        """
        准备训练数据
        
        Returns:
            特征矩阵和目标值
        """
        if not self.base_predictions or not self.actual_results:
            logger_manager.warning("没有训练数据")
            return None, None
        
        X = []  # 特征矩阵
        y_front = []  # 前区目标值
        y_back = []  # 后区目标值
        
        for base_preds, actual in zip(self.base_predictions, self.actual_results):
            # 提取特征
            features = self._extract_features(base_preds)
            
            if features is not None:
                X.append(features)
                
                # 提取目标值（归一化）
                actual_front, actual_back = actual
                y_front.append([(x - 1) / 34 for x in actual_front])  # 1-35 -> 0-1
                y_back.append([(x - 1) / 11 for x in actual_back])    # 1-12 -> 0-1
        
        if not X:
            return None, None
        
        # 转换为numpy数组
        X = np.array(X)
        y_front = np.array(y_front)
        y_back = np.array(y_back)
        
        # 合并前区和后区目标值
        y = np.hstack([y_front, y_back])
        
        return X, y
    
    def _extract_features(self, base_preds: Dict[str, List[Tuple[List[int], List[int]]]]):
        """
        从基础预测中提取特征
        
        Args:
            base_preds: 基础模型的预测结果
            
        Returns:
            特征向量
        """
        if not base_preds:
            return None
        
        features = []
        
        # 对于每个模型的预测结果
        for model_name, preds in base_preds.items():
            if not preds:
                continue
            
            # 只使用第一组预测
            front, back = preds[0]
            
            # 归一化号码
            front_norm = [(x - 1) / 34 for x in front]  # 1-35 -> 0-1
            back_norm = [(x - 1) / 11 for x in back]    # 1-12 -> 0-1
            
            # 添加到特征向量
            features.extend(front_norm)
            features.extend(back_norm)
            
            # 添加统计特征
            features.append(sum(front_norm) / len(front_norm))  # 前区平均值
            features.append(sum(back_norm) / len(back_norm))    # 后区平均值
            features.append(np.std(front_norm))                 # 前区标准差
            features.append(np.std(back_norm))                  # 后区标准差
        
        return features
    
    def train_meta_model(self):
        """训练元模型"""
        # 创建元模型
        if self.meta_model is None:
            self._create_meta_model()
        
        # 准备训练数据
        X, y = self._prepare_training_data()
        
        if X is None or y is None:
            logger_manager.warning("没有足够的训练数据，无法训练元模型")
            return False
        
        # 如果使用简单平均，不需要训练
        if self.meta_model_type == "simple_average":
            self.is_trained = True
            logger_manager.info("使用简单平均作为元模型，无需训练")
            return True
        
        try:
            # 训练元模型
            self.meta_model.fit(X, y)
            self.is_trained = True
            logger_manager.info("元模型训练完成")
            return True
        except Exception as e:
            logger_manager.error(f"元模型训练失败: {e}")
            return False
    
    def predict(self, base_predictions: Dict[str, List[Tuple[List[int], List[int]]]]) -> List[Tuple[List[int], List[int]]]:
        """
        使用元模型进行预测
        
        Args:
            base_predictions: 基础模型的预测结果
            
        Returns:
            堆叠集成的预测结果
        """
        # 如果元模型未训练，尝试训练
        if not self.is_trained:
            if not self.train_meta_model():
                logger_manager.warning("元模型未训练，使用简单平均代替")
                return self._simple_average_predict(base_predictions)
        
        # 提取特征
        features = self._extract_features(base_predictions)
        
        if features is None:
            logger_manager.warning("无法提取特征，使用简单平均代替")
            return self._simple_average_predict(base_predictions)
        
        # 转换为numpy数组
        X = np.array([features])
        
        # 如果使用简单平均
        if self.meta_model_type == "simple_average":
            return self._simple_average_predict(base_predictions)
        
        try:
            # 使用元模型预测
            y_pred = self.meta_model.predict(X)[0]
            
            # 分离前区和后区预测
            front_pred = y_pred[:5]
            back_pred = y_pred[5:7]
            
            # 转换为彩票号码
            front_balls = [max(1, min(35, int(round(x * 34 + 1)))) for x in front_pred]
            back_balls = [max(1, min(12, int(round(x * 11 + 1)))) for x in back_pred]
            
            # 确保号码唯一性
            front_balls = self._ensure_unique_numbers(front_balls, 1, 35, 5)
            back_balls = self._ensure_unique_numbers(back_balls, 1, 12, 2)
            
            return [(sorted(front_balls), sorted(back_balls))]
        except Exception as e:
            logger_manager.error(f"元模型预测失败: {e}")
            return self._simple_average_predict(base_predictions)
    
    def _simple_average_predict(self, base_predictions: Dict[str, List[Tuple[List[int], List[int]]]]) -> List[Tuple[List[int], List[int]]]:
        """
        使用简单平均进行预测
        
        Args:
            base_predictions: 基础模型的预测结果
            
        Returns:
            简单平均的预测结果
        """
        # 前区和后区号码得分字典
        front_scores = defaultdict(float)
        back_scores = defaultdict(float)
        
        # 计算每个号码的得分
        for model_name, predictions in base_predictions.items():
            if not predictions:
                continue
            
            front, back = predictions[0]
            
            # 累加前区号码得分
            for num in front:
                front_scores[num] += 1
            
            # 累加后区号码得分
            for num in back:
                back_scores[num] += 1
        
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
        
        return [(sorted(front_numbers), sorted(back_numbers))]
    
    def _ensure_unique_numbers(self, numbers: List[int], min_val: int, max_val: int, target_count: int) -> List[int]:
        """确保号码唯一性"""
        unique_numbers = list(set(numbers))
        
        # 如果数量不足，随机补充
        while len(unique_numbers) < target_count:
            candidate = np.random.randint(min_val, max_val + 1)
            if candidate not in unique_numbers:
                unique_numbers.append(candidate)
        
        # 如果数量过多，截取
        if len(unique_numbers) > target_count:
            unique_numbers = unique_numbers[:target_count]
        
        return unique_numbers


if __name__ == "__main__":
    # 测试堆叠集成
    print("🔄 测试堆叠集成...")
    
    # 创建堆叠集成
    stacking = StackingEnsemble()
    
    # 创建模拟训练数据
    base_preds1 = {
        "model1": [([1, 2, 3, 4, 5], [6, 7])],
        "model2": [([3, 4, 5, 6, 7], [8, 9])]
    }
    
    base_preds2 = {
        "model1": [([6, 7, 8, 9, 10], [10, 11])],
        "model2": [([8, 9, 10, 11, 12], [11, 12])]
    }
    
    actual1 = ([2, 3, 4, 5, 6], [7, 8])
    actual2 = ([7, 8, 9, 10, 11], [11, 12])
    
    # 添加训练数据
    stacking.add_training_data([base_preds1, base_preds2], [actual1, actual2])
    
    # 训练元模型
    stacking.train_meta_model()
    
    # 进行预测
    test_preds = {
        "model1": [([11, 12, 13, 14, 15], [6, 7])],
        "model2": [([13, 14, 15, 16, 17], [8, 9])]
    }
    
    predictions = stacking.predict(test_preds)
    
    print("堆叠集成预测结果:")
    for i, (front, back) in enumerate(predictions):
        front_str = ' '.join([str(b).zfill(2) for b in front])
        back_str = ' '.join([str(b).zfill(2) for b in back])
        print(f"第 {i+1} 注: {front_str} + {back_str}")
    
    print("堆叠集成测试完成")