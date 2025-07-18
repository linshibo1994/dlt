#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
堆叠集成方法测试
"""

import unittest
import os
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from enhanced_deep_learning.stacking_ensemble import StackingEnsemble
from enhanced_deep_learning.ensemble_manager import EnsembleManager


class MockModel:
    """模拟模型类"""
    
    def __init__(self, name, predictions=None):
        self.name = name
        self.predictions = predictions or [([1, 2, 3, 4, 5], [6, 7])]
    
    def predict(self, count, verbose=False):
        # 返回预定义的预测结果，如果不足则重复
        result = []
        for i in range(count):
            result.append(self.predictions[i % len(self.predictions)])
        return result
    
    def get_confidence(self):
        return 0.7


class TestStackingEnsemble(unittest.TestCase):
    """堆叠集成方法测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建堆叠集成
        self.stacking = StackingEnsemble()
        
        # 创建模拟训练数据
        self.base_preds1 = {
            "model1": [([1, 2, 3, 4, 5], [6, 7])],
            "model2": [([3, 4, 5, 6, 7], [8, 9])]
        }
        
        self.base_preds2 = {
            "model1": [([6, 7, 8, 9, 10], [10, 11])],
            "model2": [([8, 9, 10, 11, 12], [11, 12])]
        }
        
        self.actual1 = ([2, 3, 4, 5, 6], [7, 8])
        self.actual2 = ([7, 8, 9, 10, 11], [11, 12])
    
    def test_create_meta_model(self):
        """测试创建元模型"""
        # 创建随机森林元模型
        self.stacking.meta_model_type = "random_forest"
        self.stacking._create_meta_model()
        
        # 检查元模型类型
        if self.stacking.meta_model_type == "random_forest":
            self.assertIsNotNone(self.stacking.meta_model)
        
        # 创建简单平均元模型
        self.stacking.meta_model_type = "simple_average"
        self.stacking._create_meta_model()
        
        # 检查元模型类型
        self.assertEqual(self.stacking.meta_model_type, "simple_average")
    
    def test_add_training_data(self):
        """测试添加训练数据"""
        # 添加训练数据
        self.stacking.add_training_data([self.base_preds1, self.base_preds2], [self.actual1, self.actual2])
        
        # 检查训练数据
        self.assertEqual(len(self.stacking.base_predictions), 2)
        self.assertEqual(len(self.stacking.actual_results), 2)
    
    def test_extract_features(self):
        """测试特征提取"""
        # 提取特征
        features = self.stacking._extract_features(self.base_preds1)
        
        # 检查特征
        self.assertIsNotNone(features)
        self.assertIsInstance(features, list)
        
        # 检查空输入
        features = self.stacking._extract_features({})
        self.assertIsNone(features)
    
    def test_prepare_training_data(self):
        """测试准备训练数据"""
        # 添加训练数据
        self.stacking.add_training_data([self.base_preds1, self.base_preds2], [self.actual1, self.actual2])
        
        # 准备训练数据
        X, y = self.stacking._prepare_training_data()
        
        # 检查训练数据
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertEqual(X.shape[0], 2)  # 2个样本
        self.assertEqual(y.shape[0], 2)  # 2个样本
        self.assertEqual(y.shape[1], 7)  # 7个目标值（5前区+2后区）
    
    def test_train_meta_model(self):
        """测试训练元模型"""
        # 添加训练数据
        self.stacking.add_training_data([self.base_preds1, self.base_preds2], [self.actual1, self.actual2])
        
        # 训练元模型
        result = self.stacking.train_meta_model()
        
        # 检查训练结果
        self.assertTrue(result)
        self.assertTrue(self.stacking.is_trained)
    
    def test_simple_average_predict(self):
        """测试简单平均预测"""
        # 进行简单平均预测
        predictions = self.stacking._simple_average_predict(self.base_preds1)
        
        # 检查预测结果
        self.assertEqual(len(predictions), 1)
        front, back = predictions[0]
        self.assertEqual(len(front), 5)
        self.assertEqual(len(back), 2)
        
        # 检查前区号码（应该包含两个模型的共同号码）
        common_front = set([3, 4, 5])
        self.assertTrue(all(num in front for num in common_front))
    
    def test_predict(self):
        """测试预测"""
        # 添加训练数据
        self.stacking.add_training_data([self.base_preds1, self.base_preds2], [self.actual1, self.actual2])
        
        # 训练元模型
        self.stacking.train_meta_model()
        
        # 进行预测
        predictions = self.stacking.predict(self.base_preds1)
        
        # 检查预测结果
        self.assertEqual(len(predictions), 1)
        front, back = predictions[0]
        self.assertEqual(len(front), 5)
        self.assertEqual(len(back), 2)


class TestEnsembleManagerStacking(unittest.TestCase):
    """集成管理器堆叠集成测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建集成管理器
        self.ensemble = EnsembleManager()
        
        # 添加模拟模型
        self.model1 = MockModel("model1", [
            ([1, 2, 3, 4, 5], [6, 7]),
            ([6, 7, 8, 9, 10], [8, 9])
        ])
        
        self.model2 = MockModel("model2", [
            ([3, 4, 5, 6, 7], [8, 9]),
            ([8, 9, 10, 11, 12], [10, 11])
        ])
        
        self.ensemble.add_model("model1", self.model1, 0.6)
        self.ensemble.add_model("model2", self.model2, 0.4)
    
    def test_stacked_ensemble_predict(self):
        """测试堆叠集成预测"""
        # 进行堆叠集成预测
        predictions = self.ensemble.stacked_ensemble_predict(2, verbose=False)
        
        # 检查预测结果
        self.assertEqual(len(predictions), 2)
        
        for front, back in predictions:
            self.assertEqual(len(front), 5)
            self.assertEqual(len(back), 2)
            self.assertTrue(all(1 <= x <= 35 for x in front))
            self.assertTrue(all(1 <= x <= 12 for x in back))


if __name__ == "__main__":
    unittest.main()