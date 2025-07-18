#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
集成管理器测试
"""

import unittest
import os
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from enhanced_deep_learning.ensemble_manager import EnsembleManager
from enhanced_deep_learning.config import DEFAULT_ENSEMBLE_CONFIG


class MockModel:
    """模拟模型类"""
    
    def __init__(self, name):
        self.name = name
    
    def predict(self, count, verbose=False):
        return [([1, 2, 3, 4, 5], [6, 7]) for _ in range(count)]


class TestEnsembleManager(unittest.TestCase):
    """集成管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建集成管理器
        self.ensemble = EnsembleManager()
        
        # 添加模拟模型
        self.model1 = MockModel("model1")
        self.model2 = MockModel("model2")
        
        self.ensemble.add_model("model1", self.model1, 0.6)
        self.ensemble.add_model("model2", self.model2, 0.4)
    
    def test_initialization(self):
        """测试初始化"""
        # 创建自定义配置的集成管理器
        custom_config = {
            'weights': {
                'model1': 0.7,
                'model2': 0.3
            }
        }
        
        ensemble = EnsembleManager(config=custom_config)
        
        # 检查配置是否正确合并
        self.assertEqual(ensemble.config['weights']['model1'], 0.7)
        self.assertEqual(ensemble.config['weights']['model2'], 0.3)
    
    def test_add_model(self):
        """测试添加模型"""
        # 添加新模型
        model3 = MockModel("model3")
        result = self.ensemble.add_model("model3", model3, 0.5)
        
        # 检查添加结果
        self.assertTrue(result)
        self.assertIn("model3", self.ensemble.get_models())
        self.assertEqual(self.ensemble.get_weights()["model3"], 0.5)
        
        # 添加无效模型
        class InvalidModel:
            pass
        
        invalid_model = InvalidModel()
        result = self.ensemble.add_model("invalid", invalid_model)
        
        # 检查添加结果
        self.assertFalse(result)
        self.assertNotIn("invalid", self.ensemble.get_models())
    
    def test_remove_model(self):
        """测试移除模型"""
        # 移除已有模型
        result = self.ensemble.remove_model("model1")
        
        # 检查移除结果
        self.assertTrue(result)
        self.assertNotIn("model1", self.ensemble.get_models())
        self.assertNotIn("model1", self.ensemble.get_weights())
        
        # 移除不存在的模型
        result = self.ensemble.remove_model("non_existent")
        
        # 检查移除结果
        self.assertFalse(result)
    
    def test_get_models(self):
        """测试获取模型"""
        models = self.ensemble.get_models()
        
        # 检查模型字典
        self.assertEqual(len(models), 2)
        self.assertIn("model1", models)
        self.assertIn("model2", models)
        self.assertEqual(models["model1"], self.model1)
        self.assertEqual(models["model2"], self.model2)
    
    def test_get_weights(self):
        """测试获取权重"""
        weights = self.ensemble.get_weights()
        
        # 检查权重字典
        self.assertEqual(len(weights), 2)
        self.assertIn("model1", weights)
        self.assertIn("model2", weights)
        self.assertEqual(weights["model1"], 0.6)
        self.assertEqual(weights["model2"], 0.4)
    
    def test_set_weight(self):
        """测试设置权重"""
        # 设置已有模型的权重
        result = self.ensemble.set_weight("model1", 0.8)
        
        # 检查设置结果
        self.assertTrue(result)
        self.assertEqual(self.ensemble.get_weights()["model1"], 0.8)
        
        # 设置不存在模型的权重
        result = self.ensemble.set_weight("non_existent", 0.5)
        
        # 检查设置结果
        self.assertFalse(result)
    
    def test_normalize_weights(self):
        """测试归一化权重"""
        # 设置权重
        self.ensemble.set_weight("model1", 2.0)
        self.ensemble.set_weight("model2", 3.0)
        
        # 归一化权重
        normalized_weights = self.ensemble.normalize_weights()
        
        # 检查归一化结果
        self.assertEqual(len(normalized_weights), 2)
        self.assertAlmostEqual(normalized_weights["model1"], 0.4)
        self.assertAlmostEqual(normalized_weights["model2"], 0.6)
        self.assertAlmostEqual(sum(normalized_weights.values()), 1.0)
        
        # 测试所有权重为0的情况
        self.ensemble.set_weight("model1", 0.0)
        self.ensemble.set_weight("model2", 0.0)
        
        normalized_weights = self.ensemble.normalize_weights()
        
        # 检查归一化结果（应该平均分配）
        self.assertAlmostEqual(normalized_weights["model1"], 0.5)
        self.assertAlmostEqual(normalized_weights["model2"], 0.5)
        self.assertAlmostEqual(sum(normalized_weights.values()), 1.0)
    
    def test_collect_predictions(self):
        """测试收集预测结果"""
        # 收集预测结果
        predictions = self.ensemble._collect_predictions(2)
        
        # 检查预测结果
        self.assertEqual(len(predictions), 2)
        self.assertIn("model1", predictions)
        self.assertIn("model2", predictions)
        self.assertEqual(len(predictions["model1"]), 2)
        self.assertEqual(len(predictions["model2"]), 2)


if __name__ == "__main__":
    unittest.main()