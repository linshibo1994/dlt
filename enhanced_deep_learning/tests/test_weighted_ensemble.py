#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
加权平均集成方法测试
"""

import unittest
import os
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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


class TestWeightedEnsemble(unittest.TestCase):
    """加权平均集成方法测试类"""
    
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
    
    def test_weighted_average_predict(self):
        """测试加权平均预测"""
        # 进行加权平均预测
        predictions = self.ensemble.weighted_average_predict(2, verbose=False)
        
        # 检查预测结果
        self.assertEqual(len(predictions), 2)
        
        # 第一组预测应该偏向model1的结果
        front1, back1 = predictions[0]
        self.assertEqual(len(front1), 5)
        self.assertEqual(len(back1), 2)
        
        # 检查前区号码（应该包含更多model1的号码）
        self.assertTrue(len(set(front1) & set([1, 2, 3, 4, 5])) >= 3)
        
        # 第二组预测也应该偏向model1的结果
        front2, back2 = predictions[1]
        self.assertEqual(len(front2), 5)
        self.assertEqual(len(back2), 2)
    
    def test_update_model_contributions(self):
        """测试更新模型贡献度"""
        # 模拟预测结果
        model_predictions = {
            "model1": [([1, 2, 3, 4, 5], [6, 7])],
            "model2": [([3, 4, 5, 6, 7], [8, 9])]
        }
        
        # 集成结果
        front_numbers = [1, 3, 4, 5, 7]
        back_numbers = [6, 9]
        
        # 初始化模型贡献度
        self.ensemble._initialize_model_contributions()
        
        # 更新模型贡献度
        self.ensemble._update_model_contributions(model_predictions, 0, front_numbers, back_numbers)
        
        # 检查模型贡献度
        contributions = self.ensemble.get_model_contributions()
        
        # model1: 前区匹配4/5，后区匹配1/2
        # 贡献度 = (4/5)*0.7 + (1/2)*0.3 = 0.56 + 0.15 = 0.71
        self.assertAlmostEqual(contributions["model1"], 0.71)
        
        # model2: 前区匹配4/5，后区匹配1/2
        # 贡献度 = (4/5)*0.7 + (1/2)*0.3 = 0.56 + 0.15 = 0.71
        self.assertAlmostEqual(contributions["model2"], 0.71)
    
    def test_get_confidence_intervals(self):
        """测试获取置信区间"""
        # 创建预测结果
        predictions = [
            ([1, 2, 3, 4, 5], [6, 7]),
            ([6, 7, 8, 9, 10], [8, 9])
        ]
        
        # 获取置信区间
        intervals = self.ensemble.get_confidence_intervals(predictions)
        
        # 检查置信区间
        self.assertEqual(len(intervals), 2)
        
        for lower, upper in intervals:
            self.assertTrue(0 <= lower <= upper <= 1)
            self.assertLess(lower, upper)
    
    def test_empty_ensemble(self):
        """测试空集成"""
        # 创建空集成
        empty_ensemble = EnsembleManager()
        
        # 进行加权平均预测
        predictions = empty_ensemble.weighted_average_predict(2, verbose=False)
        
        # 检查预测结果
        self.assertEqual(len(predictions), 0)


if __name__ == "__main__":
    unittest.main()