#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动权重调整系统测试
"""

import unittest
import os
import sys
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from enhanced_deep_learning.weight_optimizer import WeightOptimizer, AdaptiveWeightOptimizer
from enhanced_deep_learning.performance_tracker import PerformanceTracker


class TestWeightOptimizer(unittest.TestCase):
    """权重优化器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建性能跟踪器
        self.tracker = PerformanceTracker()
        
        # 跟踪模型性能
        self.tracker.track_performance("model1", {
            'overall_score': 0.6
        })
        
        self.tracker.track_performance("model2", {
            'overall_score': 0.8
        })
        
        # 创建权重优化器
        self.optimizer = WeightOptimizer(self.tracker)
    
    def test_initialize_weights(self):
        """测试初始化权重"""
        # 使用默认初始化
        weights = self.optimizer.initialize_weights(["model1", "model2"])
        
        # 检查权重
        self.assertEqual(len(weights), 2)
        self.assertAlmostEqual(weights["model1"], 0.5)
        self.assertAlmostEqual(weights["model2"], 0.5)
        
        # 使用自定义初始化
        weights = self.optimizer.initialize_weights(["model1", "model2"], {
            "model1": 0.3,
            "model2": 0.7
        })
        
        # 检查权重
        self.assertEqual(len(weights), 2)
        self.assertAlmostEqual(weights["model1"], 0.3)
        self.assertAlmostEqual(weights["model2"], 0.7)
    
    def test_update_weights(self):
        """测试更新权重"""
        # 初始化权重
        self.optimizer.initialize_weights(["model1", "model2"])
        
        # 更新权重
        weights = self.optimizer.update_weights()
        
        # 检查权重
        self.assertEqual(len(weights), 2)
        
        # 由于model2的得分更高，它的权重应该增加
        self.assertLess(weights["model1"], weights["model2"])
        
        # 权重总和应该为1
        self.assertAlmostEqual(sum(weights.values()), 1.0)
    
    def test_weight_history(self):
        """测试权重历史"""
        # 初始化权重
        self.optimizer.initialize_weights(["model1", "model2"])
        
        # 更新权重多次
        for _ in range(3):
            self.optimizer.update_weights()
        
        # 获取权重历史
        history1 = self.optimizer.get_weight_history("model1")
        history2 = self.optimizer.get_weight_history("model2")
        
        # 检查历史长度
        self.assertEqual(len(history1), 4)  # 初始权重 + 3次更新
        self.assertEqual(len(history2), 4)
        
        # 检查权重变化趋势
        self.assertGreater(history2[-1], history2[0])  # model2的权重应该增加
    
    def test_get_weight_explanation(self):
        """测试获取权重解释"""
        # 初始化权重
        self.optimizer.initialize_weights(["model1", "model2"])
        
        # 更新权重
        self.optimizer.update_weights()
        
        # 获取权重解释
        explanation = self.optimizer.get_weight_explanation()
        
        # 检查解释
        self.assertIn('explanation', explanation)
        self.assertIn('weights', explanation)
        self.assertIn('scores', explanation)
        self.assertIn('model1', explanation['weights'])
        self.assertIn('model2', explanation['weights'])
    
    def test_save_and_load_weights(self):
        """测试保存和加载权重"""
        # 初始化权重
        self.optimizer.initialize_weights(["model1", "model2"])
        
        # 更新权重
        original_weights = self.optimizer.update_weights()
        
        # 保存权重
        temp_file = "temp_weights.json"
        self.optimizer.save_weights(temp_file)
        
        # 创建新的优化器
        new_optimizer = WeightOptimizer(self.tracker)
        
        # 加载权重
        new_optimizer.load_weights(temp_file)
        
        # 检查加载的权重
        loaded_weights = new_optimizer.get_weights()
        self.assertEqual(len(loaded_weights), len(original_weights))
        self.assertAlmostEqual(loaded_weights["model1"], original_weights["model1"])
        self.assertAlmostEqual(loaded_weights["model2"], original_weights["model2"])
        
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)


class TestAdaptiveWeightOptimizer(unittest.TestCase):
    """自适应权重优化器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建性能跟踪器
        self.tracker = PerformanceTracker()
        
        # 跟踪模型性能
        self.tracker.track_performance("model1", {
            'overall_score': 0.6
        })
        
        self.tracker.track_performance("model2", {
            'overall_score': 0.8
        })
        
        # 创建自适应权重优化器
        self.optimizer = AdaptiveWeightOptimizer(
            self.tracker,
            initial_learning_rate=0.1,
            min_learning_rate=0.01,
            max_learning_rate=0.5
        )
    
    def test_adjust_learning_rate(self):
        """测试调整学习率"""
        # 初始化权重
        self.optimizer.initialize_weights(["model1", "model2"])
        
        # 更新权重多次
        for _ in range(5):
            self.optimizer.update_weights()
        
        # 获取学习率历史
        lr_history = self.optimizer.get_learning_rate_history()
        
        # 检查学习率历史
        self.assertGreaterEqual(len(lr_history), 3)  # 至少有3次调整
        
        # 检查学习率范围
        for lr in lr_history:
            self.assertGreaterEqual(lr, self.optimizer.min_learning_rate)
            self.assertLessEqual(lr, self.optimizer.max_learning_rate)
    
    def test_adaptive_update_weights(self):
        """测试自适应更新权重"""
        # 初始化权重
        self.optimizer.initialize_weights(["model1", "model2"])
        
        # 记录初始学习率
        initial_lr = self.optimizer.learning_rate
        
        # 更新权重
        weights = self.optimizer.update_weights()
        
        # 检查权重
        self.assertEqual(len(weights), 2)
        
        # 由于model2的得分更高，它的权重应该增加
        self.assertLess(weights["model1"], weights["model2"])
        
        # 权重总和应该为1
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        
        # 更新多次，使学习率调整生效
        for _ in range(5):
            # 交替更新模型性能，使权重变化较大
            if _ % 2 == 0:
                self.tracker.track_performance("model1", {'overall_score': 0.9})
                self.tracker.track_performance("model2", {'overall_score': 0.5})
            else:
                self.tracker.track_performance("model1", {'overall_score': 0.4})
                self.tracker.track_performance("model2", {'overall_score': 0.9})
            
            self.optimizer.update_weights()
        
        # 获取最终学习率
        final_lr = self.optimizer.learning_rate
        
        # 由于权重变化较大，学习率应该减小
        self.assertNotEqual(final_lr, initial_lr)


if __name__ == "__main__":
    unittest.main()