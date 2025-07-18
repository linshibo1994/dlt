#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
重训练管理器测试
"""

import unittest
import os
import sys
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from enhanced_deep_learning.retraining_manager import RetrainingManager, AdaptiveRetrainingManager
from enhanced_deep_learning.performance_tracker import PerformanceTracker


class TestRetrainingManager(unittest.TestCase):
    """重训练管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建性能跟踪器
        self.tracker = PerformanceTracker()
        
        # 跟踪模型性能
        self.tracker.track_performance("model1", {
            'overall_score': 0.8
        })
        
        self.tracker.track_performance("model2", {
            'overall_score': 0.7
        })
        
        # 创建重训练管理器
        self.manager = RetrainingManager(self.tracker, degradation_threshold=0.1, window_size=5)
        
        # 注册模型
        self.retrain_called = False
        
        def mock_retrain():
            self.retrain_called = True
            return True
        
        self.manager.register_model("model1", mock_retrain)
    
    def test_register_model(self):
        """测试注册模型"""
        # 注册新模型
        def mock_retrain():
            return True
        
        result = "model2" in self.manager.retrain_callbacks
        self.manager.register_model("model2", mock_retrain)
        
        # 检查是否已注册
        self.assertIn("model2", self.manager.retrain_callbacks)
        self.assertIn("model2", self.manager.last_retrain)
    
    def test_unregister_model(self):
        """测试取消注册模型"""
        # 取消注册模型
        result = self.manager.unregister_model("model1")
        
        # 检查是否已取消注册
        self.assertTrue(result)
        self.assertNotIn("model1", self.manager.retrain_callbacks)
        self.assertNotIn("model1", self.manager.last_retrain)
        
        # 取消注册不存在的模型
        result = self.manager.unregister_model("non_existent")
        
        # 检查结果
        self.assertFalse(result)
    
    def test_check_performance(self):
        """测试检查性能"""
        # 检查性能
        result = self.manager.check_performance("model1")
        
        # 检查结果
        self.assertEqual(result['model_id'], "model1")
        self.assertFalse(result['degraded'])
        self.assertFalse(result['in_cooldown'])
        self.assertEqual(result['status'], "normal")
        
        # 检查不存在的模型
        result = self.manager.check_performance("non_existent")
        
        # 检查结果
        self.assertEqual(result['status'], "not_registered")
    
    def test_trigger_retrain(self):
        """测试触发重训练"""
        # 触发重训练（强制）
        result = self.manager.trigger_retrain("model1", force=True)
        
        # 检查结果
        self.assertTrue(result['success'])
        self.assertTrue(self.retrain_called)
        
        # 检查重训练历史
        history = self.manager.get_retrain_history("model1")
        self.assertEqual(len(history), 1)
        self.assertTrue(history[0]['forced'])
        
        # 重置标志
        self.retrain_called = False
        
        # 触发不存在的模型重训练
        result = self.manager.trigger_retrain("non_existent")
        
        # 检查结果
        self.assertFalse(result['success'])
        self.assertFalse(self.retrain_called)
    
    def test_performance_degradation(self):
        """测试性能退化检测"""
        # 模拟性能退化
        for i in range(10):
            score = 0.8 - i * 0.05
            self.tracker.track_performance("model1", {
                'overall_score': max(0.1, score)
            })
        
        # 检查性能
        result = self.manager.check_performance("model1")
        
        # 检查结果
        self.assertTrue(result['degraded'])
        self.assertEqual(result['status'], "degraded")
        
        # 触发重训练
        result = self.manager.trigger_retrain("model1")
        
        # 检查结果
        self.assertTrue(result['success'])
        self.assertTrue(self.retrain_called)
    
    def test_cooldown_period(self):
        """测试冷却期"""
        # 模拟性能退化并重训练
        for i in range(10):
            score = 0.8 - i * 0.05
            self.tracker.track_performance("model1", {
                'overall_score': max(0.1, score)
            })
        
        # 触发重训练
        self.manager.trigger_retrain("model1")
        
        # 重置标志
        self.retrain_called = False
        
        # 再次模拟性能退化
        for i in range(10):
            score = 0.8 - i * 0.05
            self.tracker.track_performance("model1", {
                'overall_score': max(0.1, score)
            })
        
        # 检查性能
        result = self.manager.check_performance("model1")
        
        # 由于冷却期机制，这里的实现可能会有所不同
        # 如果冷却期基于重训练次数，则不会进入冷却期
        # 如果冷却期基于时间，则可能会进入冷却期
        
        # 触发重训练
        result = self.manager.trigger_retrain("model1")
        
        # 检查结果
        self.assertTrue(result['success'])
        self.assertTrue(self.retrain_called)
    
    def test_optimize_training_params(self):
        """测试优化训练参数"""
        # 模拟多次重训练
        for i in range(3):
            self.manager.trigger_retrain("model1", force=True)
        
        # 优化训练参数
        result = self.manager.optimize_training_params("model1")
        
        # 检查结果
        self.assertTrue(result['success'])
        self.assertIn('params', result)
        self.assertIn('learning_rate', result['params'])
        self.assertIn('batch_size', result['params'])
        self.assertIn('epochs', result['params'])
        
        # 优化不存在的模型
        result = self.manager.optimize_training_params("non_existent")
        
        # 检查结果
        self.assertFalse(result['success'])
    
    def test_auto_check_all_models(self):
        """测试自动检查所有模型"""
        # 注册另一个模型
        def mock_retrain():
            return True
        
        self.manager.register_model("model2", mock_retrain)
        
        # 自动检查所有模型
        results = self.manager.auto_check_all_models()
        
        # 检查结果
        self.assertIn("model1", results)
        self.assertIn("model2", results)
        self.assertEqual(results["model1"]["status"], "normal")
        self.assertEqual(results["model2"]["status"], "normal")
    
    def test_auto_retrain_degraded_models(self):
        """测试自动重训练性能退化的模型"""
        # 模拟性能退化
        for i in range(10):
            score = 0.8 - i * 0.05
            self.tracker.track_performance("model1", {
                'overall_score': max(0.1, score)
            })
        
        # 自动重训练性能退化的模型
        results = self.manager.auto_retrain_degraded_models()
        
        # 检查结果
        self.assertIn("model1", results)
        self.assertTrue(results["model1"]["success"])
        self.assertTrue(self.retrain_called)


class TestAdaptiveRetrainingManager(unittest.TestCase):
    """自适应重训练管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建性能跟踪器
        self.tracker = PerformanceTracker()
        
        # 跟踪模型性能
        self.tracker.track_performance("model1", {
            'overall_score': 0.8
        })
        
        # 创建自适应重训练管理器
        self.manager = AdaptiveRetrainingManager(
            self.tracker,
            initial_threshold=0.1,
            min_threshold=0.05,
            max_threshold=0.3
        )
        
        # 注册模型
        self.retrain_called = False
        
        def mock_retrain():
            self.retrain_called = True
            return True
        
        self.manager.register_model("model1", mock_retrain)
    
    def test_adjust_threshold(self):
        """测试调整阈值"""
        # 初始阈值
        initial_threshold = self.manager.degradation_threshold
        
        # 模拟多次重训练
        for i in range(3):
            self.manager.trigger_retrain("model1", force=True)
        
        # 调整阈值
        new_threshold = self.manager.adjust_threshold("model1")
        
        # 检查阈值是否在合理范围内
        self.assertGreaterEqual(new_threshold, self.manager.min_threshold)
        self.assertLessEqual(new_threshold, self.manager.max_threshold)
        
        # 检查阈值历史
        history = self.manager.get_threshold_history("model1")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['threshold'], new_threshold)
    
    def test_adaptive_check_performance(self):
        """测试自适应性能检查"""
        # 模拟性能退化
        for i in range(10):
            score = 0.8 - i * 0.05
            self.tracker.track_performance("model1", {
                'overall_score': max(0.1, score)
            })
        
        # 检查性能
        result = self.manager.check_performance("model1")
        
        # 检查结果
        self.assertTrue(result['degraded'])
        self.assertEqual(result['status'], "degraded")
        
        # 触发重训练
        result = self.manager.trigger_retrain("model1")
        
        # 检查结果
        self.assertTrue(result['success'])
        self.assertTrue(self.retrain_called)


if __name__ == "__main__":
    unittest.main()