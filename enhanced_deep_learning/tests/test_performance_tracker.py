#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
性能跟踪系统测试
"""

import unittest
import os
import sys
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from enhanced_deep_learning.performance_tracker import ModelPerformanceMetrics, PerformanceTracker


class TestModelPerformanceMetrics(unittest.TestCase):
    """模型性能指标测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.metrics = ModelPerformanceMetrics("test_model")
    
    def test_add_metrics(self):
        """测试添加性能指标"""
        # 添加性能指标
        self.metrics.add_metrics({
            'hit_rate': 0.6,
            'accuracy': 0.7,
            'confidence': 0.8,
            'prize_rate': 0.2,
            'front_matches': 2.5,
            'back_matches': 1.0
        })
        
        # 检查指标是否已添加
        self.assertEqual(len(self.metrics.hit_rates), 1)
        self.assertEqual(len(self.metrics.accuracies), 1)
        self.assertEqual(len(self.metrics.confidence_scores), 1)
        self.assertEqual(len(self.metrics.prize_rates), 1)
        self.assertEqual(len(self.metrics.front_matches), 1)
        self.assertEqual(len(self.metrics.back_matches), 1)
        self.assertEqual(len(self.metrics.timestamps), 1)
        self.assertEqual(len(self.metrics.overall_scores), 1)
        
        # 检查指标值
        self.assertEqual(self.metrics.hit_rates[0], 0.6)
        self.assertEqual(self.metrics.accuracies[0], 0.7)
        self.assertEqual(self.metrics.confidence_scores[0], 0.8)
        self.assertEqual(self.metrics.prize_rates[0], 0.2)
        self.assertEqual(self.metrics.front_matches[0], 2.5)
        self.assertEqual(self.metrics.back_matches[0], 1.0)
    
    def test_calculate_overall_score(self):
        """测试计算综合得分"""
        # 创建测试指标
        metrics = {
            'hit_rate': 0.6,
            'accuracy': 0.7,
            'confidence': 0.8,
            'prize_rate': 0.2,
            'front_matches': 2.5 / 5.0,  # 归一化
            'back_matches': 1.0 / 2.0    # 归一化
        }
        
        # 计算综合得分
        score = self.metrics._calculate_overall_score(metrics)
        
        # 检查得分是否在合理范围内
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # 检查得分是否正确
        expected_score = (
            0.6 * 0.2 +  # hit_rate
            0.7 * 0.3 +  # accuracy
            0.8 * 0.1 +  # confidence
            0.2 * 0.3 +  # prize_rate
            0.5 * 0.05 + # front_matches
            0.5 * 0.05   # back_matches
        )
        self.assertAlmostEqual(score, expected_score)
    
    def test_get_average_metrics(self):
        """测试获取平均性能指标"""
        # 添加多组性能指标
        self.metrics.add_metrics({
            'hit_rate': 0.6,
            'accuracy': 0.7
        })
        
        self.metrics.add_metrics({
            'hit_rate': 0.8,
            'accuracy': 0.9
        })
        
        # 获取平均指标
        avg_metrics = self.metrics.get_average_metrics()
        
        # 检查平均指标
        self.assertAlmostEqual(avg_metrics['hit_rate'], 0.7)
        self.assertAlmostEqual(avg_metrics['accuracy'], 0.8)
        
        # 测试窗口参数
        avg_metrics = self.metrics.get_average_metrics(window=1)
        self.assertAlmostEqual(avg_metrics['hit_rate'], 0.8)
        self.assertAlmostEqual(avg_metrics['accuracy'], 0.9)
    
    def test_get_trend(self):
        """测试获取指标趋势"""
        # 添加多组性能指标
        self.metrics.add_metrics({
            'hit_rate': 0.6,
            'accuracy': 0.7
        })
        
        self.metrics.add_metrics({
            'hit_rate': 0.7,
            'accuracy': 0.8
        })
        
        self.metrics.add_metrics({
            'hit_rate': 0.8,
            'accuracy': 0.9
        })
        
        # 获取趋势
        hit_rate_trend = self.metrics.get_trend('hit_rate')
        accuracy_trend = self.metrics.get_trend('accuracy')
        
        # 检查趋势
        self.assertEqual(hit_rate_trend, [0.6, 0.7, 0.8])
        self.assertEqual(accuracy_trend, [0.7, 0.8, 0.9])
        
        # 测试窗口参数
        hit_rate_trend = self.metrics.get_trend('hit_rate', window=2)
        self.assertEqual(hit_rate_trend, [0.7, 0.8])
    
    def test_to_dict_and_from_dict(self):
        """测试转换为字典和从字典创建实例"""
        # 添加性能指标
        self.metrics.add_metrics({
            'hit_rate': 0.6,
            'accuracy': 0.7
        })
        
        # 转换为字典
        data = self.metrics.to_dict()
        
        # 检查字典
        self.assertEqual(data['model_id'], "test_model")
        self.assertEqual(data['hit_rates'], [0.6])
        self.assertEqual(data['accuracies'], [0.7])
        
        # 从字典创建实例
        new_metrics = ModelPerformanceMetrics.from_dict(data)
        
        # 检查实例
        self.assertEqual(new_metrics.model_id, "test_model")
        self.assertEqual(new_metrics.hit_rates, [0.6])
        self.assertEqual(new_metrics.accuracies, [0.7])


class TestPerformanceTracker(unittest.TestCase):
    """性能跟踪器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.tracker = PerformanceTracker(history_size=10)
    
    def test_track_performance(self):
        """测试跟踪模型性能"""
        # 跟踪模型性能
        self.tracker.track_performance("model1", {
            'hit_rate': 0.6,
            'accuracy': 0.7
        })
        
        # 检查模型是否已添加
        self.assertIn("model1", self.tracker.model_metrics)
        
        # 检查性能指标
        metrics = self.tracker.get_model_metrics("model1")
        self.assertEqual(metrics.hit_rates, [0.6])
        self.assertEqual(metrics.accuracies, [0.7])
    
    def test_limit_history_size(self):
        """测试限制历史记录大小"""
        # 添加超过历史记录大小的性能指标
        for i in range(15):
            self.tracker.track_performance("model1", {
                'hit_rate': 0.5 + i * 0.01,
                'accuracy': 0.6 + i * 0.01
            })
        
        # 检查历史记录大小
        metrics = self.tracker.get_model_metrics("model1")
        self.assertEqual(len(metrics.hit_rates), 10)
        self.assertEqual(len(metrics.accuracies), 10)
        
        # 检查是否保留了最新的记录
        self.assertAlmostEqual(metrics.hit_rates[-1], 0.5 + 14 * 0.01)
        self.assertAlmostEqual(metrics.accuracies[-1], 0.6 + 14 * 0.01)
    
    def test_get_best_model(self):
        """测试获取最佳模型"""
        # 添加多个模型的性能指标
        self.tracker.track_performance("model1", {
            'hit_rate': 0.6,
            'accuracy': 0.7,
            'overall_score': 0.65
        })
        
        self.tracker.track_performance("model2", {
            'hit_rate': 0.7,
            'accuracy': 0.8,
            'overall_score': 0.75
        })
        
        # 获取最佳模型
        best_model = self.tracker.get_best_model()
        
        # 检查最佳模型
        self.assertEqual(best_model, "model2")
    
    def test_compare_models(self):
        """测试比较多个模型"""
        # 添加多个模型的性能指标
        self.tracker.track_performance("model1", {
            'hit_rate': 0.6,
            'accuracy': 0.7,
            'overall_score': 0.65
        })
        
        self.tracker.track_performance("model2", {
            'hit_rate': 0.7,
            'accuracy': 0.8,
            'overall_score': 0.75
        })
        
        # 比较模型
        comparison = self.tracker.compare_models(["model1", "model2"], metric_name='accuracy')
        
        # 检查比较结果
        self.assertEqual(comparison["model1"], 0.7)
        self.assertEqual(comparison["model2"], 0.8)
    
    def test_detect_performance_degradation(self):
        """测试检测性能退化"""
        # 添加性能稳定的指标
        for i in range(20):
            self.tracker.track_performance("stable_model", {
                'overall_score': 0.7
            })
        
        # 添加性能退化的指标
        for i in range(10):
            self.tracker.track_performance("degrading_model", {
                'overall_score': 0.7
            })
        
        for i in range(10):
            self.tracker.track_performance("degrading_model", {
                'overall_score': 0.5
            })
        
        # 检测性能退化
        stable_degraded = self.tracker.detect_performance_degradation("stable_model", threshold=0.1, window=5)
        degrading_degraded = self.tracker.detect_performance_degradation("degrading_model", threshold=0.1, window=5)
        
        # 检查检测结果
        self.assertFalse(stable_degraded)
        self.assertTrue(degrading_degraded)


if __name__ == "__main__":
    unittest.main()