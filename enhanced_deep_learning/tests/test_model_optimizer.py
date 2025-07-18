#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型优化器测试
"""

import unittest
import os
import sys
import numpy as np
import time
import tempfile
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from enhanced_deep_learning.model_optimizer import ModelOptimizer, CachedModel


class TestModelOptimizer(unittest.TestCase):
    """模型优化器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时缓存目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建模型优化器
        self.optimizer = ModelOptimizer(cache_dir=self.temp_dir)
    
    def tearDown(self):
        """测试后清理"""
        # 清理临时目录
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cache_result_decorator(self):
        """测试结果缓存装饰器"""
        # 创建测试函数
        call_count = [0]  # 使用列表以便在闭包中修改
        
        @self.optimizer.cache_result
        def test_function(x, y=1):
            call_count[0] += 1
            return x * y
        
        # 第一次调用
        result1 = test_function(10)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count[0], 1)
        
        # 第二次调用（相同参数，应该使用缓存）
        result2 = test_function(10)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count[0], 1)  # 不应该增加
        
        # 不同参数的调用
        result3 = test_function(10, y=2)
        self.assertEqual(result3, 20)
        self.assertEqual(call_count[0], 2)
        
        # 清除缓存
        self.optimizer.clear_result_cache()
        
        # 再次调用（应该重新计算）
        result4 = test_function(10)
        self.assertEqual(result4, 10)
        self.assertEqual(call_count[0], 3)
    
    def test_resource_monitoring(self):
        """测试资源监控"""
        # 开始资源监控
        self.optimizer.start_resource_monitoring()
        
        # 记录资源使用情况
        for _ in range(3):
            usage = self.optimizer.record_resource_usage()
            
            # 检查返回值
            self.assertIn('cpu', usage)
            self.assertIn('memory', usage)
            self.assertIn('disk', usage)
            self.assertIn('time', usage)
            
            # 检查值的范围
            self.assertGreaterEqual(usage['cpu'], 0)
            self.assertLessEqual(usage['cpu'], 100)
            self.assertGreaterEqual(usage['memory'], 0)
            self.assertLessEqual(usage['memory'], 100)
            self.assertGreaterEqual(usage['disk'], 0)
            self.assertLessEqual(usage['disk'], 100)
            
            time.sleep(0.1)
        
        # 获取资源使用情况摘要
        summary = self.optimizer.get_resource_usage_summary()
        
        # 检查摘要
        self.assertIn('cpu', summary)
        self.assertIn('memory', summary)
        self.assertIn('disk', summary)
        
        for resource in ['cpu', 'memory', 'disk']:
            self.assertIn('min', summary[resource])
            self.assertIn('max', summary[resource])
            self.assertIn('avg', summary[resource])
            self.assertIn('current', summary[resource])
        
        # 保存资源使用情况报告
        report_path = self.optimizer.save_resource_usage_report()
        
        # 检查报告文件是否存在
        self.assertTrue(os.path.exists(report_path))
        
        # 检查报告内容
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        self.assertIn('summary', report)
        self.assertIn('details', report)
    
    def test_cached_model(self):
        """测试缓存模型"""
        # 创建模拟模型
        class MockModel:
            def __init__(self):
                self.call_count = 0
            
            def predict(self, inputs):
                self.call_count += 1
                return inputs * 2
        
        mock_model = MockModel()
        
        # 创建缓存模型
        cached_model = CachedModel(mock_model, self.optimizer, cache_size=10)
        
        # 创建测试数据
        test_data1 = np.array([[1, 2, 3]])
        test_data2 = np.array([[4, 5, 6]])
        
        # 第一次预测
        result1 = cached_model.predict(test_data1)
        np.testing.assert_array_equal(result1, test_data1 * 2)
        self.assertEqual(mock_model.call_count, 1)
        
        # 第二次预测（相同输入，应该使用缓存）
        result2 = cached_model.predict(test_data1)
        np.testing.assert_array_equal(result2, test_data1 * 2)
        self.assertEqual(mock_model.call_count, 1)  # 不应该增加
        
        # 不同输入的预测
        result3 = cached_model.predict(test_data2)
        np.testing.assert_array_equal(result3, test_data2 * 2)
        self.assertEqual(mock_model.call_count, 2)
        
        # 获取缓存统计信息
        stats = cached_model.get_cache_stats()
        
        # 检查统计信息
        self.assertEqual(stats['cache_size'], 2)
        self.assertEqual(stats['max_cache_size'], 10)
        self.assertEqual(stats['cache_hits'], 1)
        self.assertEqual(stats['cache_misses'], 2)
        self.assertAlmostEqual(stats['hit_rate'], 1/3)
        
        # 清除缓存
        cached_model.clear_cache()
        
        # 再次预测（应该重新计算）
        result4 = cached_model.predict(test_data1)
        np.testing.assert_array_equal(result4, test_data1 * 2)
        self.assertEqual(mock_model.call_count, 3)
        
        # 检查缓存统计信息
        stats = cached_model.get_cache_stats()
        self.assertEqual(stats['cache_size'], 1)
        self.assertEqual(stats['cache_hits'], 1)  # 不变
        self.assertEqual(stats['cache_misses'], 3)  # 增加


if __name__ == "__main__":
    unittest.main()