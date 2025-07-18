#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据预处理器测试
"""

import unittest
import os
import sys
import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from enhanced_deep_learning.data_preprocessor import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """数据预处理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建数据预处理器
        self.preprocessor = DataPreprocessor(cache_enabled=True)
        
        # 创建测试数据
        self.test_data = np.random.random((100, 10))
        
        # 添加一些异常值
        self.test_data[0] = 10.0  # 异常值
    
    def test_normalize_data(self):
        """测试数据标准化"""
        # MinMax标准化
        normalized_data = self.preprocessor.normalize_data(self.test_data, method='minmax')
        
        # 检查标准化结果
        self.assertEqual(normalized_data.shape, self.test_data.shape)
        self.assertTrue(np.all(normalized_data >= 0))
        self.assertTrue(np.all(normalized_data <= 1))
        
        # Standard标准化
        normalized_data = self.preprocessor.normalize_data(self.test_data, method='standard')
        
        # 检查标准化结果
        self.assertEqual(normalized_data.shape, self.test_data.shape)
        
        # Robust标准化
        normalized_data = self.preprocessor.normalize_data(self.test_data, method='robust')
        
        # 检查标准化结果
        self.assertEqual(normalized_data.shape, self.test_data.shape)
    
    def test_denormalize_data(self):
        """测试数据反标准化"""
        # 先标准化
        normalized_data = self.preprocessor.normalize_data(self.test_data, method='minmax')
        
        # 再反标准化
        denormalized_data = self.preprocessor.denormalize_data(normalized_data, method='minmax')
        
        # 检查反标准化结果
        self.assertEqual(denormalized_data.shape, self.test_data.shape)
        
        # 检查反标准化后的数据是否接近原始数据
        np.testing.assert_allclose(denormalized_data, self.test_data, rtol=1e-5)
    
    def test_detect_anomalies(self):
        """测试异常检测"""
        # 隔离森林异常检测
        normal_data, anomaly_data = self.preprocessor.detect_anomalies(self.test_data, method='isolation_forest')
        
        # 检查异常检测结果
        self.assertEqual(len(normal_data) + len(anomaly_data), len(self.test_data))
        
        # Z分数异常检测
        normal_data, anomaly_data = self.preprocessor.detect_anomalies(self.test_data, method='zscore')
        
        # 检查异常检测结果
        self.assertEqual(len(normal_data) + len(anomaly_data), len(self.test_data))
        
        # 检查异常值是否被检测出来
        self.assertGreater(len(anomaly_data), 0)
    
    def test_reduce_dimensions(self):
        """测试降维"""
        # PCA降维
        reduced_data = self.preprocessor.reduce_dimensions(self.test_data, n_components=5)
        
        # 检查降维结果
        self.assertEqual(reduced_data.shape[0], self.test_data.shape[0])
        self.assertEqual(reduced_data.shape[1], 5)
        
        # 测试目标维度大于原始维度的情况
        reduced_data = self.preprocessor.reduce_dimensions(self.test_data, n_components=20)
        
        # 检查结果（应该返回原始数据）
        self.assertEqual(reduced_data.shape, self.test_data.shape)


if __name__ == "__main__":
    unittest.main()