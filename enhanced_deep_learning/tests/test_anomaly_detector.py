#!/usr/bin/env python3
"""
异常检测器测试
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# 添加项目路径
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from enhanced_deep_learning.anomaly_detector import AnomalyDetector


class TestAnomalyDetector(unittest.TestCase):
    """异常检测器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = {
            'contamination': 0.1,
            'z_threshold': 3.0,
            'iqr_factor': 1.5
        }
        
        self.detector = AnomalyDetector(self.config)
        
        # 创建测试数据
        self.test_data = self._create_test_data()
        self.anomaly_data = self._create_anomaly_data()
    
    def _create_test_data(self) -> pd.DataFrame:
        """创建正常测试数据"""
        data = []
        
        for i in range(100):
            front_balls = sorted(np.random.choice(range(1, 36), 5, replace=False))
            back_balls = sorted(np.random.choice(range(1, 13), 2, replace=False))
            
            data.append({
                'issue': f'2024{i:03d}',
                'date': f'2024-01-{(i % 30) + 1:02d}',
                'front_balls': ','.join(map(str, front_balls)),
                'back_balls': ','.join(map(str, back_balls))
            })
        
        return pd.DataFrame(data)
    
    def _create_anomaly_data(self) -> pd.DataFrame:
        """创建包含异常的测试数据"""
        data = []
        
        for i in range(50):
            if i == 10:
                # 添加超出范围的异常
                front_balls = [1, 2, 3, 4, 50]  # 50超出范围
                back_balls = [1, 15]  # 15超出范围
            elif i == 20:
                # 添加重复号码异常
                front_balls = [1, 1, 2, 3, 4]  # 重复的1
                back_balls = [1, 1]  # 重复的1
            elif i == 30:
                # 添加连续号码异常
                front_balls = [1, 2, 3, 4, 5]  # 5个连续号码
                back_balls = [1, 2]
            elif i == 40:
                # 添加和值异常
                front_balls = [35, 34, 33, 32, 31]  # 和值过大
                back_balls = [12, 11]
            else:
                # 正常数据
                front_balls = sorted(np.random.choice(range(1, 36), 5, replace=False))
                back_balls = sorted(np.random.choice(range(1, 13), 2, replace=False))
            
            data.append({
                'issue': f'2024{i:03d}',
                'date': f'2024-01-{(i % 30) + 1:02d}',
                'front_balls': ','.join(map(str, front_balls)),
                'back_balls': ','.join(map(str, back_balls))
            })
        
        return pd.DataFrame(data)
    
    def test_init(self):
        """测试初始化"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.contamination, 0.1)
        self.assertEqual(self.detector.z_threshold, 3.0)
        self.assertEqual(self.detector.iqr_factor, 1.5)
    
    @patch('enhanced_deep_learning.anomaly_detector.logger_manager')
    def test_analyze_normal_data(self, mock_logger):
        """测试分析正常数据"""
        result = self.detector.analyze(self.test_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('data_quality', result)
        self.assertIn('statistical_anomalies', result)
        self.assertIn('pattern_anomalies', result)
        self.assertIn('isolation_forest_anomalies', result)
        self.assertIn('clustering_anomalies', result)
        self.assertIn('summary', result)
        
        # 检查摘要信息
        summary = result['summary']
        self.assertEqual(summary['total_records'], len(self.test_data))
        self.assertIn('total_anomalies', summary)
        self.assertIn('anomaly_rate', summary)
        self.assertIn('data_quality_score', summary)
    
    @patch('enhanced_deep_learning.anomaly_detector.logger_manager')
    def test_analyze_anomaly_data(self, mock_logger):
        """测试分析异常数据"""
        result = self.detector.analyze(self.anomaly_data)
        
        self.assertIsInstance(result, dict)
        
        # 应该检测到异常
        summary = result['summary']
        self.assertGreater(summary['total_anomalies'], 0)
        self.assertGreater(summary['anomaly_rate'], 0)
    
    def test_check_data_quality(self):
        """测试数据质量检查"""
        result = self.detector._check_data_quality(self.anomaly_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('quality_issues', result)
        self.assertIn('anomaly_count', result)
        
        # 应该检测到质量问题
        self.assertGreater(result['anomaly_count'], 0)
    
    def test_check_number_ranges(self):
        """测试号码范围检查"""
        issues = self.detector._check_number_ranges(self.anomaly_data)
        
        self.assertIsInstance(issues, list)
        # 应该检测到范围问题
        self.assertGreater(len(issues), 0)
    
    def test_detect_statistical_anomalies(self):
        """测试统计异常检测"""
        result = self.detector._detect_statistical_anomalies(self.test_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('anomalies', result)
        self.assertIn('anomaly_count', result)
    
    def test_detect_pattern_anomalies(self):
        """测试模式异常检测"""
        result = self.detector._detect_pattern_anomalies(self.anomaly_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('anomalies', result)
        self.assertIn('anomaly_count', result)
        
        # 应该检测到模式异常
        self.assertGreater(result['anomaly_count'], 0)
    
    def test_detect_consecutive_anomalies(self):
        """测试连续号码异常检测"""
        anomalies = self.detector._detect_consecutive_anomalies(self.anomaly_data)
        
        self.assertIsInstance(anomalies, list)
        # 应该检测到连续号码异常
        self.assertGreater(len(anomalies), 0)
        
        # 检查异常信息
        for anomaly in anomalies:
            self.assertIn('type', anomaly)
            self.assertIn('consecutive_count', anomaly)
            self.assertEqual(anomaly['type'], 'consecutive_numbers')
    
    def test_detect_repeat_pattern_anomalies(self):
        """测试重复模式异常检测"""
        # 创建包含重复组合的数据
        duplicate_data = self.test_data.copy()
        duplicate_data.loc[1] = duplicate_data.loc[0]  # 复制第一行
        
        anomalies = self.detector._detect_repeat_pattern_anomalies(duplicate_data)
        
        self.assertIsInstance(anomalies, list)
        # 应该检测到重复组合
        self.assertGreater(len(anomalies), 0)
    
    def test_detect_sum_anomalies(self):
        """测试和值异常检测"""
        anomalies = self.detector._detect_sum_anomalies(self.anomaly_data)
        
        self.assertIsInstance(anomalies, list)
        # 应该检测到和值异常
        self.assertGreater(len(anomalies), 0)
        
        # 检查异常信息
        for anomaly in anomalies:
            self.assertIn('type', anomaly)
            self.assertIn('sum', anomaly)
            self.assertIn('z_score', anomaly)
            self.assertEqual(anomaly['type'], 'sum_anomaly')
    
    def test_detect_isolation_forest_anomalies(self):
        """测试孤立森林异常检测"""
        result = self.detector._detect_isolation_forest_anomalies(self.test_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('anomalies', result)
        self.assertIn('anomaly_count', result)
    
    def test_detect_clustering_anomalies(self):
        """测试聚类异常检测"""
        result = self.detector._detect_clustering_anomalies(self.test_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('anomalies', result)
        self.assertIn('anomaly_count', result)
        self.assertIn('n_clusters', result)
    
    def test_extract_numerical_features(self):
        """测试数值特征提取"""
        features = self.detector._extract_numerical_features(self.test_data)
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features), 0)
        
        # 检查特征列
        expected_columns = [
            'front_1', 'front_2', 'front_3', 'front_4', 'front_5',
            'back_1', 'back_2',
            'front_sum', 'back_sum', 'front_span', 'back_span'
        ]
        
        for col in expected_columns:
            self.assertIn(col, features.columns)
    
    def test_calculate_quality_score(self):
        """测试质量分数计算"""
        # 创建测试结果
        results = {
            'summary': {
                'total_records': 100,
                'total_anomalies': 10
            },
            'data_quality': {
                'anomaly_count': 5
            }
        }
        
        score = self.detector._calculate_quality_score(results)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
    
    def test_empty_data(self):
        """测试空数据处理"""
        empty_data = pd.DataFrame()
        
        result = self.detector.analyze(empty_data)
        
        # 应该返回空结果而不是崩溃
        self.assertIsInstance(result, dict)
    
    def test_malformed_data(self):
        """测试格式错误的数据"""
        malformed_data = pd.DataFrame({
            'issue': ['2024001'],
            'front_balls': ['invalid'],
            'back_balls': ['invalid']
        })
        
        # 应该能处理格式错误的数据而不崩溃
        result = self.detector.analyze(malformed_data)
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
