#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据增强器测试
"""

import unittest
import os
import sys
import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from enhanced_deep_learning.data_augmentor import DataAugmentor


class TestDataAugmentor(unittest.TestCase):
    """数据增强器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建数据增强器
        self.augmentor = DataAugmentor()
        
        # 创建测试数据
        self.test_data = np.random.random((100, 10))
        
        # 创建测试标签
        self.test_labels = np.random.randint(0, 3, 100)
    
    def test_augment_data(self):
        """测试数据增强"""
        # 噪声增强
        augmented_data = self.augmentor.augment_data(self.test_data, factor=1.5, method='noise')
        
        # 检查增强结果
        self.assertEqual(augmented_data.shape[0], int(len(self.test_data) * 1.5))
        self.assertEqual(augmented_data.shape[1], self.test_data.shape[1])
        
        # Bootstrap增强
        augmented_data = self.augmentor.augment_data(self.test_data, factor=2.0, method='bootstrap')
        
        # 检查增强结果
        self.assertEqual(augmented_data.shape[0], int(len(self.test_data) * 2.0))
        self.assertEqual(augmented_data.shape[1], self.test_data.shape[1])
        
        # 测试增强因子小于等于1.0的情况
        augmented_data = self.augmentor.augment_data(self.test_data, factor=1.0)
        
        # 检查结果（应该返回原始数据）
        self.assertEqual(augmented_data.shape, self.test_data.shape)
    
    def test_generate_synthetic_data(self):
        """测试合成数据生成"""
        # 随机生成
        synthetic_data = self.augmentor.generate_synthetic_data(50, 10, method='random')
        
        # 检查生成结果
        self.assertEqual(synthetic_data.shape, (50, 10))
        self.assertTrue(np.all(synthetic_data >= 0))
        self.assertTrue(np.all(synthetic_data <= 1))
        
        # 高斯分布生成
        synthetic_data = self.augmentor.generate_synthetic_data(50, 10, method='gaussian')
        
        # 检查生成结果
        self.assertEqual(synthetic_data.shape, (50, 10))
        self.assertTrue(np.all(synthetic_data >= 0))
        self.assertTrue(np.all(synthetic_data <= 1))
        
        # 均匀分布生成
        synthetic_data = self.augmentor.generate_synthetic_data(50, 10, method='uniform')
        
        # 检查生成结果
        self.assertEqual(synthetic_data.shape, (50, 10))
        self.assertTrue(np.all(synthetic_data >= 0))
        self.assertTrue(np.all(synthetic_data <= 1))
    
    def test_balance_data(self):
        """测试数据平衡"""
        # 平衡数据
        balanced_data, balanced_labels = self.augmentor.balance_data(self.test_data, self.test_labels)
        
        # 检查平衡结果
        self.assertEqual(balanced_data.shape[1], self.test_data.shape[1])
        
        # 检查各类别样本数量是否相等
        unique_labels, counts = np.unique(balanced_labels, return_counts=True)
        self.assertTrue(np.all(counts == counts[0]))
        
        # 测试数据和标签长度不一致的情况
        balanced_data, balanced_labels = self.augmentor.balance_data(self.test_data, self.test_labels[:50])
        
        # 检查结果（应该返回原始数据和标签）
        self.assertEqual(len(balanced_data), len(balanced_labels))
    
    def test_mix_data(self):
        """测试数据混合"""
        # 创建第二组测试数据
        test_data2 = np.random.random((50, 10))
        
        # 混合数据
        mixed_data = self.augmentor.mix_data(self.test_data, test_data2, ratio=0.7)
        
        # 检查混合结果
        self.assertEqual(mixed_data.shape[1], self.test_data.shape[1])
        self.assertEqual(len(mixed_data), len(self.test_data) + len(test_data2))
        
        # 测试第一组数据为空的情况
        mixed_data = self.augmentor.mix_data(np.array([]), test_data2)
        
        # 检查结果（应该返回第二组数据）
        self.assertEqual(mixed_data.shape, test_data2.shape)
        
        # 测试第二组数据为空的情况
        mixed_data = self.augmentor.mix_data(self.test_data, np.array([]))
        
        # 检查结果（应该返回第一组数据）
        self.assertEqual(mixed_data.shape, self.test_data.shape)
        
        # 测试维度不一致的情况
        mixed_data = self.augmentor.mix_data(self.test_data, np.random.random((50, 5)))
        
        # 检查结果（应该返回第一组数据）
        self.assertEqual(mixed_data.shape, self.test_data.shape)


if __name__ == "__main__":
    unittest.main()