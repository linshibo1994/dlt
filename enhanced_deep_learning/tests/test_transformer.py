#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transformer预测器测试
"""

import unittest
import numpy as np
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from enhanced_deep_learning.transformer_predictor import TransformerPredictor
from enhanced_deep_learning.config import DEFAULT_TRANSFORMER_CONFIG


@unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow未安装，跳过测试")
class TestTransformerPredictor(unittest.TestCase):
    """Transformer预测器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 使用测试配置
        self.test_config = DEFAULT_TRANSFORMER_CONFIG.copy()
        self.test_config.update({
            'd_model': 32,
            'num_heads': 2,
            'num_layers': 1,
            'epochs': 2,
            'batch_size': 16
        })
        
        # 创建预测器实例
        self.predictor = TransformerPredictor(config=self.test_config)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.predictor.name, "Transformer")
        self.assertEqual(self.predictor.d_model, 32)
        self.assertEqual(self.predictor.num_heads, 2)
        self.assertEqual(self.predictor.num_layers, 1)
        self.assertFalse(self.predictor.is_trained)
    
    def test_build_model(self):
        """测试模型构建"""
        model = self.predictor._build_model()
        
        # 检查模型类型
        self.assertIsInstance(model, tf.keras.Model)
        
        # 检查输入形状
        self.assertEqual(model.input_shape, (None, self.predictor.sequence_length, self.predictor.feature_dim))
        
        # 检查输出形状
        self.assertEqual(model.output_shape, (None, 7))
        
        # 检查模型层数
        # 输入层 + 嵌入层 + (注意力+规范化+前馈+规范化)*层数 + 池化 + 全连接 + Dropout + 输出
        expected_layers = 1 + 1 + (4 * self.predictor.num_layers) + 1 + 1 + 1 + 1
        self.assertGreaterEqual(len(model.layers), expected_layers)
    
    def test_fallback_config(self):
        """测试备用配置"""
        original_d_model = self.predictor.d_model
        self.predictor.use_fallback_config()
        
        # 检查配置是否已更改
        self.assertNotEqual(self.predictor.d_model, original_d_model)
        self.assertEqual(self.predictor.d_model, 64)
        self.assertEqual(self.predictor.num_heads, 4)
    
    def test_simple_model(self):
        """测试简单模型"""
        self.predictor.use_simple_model()
        
        # 检查配置是否已更改为简单模型
        self.assertEqual(self.predictor.d_model, 32)
        self.assertEqual(self.predictor.num_heads, 2)
        self.assertEqual(self.predictor.num_layers, 1)
    
    def test_get_confidence(self):
        """测试置信度计算"""
        # 未训练时置信度应为0
        self.assertEqual(self.predictor.get_confidence(), 0.0)
        
        # 模拟训练完成
        self.predictor.is_trained = True
        
        # 检查置信度是否在合理范围内
        confidence = self.predictor.get_confidence()
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 0.95)


if __name__ == "__main__":
    unittest.main()