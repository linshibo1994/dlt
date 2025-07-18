#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transformer预测器训练功能测试
"""

import unittest
import os
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from enhanced_deep_learning.transformer_predictor import TransformerPredictor
from enhanced_deep_learning.data_manager import DeepLearningDataManager
from enhanced_deep_learning.config import DEFAULT_TRANSFORMER_CONFIG


@unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow未安装，跳过测试")
class TestTransformerTraining(unittest.TestCase):
    """Transformer预测器训练功能测试类"""
    
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
    
    def test_data_preparation(self):
        """测试数据准备"""
        # 创建数据管理器
        data_manager = DeepLearningDataManager()
        
        # 准备批处理数据
        batch_data = data_manager.prepare_batch_data(
            sequence_length=self.predictor.sequence_length,
            batch_size=16,
            validation_split=0.2
        )
        
        # 检查返回的数据
        self.assertIn('train_dataset', batch_data)
        self.assertIn('val_dataset', batch_data)
        self.assertIn('X_train', batch_data)
        self.assertIn('y_train', batch_data)
        self.assertIn('feature_dim', batch_data)
        
        # 检查数据形状
        self.assertEqual(batch_data['X_train'].shape[1], self.predictor.sequence_length)
        self.assertEqual(batch_data['X_train'].shape[2], batch_data['feature_dim'])
        self.assertEqual(batch_data['y_train'].shape[1], 7)  # 7个输出（5前区+2后区）
    
    def test_mini_training(self):
        """测试小规模训练"""
        # 设置极小的训练参数
        mini_epochs = 1
        mini_batch_size = 8
        
        # 执行训练
        result = self.predictor.train(epochs=mini_epochs, batch_size=mini_batch_size)
        
        # 检查训练结果
        self.assertTrue(result)
        self.assertTrue(self.predictor.is_trained)
        self.assertIsNotNone(self.predictor.model)
    
    def test_prediction_after_training(self):
        """测试训练后的预测"""
        # 先进行小规模训练
        self.predictor.train(epochs=1, batch_size=8)
        
        # 执行预测
        predictions = self.predictor.predict(count=2)
        
        # 检查预测结果
        self.assertEqual(len(predictions), 2)
        
        for front, back in predictions:
            # 检查前区号码
            self.assertEqual(len(front), 5)
            self.assertTrue(all(1 <= x <= 35 for x in front))
            self.assertEqual(len(set(front)), 5)  # 确保唯一性
            
            # 检查后区号码
            self.assertEqual(len(back), 2)
            self.assertTrue(all(1 <= x <= 12 for x in back))
            self.assertEqual(len(set(back)), 2)  # 确保唯一性


if __name__ == "__main__":
    unittest.main()