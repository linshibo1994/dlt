#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GAN预测器训练功能测试
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

from enhanced_deep_learning.gan_predictor import GANPredictor
from enhanced_deep_learning.config import DEFAULT_GAN_CONFIG


@unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow未安装，跳过测试")
class TestGANTraining(unittest.TestCase):
    """GAN预测器训练功能测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 使用测试配置
        self.test_config = DEFAULT_GAN_CONFIG.copy()
        self.test_config.update({
            'latent_dim': 20,
            'generator_layers': [32, 64],
            'discriminator_layers': [32],
            'epochs': 2,
            'batch_size': 16
        })
        
        # 创建预测器实例
        self.predictor = GANPredictor(config=self.test_config)
    
    def test_prepare_training_data(self):
        """测试训练数据准备"""
        # 准备训练数据
        real_samples = self.predictor._prepare_training_data()
        
        # 检查数据形状
        self.assertIsInstance(real_samples, np.ndarray)
        self.assertEqual(real_samples.shape[1], 7)  # 5前区+2后区
        
        # 检查数据范围
        self.assertTrue(np.all(real_samples >= 0))
        self.assertTrue(np.all(real_samples <= 1))
    
    def test_generate_samples(self):
        """测试样本生成"""
        # 构建模型
        self.predictor._build_model()
        
        # 生成样本
        samples = self.predictor._generate_samples(count=5)
        
        # 检查样本形状
        self.assertEqual(samples.shape, (5, 7))
        
        # 检查样本范围
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples <= 1))
    
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
        self.assertIsNotNone(self.predictor.generator)
        self.assertIsNotNone(self.predictor.discriminator)
        self.assertIsNotNone(self.predictor.gan)
    
    def test_apply_gradient_penalty(self):
        """测试梯度惩罚"""
        # 构建模型
        self.predictor._build_model()
        
        # 创建测试数据
        real_samples = np.random.rand(10, 7)
        fake_samples = np.random.rand(10, 7)
        
        # 应用梯度惩罚
        try:
            penalty = self.predictor._apply_gradient_penalty(real_samples, fake_samples)
            self.assertIsNotNone(penalty)
        except Exception as e:
            self.fail(f"应用梯度惩罚失败: {e}")


if __name__ == "__main__":
    unittest.main()