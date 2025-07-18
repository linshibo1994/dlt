#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GAN预测器测试
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
class TestGANPredictor(unittest.TestCase):
    """GAN预测器测试类"""
    
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
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.predictor.name, "GAN")
        self.assertEqual(self.predictor.latent_dim, 20)
        self.assertEqual(self.predictor.generator_layers, [32, 64])
        self.assertEqual(self.predictor.discriminator_layers, [32])
        self.assertFalse(self.predictor.is_trained)
    
    def test_build_generator(self):
        """测试生成器构建"""
        generator = self.predictor._build_generator()
        
        # 检查模型类型
        self.assertIsInstance(generator, tf.keras.Model)
        
        # 检查输入形状
        self.assertEqual(generator.input_shape, (None, self.predictor.latent_dim))
        
        # 检查输出形状
        self.assertEqual(generator.output_shape, (None, 7))
        
        # 检查模型层数
        # 输入层 + (Dense+BN+LeakyReLU)*层数 + 输出层
        expected_layers = 1 + (3 * len(self.predictor.generator_layers)) + 1
        self.assertEqual(len(generator.layers), expected_layers)
    
    def test_build_discriminator(self):
        """测试判别器构建"""
        discriminator = self.predictor._build_discriminator()
        
        # 检查模型类型
        self.assertIsInstance(discriminator, tf.keras.Model)
        
        # 检查输入形状
        self.assertEqual(discriminator.input_shape, (None, 7))
        
        # 检查输出形状
        self.assertEqual(discriminator.output_shape, (None, 1))
        
        # 检查模型层数
        # 输入层 + (Dense+LeakyReLU+Dropout)*层数 + 输出层
        expected_layers = 1 + (3 * len(self.predictor.discriminator_layers)) + 1
        self.assertEqual(len(discriminator.layers), expected_layers)
    
    def test_build_gan(self):
        """测试GAN构建"""
        # 先构建生成器和判别器
        self.predictor.generator = self.predictor._build_generator()
        self.predictor.discriminator = self.predictor._build_discriminator()
        
        # 构建GAN
        gan = self.predictor._build_gan()
        
        # 检查模型类型
        self.assertIsInstance(gan, tf.keras.Model)
        
        # 检查输入形状
        self.assertEqual(gan.input_shape, (None, self.predictor.latent_dim))
        
        # 检查输出形状
        self.assertEqual(gan.output_shape, (None, 1))
        
        # 检查判别器是否被冻结
        self.assertFalse(self.predictor.discriminator.trainable)
    
    def test_build_model(self):
        """测试模型构建"""
        model = self.predictor._build_model()
        
        # 检查是否返回生成器
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.output_shape, (None, 7))
        
        # 检查是否创建了所有组件
        self.assertIsNotNone(self.predictor.generator)
        self.assertIsNotNone(self.predictor.discriminator)
        self.assertIsNotNone(self.predictor.gan)
    
    def test_fallback_config(self):
        """测试备用配置"""
        original_latent_dim = self.predictor.latent_dim
        self.predictor.use_fallback_config()
        
        # 检查配置是否已更改
        self.assertNotEqual(self.predictor.latent_dim, original_latent_dim)
        self.assertEqual(self.predictor.latent_dim, 50)
        self.assertEqual(self.predictor.generator_layers, [64, 128, 64])
    
    def test_simple_model(self):
        """测试简单模型"""
        self.predictor.use_simple_model()
        
        # 检查配置是否已更改为简单模型
        self.assertEqual(self.predictor.latent_dim, 20)
        self.assertEqual(self.predictor.generator_layers, [32, 64])
        self.assertEqual(self.predictor.discriminator_layers, [32])
    
    def test_get_confidence(self):
        """测试置信度计算"""
        # 未训练时置信度应为0
        self.assertEqual(self.predictor.get_confidence(), 0.0)
        
        # 模拟训练完成
        self.predictor.is_trained = True
        
        # 检查置信度是否在合理范围内
        confidence = self.predictor.get_confidence()
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 0.85)


if __name__ == "__main__":
    unittest.main()