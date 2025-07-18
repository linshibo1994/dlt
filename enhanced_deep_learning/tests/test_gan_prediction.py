#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GAN预测器预测功能测试
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
class TestGANPrediction(unittest.TestCase):
    """GAN预测器预测功能测试类"""
    
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
        
        # 训练模型（如果需要）
        if not self.predictor.is_trained:
            self.predictor.train(epochs=1, batch_size=8)
    
    def test_select_best_samples(self):
        """测试最佳样本选择"""
        # 生成测试样本
        samples = np.random.rand(10, 7)
        
        # 选择最佳样本
        best_samples = self.predictor._select_best_samples(samples, count=3)
        
        # 检查结果
        self.assertEqual(len(best_samples), 3)
        self.assertEqual(best_samples.shape, (3, 7))
    
    def test_prediction(self):
        """测试预测功能"""
        # 执行预测
        predictions = self.predictor.predict(count=2, verbose=False)
        
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
    
    def test_prediction_with_details(self):
        """测试带详细信息的预测"""
        # 执行预测
        details = self.predictor.predict_with_details(count=2)
        
        # 检查返回结果
        self.assertEqual(details['model_name'], "GAN")
        self.assertEqual(details['count'], 2)
        self.assertEqual(len(details['predictions']), 2)
        self.assertIn('confidence', details)
        self.assertIn('model_config', details)
        self.assertIn('timestamp', details)
        
        # 检查预测结果格式
        for pred in details['predictions']:
            self.assertIn('index', pred)
            self.assertIn('front_balls', pred)
            self.assertIn('back_balls', pred)
            self.assertIn('formatted', pred)
    
    def test_prediction_evaluation(self):
        """测试预测评估"""
        # 执行预测
        predictions = self.predictor.predict(count=2, verbose=False)
        
        # 创建模拟实际结果
        actuals = [
            ([1, 3, 5, 7, 9], [2, 4]),
            ([2, 4, 6, 8, 10], [3, 6])
        ]
        
        # 评估预测结果
        eval_results = self.predictor.evaluate_predictions(predictions, actuals)
        
        # 检查评估结果
        self.assertIn('avg_front_matches', eval_results)
        self.assertIn('avg_back_matches', eval_results)
        self.assertIn('prize_rate', eval_results)


if __name__ == "__main__":
    unittest.main()