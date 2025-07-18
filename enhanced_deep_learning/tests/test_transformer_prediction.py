#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transformer预测器预测功能测试
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
from enhanced_deep_learning.prediction_utils import PredictionProcessor, PredictionEvaluator
from enhanced_deep_learning.config import DEFAULT_TRANSFORMER_CONFIG


@unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow未安装，跳过测试")
class TestTransformerPrediction(unittest.TestCase):
    """Transformer预测器预测功能测试类"""
    
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
        
        # 训练模型（如果需要）
        if not self.predictor.is_trained:
            self.predictor.train(epochs=1, batch_size=8)
    
    def test_prediction_processor(self):
        """测试预测处理器"""
        processor = PredictionProcessor()
        
        # 测试原始预测处理
        raw_prediction = np.array([10.2, 15.7, 20.3, 25.8, 30.1, 5.4, 8.9])
        front_balls, back_balls = processor.process_raw_prediction(raw_prediction)
        
        # 检查处理结果
        self.assertEqual(len(front_balls), 5)
        self.assertEqual(len(back_balls), 2)
        self.assertTrue(all(1 <= x <= 35 for x in front_balls))
        self.assertTrue(all(1 <= x <= 12 for x in back_balls))
        
        # 测试格式化
        formatted = processor.format_prediction((front_balls, back_balls))
        self.assertIsInstance(formatted, str)
        self.assertIn("+", formatted)
    
    def test_prediction_with_details(self):
        """测试带详细信息的预测"""
        # 执行预测
        details = self.predictor.predict_with_details(count=2)
        
        # 检查返回结果
        self.assertEqual(details['model_name'], "Transformer")
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
        # 创建评估器
        evaluator = PredictionEvaluator()
        
        # 创建测试数据
        predictions = [
            ([1, 2, 3, 4, 5], [6, 7]),
            ([8, 9, 10, 11, 12], [6, 8])
        ]
        
        actuals = [
            ([1, 3, 5, 7, 9], [6, 8]),
            ([8, 10, 12, 14, 16], [7, 9])
        ]
        
        # 评估单个预测
        single_eval = evaluator.evaluate_prediction(predictions[0], actuals[0])
        self.assertIn('front_matches', single_eval)
        self.assertIn('back_matches', single_eval)
        self.assertIn('prize_level', single_eval)
        
        # 评估多个预测
        multi_eval = evaluator.evaluate_multiple_predictions(predictions, actuals)
        self.assertIn('avg_front_matches', multi_eval)
        self.assertIn('avg_back_matches', multi_eval)
        self.assertIn('prize_rate', multi_eval)
    
    def test_predictor_evaluation(self):
        """测试预测器评估功能"""
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