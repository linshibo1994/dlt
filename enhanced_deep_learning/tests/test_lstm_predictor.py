#!/usr/bin/env python3
"""
LSTM预测器测试
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# 添加项目路径
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from enhanced_deep_learning.lstm_predictor import LSTMPredictor
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


@unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow not available")
class TestLSTMPredictor(unittest.TestCase):
    """LSTM预测器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        
        self.config = {
            'sequence_length': 10,
            'lstm_units': [32, 16],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 16,
            'epochs': 5,
            'model_dir': self.temp_dir
        }
        
        self.predictor = LSTMPredictor(self.config)
        
        # 创建测试数据
        self.test_data = self._create_test_data()
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_data(self) -> pd.DataFrame:
        """创建测试数据"""
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
    
    def test_init(self):
        """测试初始化"""
        self.assertIsNotNone(self.predictor)
        self.assertEqual(self.predictor.sequence_length, 10)
        self.assertEqual(self.predictor.lstm_units, [32, 16])
        self.assertEqual(self.predictor.dropout_rate, 0.2)
    
    def test_prepare_data(self):
        """测试数据准备"""
        X_front, y_front, X_back, y_back = self.predictor.prepare_data(self.test_data)
        
        self.assertIsInstance(X_front, np.ndarray)
        self.assertIsInstance(y_front, np.ndarray)
        self.assertIsInstance(X_back, np.ndarray)
        self.assertIsInstance(y_back, np.ndarray)
        
        # 检查形状
        self.assertEqual(X_front.shape[1], self.predictor.sequence_length)
        self.assertEqual(X_front.shape[2], 5)  # 前区5个号码
        self.assertEqual(X_back.shape[2], 2)   # 后区2个号码
    
    def test_build_model(self):
        """测试模型构建"""
        # 前区模型
        front_model = self.predictor.build_model((10, 5), 5)
        self.assertIsNotNone(front_model)
        
        # 后区模型
        back_model = self.predictor.build_model((10, 2), 2)
        self.assertIsNotNone(back_model)
    
    @patch('enhanced_deep_learning.lstm_predictor.logger_manager')
    def test_train(self, mock_logger):
        """测试训练"""
        # 使用较小的配置加速测试
        self.predictor.epochs = 2
        self.predictor.batch_size = 8
        
        result = self.predictor.train(self.test_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('front_loss', result)
        self.assertIn('back_loss', result)
        self.assertIn('training_samples', result)
        
        # 检查模型是否已训练
        self.assertIsNotNone(self.predictor.front_model)
        self.assertIsNotNone(self.predictor.back_model)
    
    def test_predict_without_model(self):
        """测试未训练模型的预测"""
        predictions = self.predictor.predict(self.test_data, count=1)
        
        # 应该返回空列表或尝试加载模型
        self.assertIsInstance(predictions, list)
    
    @patch('enhanced_deep_learning.lstm_predictor.logger_manager')
    def test_predict_with_model(self, mock_logger):
        """测试已训练模型的预测"""
        # 先训练模型
        self.predictor.epochs = 2
        self.predictor.train(self.test_data)
        
        # 预测
        predictions = self.predictor.predict(self.test_data, count=2)
        
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 2)
        
        for front, back in predictions:
            self.assertEqual(len(front), 5)
            self.assertEqual(len(back), 2)
            
            # 检查号码范围
            self.assertTrue(all(1 <= x <= 35 for x in front))
            self.assertTrue(all(1 <= x <= 12 for x in back))
    
    def test_save_and_load_models(self):
        """测试模型保存和加载"""
        # 先训练模型
        self.predictor.epochs = 2
        self.predictor.train(self.test_data)
        
        # 保存模型
        save_result = self.predictor.save_models()
        self.assertTrue(save_result)
        
        # 检查文件是否存在
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'front_lstm_model.h5')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'back_lstm_model.h5')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'front_scaler.pkl')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'back_scaler.pkl')))
        
        # 创建新的预测器并加载模型
        new_predictor = LSTMPredictor(self.config)
        load_result = new_predictor.load_models()
        self.assertTrue(load_result)
        
        # 检查模型是否加载成功
        self.assertIsNotNone(new_predictor.front_model)
        self.assertIsNotNone(new_predictor.back_model)
    
    def test_evaluate(self):
        """测试模型评估"""
        # 先训练模型
        self.predictor.epochs = 2
        self.predictor.train(self.test_data)
        
        # 评估模型
        result = self.predictor.evaluate(self.test_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('front_mse', result)
        self.assertIn('front_mae', result)
        self.assertIn('back_mse', result)
        self.assertIn('back_mae', result)
    
    def test_create_sequences(self):
        """测试序列创建"""
        # 创建测试数据
        data = np.random.rand(50, 5)
        
        X, y = self.predictor._create_sequences(data)
        
        self.assertEqual(X.shape[0], 50 - self.predictor.sequence_length)
        self.assertEqual(X.shape[1], self.predictor.sequence_length)
        self.assertEqual(X.shape[2], 5)
        self.assertEqual(y.shape[0], 50 - self.predictor.sequence_length)
        self.assertEqual(y.shape[1], 5)
    
    def test_empty_data(self):
        """测试空数据处理"""
        empty_data = pd.DataFrame()
        
        with self.assertRaises(Exception):
            self.predictor.prepare_data(empty_data)
    
    def test_insufficient_data(self):
        """测试数据不足的情况"""
        # 创建少量数据
        small_data = self.test_data.head(5)
        
        with self.assertRaises(Exception):
            self.predictor.prepare_data(small_data)


if __name__ == "__main__":
    unittest.main()
