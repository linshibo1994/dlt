#!/usr/bin/env python3
"""
Advanced LSTM Predictor Module
提供基于TensorFlow的高级LSTM预测功能
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional

# 尝试导入TensorFlow相关依赖
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from enhanced_deep_learning.models.lstm_predictor import LSTMPredictor
    TENSORFLOW_AVAILABLE = True
except ImportError:
    pass

from core_modules import logger_manager, cache_manager


class AdvancedLSTMPredictor:
    """高级LSTM预测器，提供增强的深度学习预测功能"""
    
    def __init__(self, data_file: str = None):
        """
        初始化高级LSTM预测器
        
        Args:
            data_file: 数据文件路径，默认为None使用系统默认数据
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow未安装，无法使用LSTM预测器")
            
        self.data_file = data_file
        self.base_predictor = None
        self.model_cache_dir = os.path.join('cache', 'models', 'lstm')
        
        # 确保缓存目录存在
        os.makedirs(self.model_cache_dir, exist_ok=True)
        
        # 初始化基础LSTM预测器
        self._init_base_predictor()
        
        logger_manager.info("高级LSTM预测器初始化完成")
    
    def _init_base_predictor(self):
        """
        初始化基础LSTM预测器
        """
        # 高级配置参数
        config = {
            'sequence_length': 50,  # 增加序列长度以捕获更长期的模式
            'lstm_units': [256, 128, 64],  # 更深的网络结构
            'dropout_rate': 0.3,  # 增加dropout以减少过拟合
            'learning_rate': 0.0005,  # 降低学习率以获得更稳定的训练
            'batch_size': 64,  # 增加批量大小
            'epochs': 200,  # 增加训练轮次
            'use_bidirectional': True,  # 使用双向LSTM
            'use_attention': True,  # 使用注意力机制
            'use_residual': True,  # 使用残差连接
            'attention_heads': 8,  # 增加注意力头数
            'l1_reg': 0.005,  # 调整L1正则化
            'l2_reg': 0.005,  # 调整L2正则化
            'gradient_clip_norm': 1.0  # 梯度裁剪
        }
        
        self.base_predictor = LSTMPredictor(config=config)
    
    def lstm_predict(self, count: int = 3, periods: int = 100) -> List[Tuple[List[int], List[int]]]:
        """
        使用LSTM模型进行预测
        
        Args:
            count: 预测数量
            periods: 使用的历史数据期数
            
        Returns:
            预测结果列表，每个元素为(前区号码, 后区号码)的元组
        """
        try:
            # 加载数据
            if self.data_file:
                data = pd.read_csv(self.data_file)
            else:
                data_path = os.path.join('data', 'dlt_data_all.csv')
                data = pd.read_csv(data_path)
            
            # 使用最近的periods期数据
            recent_data = data.tail(periods)
            
            # 检查模型是否已训练
            model_path = os.path.join(self.model_cache_dir, 'lstm_model.h5')
            if not os.path.exists(model_path):
                logger_manager.info(f"LSTM模型不存在，开始训练 (使用{periods}期数据)...")
                self._train_model(recent_data)
            
            # 进行预测
            logger_manager.info(f"使用LSTM模型预测{count}组号码...")
            predictions = self.base_predictor.predict(recent_data, count)
            
            # 确保预测结果格式正确
            formatted_predictions = []
            for front_balls, back_balls in predictions:
                # 确保前区有5个不重复的号码
                if len(set(front_balls)) < 5:
                    front_balls = list(set(front_balls))
                    while len(front_balls) < 5:
                        new_num = np.random.randint(1, 36)
                        if new_num not in front_balls:
                            front_balls.append(new_num)
                    front_balls.sort()
                
                # 确保后区有2个不重复的号码
                if len(set(back_balls)) < 2:
                    back_balls = list(set(back_balls))
                    while len(back_balls) < 2:
                        new_num = np.random.randint(1, 13)
                        if new_num not in back_balls:
                            back_balls.append(new_num)
                    back_balls.sort()
                
                formatted_predictions.append((front_balls, back_balls))
            
            return formatted_predictions
            
        except Exception as e:
            logger_manager.error(f"LSTM预测失败: {e}")
            # 返回空列表作为备选
            return []
    
    def _train_model(self, data: pd.DataFrame):
        """
        训练LSTM模型
        
        Args:
            data: 训练数据
        """
        try:
            # 训练模型
            self.base_predictor.train(data)
            
            # 保存模型
            model_path = os.path.join(self.model_cache_dir, 'lstm_model.h5')
            self.base_predictor.save(model_path)
            
            logger_manager.info("LSTM模型训练完成并保存")
            
        except Exception as e:
            logger_manager.error(f"LSTM模型训练失败: {e}")
            raise


# 如果直接运行此模块，进行测试
if __name__ == "__main__":
    if TENSORFLOW_AVAILABLE:
        try:
            predictor = AdvancedLSTMPredictor()
            results = predictor.lstm_predict(count=5)
            
            print("\nLSTM预测结果:")
            for i, (front, back) in enumerate(results, 1):
                print(f"预测 {i}: 前区 {front}, 后区 {back}")
                
        except Exception as e:
            print(f"测试失败: {e}")
    else:
        print("TensorFlow未安装，无法使用LSTM预测器")