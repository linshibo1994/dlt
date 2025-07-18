#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高级LSTM预测器
基于深度学习的时间序列预测
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from datetime import datetime

# 检查TensorFlow可用性
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, MultiHeadAttention
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from core_modules import logger_manager, data_manager, cache_manager


class AdvancedLSTMPredictor:
    """高级LSTM预测器"""
    
    def __init__(self, data_file="data/dlt_data_all.csv"):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow未安装，无法使用LSTM预测器")
        
        self.data_file = data_file
        self.df = data_manager.get_data()
        self.sequence_length = 20  # 时间序列长度
        self.feature_dim = 21  # 特征维度
        
        # 模型组件
        self.scaler = MinMaxScaler()
        self.model = None
        self.is_trained = False
        
        if self.df is None:
            logger_manager.error("数据未加载")
    
    def _extract_features(self, df_subset) -> np.ndarray:
        """提取深度特征"""
        features = []
        
        for _, row in df_subset.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            
            # 基础特征 (7维)
            feature_vector = front_balls + back_balls
            
            # 统计特征 (8维)
            front_sum = sum(front_balls)
            back_sum = sum(back_balls)
            total_sum = front_sum + back_sum
            front_mean = np.mean(front_balls)
            back_mean = np.mean(back_balls)
            front_std = np.std(front_balls)
            back_std = np.std(back_balls)
            span = max(front_balls) - min(front_balls)
            
            # 模式特征 (6维)
            odd_count = sum(1 for x in front_balls if x % 2 == 1)
            even_count = 5 - odd_count
            big_count = sum(1 for x in front_balls if x > 17)  # 大号(18-35)
            small_count = 5 - big_count  # 小号(1-17)
            consecutive_count = self._count_consecutive(front_balls)
            prime_count = sum(1 for x in front_balls if self._is_prime(x))
            
            # 组合所有特征
            all_features = (
                feature_vector +  # 7维
                [front_sum, back_sum, total_sum, front_mean, back_mean, front_std, back_std, span] +  # 8维
                [odd_count, even_count, big_count, small_count, consecutive_count, prime_count]  # 6维
            )
            
            features.append(all_features)
        
        return np.array(features)
    
    def _count_consecutive(self, numbers):
        """计算连续号码数量"""
        sorted_nums = sorted(numbers)
        consecutive = 0
        for i in range(len(sorted_nums) - 1):
            if sorted_nums[i+1] - sorted_nums[i] == 1:
                consecutive += 1
        return consecutive
    
    def _is_prime(self, n):
        """判断是否为质数"""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def _prepare_sequences(self, features):
        """准备时间序列数据"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            y.append(features[i][:7])  # 只预测前5个前区号码和2个后区号码
        
        return np.array(X), np.array(y)
    
    def _build_model(self):
        """构建LSTM模型"""
        model = Sequential([
            # 第一层LSTM
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.feature_dim)),
            Dropout(0.2),
            
            # 第二层LSTM
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            
            # 第三层LSTM
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            # 全连接层
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            
            # 输出层
            Dense(7, activation='sigmoid')  # 5前区+2后区，归一化到0-1
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, epochs=100, validation_split=0.2):
        """训练LSTM模型"""
        logger_manager.info("开始训练LSTM模型...")
        
        # 提取特征
        features = self._extract_features(self.df)
        
        # 数据标准化
        features_scaled = self.scaler.fit_transform(features)
        
        # 准备序列数据
        X, y = self._prepare_sequences(features_scaled)
        
        if len(X) == 0:
            logger_manager.error("序列数据不足，无法训练模型")
            return False
        
        # 构建模型
        self.model = self._build_model()
        
        # 设置回调
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0001)
        ]
        
        # 训练模型
        history = self.model.fit(
            X, y,
            epochs=epochs,
            validation_split=validation_split,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        logger_manager.info("LSTM模型训练完成")
        
        return True
    
    def lstm_predict(self, count=1) -> List[Tuple[List[int], List[int]]]:
        """LSTM预测"""
        if not self.is_trained:
            logger_manager.info("模型未训练，开始训练...")
            if not self.train_model():
                logger_manager.error("模型训练失败")
                return []
        
        # 获取最近的序列数据
        recent_features = self._extract_features(self.df.head(self.sequence_length))
        recent_scaled = self.scaler.transform(recent_features)
        
        # 准备输入序列
        input_sequence = recent_scaled.reshape(1, self.sequence_length, self.feature_dim)
        
        predictions = []
        
        for _ in range(count):
            # 预测
            pred_scaled = self.model.predict(input_sequence, verbose=0)
            
            # 反标准化
            # 创建完整特征向量用于反标准化
            full_pred = np.zeros((1, self.feature_dim))
            full_pred[0, :7] = pred_scaled[0]
            pred_original = self.scaler.inverse_transform(full_pred)[0, :7]
            
            # 转换为彩票号码
            front_balls = [max(1, min(35, int(round(x)))) for x in pred_original[:5]]
            back_balls = [max(1, min(12, int(round(x)))) for x in pred_original[5:7]]
            
            # 确保号码唯一性
            front_balls = self._ensure_unique_numbers(front_balls, 1, 35, 5)
            back_balls = self._ensure_unique_numbers(back_balls, 1, 12, 2)
            
            predictions.append((sorted(front_balls), sorted(back_balls)))
            
            # 更新输入序列用于下一次预测
            new_feature = np.concatenate([pred_original, recent_scaled[-1, 7:]])
            input_sequence = np.roll(input_sequence, -1, axis=1)
            input_sequence[0, -1] = new_feature
        
        return predictions
    
    def _ensure_unique_numbers(self, numbers, min_val, max_val, target_count):
        """确保号码唯一性"""
        unique_numbers = list(set(numbers))
        
        # 如果数量不足，随机补充
        while len(unique_numbers) < target_count:
            candidate = np.random.randint(min_val, max_val + 1)
            if candidate not in unique_numbers:
                unique_numbers.append(candidate)
        
        return unique_numbers[:target_count]


# 兼容性检查
if not TENSORFLOW_AVAILABLE:
    class AdvancedLSTMPredictor:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow未安装，无法使用LSTM预测器")
        
        def lstm_predict(self, *args, **kwargs):
            return []


if __name__ == "__main__":
    # 测试LSTM预测器
    if TENSORFLOW_AVAILABLE:
        print("🧠 测试LSTM预测器...")
        predictor = AdvancedLSTMPredictor()
        
        # 训练模型
        predictor.train_model(epochs=50)
        
        # 进行预测
        predictions = predictor.lstm_predict(3)
        
        print("LSTM预测结果:")
        for i, (front, back) in enumerate(predictions):
            front_str = ' '.join([str(b).zfill(2) for b in front])
            back_str = ' '.join([str(b).zfill(2) for b in back])
            print(f"第 {i+1} 注: {front_str} + {back_str}")
    else:
        print("❌ TensorFlow未安装，无法测试LSTM预测器")