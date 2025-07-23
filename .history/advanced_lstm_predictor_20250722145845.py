#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高级LSTM预测器模块
基于TensorFlow的LSTM深度学习预测模型
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# 检查TensorFlow是否可用
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# 导入项目依赖
from core_modules import logger_manager, data_manager, cache_manager


class AdvancedLSTMPredictor:
    """高级LSTM预测器"""
    
    def __init__(self, data_file="data/dlt_data_all.csv"):
        """初始化高级LSTM预测器
        
        Args:
            data_file: 数据文件路径
        """
        self.data_file = data_file
        self.df = data_manager.get_data()
        
        # 模型参数
        self.sequence_length = 30
        self.lstm_units = [128, 64, 32]
        self.dropout_rate = 0.2
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 100
        
        # 高级参数
        self.use_bidirectional = True
        self.use_attention = True
        self.use_residual = True
        self.attention_heads = 4
        
        # 模型和缩放器
        self.front_model = None
        self.back_model = None
        self.front_scaler = MinMaxScaler()
        self.back_scaler = MinMaxScaler()
        
        # 模型路径
        self.model_dir = os.path.join('cache', 'models')
        self.front_model_path = os.path.join(self.model_dir, 'lstm_front_model.h5')
        self.back_model_path = os.path.join(self.model_dir, 'lstm_back_model.h5')
        self.front_scaler_path = os.path.join(self.model_dir, 'lstm_front_scaler.pkl')
        self.back_scaler_path = os.path.join(self.model_dir, 'lstm_back_scaler.pkl')
        
        # 尝试加载模型
        self._try_load_models()
    
    def _try_load_models(self):
        """尝试加载已保存的模型"""
        try:
            if os.path.exists(self.front_model_path) and os.path.exists(self.back_model_path):
                self.front_model = load_model(self.front_model_path)
                self.back_model = load_model(self.back_model_path)
                
                # 加载缩放器
                if os.path.exists(self.front_scaler_path) and os.path.exists(self.back_scaler_path):
                    import joblib
                    self.front_scaler = joblib.load(self.front_scaler_path)
                    self.back_scaler = joblib.load(self.back_scaler_path)
                
                logger_manager.info("LSTM模型加载成功")
                return True
        except Exception as e:
            logger_manager.error(f"加载LSTM模型失败: {e}")
        
        return False
    
    def _prepare_data(self, data: pd.DataFrame, periods: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """准备训练数据
        
        Args:
            data: 历史数据
            periods: 使用的期数
            
        Returns:
            (X_front, y_front, X_back, y_back)
        """
        # 使用最近的periods期数据
        recent_data = data.tail(periods)
        
        # 提取前区和后区号码
        front_numbers = []
        back_numbers = []
        
        for _, row in recent_data.iterrows():
            front_balls = [int(x) for x in row['front_balls'].split(',')]
            back_balls = [int(x) for x in row['back_balls'].split(',')]
            
            front_numbers.append(front_balls)
            back_numbers.append(back_balls)
        
        # 转换为numpy数组
        front_array = np.array(front_numbers)
        back_array = np.array(back_numbers)
        
        # 数据标准化
        front_scaled = self.front_scaler.fit_transform(front_array)
        back_scaled = self.back_scaler.fit_transform(back_array)
        
        # 创建序列数据
        X_front, y_front = self._create_sequences(front_scaled)
        X_back, y_back = self._create_sequences(back_scaled)
        
        return X_front, y_front, X_back, y_back
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """创建序列数据
        
        Args:
            data: 标准化后的数据
            
        Returns:
            (X, y) 序列数据和标签
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: Tuple[int, ...], output_dim: int):
        """构建LSTM模型
        
        Args:
            input_shape: 输入形状
            output_dim: 输出维度
            
        Returns:
            LSTM模型
        """
        model = tf.keras.Sequential()
        
        # 第一层LSTM（返回序列）
        if self.use_bidirectional:
            model.add(tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(self.lstm_units[0], return_sequences=True),
                input_shape=input_shape[1:]
            ))
        else:
            model.add(tf.keras.layers.LSTM(
                self.lstm_units[0], return_sequences=True,
                input_shape=input_shape[1:]
            ))
        
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.BatchNormalization())
        
        # 中间层LSTM
        for units in self.lstm_units[1:-1]:
            if self.use_bidirectional:
                model.add(tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(units, return_sequences=True)
                ))
            else:
                model.add(tf.keras.layers.LSTM(units, return_sequences=True))
            
            model.add(tf.keras.layers.Dropout(self.dropout_rate))
            model.add(tf.keras.layers.BatchNormalization())
        
        # 最后一层LSTM（不返回序列）
        if self.use_bidirectional:
            model.add(tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(self.lstm_units[-1])
            ))
        else:
            model.add(tf.keras.layers.LSTM(self.lstm_units[-1]))
        
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.BatchNormalization())
        
        # 输出层
        model.add(tf.keras.layers.Dense(output_dim, activation='sigmoid'))
        
        # 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _train_models(self, X_front, y_front, X_back, y_back):
        """训练前区和后区模型
        
        Args:
            X_front: 前区序列数据
            y_front: 前区标签
            X_back: 后区序列数据
            y_back: 后区标签
        """
        # 创建回调函数
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7
            )
        ]
        
        # 构建前区模型
        self.front_model = self._build_model(X_front.shape, y_front.shape[1])
        
        # 构建后区模型
        self.back_model = self._build_model(X_back.shape, y_back.shape[1])
        
        # 训练前区模型
        logger_manager.info("训练前区LSTM模型")
        self.front_model.fit(
            X_front, y_front,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # 训练后区模型
        logger_manager.info("训练后区LSTM模型")
        self.back_model.fit(
            X_back, y_back,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # 保存模型
        self._save_models()
    
    def _save_models(self):
        """保存模型和缩放器"""
        try:
            # 确保目录存在
            os.makedirs(self.model_dir, exist_ok=True)
            
            # 保存模型
            self.front_model.save(self.front_model_path)
            self.back_model.save(self.back_model_path)
            
            # 保存缩放器
            import joblib
            joblib.dump(self.front_scaler, self.front_scaler_path)
            joblib.dump(self.back_scaler, self.back_scaler_path)
            
            logger_manager.info("LSTM模型保存成功")
        except Exception as e:
            logger_manager.error(f"保存LSTM模型失败: {e}")
    
    def lstm_predict(self, count: int = 1, periods: int = 500) -> List[Tuple[List[int], List[int]]]:
        """LSTM预测
        
        Args:
            count: 预测注数
            periods: 使用的期数
            
        Returns:
            预测结果列表
        """
        try:
            # 检查TensorFlow是否可用
            if not TENSORFLOW_AVAILABLE:
                logger_manager.error("TensorFlow未安装，无法使用LSTM预测")
                return []
            
            # 检查模型是否已加载
            if self.front_model is None or self.back_model is None:
                # 准备数据并训练模型
                X_front, y_front, X_back, y_back = self._prepare_data(self.df, periods)
                self._train_models(X_front, y_front, X_back, y_back)
            
            # 准备最近的序列数据
            recent_data = self.df.tail(self.sequence_length)
            
            # 提取号码
            front_numbers = []
            back_numbers = []
            
            for _, row in recent_data.iterrows():
                front_balls = [int(x) for x in row['front_balls'].split(',')]
                back_balls = [int(x) for x in row['back_balls'].split(',')]
                
                front_numbers.append(front_balls)
                back_numbers.append(back_balls)
            
            # 标准化
            front_scaled = self.front_scaler.transform(np.array(front_numbers))
            back_scaled = self.back_scaler.transform(np.array(back_numbers))
            
            # 预测
            predictions = []
            
            for _ in range(count):
                # 前区预测
                front_input = front_scaled.reshape(1, self.sequence_length, 5)
                front_pred = self.front_model.predict(front_input, verbose=0)
                front_pred = self.front_scaler.inverse_transform(front_pred)
                
                # 后区预测
                back_input = back_scaled.reshape(1, self.sequence_length, 2)
                back_pred = self.back_model.predict(back_input, verbose=0)
                back_pred = self.back_scaler.inverse_transform(back_pred)
                
                # 转换为整数并排序
                front_balls = sorted([max(1, min(35, int(round(x)))) for x in front_pred[0]])
                back_balls = sorted([max(1, min(12, int(round(x)))) for x in back_pred[0]])
                
                # 确保没有重复号码
                front_balls = list(dict.fromkeys(front_balls))
                back_balls = list(dict.fromkeys(back_balls))
                
                # 确保号码数量正确
                while len(front_balls) < 5:
                    # 添加随机号码
                    new_ball = np.random.randint(1, 36)
                    if new_ball not in front_balls:
                        front_balls.append(new_ball)
                front_balls = sorted(front_balls[:5])  # 只取前5个
                
                while len(back_balls) < 2:
                    # 添加随机号码
                    new_ball = np.random.randint(1, 13)
                    if new_ball not in back_balls:
                        back_balls.append(new_ball)
                back_balls = sorted(back_balls[:2])  # 只取前2个
                
                predictions.append((front_balls, back_balls))
            
            logger_manager.info(f"LSTM预测完成，生成 {len(predictions)} 注预测")
            
            return predictions
            
        except Exception as e:
            logger_manager.error(f"LSTM预测失败: {e}")
            return []


# 测试代码
if __name__ == "__main__":
    predictor = AdvancedLSTMPredictor()
    results = predictor.lstm_predict(count=3)
    
    print("\nLSTM预测结果:")
    for i, (front, back) in enumerate(results):
        print(f"第{i+1}注: 前区 {front}, 后区 {back}")