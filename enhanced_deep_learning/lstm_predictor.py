#!/usr/bin/env python3
"""
LSTM预测模型
基于LSTM神经网络的彩票号码预测
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List, Tuple, Dict, Any, Optional
import joblib
import os
from datetime import datetime

from core_modules import logger_manager, cache_manager
from interfaces import PredictorInterface


class LSTMPredictor(PredictorInterface):
    """LSTM预测器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化LSTM预测器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 模型参数
        self.sequence_length = self.config.get('sequence_length', 30)
        self.lstm_units = self.config.get('lstm_units', [64, 32])
        self.dropout_rate = self.config.get('dropout_rate', 0.2)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 32)
        self.epochs = self.config.get('epochs', 100)
        
        # 模型和缩放器
        self.front_model = None
        self.back_model = None
        self.front_scaler = MinMaxScaler()
        self.back_scaler = MinMaxScaler()
        
        # 模型保存路径
        self.model_dir = self.config.get('model_dir', 'models/lstm')
        os.makedirs(self.model_dir, exist_ok=True)
        
        logger_manager.info("LSTM预测器初始化完成")
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备训练数据
        
        Args:
            data: 历史数据
            
        Returns:
            (X_front, y_front, X_back, y_back)
        """
        try:
            # 提取前区和后区号码
            front_numbers = []
            back_numbers = []
            
            for _, row in data.iterrows():
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
            
            logger_manager.info(f"数据准备完成: 前区序列 {X_front.shape}, 后区序列 {X_back.shape}")
            
            return X_front, y_front, X_back, y_back
            
        except Exception as e:
            logger_manager.error(f"数据准备失败: {e}")
            raise
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建序列数据
        
        Args:
            data: 标准化后的数据
            
        Returns:
            (X, y) 序列数据
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int], output_dim: int) -> Sequential:
        """
        构建LSTM模型
        
        Args:
            input_shape: 输入形状
            output_dim: 输出维度
            
        Returns:
            LSTM模型
        """
        model = Sequential()
        
        # 第一层LSTM
        model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=True if len(self.lstm_units) > 1 else False,
            input_shape=input_shape
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # 额外的LSTM层
        for i, units in enumerate(self.lstm_units[1:]):
            return_sequences = i < len(self.lstm_units) - 2
            model.add(LSTM(units=units, return_sequences=return_sequences))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # 输出层
        model.add(Dense(output_dim, activation='linear'))
        
        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        训练LSTM模型
        
        Args:
            data: 训练数据
            
        Returns:
            训练结果
        """
        try:
            logger_manager.info("开始训练LSTM模型")
            
            # 准备数据
            X_front, y_front, X_back, y_back = self.prepare_data(data)
            
            # 构建模型
            self.front_model = self.build_model(
                input_shape=(X_front.shape[1], X_front.shape[2]),
                output_dim=5  # 前区5个号码
            )
            
            self.back_model = self.build_model(
                input_shape=(X_back.shape[1], X_back.shape[2]),
                output_dim=2  # 后区2个号码
            )
            
            # 训练回调
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
            ]
            
            # 训练前区模型
            logger_manager.info("训练前区LSTM模型")
            front_history = self.front_model.fit(
                X_front, y_front,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            # 训练后区模型
            logger_manager.info("训练后区LSTM模型")
            back_history = self.back_model.fit(
                X_back, y_back,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            # 保存模型
            self.save_models()
            
            # 计算训练结果
            front_loss = min(front_history.history['val_loss'])
            back_loss = min(back_history.history['val_loss'])
            
            result = {
                'front_loss': front_loss,
                'back_loss': back_loss,
                'front_epochs': len(front_history.history['loss']),
                'back_epochs': len(back_history.history['loss']),
                'training_samples': len(X_front)
            }
            
            logger_manager.info(f"LSTM模型训练完成: 前区损失 {front_loss:.4f}, 后区损失 {back_loss:.4f}")
            
            return result
            
        except Exception as e:
            logger_manager.error(f"LSTM模型训练失败: {e}")
            raise
    
    def predict(self, data: pd.DataFrame, count: int = 1) -> List[Tuple[List[int], List[int]]]:
        """
        使用LSTM模型预测
        
        Args:
            data: 历史数据
            count: 预测数量
            
        Returns:
            预测结果列表
        """
        try:
            if self.front_model is None or self.back_model is None:
                # 尝试加载模型
                if not self.load_models():
                    raise ValueError("LSTM模型未训练或加载失败")
            
            # 准备最近的序列数据
            recent_data = data.tail(self.sequence_length)
            
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
                
                predictions.append((front_balls, back_balls))
            
            logger_manager.info(f"LSTM预测完成，生成 {len(predictions)} 注预测")
            
            return predictions
            
        except Exception as e:
            logger_manager.error(f"LSTM预测失败: {e}")
            return []
    
    def save_models(self) -> bool:
        """保存模型"""
        try:
            # 保存Keras模型
            self.front_model.save(os.path.join(self.model_dir, 'front_lstm_model.h5'))
            self.back_model.save(os.path.join(self.model_dir, 'back_lstm_model.h5'))
            
            # 保存缩放器
            joblib.dump(self.front_scaler, os.path.join(self.model_dir, 'front_scaler.pkl'))
            joblib.dump(self.back_scaler, os.path.join(self.model_dir, 'back_scaler.pkl'))
            
            logger_manager.info("LSTM模型保存成功")
            return True
            
        except Exception as e:
            logger_manager.error(f"LSTM模型保存失败: {e}")
            return False
    
    def load_models(self) -> bool:
        """加载模型"""
        try:
            front_model_path = os.path.join(self.model_dir, 'front_lstm_model.h5')
            back_model_path = os.path.join(self.model_dir, 'back_lstm_model.h5')
            front_scaler_path = os.path.join(self.model_dir, 'front_scaler.pkl')
            back_scaler_path = os.path.join(self.model_dir, 'back_scaler.pkl')
            
            if not all(os.path.exists(p) for p in [front_model_path, back_model_path, front_scaler_path, back_scaler_path]):
                logger_manager.warning("LSTM模型文件不存在")
                return False
            
            # 加载模型
            self.front_model = tf.keras.models.load_model(front_model_path)
            self.back_model = tf.keras.models.load_model(back_model_path)
            
            # 加载缩放器
            self.front_scaler = joblib.load(front_scaler_path)
            self.back_scaler = joblib.load(back_scaler_path)
            
            logger_manager.info("LSTM模型加载成功")
            return True
            
        except Exception as e:
            logger_manager.error(f"LSTM模型加载失败: {e}")
            return False
    
    def evaluate(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            data: 测试数据
            
        Returns:
            评估结果
        """
        try:
            if self.front_model is None or self.back_model is None:
                raise ValueError("模型未训练")
            
            # 准备测试数据
            X_front, y_front, X_back, y_back = self.prepare_data(data)
            
            # 预测
            front_pred = self.front_model.predict(X_front, verbose=0)
            back_pred = self.back_model.predict(X_back, verbose=0)
            
            # 计算评估指标
            front_mse = mean_squared_error(y_front, front_pred)
            front_mae = mean_absolute_error(y_front, front_pred)
            back_mse = mean_squared_error(y_back, back_pred)
            back_mae = mean_absolute_error(y_back, back_pred)
            
            result = {
                'front_mse': front_mse,
                'front_mae': front_mae,
                'back_mse': back_mse,
                'back_mae': back_mae
            }
            
            logger_manager.info(f"LSTM模型评估完成: {result}")
            
            return result
            
        except Exception as e:
            logger_manager.error(f"LSTM模型评估失败: {e}")
            return {}
