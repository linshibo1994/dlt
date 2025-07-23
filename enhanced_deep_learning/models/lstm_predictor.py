#!/usr/bin/env python3
"""
LSTM预测模型
基于LSTM神经网络的彩票号码预测
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Bidirectional,
    Input, Add, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Concatenate, Attention
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List, Tuple, Dict, Any, Optional
import joblib
import os
import math
from datetime import datetime

from core_modules import logger_manager, cache_manager
from . import BaseDeepLearningModel, ModelMetadata


class LSTMPredictor(BaseDeepLearningModel):
    """LSTM预测器"""
    
    def __init__(self, config: Dict[str, Any] = None, metadata: ModelMetadata = None):
        """
        初始化LSTM预测器

        Args:
            config: 配置参数
            metadata: 模型元数据
        """
        # 创建默认元数据
        if metadata is None:
            metadata = ModelMetadata(
                name="LSTMPredictor",
                version="2.0.0",
                description="基于LSTM神经网络的彩票号码预测模型",
                dependencies=["tensorflow", "scikit-learn", "numpy", "pandas"]
            )

        # 调用父类初始化
        super().__init__(config, metadata)

        # 模型参数
        self.sequence_length = self.config.get('sequence_length', 30)
        self.lstm_units = self.config.get('lstm_units', [128, 64, 32])
        self.dropout_rate = self.config.get('dropout_rate', 0.2)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 32)
        self.epochs = self.config.get('epochs', 100)

        # 高级LSTM参数
        self.use_bidirectional = self.config.get('use_bidirectional', True)
        self.use_attention = self.config.get('use_attention', True)
        self.use_residual = self.config.get('use_residual', True)
        self.attention_heads = self.config.get('attention_heads', 4)
        self.l1_reg = self.config.get('l1_reg', 0.01)
        self.l2_reg = self.config.get('l2_reg', 0.01)
        self.gradient_clip_norm = self.config.get('gradient_clip_norm', 1.0)

        # 模型和缩放器
        self.front_model = None
        self.back_model = None
        self.front_scaler = MinMaxScaler()
        self.back_scaler = MinMaxScaler()

        logger_manager.info("增强LSTM预测器初始化完成")

    def _create_learning_rate_scheduler(self):
        """创建学习率调度器"""
        def scheduler(epoch, lr):
            # 预热阶段
            if epoch < 10:
                return lr * (epoch + 1) / 10
            # 余弦退火
            elif epoch < self.epochs * 0.8:
                return lr * 0.95
            # 最后阶段快速下降
            else:
                return lr * 0.9

        return LearningRateScheduler(scheduler, verbose=0)

    def _get_advanced_callbacks(self):
        """获取高级回调函数"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            self._create_learning_rate_scheduler()
        ]

        return callbacks

    def _build_model(self) -> Any:
        """构建LSTM模型架构"""
        # 这个方法在build_model中实现
        return None

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        X_front, y_front, X_back, y_back = self.prepare_data(data)
        # 合并前区和后区数据
        X = np.concatenate([X_front.reshape(X_front.shape[0], -1),
                           X_back.reshape(X_back.shape[0], -1)], axis=1)
        y = np.concatenate([y_front, y_back], axis=1)
        return X, y

    def _train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """训练模型"""
        # 重新分离前区和后区数据
        front_features = X.shape[1] // 7 * 5  # 假设前区占5/7
        X_front = X[:, :front_features].reshape(X.shape[0], self.sequence_length, 5)
        X_back = X[:, front_features:].reshape(X.shape[0], self.sequence_length, 2)
        y_front = y[:, :5]
        y_back = y[:, 5:]

        # 构建和训练模型
        return self.train_models(X_front, y_front, X_back, y_back)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """模型预测"""
        # 重新分离前区和后区数据
        front_features = X.shape[1] // 7 * 5
        X_front = X[:, :front_features].reshape(X.shape[0], self.sequence_length, 5)
        X_back = X[:, front_features:].reshape(X.shape[0], self.sequence_length, 2)

        # 预测
        front_pred = self.front_model.predict(X_front, verbose=0)
        back_pred = self.back_model.predict(X_back, verbose=0)

        # 反标准化
        front_pred = self.front_scaler.inverse_transform(front_pred)
        back_pred = self.back_scaler.inverse_transform(back_pred)

        # 合并结果
        return np.concatenate([front_pred, back_pred], axis=1)

    def _save_model_file(self, file_path: str):
        """保存模型文件"""
        model_data = {
            'front_model': self.front_model,
            'back_model': self.back_model,
            'front_scaler': self.front_scaler,
            'back_scaler': self.back_scaler,
            'sequence_length': self.sequence_length
        }

        import joblib
        joblib.dump(model_data, file_path)

    def _load_model_file(self, file_path: str):
        """加载模型文件"""
        import joblib
        model_data = joblib.load(file_path)

        self.front_model = model_data['front_model']
        self.back_model = model_data['back_model']
        self.front_scaler = model_data['front_scaler']
        self.back_scaler = model_data['back_scaler']
        self.sequence_length = model_data.get('sequence_length', self.sequence_length)

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

    def train_models(self, X_front: np.ndarray, y_front: np.ndarray,
                    X_back: np.ndarray, y_back: np.ndarray) -> Dict[str, Any]:
        """训练前区和后区模型"""
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

        # 计算训练结果
        front_loss = min(front_history.history['val_loss'])
        back_loss = min(back_history.history['val_loss'])

        return {
            'front_loss': front_loss,
            'back_loss': back_loss,
            'front_epochs': len(front_history.history['loss']),
            'back_epochs': len(back_history.history['loss']),
            'training_samples': len(X_front)
        }

    def build_model(self, input_shape: Tuple[int, int], output_dim: int) -> Model:
        """
        构建增强LSTM模型（支持双向、注意力、残差连接）

        Args:
            input_shape: 输入形状
            output_dim: 输出维度

        Returns:
            增强LSTM模型
        """
        # 输入层
        inputs = Input(shape=input_shape, name='input_layer')
        x = inputs

        # 多层LSTM with 残差连接
        lstm_outputs = []

        for i, units in enumerate(self.lstm_units):
            # 是否返回序列
            return_sequences = (i < len(self.lstm_units) - 1) or self.use_attention

            # LSTM层（支持双向）
            if self.use_bidirectional:
                lstm_layer = Bidirectional(
                    LSTM(
                        units=units,
                        return_sequences=return_sequences,
                        dropout=self.dropout_rate,
                        recurrent_dropout=self.dropout_rate,
                        kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                        name=f'bidirectional_lstm_{i}'
                    ),
                    name=f'bidirectional_wrapper_{i}'
                )(x)
            else:
                lstm_layer = LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate,
                    kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                    name=f'lstm_{i}'
                )(x)

            # 批归一化
            lstm_layer = BatchNormalization(name=f'bn_{i}')(lstm_layer)

            # 残差连接（如果维度匹配）
            if self.use_residual and i > 0 and return_sequences:
                try:
                    # 检查维度是否匹配
                    if x.shape[-1] == lstm_layer.shape[-1]:
                        lstm_layer = Add(name=f'residual_{i}')([x, lstm_layer])
                    else:
                        # 维度不匹配时使用投影
                        projected_x = Dense(lstm_layer.shape[-1], name=f'projection_{i}')(x)
                        lstm_layer = Add(name=f'residual_projected_{i}')([projected_x, lstm_layer])
                except:
                    # 如果残差连接失败，继续使用原始输出
                    pass

            # 层归一化
            lstm_layer = LayerNormalization(name=f'ln_{i}')(lstm_layer)

            x = lstm_layer
            lstm_outputs.append(x)

        # 注意力机制
        if self.use_attention and len(lstm_outputs) > 0:
            # 确保最后一层返回序列
            if not return_sequences:
                # 如果最后一层没有返回序列，使用倒数第二层
                attention_input = lstm_outputs[-2] if len(lstm_outputs) > 1 else lstm_outputs[-1]
            else:
                attention_input = lstm_outputs[-1]

            # 多头自注意力
            attention_output = MultiHeadAttention(
                num_heads=self.attention_heads,
                key_dim=attention_input.shape[-1] // self.attention_heads,
                name='multi_head_attention'
            )(attention_input, attention_input)

            # 残差连接和层归一化
            attention_output = Add(name='attention_residual')([attention_input, attention_output])
            attention_output = LayerNormalization(name='attention_ln')(attention_output)

            # 全局平均池化
            x = GlobalAveragePooling1D(name='global_avg_pool')(attention_output)
        else:
            # 如果没有注意力机制，确保输出是2D
            if len(x.shape) > 2:
                x = GlobalAveragePooling1D(name='global_avg_pool_fallback')(x)

        # 全连接层
        x = Dense(128, activation='relu', name='dense_1')(x)
        x = BatchNormalization(name='bn_dense_1')(x)
        x = Dropout(self.dropout_rate, name='dropout_dense_1')(x)

        x = Dense(64, activation='relu', name='dense_2')(x)
        x = BatchNormalization(name='bn_dense_2')(x)
        x = Dropout(self.dropout_rate, name='dropout_dense_2')(x)

        # 输出层
        outputs = Dense(output_dim, activation='linear', name='output_layer')(x)

        # 构建模型
        model = Model(inputs=inputs, outputs=outputs, name='Enhanced_LSTM')

        # 编译模型（支持梯度裁剪）
        optimizer = Adam(
            learning_rate=self.learning_rate,
            clipnorm=self.gradient_clip_norm
        )

        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
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
            
            # 高级训练回调
            callbacks = self._get_advanced_callbacks()
            
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
