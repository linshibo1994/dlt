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
import json
import math
from datetime import datetime

# 导入核心模块
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import core_modules as cm
logger_manager = cm.logger_manager
cache_manager = cm.cache_manager
data_manager = cm.data_manager

from compound_modules.compound_predictor import CompoundPredictorMixin, CompoundConfig, CompoundResult
from .base_model import BaseModel as BaseDeepLearningModel, ModelConfig, ModelType
from .metadata import ModelMetadata
from ..training.smart_epochs_calculator import SmartEpochsCalculator, TrainingConfig, ModelType as SmartModelType, PerformanceMode
from ..utils.intelligent_early_stopping import IntelligentEarlyStopping, create_intelligent_callbacks


class LSTMPredictor(BaseDeepLearningModel, CompoundPredictorMixin):
    """LSTM预测器（支持复式预测）"""
    
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
                description="基于LSTM神经网络的彩票号码预测模型"
            )

        # 创建ModelConfig对象
        if config is None or isinstance(config, dict):
            model_config = ModelConfig(
                model_type=ModelType.LSTM,
                model_name=metadata.name if metadata else "LSTMPredictor",
                version=metadata.version if metadata else "2.0.0",
                description=metadata.description if metadata else "LSTM预测器"
            )
            # 保存原始配置参数
            self.config_params = config or {}
        else:
            model_config = config
            self.config_params = {}

        # 提取GPU配置参数
        self.use_gpu = self.config_params.get('use_gpu', False)
        self.gpu_device = self.config_params.get('gpu_device', 0)
        self.gpu_memory_limit = self.config_params.get('gpu_memory_limit', None)
        self.mixed_precision = self.config_params.get('mixed_precision', False)

        # GPU配置状态
        self._gpu_configured = False

        # 调用父类初始化
        super().__init__(model_config)

        # 初始化复式预测功能
        CompoundPredictorMixin.__init__(self)

        # 保存元数据
        self.metadata = metadata

        # 模型参数
        self.sequence_length = self.config_params.get('sequence_length', 50)
        self.lstm_units = self.config_params.get('lstm_units', [128, 64, 32])
        self.dropout_rate = self.config_params.get('dropout_rate', 0.2)
        self.learning_rate = self.config_params.get('learning_rate', 0.001)
        self.batch_size = self.config_params.get('batch_size', 64)
        self.epochs = self.config_params.get('epochs', 200)
        # 单独 LSTM 默认不启用“智能早停”，避免复用集成/优化场景的早停策略。
        self.enable_early_stopping = self.config_params.get('enable_early_stopping', False)
        self.early_stopping_patience = self.config_params.get('early_stopping_patience', 20)

        # 高级LSTM参数
        self.use_bidirectional = self.config_params.get('use_bidirectional', True)
        self.use_attention = self.config_params.get('use_attention', True)
        self.use_residual = self.config_params.get('use_residual', True)
        self.attention_heads = self.config_params.get('attention_heads', 4)
        self.l1_reg = self.config_params.get('l1_reg', 0.01)
        self.l2_reg = self.config_params.get('l2_reg', 0.01)
        self.gradient_clip_norm = self.config_params.get('gradient_clip_norm', 1.0)

        # 模型和缩放器
        self.front_model = None
        self.back_model = None
        self.front_scaler = MinMaxScaler()
        self.back_scaler = MinMaxScaler()

        # 训练状态
        self.is_trained = False

        # 模型保存目录
        self.model_dir = self.config_params.get('model_dir', 'artifacts/models/lstm')
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_cache_version = 2
        self.model_data_order = 'chronological_ascending'

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
        """获取训练回调函数"""
        if self.enable_early_stopping:
            callbacks = create_intelligent_callbacks(
                patience=self.early_stopping_patience,
                min_delta=1e-6,
                monitor='val_loss',
                reduce_lr_patience=10
            )
        else:
            callbacks = [
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7,
                    verbose=0
                )
            ]
        callbacks.append(self._create_learning_rate_scheduler())

        return callbacks

    def _order_lottery_dataframe(self, data: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
        """
        按开奖时间排序数据。

        DataManager 统一返回期号降序（最新在前），但 LSTM 训练序列必须是时间正序：
        旧数据 -> 新数据。这里集中兜底，避免不同入口传入不同顺序。
        """
        if data is None or data.empty:
            return data

        ordered = data.copy()

        if 'issue' in ordered.columns:
            issue_series = pd.to_numeric(ordered['issue'], errors='coerce')
            if issue_series.notna().any():
                return (
                    ordered.assign(_issue_num=issue_series)
                    .sort_values('_issue_num', ascending=ascending)
                    .drop(columns=['_issue_num'])
                    .reset_index(drop=True)
                )
            return ordered.sort_values('issue', ascending=ascending).reset_index(drop=True)

        if 'date' in ordered.columns:
            date_series = pd.to_datetime(ordered['date'], errors='coerce')
            if date_series.notna().any():
                return (
                    ordered.assign(_date_sort=date_series)
                    .sort_values('_date_sort', ascending=ascending)
                    .drop(columns=['_date_sort'])
                    .reset_index(drop=True)
                )

        return ordered.reset_index(drop=True)

    def _latest_chronological_window(self, data: pd.DataFrame, window_size: int = None) -> pd.DataFrame:
        """获取最新 N 期，并保持时间正序。"""
        window_size = window_size or self.sequence_length
        ordered = self._order_lottery_dataframe(data, ascending=True)
        return ordered.tail(window_size).reset_index(drop=True)

    def _configure_gpu(self):
        """配置GPU设备"""
        if self._gpu_configured:
            return

        try:
            import tensorflow as tf

            # 检查GPU可用性
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                logger_manager.warning("未检测到GPU设备，将使用CPU")
                self.use_gpu = False
                return

            # 选择GPU设备
            if self.gpu_device < len(gpus):
                gpu = gpus[self.gpu_device]
                logger_manager.info(f"使用GPU设备: {gpu}")

                # 配置GPU内存增长
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger_manager.info("GPU内存增长已启用")
                except RuntimeError as e:
                    logger_manager.warning(f"GPU内存增长配置失败: {e}")

                # 配置GPU内存限制
                if self.gpu_memory_limit:
                    try:
                        memory_limit = int(self.gpu_memory_limit * 1024)  # 转换为MB
                        tf.config.experimental.set_memory_limit(gpu, memory_limit)
                        logger_manager.info(f"GPU内存限制设置为: {self.gpu_memory_limit}GB")
                    except Exception as e:
                        logger_manager.warning(f"GPU内存限制配置失败: {e}")

                # 配置混合精度
                if self.mixed_precision:
                    try:
                        policy = tf.keras.mixed_precision.Policy('mixed_float16')
                        tf.keras.mixed_precision.set_global_policy(policy)
                        logger_manager.info("混合精度训练已启用")
                    except Exception as e:
                        logger_manager.warning(f"混合精度配置失败: {e}")
            else:
                logger_manager.warning(f"GPU设备ID {self.gpu_device} 超出范围，使用CPU")
                self.use_gpu = False

            self._gpu_configured = True

        except ImportError:
            logger_manager.warning("TensorFlow未安装，无法使用GPU")
            self.use_gpu = False
        except Exception as e:
            logger_manager.error(f"GPU配置失败: {e}")
            self.use_gpu = False

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

    def prepare_data(self, data) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备训练数据

        Args:
            data: 历史数据 (可以是 pd.DataFrame 或 np.ndarray)

        Returns:
            (X_front, y_front, X_back, y_back)
        """
        try:
            # 处理不同类型的输入数据
            if isinstance(data, np.ndarray):
                # 如果是numpy数组，假设已经是处理好的数字格式
                if data.shape[1] >= 7:  # 至少包含前区5个+后区2个号码
                    front_numbers = data[:, :5].tolist()
                    back_numbers = data[:, 5:7].tolist()
                else:
                    raise ValueError(f"数据维度不正确，期望至少7列，实际{data.shape[1]}列")
            elif isinstance(data, pd.DataFrame):
                data = self._order_lottery_dataframe(data, ascending=True)
                # 提取前区和后区号码
                front_numbers = []
                back_numbers = []

                for _, row in data.iterrows():
                    front_balls = [int(x) for x in row['front_balls'].split(',')]
                    back_balls = [int(x) for x in row['back_balls'].split(',')]

                    front_numbers.append(front_balls)
                    back_numbers.append(back_balls)
            else:
                raise ValueError(f"不支持的数据类型: {type(data)}")
            
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
        # 配置GPU设备
        if self.use_gpu:
            self._configure_gpu()

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
    
    def train(self, data) -> Dict[str, Any]:
        """
        训练LSTM模型
        
        Args:
            data: 训练数据
            
        Returns:
            训练结果
        """
        try:
            logger_manager.info("开始训练LSTM模型")

            # 智能训练轮数计算
            epochs_calculator = SmartEpochsCalculator()
            # 使用智能早停机制，允许更多训练轮数以获得更好效果
            dynamic_max_epochs = min(200, max(50, len(data) // 10))  # 恢复原始训练轮数上限
            training_config = TrainingConfig(
                model_type=SmartModelType.LSTM,
                data_size=len(data),
                feature_dim=7,
                performance_mode=PerformanceMode.MEDIUM,  # 使用中等性能模式
                min_epochs=20,  # 恢复原始最小轮数
                max_epochs=dynamic_max_epochs
            )
            epochs_recommendation = epochs_calculator.calculate_optimal_epochs(training_config)

            # 使用推荐的训练轮数
            optimal_epochs = epochs_recommendation.recommended_epochs
            logger_manager.info(f"智能训练轮数推荐: {optimal_epochs} (置信度: {epochs_recommendation.confidence:.2f})")
            logger_manager.info(f"推荐理由: {epochs_recommendation.reasoning}")

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
                epochs=optimal_epochs,  # 使用智能推荐的轮数
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
                # Keras 3.x 已移除 use_multiprocessing 和 workers 参数
            )

            # 训练后区模型
            logger_manager.info("训练后区LSTM模型")
            back_history = self.back_model.fit(
                X_back, y_back,
                batch_size=self.batch_size,
                epochs=optimal_epochs,  # 使用智能推荐的轮数
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
                # Keras 3.x 已移除 use_multiprocessing 和 workers 参数
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

            # 设置训练状态
            self.is_trained = True

            return result
            
        except Exception as e:
            logger_manager.error(f"LSTM模型训练失败: {e}")
            raise
    
    def predict(self, data=None, count: int = 1) -> List[Tuple[List[int], List[int]]]:
        """
        使用LSTM模型预测

        Args:
            data: 历史数据（可选，如果不提供则自动获取）
            count: 预测数量

        Returns:
            预测结果列表
        """
        try:
            # 如果没有提供数据，自动获取
            if data is None:
                from core_modules import data_manager
                data = data_manager.get_data()
                if data is None or data.empty:
                    raise ValueError("无法获取历史数据")

            if self.front_model is None or self.back_model is None:
                # 尝试加载模型
                logger_manager.info("尝试从缓存加载模型...")
                if not self.load_models():
                    logger_manager.info("模型缓存不存在，开始训练新模型...")
                    # 如果加载失败，尝试训练新模型
                    # train方法返回字典，检查success字段或使用is_trained标志
                    train_result = self.train(data)
                    if not train_result or (isinstance(train_result, dict) and not train_result.get('training_samples')):
                        raise ValueError("LSTM模型训练失败")
                    logger_manager.info("模型训练完成，开始预测...")
                else:
                    logger_manager.info("成功从缓存加载模型，跳过训练！")

            # 处理不同类型的输入数据
            front_numbers = []
            back_numbers = []

            if isinstance(data, np.ndarray):
                # 如果是numpy数组，取最后几行作为序列数据
                if len(data) >= self.sequence_length:
                    recent_data = data[-self.sequence_length:]
                else:
                    recent_data = data

                # 直接从numpy数组提取号码
                if recent_data.shape[1] >= 7:
                    front_numbers = recent_data[:, :5].tolist()
                    back_numbers = recent_data[:, 5:7].tolist()
                else:
                    raise ValueError(f"数据维度不正确，期望至少7列，实际{recent_data.shape[1]}列")

            elif isinstance(data, pd.DataFrame):
                # 准备最近的序列数据
                recent_data = self._latest_chronological_window(data, self.sequence_length)

                # 提取号码 - 使用data_manager的parse_balls方法
                import core_modules as cm

                for _, row in recent_data.iterrows():
                    front_balls, back_balls = cm.data_manager.parse_balls(row)
                    if len(front_balls) == 5 and len(back_balls) == 2:
                        front_numbers.append(front_balls)
                        back_numbers.append(back_balls)
            else:
                raise ValueError(f"不支持的数据类型: {type(data)}")

            # 确保有足够的数据
            if len(front_numbers) < self.sequence_length:
                logger_manager.warning(f"数据不足，需要{self.sequence_length}期，实际{len(front_numbers)}期")
                return []
            
            # 标准化 - 确保数据类型正确
            front_array = np.array(front_numbers, dtype=np.float32)
            back_array = np.array(back_numbers, dtype=np.float32)

            # 检查数据维度
            logger_manager.info(f"前区数据形状: {front_array.shape}, 后区数据形状: {back_array.shape}")

            try:
                # 如果scaler没有被训练，先训练它
                if not hasattr(self.front_scaler, 'scale_') or self.front_scaler.scale_ is None:
                    logger_manager.info("训练前区scaler...")
                    self.front_scaler.fit(front_array)
                if not hasattr(self.back_scaler, 'scale_') or self.back_scaler.scale_ is None:
                    logger_manager.info("训练后区scaler...")
                    self.back_scaler.fit(back_array)

                front_scaled = self.front_scaler.transform(front_array)
                back_scaled = self.back_scaler.transform(back_array)

            except Exception as e:
                logger_manager.error(f"数据标准化失败: {e}")
                # 使用简单的归一化作为回退
                front_scaled = front_array / 35.0
                back_scaled = back_array / 12.0
            
            # 预测
            predictions = []
            
            for _ in range(count):
                try:
                    # 前区预测 - 使用最后几期数据
                    front_recent = front_scaled[-self.sequence_length:]
                    # 注入小幅随机噪声，避免确定性输入导致多次预测结果完全一致
                    front_recent = np.clip(
                        front_recent + np.random.normal(0, 0.02, front_recent.shape),
                        0,
                        1,
                    )
                    front_input = front_recent.reshape(1, self.sequence_length, 5)
                    front_pred = self.front_model.predict(front_input, verbose=0)

                    # 反标准化
                    if hasattr(self.front_scaler, 'scale_') and self.front_scaler.scale_ is not None:
                        front_pred = self.front_scaler.inverse_transform(front_pred)
                    else:
                        front_pred = front_pred * 35.0

                    # 后区预测 - 使用最后几期数据
                    back_recent = back_scaled[-self.sequence_length:]
                    # 注入小幅随机噪声，避免确定性输入导致多次预测结果完全一致
                    back_recent = np.clip(
                        back_recent + np.random.normal(0, 0.02, back_recent.shape),
                        0,
                        1,
                    )
                    back_input = back_recent.reshape(1, self.sequence_length, 2)
                    back_pred = self.back_model.predict(back_input, verbose=0)

                    # 反标准化
                    if hasattr(self.back_scaler, 'scale_') and self.back_scaler.scale_ is not None:
                        back_pred = self.back_scaler.inverse_transform(back_pred)
                    else:
                        back_pred = back_pred * 12.0

                except Exception as e:
                    logger_manager.error(f"LSTM预测过程出错: {e}")
                    import traceback
                    logger_manager.error(f"错误堆栈: {traceback.format_exc()}")
                    # 使用简单的预测作为回退
                    front_pred = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
                    back_pred = np.array([[1, 2]], dtype=np.float32)
                
                # 转换为整数并确保无重复 - 修复重复号码问题
                front_pred_flat = front_pred.flatten() if front_pred.ndim > 1 else front_pred
                back_pred_flat = back_pred.flatten() if back_pred.ndim > 1 else back_pred

                # 转换为有效范围内的整数
                front_candidates = [max(1, min(35, int(round(float(x))))) for x in front_pred_flat]
                back_candidates = [max(1, min(12, int(round(float(x))))) for x in back_pred_flat]

                # 去除重复并确保数量正确 - 前区5个不重复号码
                front_balls = []
                front_set = set()
                for num in front_candidates:
                    if num not in front_set and len(front_balls) < 5:
                        front_balls.append(num)
                        front_set.add(num)

                # 如果前区号码不够5个，随机补充
                if len(front_balls) < 5:
                    import random
                    available_front = [i for i in range(1, 36) if i not in front_set]
                    random.shuffle(available_front)
                    front_balls.extend(available_front[:5-len(front_balls)])

                # 去除重复并确保数量正确 - 后区2个不重复号码
                back_balls = []
                back_set = set()
                for num in back_candidates:
                    if num not in back_set and len(back_balls) < 2:
                        back_balls.append(num)
                        back_set.add(num)

                # 如果后区号码不够2个，随机补充
                if len(back_balls) < 2:
                    import random
                    available_back = [i for i in range(1, 13) if i not in back_set]
                    random.shuffle(available_back)
                    back_balls.extend(available_back[:2-len(back_balls)])

                # 排序
                front_balls = sorted(front_balls)
                back_balls = sorted(back_balls)
                
                predictions.append((front_balls, back_balls))
            
            logger_manager.info(f"LSTM预测完成，生成 {len(predictions)} 注预测")
            
            return predictions
            
        except Exception as e:
            logger_manager.error(f"LSTM预测失败: {e}")
            logger_manager.error(f"错误类型: {type(e).__name__}")
            import traceback
            logger_manager.error(f"错误堆栈: {traceback.format_exc()}")
            # 重新抛出异常，不使用回退机制
            raise e


    
    def save_models(self) -> bool:
        """保存模型"""
        try:
            # 保存Keras模型。优先使用 Keras 3 原生格式，避免旧 HDF5 编译配置反序列化问题。
            self.front_model.save(os.path.join(self.model_dir, 'front_lstm_model.keras'))
            self.back_model.save(os.path.join(self.model_dir, 'back_lstm_model.keras'))
            
            # 保存缩放器
            joblib.dump(self.front_scaler, os.path.join(self.model_dir, 'front_scaler.pkl'))
            joblib.dump(self.back_scaler, os.path.join(self.model_dir, 'back_scaler.pkl'))

            metadata = {
                'cache_version': self.model_cache_version,
                'data_order': self.model_data_order,
                'sequence_length': self.sequence_length,
                'saved_at': datetime.now().isoformat()
            }
            with open(os.path.join(self.model_dir, 'model_metadata.json'), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger_manager.info("LSTM模型保存成功")
            return True
            
        except Exception as e:
            logger_manager.error(f"LSTM模型保存失败: {e}")
            return False
    
    def load_models(self) -> bool:
        """加载模型"""
        try:
            model_pairs = [
                (
                    os.path.join(self.model_dir, 'front_lstm_model.keras'),
                    os.path.join(self.model_dir, 'back_lstm_model.keras')
                ),
                (
                    os.path.join(self.model_dir, 'front_lstm_model.h5'),
                    os.path.join(self.model_dir, 'back_lstm_model.h5')
                )
            ]
            front_scaler_path = os.path.join(self.model_dir, 'front_scaler.pkl')
            back_scaler_path = os.path.join(self.model_dir, 'back_scaler.pkl')

            selected_pair = next(
                ((front_path, back_path) for front_path, back_path in model_pairs
                 if os.path.exists(front_path) and os.path.exists(back_path)),
                None
            )

            if selected_pair is None or not all(os.path.exists(p) for p in [front_scaler_path, back_scaler_path]):
                logger_manager.warning("LSTM模型文件不存在")
                return False

            metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
            if not os.path.exists(metadata_path):
                logger_manager.warning("LSTM模型缺少元数据，将重新训练以修正时间序列方向")
                return False

            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            if (
                metadata.get('cache_version') != self.model_cache_version or
                metadata.get('data_order') != self.model_data_order or
                metadata.get('sequence_length') != self.sequence_length
            ):
                logger_manager.warning("LSTM模型元数据不匹配，将重新训练")
                return False

            front_model_path, back_model_path = selected_pair
            
            # compile=False 可兼容旧模型中 loss='mse' 的 Keras 3 反序列化问题。
            self.front_model = tf.keras.models.load_model(front_model_path, compile=False)
            self.back_model = tf.keras.models.load_model(back_model_path, compile=False)
            
            # 加载缩放器
            self.front_scaler = joblib.load(front_scaler_path)
            self.back_scaler = joblib.load(back_scaler_path)
            
            logger_manager.info("LSTM模型加载成功")
            self.is_trained = True
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

    def predict_compound(self, config: Optional[CompoundConfig] = None) -> CompoundResult:
        """
        LSTM复式预测

        Args:
            config: 复式预测配置

        Returns:
            复式预测结果
        """
        if config is None:
            config = self.compound_config or CompoundConfig()

        # 验证参数
        if not self.validate_compound_params(config.front_count, config.back_count, config.max_cost):
            raise ValueError("LSTM复式预测参数验证失败")

        logger_manager.info(f"开始LSTM复式预测: {config.front_count}+{config.back_count}")

        try:
            # 确保模型已训练
            if not self.is_trained:
                if not self.load_models():
                    logger_manager.info("LSTM模型未训练，开始训练...")
                    # 获取数据进行训练
                    data = data_manager.get_data()
                    if not self.train(data):
                        raise Exception("LSTM模型训练失败")

            # 生成多个候选预测
            candidate_count = max(config.front_count * 2, 20)
            candidates = self._generate_compound_candidates(candidate_count, config.periods)

            # 基于置信度和多样性选择最优组合
            front_balls, back_balls = self._select_optimal_compound(
                candidates, config.front_count, config.back_count
            )

            # 计算组合数和成本
            combinations = self.calculate_combinations(config.front_count, config.back_count)
            cost = self.calculate_cost(combinations)

            # 计算置信度
            confidence = self._calculate_compound_confidence(front_balls, back_balls, candidates)

            # 创建结果
            from datetime import datetime
            result = CompoundResult(
                front_balls=front_balls,
                back_balls=back_balls,
                front_count=config.front_count,
                back_count=config.back_count,
                total_combinations=combinations,
                total_cost=cost,
                confidence=confidence,
                method="LSTM复式预测",
                analysis_periods=config.periods,
                timestamp=datetime.now().isoformat(),
                details={
                    'model_type': 'LSTM',
                    'candidate_count': candidate_count,
                    'selection_strategy': 'confidence_diversity'
                }
            )

            logger_manager.info(f"LSTM复式预测完成: {config.front_count}+{config.back_count}, 置信度: {confidence:.3f}")
            return result

        except Exception as e:
            logger_manager.error(f"LSTM复式预测失败: {e}")
            # 返回默认结果
            return super().predict_compound(config)

    def _generate_compound_candidates(self, count: int, periods: int) -> List[Tuple[List[int], List[int]]]:
        """生成复式候选预测"""
        candidates = []

        try:
            # 获取历史数据
            historical_data = data_manager.get_data()
            if historical_data is None or len(historical_data) < periods:
                raise Exception("历史数据不足")

            recent_data = historical_data.head(periods).iloc[::-1].reset_index(drop=True)

            # 生成多个预测
            for i in range(count):
                # 使用不同的随机种子生成多样化预测
                np.random.seed(42 + i)

                # 准备输入数据
                input_data = self._prepare_prediction_input(recent_data)
                # 注入小幅随机噪声，避免确定性输入导致多次预测结果完全一致
                input_data['front'] = np.clip(
                    input_data['front'] + np.random.normal(0, 0.02, input_data['front'].shape),
                    0,
                    1,
                )
                input_data['back'] = np.clip(
                    input_data['back'] + np.random.normal(0, 0.02, input_data['back'].shape),
                    0,
                    1,
                )

                # 模型预测
                front_pred = self.front_model.predict(input_data['front'], verbose=0)
                back_pred = self.back_model.predict(input_data['back'], verbose=0)

                # 反标准化
                front_pred = self.front_scaler.inverse_transform(front_pred)
                back_pred = self.back_scaler.inverse_transform(back_pred)

                # 转换为号码
                front_balls = self._convert_to_balls(front_pred[0], True)
                back_balls = self._convert_to_balls(back_pred[0], False)

                candidates.append((front_balls, back_balls))

            logger_manager.debug(f"生成 {len(candidates)} 个LSTM候选预测")
            return candidates

        except Exception as e:
            logger_manager.error(f"生成LSTM候选预测失败: {e}")
            return []

    def _select_optimal_compound(self, candidates: List[Tuple[List[int], List[int]]],
                               front_count: int, back_count: int) -> Tuple[List[int], List[int]]:
        """选择最优复式组合"""
        if not candidates:
            # 如果没有候选，使用随机生成
            import random
            front_balls = sorted(random.sample(range(1, 36), front_count))
            back_balls = sorted(random.sample(range(1, 13), back_count))
            return front_balls, back_balls

        # 统计号码出现频率
        front_freq = {}
        back_freq = {}

        for front_balls, back_balls in candidates:
            for ball in front_balls:
                front_freq[ball] = front_freq.get(ball, 0) + 1
            for ball in back_balls:
                back_freq[ball] = back_freq.get(ball, 0) + 1

        # 按频率排序选择
        front_sorted = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)
        back_sorted = sorted(back_freq.items(), key=lambda x: x[1], reverse=True)

        # 选择频率最高的号码
        selected_front = [ball for ball, freq in front_sorted[:front_count]]
        selected_back = [ball for ball, freq in back_sorted[:back_count]]

        return sorted(selected_front), sorted(selected_back)

    def _calculate_compound_confidence(self, front_balls: List[int], back_balls: List[int],
                                     candidates: List[Tuple[List[int], List[int]]]) -> float:
        """计算复式预测置信度"""
        if not candidates:
            return 0.5

        # 计算选中号码在候选中的出现频率
        front_appearances = sum(1 for front, _ in candidates if any(ball in front for ball in front_balls))
        back_appearances = sum(1 for _, back in candidates if any(ball in back for ball in back_balls))

        front_confidence = front_appearances / len(candidates)
        back_confidence = back_appearances / len(candidates)

        # 综合置信度
        overall_confidence = (front_confidence + back_confidence) / 2

        return min(0.95, max(0.1, overall_confidence))

    def _convert_to_balls(self, predictions: np.ndarray, is_front: bool) -> List[int]:
        """将预测值转换为号码"""
        if is_front:
            # 前区：1-35
            balls = [max(1, min(35, int(round(x)))) for x in predictions]
            # 确保5个不重复的号码
            unique_balls = list(set(balls))
            while len(unique_balls) < 5:
                new_ball = np.random.randint(1, 36)
                if new_ball not in unique_balls:
                    unique_balls.append(new_ball)
            return sorted(unique_balls[:5])
        else:
            # 后区：1-12
            balls = [max(1, min(12, int(round(x)))) for x in predictions]
            # 确保2个不重复的号码
            unique_balls = list(set(balls))
            while len(unique_balls) < 2:
                new_ball = np.random.randint(1, 13)
                if new_ball not in unique_balls:
                    unique_balls.append(new_ball)
            return sorted(unique_balls[:2])

    def _prepare_prediction_input(self, data) -> Dict[str, np.ndarray]:
        """准备预测输入数据"""
        # 简化的输入准备，实际应该与训练时保持一致
        sequences = []
        for i in range(min(self.sequence_length, len(data))):
            row = data.iloc[i]
            front_balls = [int(x) for x in str(row.get('front_balls', '')).split(',') if x.strip().isdigit()]
            back_balls = [int(x) for x in str(row.get('back_balls', '')).split(',') if x.strip().isdigit()]

            if len(front_balls) == 5 and len(back_balls) == 2:
                # 归一化
                front_normalized = [x / 35.0 for x in front_balls]
                back_normalized = [x / 12.0 for x in back_balls]
                sequences.append(front_normalized + back_normalized)

        if len(sequences) < self.sequence_length:
            # 填充序列
            while len(sequences) < self.sequence_length:
                sequences.append([0.5] * 7)  # 使用中位数填充

        sequence_array = np.array(sequences[-self.sequence_length:]).reshape(1, self.sequence_length, 7)

        # 分离前区和后区
        front_input = sequence_array[:, :, :5]
        back_input = sequence_array[:, :, 5:]

        return {'front': front_input, 'back': back_input}
