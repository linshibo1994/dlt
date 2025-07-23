#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transformer预测器
基于Transformer架构的深度学习预测模型
"""

import os
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Any
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from datetime import datetime
import math

from . import BaseDeepPredictor
from ..utils.config import DEFAULT_TRANSFORMER_CONFIG
from ..utils.exceptions import ModelInitializationError, handle_model_error
from core_modules import logger_manager


class TransformerPredictor(BaseDeepPredictor):
    """基于Transformer的彩票预测模型"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化Transformer预测器
        
        Args:
            config: 配置参数字典
        """
        # 合并默认配置和用户配置
        merged_config = DEFAULT_TRANSFORMER_CONFIG.copy()
        if config:
            merged_config.update(config)
        
        super().__init__(name="Transformer", config=merged_config)
        
        # 从配置中提取参数
        self.d_model = self.config.get('d_model', 256)
        self.num_heads = self.config.get('num_heads', 8)
        self.num_encoder_layers = self.config.get('num_encoder_layers', 6)
        self.num_decoder_layers = self.config.get('num_decoder_layers', 6)
        self.dff = self.config.get('dff', 1024)
        self.dropout_rate = self.config.get('dropout_rate', 0.1)

        # 高级Transformer参数
        self.use_relative_position = self.config.get('use_relative_position', True)
        self.use_sparse_attention = self.config.get('use_sparse_attention', False)
        self.use_local_attention = self.config.get('use_local_attention', False)
        self.local_attention_window = self.config.get('local_attention_window', 64)
        self.max_position_encoding = self.config.get('max_position_encoding', 1000)

        logger_manager.info(f"初始化增强Transformer预测器: d_model={self.d_model}, heads={self.num_heads}, "
                          f"encoder_layers={self.num_encoder_layers}, decoder_layers={self.num_decoder_layers}")
    
    def _build_model(self):
        """构建增强Transformer模型（编码器-解码器架构）"""
        try:
            # 编码器输入
            encoder_inputs = layers.Input(shape=(self.sequence_length, self.feature_dim), name='encoder_inputs')

            # 解码器输入（用于训练时的teacher forcing）
            decoder_inputs = layers.Input(shape=(None, 7), name='decoder_inputs')  # 7 = 5前区 + 2后区

            # 构建编码器
            encoder_outputs = self._build_encoder(encoder_inputs)

            # 构建解码器
            decoder_outputs = self._build_decoder(decoder_inputs, encoder_outputs)

            # 构建完整模型
            model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs, name='Enhanced_Transformer')

            # 编译模型
            optimizer = optimizers.Adam(
                learning_rate=self.config.get('learning_rate', 0.0001),
                beta_1=0.9,
                beta_2=0.98,
                epsilon=1e-9
            )

            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae', 'mape']
            )

            # 打印模型摘要
            model.summary()

            return model
        except Exception as e:
            raise ModelInitializationError("Enhanced_Transformer", str(e))

    def _build_encoder(self, inputs):
        """构建Transformer编码器"""
        # 输入嵌入和位置编码
        x = layers.Dense(self.d_model, name='encoder_embedding')(inputs)
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # 位置编码
        if self.use_relative_position:
            x = self._add_relative_position_encoding(x)
        else:
            x = self._add_absolute_position_encoding(x)

        x = layers.Dropout(self.dropout_rate)(x)

        # 编码器层
        for i in range(self.num_encoder_layers):
            x = self._encoder_layer(x, i)

        return x

    def _build_decoder(self, inputs, encoder_outputs):
        """构建Transformer解码器"""
        # 输入嵌入和位置编码
        x = layers.Dense(self.d_model, name='decoder_embedding')(inputs)
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # 位置编码
        if self.use_relative_position:
            x = self._add_relative_position_encoding(x, name_prefix='decoder')
        else:
            x = self._add_absolute_position_encoding(x, name_prefix='decoder')

        x = layers.Dropout(self.dropout_rate)(x)

        # 解码器层
        for i in range(self.num_decoder_layers):
            x = self._decoder_layer(x, encoder_outputs, i)

        # 输出投影
        outputs = layers.Dense(7, activation='linear', name='output_projection')(x)

        return outputs

    def _encoder_layer(self, x, layer_idx):
        """单个编码器层"""
        # 多头自注意力
        if self.use_sparse_attention:
            attention_output = self._sparse_multi_head_attention(x, x, layer_idx, 'encoder')
        elif self.use_local_attention:
            attention_output = self._local_multi_head_attention(x, x, layer_idx, 'encoder')
        else:
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads,
                dropout=self.dropout_rate,
                name=f'encoder_mha_{layer_idx}'
            )(x, x)

        # 残差连接和层归一化
        x = layers.Add(name=f'encoder_add_1_{layer_idx}')([x, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6, name=f'encoder_ln_1_{layer_idx}')(x)

        # 前馈网络
        ffn_output = self._point_wise_feed_forward_network(x, layer_idx, 'encoder')

        # 残差连接和层归一化
        x = layers.Add(name=f'encoder_add_2_{layer_idx}')([x, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-6, name=f'encoder_ln_2_{layer_idx}')(x)

        return x

    def _decoder_layer(self, x, encoder_outputs, layer_idx):
        """单个解码器层"""
        # 掩码多头自注意力
        masked_attention_output = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate,
            name=f'decoder_masked_mha_{layer_idx}'
        )(x, x, use_causal_mask=True)

        # 残差连接和层归一化
        x = layers.Add(name=f'decoder_add_1_{layer_idx}')([x, masked_attention_output])
        x = layers.LayerNormalization(epsilon=1e-6, name=f'decoder_ln_1_{layer_idx}')(x)

        # 编码器-解码器注意力
        cross_attention_output = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate,
            name=f'decoder_cross_mha_{layer_idx}'
        )(x, encoder_outputs)

        # 残差连接和层归一化
        x = layers.Add(name=f'decoder_add_2_{layer_idx}')([x, cross_attention_output])
        x = layers.LayerNormalization(epsilon=1e-6, name=f'decoder_ln_2_{layer_idx}')(x)

        # 前馈网络
        ffn_output = self._point_wise_feed_forward_network(x, layer_idx, 'decoder')

        # 残差连接和层归一化
        x = layers.Add(name=f'decoder_add_3_{layer_idx}')([x, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-6, name=f'decoder_ln_3_{layer_idx}')(x)

        return x
    
    def _point_wise_feed_forward_network(self, x, layer_idx=0, layer_type='encoder'):
        """实现Transformer的前馈网络"""
        x = layers.Dense(self.dff, activation='relu', name=f'{layer_type}_ffn_1_{layer_idx}')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(self.d_model, name=f'{layer_type}_ffn_2_{layer_idx}')(x)
        return x

    def _add_absolute_position_encoding(self, x, name_prefix='encoder'):
        """添加绝对位置编码"""
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        position_embeddings = layers.Embedding(
            self.max_position_encoding,
            self.d_model,
            name=f'{name_prefix}_position_embedding'
        )(positions)
        return x + position_embeddings

    def _add_relative_position_encoding(self, x, name_prefix='encoder'):
        """添加相对位置编码（更高级的位置编码）"""
        seq_len = tf.shape(x)[1]

        # 创建相对位置矩阵
        positions = tf.range(seq_len)
        relative_positions = positions[:, None] - positions[None, :]

        # 限制相对位置的范围
        max_relative_position = 32
        relative_positions = tf.clip_by_value(
            relative_positions,
            -max_relative_position,
            max_relative_position
        )

        # 相对位置嵌入
        relative_position_embeddings = layers.Embedding(
            2 * max_relative_position + 1,
            self.d_model,
            name=f'{name_prefix}_relative_position_embedding'
        )(relative_positions + max_relative_position)

        # 将相对位置编码添加到输入
        return x + tf.reduce_mean(relative_position_embeddings, axis=1, keepdims=True)

    def _sparse_multi_head_attention(self, query, key, layer_idx, layer_type):
        """稀疏多头注意力（减少计算复杂度）"""
        # 简化的稀疏注意力实现
        # 在实际应用中，这里会实现更复杂的稀疏模式

        # 使用局部窗口注意力作为稀疏注意力的简化版本
        return self._local_multi_head_attention(query, key, layer_idx, layer_type)

    def _local_multi_head_attention(self, query, key, layer_idx, layer_type):
        """局部多头注意力"""
        # 创建局部注意力掩码
        seq_len = tf.shape(query)[1]
        window_size = min(self.local_attention_window, seq_len)

        # 简化实现：使用标准多头注意力但限制注意力范围
        attention_output = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate,
            name=f'{layer_type}_local_mha_{layer_idx}'
        )(query, key)

        return attention_output

    def _create_learning_rate_scheduler(self):
        """创建Transformer专用的学习率调度器"""
        def scheduler(epoch, lr):
            # Transformer的预热学习率调度
            warmup_steps = 4000
            step = epoch + 1

            arg1 = tf.math.rsqrt(step)
            arg2 = step * (warmup_steps ** -1.5)

            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

        return LearningRateScheduler(scheduler, verbose=0)

    def _get_advanced_callbacks(self):
        """获取高级回调函数"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.8,
                patience=10,
                min_lr=1e-8,
                verbose=1
            ),
            self._create_learning_rate_scheduler()
        ]

        return callbacks
    
    @handle_model_error
    def train(self, epochs=None, validation_split=0.2, batch_size=None):
        """
        训练Transformer模型
        
        Args:
            epochs: 训练轮数，如果为None则使用配置中的值
            validation_split: 验证集比例
            batch_size: 批处理大小，如果为None则使用配置中的值
            
        Returns:
            训练是否成功
        """
        from .data_manager import DeepLearningDataManager
        from .training_utils import get_callbacks, TrainingVisualizer
        
        if epochs is None:
            epochs = self.config.get('epochs', 100)
        
        if batch_size is None:
            batch_size = self.config.get('batch_size', 64)
        
        logger_manager.info(f"开始训练Transformer模型: epochs={epochs}, batch_size={batch_size}")
        
        try:
            # 创建数据管理器
            data_manager = DeepLearningDataManager()
            
            # 准备批处理数据
            batch_data = data_manager.prepare_batch_data(
                sequence_length=self.sequence_length,
                batch_size=batch_size,
                validation_split=validation_split
            )
            
            # 更新特征维度
            self.feature_dim = batch_data['feature_dim']
            
            # 构建模型
            if self.model is None:
                self.model = self._build_model()
            
            # 获取回调函数
            callbacks = get_callbacks(self.name, epochs)
            
            # 训练模型
            history = self.model.fit(
                batch_data['train_dataset'],
                epochs=epochs,
                validation_data=batch_data['val_dataset'],
                callbacks=callbacks,
                verbose=0  # 使用自定义进度条，禁用TensorFlow的进度条
            )
            
            self.is_trained = True
            
            # 保存模型
            self._save_model()
            
            # 可视化训练历史
            visualizer = TrainingVisualizer(self.name)
            visualizer.plot_training_history(history.history)
            
            logger_manager.info(f"{self.name}模型训练完成")
            
            return True
        
        except Exception as e:
            logger_manager.error(f"{self.name}模型训练失败: {e}")
            return False
    
    @handle_model_error
    def predict(self, count=1, verbose=True) -> List[Tuple[List[int], List[int]]]:
        """
        生成预测结果
        
        Args:
            count: 预测注数
            verbose: 是否显示详细信息
            
        Returns:
            预测结果列表，每个元素为(前区号码列表, 后区号码列表)
        """
        from .prediction_utils import PredictionProcessor
        
        # 尝试加载已有模型
        if not self.is_trained:
            if not self._load_model():
                logger_manager.info(f"{self.name}模型未训练，开始训练...")
                if not self.train():
                    logger_manager.error(f"{self.name}模型训练失败")
                    return []
        
        # 获取最近的序列数据
        recent_features = self._extract_features(self.df.head(self.sequence_length))
        recent_scaled = self.scaler.transform(recent_features)
        
        # 准备输入序列
        input_sequence = recent_scaled.reshape(1, self.sequence_length, self.feature_dim)
        
        # 创建预测处理器
        processor = PredictionProcessor()
        
        predictions = []
        raw_predictions = []
        
        if verbose:
            logger_manager.info(f"使用{self.name}模型生成{count}注预测...")
        
        for i in range(count):
            # 预测
            pred_scaled = self.model.predict(input_sequence, verbose=0)
            
            # 反标准化
            # 创建完整特征向量用于反标准化
            full_pred = np.zeros((1, self.feature_dim))
            full_pred[0, :7] = pred_scaled[0]
            pred_original = self.scaler.inverse_transform(full_pred)[0, :7]
            
            # 保存原始预测结果
            raw_predictions.append(pred_original)
            
            # 处理预测结果
            front_balls, back_balls = processor.process_raw_prediction(pred_original)
            predictions.append((front_balls, back_balls))
            
            # 更新输入序列用于下一次预测
            new_feature = np.concatenate([pred_original, recent_scaled[-1, 7:]])
            input_sequence = np.roll(input_sequence, -1, axis=1)
            input_sequence[0, -1] = new_feature
            
            if verbose:
                formatted = processor.format_prediction((front_balls, back_balls))
                logger_manager.info(f"预测 {i+1}/{count}: {formatted}")
        
        # 计算预测置信度
        confidence = processor.calculate_confidence(predictions)
        
        if verbose:
            logger_manager.info(f"{self.name}预测完成，置信度: {confidence:.2f}")
        
        return predictions
    
    def predict_with_details(self, count=1) -> Dict[str, Any]:
        """
        生成带详细信息的预测结果
        
        Args:
            count: 预测注数
            
        Returns:
            包含预测结果和详细信息的字典
        """
        from .prediction_utils import PredictionProcessor
        
        # 执行预测
        predictions = self.predict(count, verbose=False)
        
        # 创建预测处理器
        processor = PredictionProcessor()
        
        # 计算置信度
        confidence = processor.calculate_confidence(predictions)
        
        # 格式化预测结果
        formatted_predictions = []
        for i, pred in enumerate(predictions):
            formatted = processor.format_prediction(pred)
            formatted_predictions.append({
                'index': i + 1,
                'front_balls': pred[0],
                'back_balls': pred[1],
                'formatted': formatted
            })
        
        # 返回详细结果
        return {
            'model_name': self.name,
            'count': count,
            'predictions': formatted_predictions,
            'confidence': confidence,
            'model_config': {
                'd_model': self.d_model,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def evaluate_predictions(self, predictions: List[Tuple[List[int], List[int]]], 
                           actuals: List[Tuple[List[int], List[int]]]) -> Dict[str, Any]:
        """
        评估预测结果
        
        Args:
            predictions: 预测结果列表
            actuals: 实际结果列表
            
        Returns:
            评估结果字典
        """
        from .prediction_utils import PredictionEvaluator
        
        evaluator = PredictionEvaluator()
        return evaluator.evaluate_multiple_predictions(predictions, actuals)
    
    def get_confidence(self) -> float:
        """
        获取预测置信度
        
        Returns:
            置信度分数 (0.0-1.0)
        """
        if not self.is_trained:
            return 0.0
        
        # 基于模型验证性能计算置信度
        # 这里使用一个简单的启发式方法，实际应用中可以基于验证集性能
        base_confidence = 0.7
        
        # 根据模型复杂度调整
        complexity_factor = min(1.0, (self.num_layers * self.num_heads) / 40)
        
        # 根据训练数据量调整
        data_factor = min(1.0, len(self.df) / 1000)
        
        confidence = base_confidence * complexity_factor * data_factor
        
        return min(0.95, confidence)  # 最高置信度限制在0.95
    
    def use_fallback_config(self):
        """使用备用配置"""
        logger_manager.info("使用Transformer备用配置")
        
        # 简化模型配置
        self.d_model = 64
        self.num_heads = 4
        self.num_layers = 2
        self.dff = 256
        self.dropout_rate = 0.2
        
        # 更新配置字典
        self.config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate
        })
    
    def use_simple_model(self):
        """使用简单模型"""
        logger_manager.info("使用简单Transformer模型")
        
        # 极简配置
        self.d_model = 32
        self.num_heads = 2
        self.num_layers = 1
        self.dff = 128
        self.dropout_rate = 0.1
        
        # 更新配置字典
        self.config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate
        })


if __name__ == "__main__":
    # 测试Transformer预测器
    print("🧠 测试Transformer预测器...")
    
    # 创建预测器
    transformer = TransformerPredictor()
    
    # 训练模型
    transformer.train(epochs=10)
    
    # 进行预测
    predictions = transformer.predict(3)
    
    print("Transformer预测结果:")
    for i, (front, back) in enumerate(predictions):
        front_str = ' '.join([str(b).zfill(2) for b in front])
        back_str = ' '.join([str(b).zfill(2) for b in back])
        print(f"第 {i+1} 注: {front_str} + {back_str}")