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
from datetime import datetime

from .base import BaseDeepPredictor
from .config import DEFAULT_TRANSFORMER_CONFIG
from .exceptions import ModelInitializationError, handle_model_error
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
        self.d_model = self.config.get('d_model', 128)
        self.num_heads = self.config.get('num_heads', 8)
        self.num_layers = self.config.get('num_layers', 4)
        self.dff = self.config.get('dff', 512)
        self.dropout_rate = self.config.get('dropout_rate', 0.1)
        
        logger_manager.info(f"初始化Transformer预测器: d_model={self.d_model}, heads={self.num_heads}, layers={self.num_layers}")
    
    def _build_model(self):
        """构建Transformer模型"""
        try:
            # 输入层
            inputs = layers.Input(shape=(self.sequence_length, self.feature_dim))
            
            # 位置编码
            positions = tf.range(start=0, limit=self.sequence_length, delta=1)
            positions = layers.Embedding(self.sequence_length, self.d_model)(positions)
            
            # 输入嵌入
            x = layers.Dense(self.d_model)(inputs)
            x = x + positions
            
            # Transformer层
            for i in range(self.num_layers):
                # 多头注意力
                attention_output = layers.MultiHeadAttention(
                    num_heads=self.num_heads, 
                    key_dim=self.d_model // self.num_heads
                )(x, x)
                x = layers.Add()([x, attention_output])
                x = layers.LayerNormalization(epsilon=1e-6)(x)
                
                # 前馈网络
                ffn_output = self._point_wise_feed_forward_network(x)
                x = layers.Add()([x, ffn_output])
                x = layers.LayerNormalization(epsilon=1e-6)(x)
            
            # 全局平均池化
            x = layers.GlobalAveragePooling1D()(x)
            
            # 全连接层
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dropout(self.dropout_rate)(x)
            
            # 输出层 - 使用单一输出，前5个值为前区，后2个值为后区
            outputs = layers.Dense(7, activation='sigmoid')(x)
            
            # 构建模型
            model = Model(inputs=inputs, outputs=outputs)
            
            # 编译模型
            model.compile(
                optimizer=optimizers.Adam(learning_rate=self.config.get('learning_rate', 0.001)),
                loss='mse',
                metrics=['mae']
            )
            
            # 打印模型摘要
            model.summary()
            
            return model
        except Exception as e:
            raise ModelInitializationError("Transformer", str(e))
    
    def _point_wise_feed_forward_network(self, x):
        """实现Transformer的前馈网络"""
        x = layers.Dense(self.dff, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(self.d_model)(x)
        return x
    
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