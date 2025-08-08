#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transformer预测器
基于Transformer架构的深度学习预测模型
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Tuple, Dict, Any, Optional
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import math

from .base_model import BaseModel as BaseDeepPredictor, ModelConfig, ModelType
from .metadata import ModelMetadata
from ..utils.config import DEFAULT_TRANSFORMER_CONFIG
from ..utils.exceptions import ModelInitializationError, handle_model_error
# 导入核心模块
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import core_modules as cm
from core_modules import data_manager as core_data_manager
logger_manager = cm.logger_manager
core_data_manager = cm.data_manager

from compound_modules.compound_predictor import CompoundPredictorMixin, CompoundConfig, CompoundResult

# 导入智能训练轮数计算器
try:
    from ..training.smart_epochs_calculator import SmartEpochsCalculator, TrainingConfig, ModelType as SmartModelType, PerformanceMode
except ImportError:
    SmartEpochsCalculator = None
    TrainingConfig = None
    SmartModelType = None
    PerformanceMode = None

# 导入智能早停机制
from ..utils.intelligent_early_stopping import IntelligentEarlyStopping, create_intelligent_callbacks


class TransformerPredictor(BaseDeepPredictor, CompoundPredictorMixin):
    """基于Transformer的彩票预测模型（支持复式预测）"""
    
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
        
        # 创建ModelConfig对象
        model_config = ModelConfig(
            model_type=ModelType.TRANSFORMER,
            model_name="TransformerPredictor",
            version="2.0.0",
            description="基于Transformer的彩票号码预测模型"
        )
        super().__init__(model_config)
        CompoundPredictorMixin.__init__(self)

        # 保存配置参数
        self.config_params = merged_config
        
        # 从配置中提取参数
        self.d_model = self.config_params.get('d_model', 256)
        self.num_heads = self.config_params.get('num_heads', 8)
        self.num_encoder_layers = self.config_params.get('num_encoder_layers', 6)
        self.num_decoder_layers = self.config_params.get('num_decoder_layers', 6)
        self.dff = self.config_params.get('dff', 1024)
        self.dropout_rate = self.config_params.get('dropout_rate', 0.1)

        # 高级Transformer参数
        self.use_relative_position = self.config_params.get('use_relative_position', True)
        self.use_sparse_attention = self.config_params.get('use_sparse_attention', False)
        self.use_local_attention = self.config_params.get('use_local_attention', False)
        self.local_attention_window = self.config_params.get('local_attention_window', 64)
        self.max_position_encoding = self.config_params.get('max_position_encoding', 1000)

        # 添加缺失的序列和特征维度参数
        self.sequence_length = self.config_params.get('sequence_length', 20)
        self.feature_dim = self.config_params.get('feature_dim', 7)  # 5前区 + 2后区

        # 模型和缩放器
        self.front_model = None
        self.back_model = None
        self.front_scaler = None
        self.back_scaler = None
        self.scaler = StandardScaler()  # 添加通用scaler

        # 训练状态
        self.is_trained = False

        # 模型名称
        self.name = "TransformerPredictor"

        # 模型保存目录
        self.model_dir = self.config_params.get('model_dir', 'models/transformer')
        import os
        os.makedirs(self.model_dir, exist_ok=True)

        logger_manager.info(f"初始化增强Transformer预测器: d_model={self.d_model}, heads={self.num_heads}, "
                          f"encoder_layers={self.num_encoder_layers}, decoder_layers={self.num_decoder_layers}")

    def build_model(self):
        """构建模型（公共接口）"""
        return self._build_model()

    def _build_model(self):
        """构建简化Transformer模型（仅编码器架构）"""
        try:
            # 输入层
            inputs = layers.Input(shape=(self.sequence_length, self.feature_dim), name='inputs')

            # 构建编码器
            encoder_outputs = self._build_encoder(inputs)

            # 全局平均池化
            pooled = layers.GlobalAveragePooling1D()(encoder_outputs)

            # 输出层
            outputs = layers.Dense(7, activation='linear', name='output')(pooled)

            # 构建完整模型
            model = Model(inputs=inputs, outputs=outputs, name='Simplified_Transformer')

            # 编译模型
            optimizer = optimizers.Adam(
                learning_rate=self.config_params.get('learning_rate', 0.0001),
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
            raise ModelInitializationError("Simplified_Transformer", str(e))

    def _build_encoder(self, inputs):
        """构建Transformer编码器"""
        # 输入嵌入和位置编码
        x = layers.Dense(self.d_model, name='encoder_embedding')(inputs)
        # 简化实现：跳过缩放
        # x = layers.Lambda(lambda x: x * scale_factor)(x)

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
        """添加绝对位置编码（简化版本）"""
        # 简化实现：直接返回输入，跳过位置编码
        return x

    def _add_relative_position_encoding(self, x, name_prefix='encoder'):
        """添加相对位置编码（简化版本）"""
        # 简化实现：直接使用绝对位置编码
        return self._add_absolute_position_encoding(x, name_prefix)

    def _sparse_multi_head_attention(self, query, key, layer_idx, layer_type):
        """稀疏多头注意力（减少计算复杂度）"""
        # 简化的稀疏注意力实现
        # 在实际应用中，这里会实现更复杂的稀疏模式

        # 使用局部窗口注意力作为稀疏注意力的简化版本
        return self._local_multi_head_attention(query, key, layer_idx, layer_type)

    def _local_multi_head_attention(self, query, key, layer_idx, layer_type):
        """局部多头注意力（简化版本）"""
        # 简化实现：使用标准多头注意力
        attention_output = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate,
            name=f'{layer_type}_local_mha_{layer_idx}'
        )(query, key)

        return attention_output

    def _create_learning_rate_scheduler(self):
        """创建Transformer专用的学习率调度器"""
        # 简化实现：使用ReduceLROnPlateau
        return ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=0
        )

    def _get_advanced_callbacks(self):
        """获取高级回调函数，包含智能早停机制"""
        callbacks = create_intelligent_callbacks(
            patience=20,  # 连续20次相同结果时停止（按要求调整）
            min_delta=1e-6,
            monitor='val_loss',
            reduce_lr_patience=10
        )
        callbacks.append(self._create_learning_rate_scheduler())
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
        from ..data.data_manager import DeepLearningDataManager
        from .training_utils import get_callbacks, TrainingVisualizer
        
        if epochs is None:
            epochs = self.config_params.get('epochs', 100)

            # 智能训练轮数计算
            try:
                # 使用默认数据大小进行计算
                actual_data_size = 2755  # 默认历史数据大小
                dynamic_max_epochs = min(150, max(30, actual_data_size // 50))  # 使用智能早停，允许更多训练轮数
                epochs = min(epochs, dynamic_max_epochs)
                logger_manager.info(f"智能训练轮数调整: {epochs}")
            except Exception as e:
                logger_manager.warning(f"智能轮数计算失败，使用默认值: {e}")
                epochs = self.config_params.get('epochs', 100)

        if batch_size is None:
            batch_size = self.config_params.get('batch_size', 64)

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
            callbacks = get_callbacks(self.name, self.model_dir)
            
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
            visualizer.plot_history()

            # 保存scaler
            try:
                import joblib
                scaler_path = os.path.join(self.model_dir, f'{self.name}_scaler.pkl')
                joblib.dump(self.scaler, scaler_path)
                logger_manager.info(f"Scaler已保存到: {scaler_path}")
            except Exception as e:
                logger_manager.warning(f"保存scaler失败: {e}")

            logger_manager.info(f"{self.name}模型训练完成")

            return True
        
        except Exception as e:
            logger_manager.error(f"{self.name}模型训练失败: {e}")
            return False
    
    @handle_model_error
    def predict(self, data: pd.DataFrame = None, count=1, verbose=True) -> List[Tuple[List[int], List[int]]]:
        """
        生成预测结果

        Args:
            data: 历史数据（可选，如果不提供则使用内部数据）
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
        
        # 获取最近的序列数据 - 使用传入的数据或从数据管理器获取
        if data is not None:
            recent_data = data.tail(self.sequence_length)
        else:
            # 从数据管理器获取数据
            all_data = core_data_manager.get_data()
            recent_data = all_data.tail(self.sequence_length)

        # 提取特征 - 使用data_manager的parse_balls方法
        import core_modules as cm
        features = []

        for _, row in recent_data.iterrows():
            front_balls, back_balls = cm.data_manager.parse_balls(row)
            if len(front_balls) == 5 and len(back_balls) == 2:
                # 基础特征：前5个号码 + 后2个号码
                feature_vector = front_balls + back_balls

                # 扩展特征到所需维度
                while len(feature_vector) < self.feature_dim:
                    feature_vector.append(sum(feature_vector) / len(feature_vector))

                features.append(feature_vector[:self.feature_dim])

        if len(features) < self.sequence_length:
            logger_manager.warning(f"Transformer数据不足，需要{self.sequence_length}期，实际{len(features)}期")
            return []

        recent_features = np.array(features, dtype=np.float32)

        # 检查并训练scaler
        try:
            if not hasattr(self.scaler, 'scale_') or self.scaler.scale_ is None:
                logger_manager.info("训练Transformer scaler...")
                self.scaler.fit(recent_features)
            recent_scaled = self.scaler.transform(recent_features)
        except Exception as e:
            logger_manager.error(f"Transformer数据标准化失败: {e}")
            # 使用简单的归一化作为回退
            recent_scaled = recent_features / np.max(recent_features, axis=0, keepdims=True)
        
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

    def evaluate(self, data):
        """评估模型性能（公共接口）"""
        try:
            # 准备数据
            X_front, y_front, X_back, y_back = self.prepare_data(data)

            # 预测
            front_pred = self.front_model.predict(X_front, verbose=0)
            back_pred = self.back_model.predict(X_back, verbose=0)

            # 计算评估指标
            from sklearn.metrics import mean_squared_error, mean_absolute_error
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

            logger_manager.info(f"Transformer模型评估完成: {result}")

            return result

        except Exception as e:
            logger_manager.error(f"Transformer模型评估失败: {e}")
            return {'error': str(e)}

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
        self.config_params.update({
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
        self.config_params.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate
        })

    def predict_compound(self, config: Optional[CompoundConfig] = None) -> CompoundResult:
        """
        Transformer复式预测

        Args:
            config: 复式预测配置

        Returns:
            复式预测结果
        """
        if config is None:
            config = self.compound_config or CompoundConfig()

        # 验证参数
        if not self.validate_compound_params(config.front_count, config.back_count, config.max_cost):
            raise ValueError("Transformer复式预测参数验证失败")

        logger_manager.info(f"开始Transformer复式预测: {config.front_count}+{config.back_count}")

        try:
            # 生成多个候选预测
            candidate_count = max(config.front_count * 2, 15)
            candidates = []

            # 生成多样化的预测
            for i in range(candidate_count):
                predictions = self.predict(data=None, count=1)
                if predictions:
                    candidates.append(predictions[0])

            # 基于频率选择最优组合
            front_balls, back_balls = self._select_optimal_compound_simple(
                candidates, config.front_count, config.back_count
            )

            # 计算组合数和成本
            combinations = self.calculate_combinations(config.front_count, config.back_count)
            cost = self.calculate_cost(combinations)

            # 计算置信度
            confidence = min(0.8, max(0.3, len(candidates) / candidate_count))

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
                method="Transformer复式预测",
                analysis_periods=config.periods,
                timestamp=datetime.now().isoformat(),
                details={
                    'model_type': 'Transformer',
                    'candidate_count': len(candidates),
                    'selection_strategy': 'frequency_based'
                }
            )

            logger_manager.info(f"Transformer复式预测完成: {config.front_count}+{config.back_count}, 置信度: {confidence:.3f}")
            return result

        except Exception as e:
            logger_manager.error(f"Transformer复式预测失败: {e}")
            # 返回默认结果
            return super().predict_compound(config)

    def _select_optimal_compound_simple(self, candidates: List[Tuple[List[int], List[int]]],
                                      front_count: int, back_count: int) -> Tuple[List[int], List[int]]:
        """简单的复式组合选择"""
        if not candidates:
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

    def _load_model(self) -> bool:
        """加载已训练的模型"""
        try:
            import os
            model_path = os.path.join(self.model_dir, f'{self.name}_best.h5')
            scaler_path = os.path.join(self.model_dir, f'{self.name}_scaler.pkl')

            if os.path.exists(model_path):
                from tensorflow.keras.models import load_model
                self.model = load_model(model_path)

                # 加载scaler
                try:
                    import joblib
                    if os.path.exists(scaler_path):
                        self.scaler = joblib.load(scaler_path)
                        logger_manager.info(f"Scaler已从 {scaler_path} 加载")
                    else:
                        logger_manager.warning("Scaler文件不存在，将使用新的scaler")
                        from sklearn.preprocessing import StandardScaler
                        self.scaler = StandardScaler()
                except Exception as e:
                    logger_manager.warning(f"加载scaler失败: {e}，将使用新的scaler")
                    from sklearn.preprocessing import StandardScaler
                    self.scaler = StandardScaler()

                self.is_trained = True
                logger_manager.info("Transformer模型加载成功")
                return True
            else:
                logger_manager.warning("Transformer模型文件不存在")
                return False

        except Exception as e:
            logger_manager.error(f"Transformer模型加载失败: {e}")
            return False

    def _save_model(self, filepath=None):
        """内部保存模型方法"""
        try:
            if filepath is None:
                filepath = os.path.join(self.model_dir, f'{self.name}_best.h5')

            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # 保存模型
            if self.model is not None:
                self.model.save(filepath)
                logger_manager.info(f"Transformer模型已保存到: {filepath}")
                return True
            else:
                logger_manager.warning("没有可保存的模型")
                return False

        except Exception as e:
            logger_manager.error(f"保存Transformer模型失败: {e}")
            return False

    def save_model(self, filepath=None):
        """保存模型"""
        return self._save_model(filepath)


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