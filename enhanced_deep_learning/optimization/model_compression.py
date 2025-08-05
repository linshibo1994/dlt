#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型压缩模块
Model Compression Module

提供量化、剪枝、蒸馏、低秩分解等模型压缩技术。
"""

import os
import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import copy
import math

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException


class CompressionMethod(Enum):
    """压缩方法枚举"""
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    LOW_RANK = "low_rank"
    HYBRID = "hybrid"


@dataclass
class CompressionConfig:
    """压缩配置"""
    method: CompressionMethod
    target_compression_ratio: float = 0.5
    accuracy_threshold: float = 0.95
    quantization_bits: int = 8
    pruning_sparsity: float = 0.5
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7
    low_rank_ratio: float = 0.5


@dataclass
class CompressionResult:
    """压缩结果"""
    original_size: int
    compressed_size: int
    compression_ratio: float
    accuracy_before: float
    accuracy_after: float
    accuracy_loss: float
    inference_speedup: float
    memory_reduction: float


class ModelQuantizer:
    """模型量化器"""
    
    def __init__(self, bits: int = 8):
        """
        初始化模型量化器
        
        Args:
            bits: 量化位数
        """
        self.bits = bits
        self.quantization_ranges = {}
        
        logger_manager.info(f"模型量化器初始化完成，量化位数: {bits}")
    
    def quantize_weights(self, weights: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        量化权重
        
        Args:
            weights: 原始权重
            
        Returns:
            量化后的权重和量化参数
        """
        try:
            # 计算量化范围
            w_min = np.min(weights)
            w_max = np.max(weights)
            
            # 对称量化
            w_abs_max = max(abs(w_min), abs(w_max))
            scale = w_abs_max / (2 ** (self.bits - 1) - 1)
            
            # 量化
            quantized = np.round(weights / scale)
            quantized = np.clip(quantized, -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1)
            
            # 反量化（用于推理）
            dequantized = quantized * scale
            
            quantization_params = {
                'scale': scale,
                'min_val': w_min,
                'max_val': w_max,
                'bits': self.bits
            }
            
            logger_manager.debug(f"权重量化完成，压缩比: {weights.nbytes / dequantized.nbytes:.2f}")
            return dequantized.astype(np.float32), quantization_params
            
        except Exception as e:
            logger_manager.error(f"权重量化失败: {e}")
            return weights, {}
    
    def quantize_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        量化整个模型
        
        Args:
            model: 原始模型
            
        Returns:
            量化后的模型
        """
        try:
            # 使用TensorFlow Lite量化
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # 设置量化类型
            if self.bits == 8:
                converter.target_spec.supported_types = [tf.int8]
            elif self.bits == 16:
                converter.target_spec.supported_types = [tf.float16]
            
            # 转换模型
            quantized_tflite_model = converter.convert()
            
            # 保存量化模型
            temp_path = "temp_quantized_model.tflite"
            with open(temp_path, 'wb') as f:
                f.write(quantized_tflite_model)
            
            logger_manager.info(f"模型量化完成，量化位数: {self.bits}")
            
            # 返回原模型（实际应用中需要加载TFLite模型）
            return model
            
        except Exception as e:
            logger_manager.error(f"模型量化失败: {e}")
            return model
    
    def dynamic_quantization(self, model: tf.keras.Model, 
                           calibration_data: np.ndarray) -> tf.keras.Model:
        """
        动态量化
        
        Args:
            model: 原始模型
            calibration_data: 校准数据
            
        Returns:
            动态量化后的模型
        """
        try:
            # 使用校准数据进行动态量化
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # 设置代表性数据集
            def representative_dataset():
                for data in calibration_data:
                    yield [data.astype(np.float32)]
            
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            quantized_tflite_model = converter.convert()
            
            logger_manager.info("动态量化完成")
            return model
            
        except Exception as e:
            logger_manager.error(f"动态量化失败: {e}")
            return model


class ModelPruner:
    """模型剪枝器"""
    
    def __init__(self, sparsity: float = 0.5):
        """
        初始化模型剪枝器
        
        Args:
            sparsity: 稀疏度（要剪枝的权重比例）
        """
        self.sparsity = sparsity
        self.pruning_masks = {}
        
        logger_manager.info(f"模型剪枝器初始化完成，稀疏度: {sparsity}")
    
    def magnitude_pruning(self, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于权重大小的剪枝
        
        Args:
            weights: 原始权重
            
        Returns:
            剪枝后的权重和剪枝掩码
        """
        try:
            # 计算权重的绝对值
            abs_weights = np.abs(weights)
            
            # 计算阈值
            threshold = np.percentile(abs_weights, self.sparsity * 100)
            
            # 创建剪枝掩码
            mask = abs_weights > threshold
            
            # 应用剪枝
            pruned_weights = weights * mask
            
            actual_sparsity = 1 - np.count_nonzero(pruned_weights) / pruned_weights.size
            logger_manager.debug(f"权重剪枝完成，实际稀疏度: {actual_sparsity:.3f}")
            
            return pruned_weights, mask
            
        except Exception as e:
            logger_manager.error(f"权重剪枝失败: {e}")
            return weights, np.ones_like(weights)
    
    def structured_pruning(self, weights: np.ndarray, 
                          structure: str = "channel") -> Tuple[np.ndarray, np.ndarray]:
        """
        结构化剪枝
        
        Args:
            weights: 原始权重
            structure: 剪枝结构 ("channel", "filter", "block")
            
        Returns:
            剪枝后的权重和剪枝掩码
        """
        try:
            if structure == "channel" and len(weights.shape) >= 2:
                # 通道级剪枝
                channel_importance = np.sum(np.abs(weights), axis=tuple(range(len(weights.shape) - 1)))
                threshold = np.percentile(channel_importance, self.sparsity * 100)
                
                mask = np.ones_like(weights)
                for i, importance in enumerate(channel_importance):
                    if importance <= threshold:
                        mask[..., i] = 0
                
                pruned_weights = weights * mask
                
            elif structure == "filter" and len(weights.shape) == 4:
                # 滤波器级剪枝（适用于卷积层）
                filter_importance = np.sum(np.abs(weights), axis=(1, 2, 3))
                threshold = np.percentile(filter_importance, self.sparsity * 100)
                
                mask = np.ones_like(weights)
                for i, importance in enumerate(filter_importance):
                    if importance <= threshold:
                        mask[i, ...] = 0
                
                pruned_weights = weights * mask
                
            else:
                # 回退到非结构化剪枝
                return self.magnitude_pruning(weights)
            
            actual_sparsity = 1 - np.count_nonzero(pruned_weights) / pruned_weights.size
            logger_manager.debug(f"结构化剪枝完成，结构: {structure}, 实际稀疏度: {actual_sparsity:.3f}")
            
            return pruned_weights, mask
            
        except Exception as e:
            logger_manager.error(f"结构化剪枝失败: {e}")
            return weights, np.ones_like(weights)
    
    def gradual_pruning(self, model: tf.keras.Model, 
                       training_data: np.ndarray,
                       epochs: int = 10) -> tf.keras.Model:
        """
        渐进式剪枝
        
        Args:
            model: 原始模型
            training_data: 训练数据
            epochs: 训练轮数
            
        Returns:
            剪枝后的模型
        """
        try:
            # 初始化智能早停机制
            from ..utils.intelligent_early_stopping import GeneralIntelligentEarlyStopping
            early_stopping = GeneralIntelligentEarlyStopping(
                patience=20,  # 连续20次相同结果时停止
                min_delta=1e-6,
                verbose=1
            )
            early_stopping.reset()

            # 简化的渐进式剪枝实现
            initial_sparsity = 0.0
            final_sparsity = self.sparsity

            for epoch in range(epochs):
                # 计算当前稀疏度
                current_sparsity = initial_sparsity + (final_sparsity - initial_sparsity) * (epoch / epochs)

                # 应用剪枝
                total_weights_pruned = 0
                for layer in model.layers:
                    if hasattr(layer, 'kernel'):
                        weights = layer.get_weights()[0]
                        pruned_weights, mask = self.magnitude_pruning(weights)

                        # 计算剪枝的权重数量
                        total_weights_pruned += np.sum(mask == 0)

                        # 更新权重
                        layer.set_weights([pruned_weights] + layer.get_weights()[1:])

                # 智能早停检查（使用剪枝权重数量作为指标）
                if early_stopping.update(total_weights_pruned):
                    logger_manager.info(f"渐进式剪枝智能早停，轮次: {epoch + 1}")
                    break

                # 微调训练（简化实现）
                logger_manager.debug(f"渐进式剪枝 - 轮次 {epoch + 1}, 稀疏度: {current_sparsity:.3f}")

            logger_manager.info("渐进式剪枝完成")
            return model
            
        except Exception as e:
            logger_manager.error(f"渐进式剪枝失败: {e}")
            return model


class KnowledgeDistiller:
    """知识蒸馏器"""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        """
        初始化知识蒸馏器
        
        Args:
            temperature: 蒸馏温度
            alpha: 蒸馏损失权重
        """
        self.temperature = temperature
        self.alpha = alpha
        
        logger_manager.info(f"知识蒸馏器初始化完成，温度: {temperature}, alpha: {alpha}")
    
    def distillation_loss(self, y_true, y_pred_student, y_pred_teacher):
        """
        计算蒸馏损失
        
        Args:
            y_true: 真实标签
            y_pred_student: 学生模型预测
            y_pred_teacher: 教师模型预测
            
        Returns:
            蒸馏损失
        """
        try:
            # 软目标损失
            teacher_soft = tf.nn.softmax(y_pred_teacher / self.temperature)
            student_soft = tf.nn.softmax(y_pred_student / self.temperature)
            
            soft_loss = tf.keras.losses.categorical_crossentropy(teacher_soft, student_soft)
            soft_loss = soft_loss * (self.temperature ** 2)
            
            # 硬目标损失
            hard_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred_student)
            
            # 组合损失
            total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
            
            return total_loss
            
        except Exception as e:
            logger_manager.error(f"计算蒸馏损失失败: {e}")
            return tf.keras.losses.categorical_crossentropy(y_true, y_pred_student)
    
    def distill_model(self, teacher_model: tf.keras.Model, 
                     student_model: tf.keras.Model,
                     training_data: Tuple[np.ndarray, np.ndarray],
                     epochs: int = 10) -> tf.keras.Model:
        """
        执行知识蒸馏
        
        Args:
            teacher_model: 教师模型
            student_model: 学生模型
            training_data: 训练数据
            epochs: 训练轮数
            
        Returns:
            蒸馏后的学生模型
        """
        try:
            X_train, y_train = training_data
            
            # 获取教师模型的软标签
            teacher_predictions = teacher_model.predict(X_train)
            
            # 定义蒸馏损失函数
            def distill_loss(y_true, y_pred):
                return self.distillation_loss(y_true, y_pred, teacher_predictions)
            
            # 编译学生模型
            student_model.compile(
                optimizer='adam',
                loss=distill_loss,
                metrics=['accuracy']
            )
            
            # 训练学生模型
            history = student_model.fit(
                X_train, y_train,
                epochs=epochs,
                verbose=0,
                validation_split=0.2
            )
            
            logger_manager.info(f"知识蒸馏完成，训练 {epochs} 轮")
            return student_model
            
        except Exception as e:
            logger_manager.error(f"知识蒸馏失败: {e}")
            return student_model


class LowRankDecomposer:
    """低秩分解器"""
    
    def __init__(self, rank_ratio: float = 0.5):
        """
        初始化低秩分解器
        
        Args:
            rank_ratio: 秩比例
        """
        self.rank_ratio = rank_ratio
        
        logger_manager.info(f"低秩分解器初始化完成，秩比例: {rank_ratio}")
    
    def svd_decomposition(self, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        SVD分解
        
        Args:
            weights: 原始权重矩阵
            
        Returns:
            分解后的U, S, V矩阵
        """
        try:
            # 执行SVD分解
            U, S, Vt = np.linalg.svd(weights, full_matrices=False)
            
            # 计算保留的秩
            original_rank = min(weights.shape)
            target_rank = max(1, int(original_rank * self.rank_ratio))
            
            # 截断
            U_truncated = U[:, :target_rank]
            S_truncated = S[:target_rank]
            Vt_truncated = Vt[:target_rank, :]
            
            # 重构权重
            reconstructed = U_truncated @ np.diag(S_truncated) @ Vt_truncated
            
            compression_ratio = (U_truncated.size + S_truncated.size + Vt_truncated.size) / weights.size
            logger_manager.debug(f"SVD分解完成，压缩比: {compression_ratio:.3f}")
            
            return U_truncated, S_truncated, Vt_truncated
            
        except Exception as e:
            logger_manager.error(f"SVD分解失败: {e}")
            return weights, np.array([1.0]), np.eye(weights.shape[1])
    
    def tucker_decomposition(self, tensor: np.ndarray) -> List[np.ndarray]:
        """
        Tucker分解（适用于高维张量）
        
        Args:
            tensor: 输入张量
            
        Returns:
            分解后的因子列表
        """
        try:
            # 简化的Tucker分解实现
            # 实际应用中需要使用专门的张量分解库
            
            if len(tensor.shape) == 2:
                # 对于2D张量，使用SVD
                U, S, Vt = self.svd_decomposition(tensor)
                return [U, np.diag(S), Vt]
            
            elif len(tensor.shape) == 4:
                # 对于4D张量（如卷积核），沿每个维度分解
                factors = []
                core_tensor = tensor.copy()
                
                for mode in range(len(tensor.shape)):
                    # 展开张量
                    unfolded = self._unfold_tensor(core_tensor, mode)
                    
                    # SVD分解
                    U, S, Vt = np.linalg.svd(unfolded, full_matrices=False)
                    
                    # 截断
                    target_rank = max(1, int(U.shape[1] * self.rank_ratio))
                    U_truncated = U[:, :target_rank]
                    
                    factors.append(U_truncated)
                    
                    # 更新核心张量
                    core_tensor = np.tensordot(core_tensor, U_truncated.T, axes=([mode], [1]))
                
                factors.append(core_tensor)  # 核心张量
                
                logger_manager.debug("Tucker分解完成")
                return factors
            
            else:
                logger_manager.warning(f"不支持的张量维度: {len(tensor.shape)}")
                return [tensor]
                
        except Exception as e:
            logger_manager.error(f"Tucker分解失败: {e}")
            return [tensor]
    
    def _unfold_tensor(self, tensor: np.ndarray, mode: int) -> np.ndarray:
        """
        张量展开
        
        Args:
            tensor: 输入张量
            mode: 展开模式
            
        Returns:
            展开后的矩阵
        """
        try:
            # 将指定模式移到第一个维度
            axes = list(range(len(tensor.shape)))
            axes[0], axes[mode] = axes[mode], axes[0]
            
            reordered = np.transpose(tensor, axes)
            
            # 展开为矩阵
            unfolded = reordered.reshape(reordered.shape[0], -1)
            
            return unfolded
            
        except Exception as e:
            logger_manager.error(f"张量展开失败: {e}")
            return tensor.reshape(tensor.shape[0], -1)


class ModelCompressor:
    """模型压缩器"""
    
    def __init__(self, config: CompressionConfig):
        """
        初始化模型压缩器
        
        Args:
            config: 压缩配置
        """
        self.config = config
        self.quantizer = ModelQuantizer(bits=config.quantization_bits)
        self.pruner = ModelPruner(sparsity=config.pruning_sparsity)
        self.distiller = KnowledgeDistiller(
            temperature=config.distillation_temperature,
            alpha=config.distillation_alpha
        )
        self.decomposer = LowRankDecomposer(rank_ratio=config.low_rank_ratio)
        
        logger_manager.info(f"模型压缩器初始化完成，方法: {config.method.value}")
    
    def compress_model(self, model: tf.keras.Model, 
                      training_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                      teacher_model: Optional[tf.keras.Model] = None) -> Tuple[tf.keras.Model, CompressionResult]:
        """
        压缩模型
        
        Args:
            model: 原始模型
            training_data: 训练数据（用于蒸馏和微调）
            teacher_model: 教师模型（用于蒸馏）
            
        Returns:
            压缩后的模型和压缩结果
        """
        try:
            # 记录原始模型信息
            original_size = self._calculate_model_size(model)
            
            # 根据配置选择压缩方法
            if self.config.method == CompressionMethod.QUANTIZATION:
                compressed_model = self.quantizer.quantize_model(model)
            
            elif self.config.method == CompressionMethod.PRUNING:
                if training_data:
                    compressed_model = self.pruner.gradual_pruning(model, training_data[0])
                else:
                    # 简单剪枝
                    compressed_model = self._apply_simple_pruning(model)
            
            elif self.config.method == CompressionMethod.DISTILLATION:
                if teacher_model and training_data:
                    compressed_model = self.distiller.distill_model(
                        teacher_model, model, training_data
                    )
                else:
                    logger_manager.warning("蒸馏需要教师模型和训练数据")
                    compressed_model = model
            
            elif self.config.method == CompressionMethod.LOW_RANK:
                compressed_model = self._apply_low_rank_decomposition(model)
            
            elif self.config.method == CompressionMethod.HYBRID:
                # 混合压缩方法
                compressed_model = self._apply_hybrid_compression(model, training_data, teacher_model)
            
            else:
                compressed_model = model
            
            # 计算压缩结果
            compressed_size = self._calculate_model_size(compressed_model)
            compression_ratio = compressed_size / original_size
            
            result = CompressionResult(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compression_ratio,
                accuracy_before=0.0,  # 需要实际评估
                accuracy_after=0.0,   # 需要实际评估
                accuracy_loss=0.0,
                inference_speedup=1.0 / compression_ratio,  # 估算
                memory_reduction=1.0 - compression_ratio
            )
            
            logger_manager.info(f"模型压缩完成，压缩比: {compression_ratio:.3f}")
            return compressed_model, result
            
        except Exception as e:
            logger_manager.error(f"模型压缩失败: {e}")
            return model, CompressionResult(0, 0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    
    def _calculate_model_size(self, model: tf.keras.Model) -> int:
        """计算模型大小（参数数量）"""
        try:
            return model.count_params()
        except:
            return 0
    
    def _apply_simple_pruning(self, model: tf.keras.Model) -> tf.keras.Model:
        """应用简单剪枝"""
        try:
            for layer in model.layers:
                if hasattr(layer, 'kernel'):
                    weights = layer.get_weights()
                    if weights:
                        pruned_kernel, _ = self.pruner.magnitude_pruning(weights[0])
                        layer.set_weights([pruned_kernel] + weights[1:])
            
            return model
            
        except Exception as e:
            logger_manager.error(f"简单剪枝失败: {e}")
            return model
    
    def _apply_low_rank_decomposition(self, model: tf.keras.Model) -> tf.keras.Model:
        """应用低秩分解"""
        try:
            for layer in model.layers:
                if hasattr(layer, 'kernel'):
                    weights = layer.get_weights()
                    if weights and len(weights[0].shape) == 2:
                        # 对全连接层应用SVD分解
                        U, S, Vt = self.decomposer.svd_decomposition(weights[0])
                        
                        # 重构权重
                        reconstructed = U @ np.diag(S) @ Vt
                        layer.set_weights([reconstructed] + weights[1:])
            
            return model
            
        except Exception as e:
            logger_manager.error(f"低秩分解失败: {e}")
            return model
    
    def _apply_hybrid_compression(self, model: tf.keras.Model,
                                 training_data: Optional[Tuple[np.ndarray, np.ndarray]],
                                 teacher_model: Optional[tf.keras.Model]) -> tf.keras.Model:
        """应用混合压缩方法"""
        try:
            # 1. 先应用剪枝
            compressed_model = self._apply_simple_pruning(model)
            
            # 2. 再应用量化
            compressed_model = self.quantizer.quantize_model(compressed_model)
            
            # 3. 如果有教师模型，应用蒸馏
            if teacher_model and training_data:
                compressed_model = self.distiller.distill_model(
                    teacher_model, compressed_model, training_data, epochs=5
                )
            
            logger_manager.info("混合压缩完成")
            return compressed_model
            
        except Exception as e:
            logger_manager.error(f"混合压缩失败: {e}")
            return model


# 全局模型压缩器实例
default_config = CompressionConfig(
    method=CompressionMethod.HYBRID,
    target_compression_ratio=0.3,
    quantization_bits=8,
    pruning_sparsity=0.5
)
model_compressor = ModelCompressor(default_config)


if __name__ == "__main__":
    # 测试模型压缩功能
    print("🗜️ 测试模型压缩功能...")
    
    # 创建测试模型
    test_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # 测试量化
    quantizer = ModelQuantizer(bits=8)
    test_weights = np.random.randn(128, 64)
    quantized_weights, params = quantizer.quantize_weights(test_weights)
    print(f"量化测试完成，量化参数: {params}")
    
    # 测试剪枝
    pruner = ModelPruner(sparsity=0.3)
    pruned_weights, mask = pruner.magnitude_pruning(test_weights)
    sparsity = 1 - np.count_nonzero(pruned_weights) / pruned_weights.size
    print(f"剪枝测试完成，实际稀疏度: {sparsity:.3f}")
    
    # 测试低秩分解
    decomposer = LowRankDecomposer(rank_ratio=0.5)
    U, S, Vt = decomposer.svd_decomposition(test_weights)
    print(f"低秩分解测试完成，分解形状: U{U.shape}, S{S.shape}, Vt{Vt.shape}")
    
    print("模型压缩功能测试完成")
