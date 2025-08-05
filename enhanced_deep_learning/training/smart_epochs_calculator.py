#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
智能训练轮数计算器
Smart Epochs Calculator

基于硬件性能、数据量和模型复杂度智能计算最优训练轮数。
"""

import math
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from core_modules import logger_manager
from ..performance.enhanced_hardware_accelerator import EnhancedHardwareAccelerator, PerformanceBenchmark


class ModelType(Enum):
    """模型类型枚举"""
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    GAN = "gan"
    TRADITIONAL_ML = "traditional_ml"
    CLUSTERING = "clustering"
    MARKOV_CHAIN = "markov_chain"
    BAYESIAN = "bayesian"


class PerformanceMode(Enum):
    """性能模式枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class TrainingConfig:
    """训练配置"""
    model_type: ModelType
    data_size: int  # 数据量（样本数）
    feature_dim: int = 7  # 特征维度
    performance_mode: PerformanceMode = PerformanceMode.MEDIUM
    min_epochs: int = 10
    max_epochs: int = 1000
    target_accuracy: float = 0.8
    early_stopping: bool = True
    patience: int = 10


@dataclass
class EpochsRecommendation:
    """训练轮数推荐结果"""
    recommended_epochs: int
    min_epochs: int
    max_epochs: int
    estimated_time: float  # 预估训练时间（分钟）
    confidence: float
    reasoning: str
    hardware_factor: float
    data_factor: float
    complexity_factor: float


class SmartEpochsCalculator:
    """智能训练轮数计算器"""
    
    def __init__(self):
        """初始化智能训练轮数计算器"""
        self.hardware_accelerator = EnhancedHardwareAccelerator()
        self.benchmark_result = None
        
        # 基础训练轮数比例
        self.base_ratios = {
            ModelType.LSTM: 0.6,
            ModelType.TRANSFORMER: 0.5,
            ModelType.GAN: 0.7,
            ModelType.TRADITIONAL_ML: 0.3,
            ModelType.CLUSTERING: 0.2,
            ModelType.MARKOV_CHAIN: 0.1,
            ModelType.BAYESIAN: 0.25
        }
        
        # 性能模式倍数
        self.performance_multipliers = {
            PerformanceMode.LOW: 0.5,
            PerformanceMode.MEDIUM: 1.0,
            PerformanceMode.HIGH: 1.5
        }
        
        logger_manager.info("智能训练轮数计算器初始化完成")
    
    def calculate_optimal_epochs(self, config: TrainingConfig) -> EpochsRecommendation:
        """
        计算最优训练轮数
        
        Args:
            config: 训练配置
            
        Returns:
            训练轮数推荐结果
        """
        logger_manager.info(f"计算 {config.model_type.value} 模型的最优训练轮数")
        
        # 获取硬件基准测试结果
        if self.benchmark_result is None:
            self.benchmark_result = self.hardware_accelerator.benchmark_hardware()
        
        # 计算各种因子
        hardware_factor = self._calculate_hardware_factor()
        data_factor = self._calculate_data_factor(config.data_size, config.model_type)
        complexity_factor = self._calculate_complexity_factor(config.model_type, config.feature_dim)
        
        # 基础轮数计算
        base_epochs = self._calculate_base_epochs(config.data_size, config.model_type)
        
        # 应用各种因子
        adjusted_epochs = base_epochs * hardware_factor * data_factor * complexity_factor
        
        # 应用性能模式倍数
        performance_multiplier = self.performance_multipliers[config.performance_mode]
        final_epochs = int(adjusted_epochs * performance_multiplier)
        
        # 应用边界限制
        final_epochs = max(config.min_epochs, min(config.max_epochs, final_epochs))
        
        # 计算动态边界
        dynamic_min = max(config.min_epochs, int(final_epochs * 0.3))
        dynamic_max = min(config.max_epochs, int(final_epochs * 2.0))
        
        # 估算训练时间
        estimated_time = self._estimate_training_time(final_epochs, config)
        
        # 计算置信度
        confidence = self._calculate_confidence(config, hardware_factor, data_factor)
        
        # 生成推理说明
        reasoning = self._generate_reasoning(config, hardware_factor, data_factor, complexity_factor)
        
        recommendation = EpochsRecommendation(
            recommended_epochs=final_epochs,
            min_epochs=dynamic_min,
            max_epochs=dynamic_max,
            estimated_time=estimated_time,
            confidence=confidence,
            reasoning=reasoning,
            hardware_factor=hardware_factor,
            data_factor=data_factor,
            complexity_factor=complexity_factor
        )
        
        logger_manager.info(f"推荐训练轮数: {final_epochs}, 预估时间: {estimated_time:.1f}分钟, 置信度: {confidence:.2f}")
        
        return recommendation
    
    def _calculate_hardware_factor(self) -> float:
        """计算硬件因子"""
        # 基于硬件性能调整训练轮数
        overall_score = self.benchmark_result.overall_score
        
        if overall_score > 75:
            # 高性能硬件：可以进行更多轮训练
            return 1.5
        elif overall_score > 50:
            # 中等性能硬件：标准训练轮数
            return 1.0
        elif overall_score > 25:
            # 低性能硬件：减少训练轮数
            return 0.7
        else:
            # 极低性能硬件：大幅减少训练轮数
            return 0.5
    
    def _calculate_data_factor(self, data_size: int, model_type: ModelType) -> float:
        """计算数据因子"""
        # 基于数据量调整训练轮数
        if model_type in [ModelType.LSTM, ModelType.TRANSFORMER, ModelType.GAN]:
            # 深度学习模型需要更多数据
            if data_size < 500:
                return 0.8  # 数据少，减少训练轮数防止过拟合
            elif data_size < 1000:
                return 1.0  # 标准数据量
            elif data_size < 2000:
                return 1.2  # 较多数据，可以增加训练轮数
            else:
                return 1.4  # 大量数据，显著增加训练轮数
        else:
            # 传统机器学习模型
            if data_size < 200:
                return 0.6
            elif data_size < 500:
                return 0.8
            elif data_size < 1000:
                return 1.0
            else:
                return 1.2
    
    def _calculate_complexity_factor(self, model_type: ModelType, feature_dim: int) -> float:
        """计算复杂度因子"""
        # 基于模型复杂度调整训练轮数
        base_complexity = {
            ModelType.LSTM: 1.2,
            ModelType.TRANSFORMER: 1.4,
            ModelType.GAN: 1.5,
            ModelType.TRADITIONAL_ML: 0.8,
            ModelType.CLUSTERING: 0.6,
            ModelType.MARKOV_CHAIN: 0.5,
            ModelType.BAYESIAN: 0.7
        }
        
        complexity = base_complexity.get(model_type, 1.0)
        
        # 特征维度调整
        if feature_dim > 20:
            complexity *= 1.2
        elif feature_dim > 10:
            complexity *= 1.1
        
        return complexity
    
    def _calculate_base_epochs(self, data_size: int, model_type: ModelType) -> int:
        """计算基础训练轮数"""
        base_ratio = self.base_ratios.get(model_type, 0.5)
        base_epochs = int(data_size * base_ratio)
        
        # 模型特定的最小轮数
        min_epochs_by_type = {
            ModelType.LSTM: 50,
            ModelType.TRANSFORMER: 30,
            ModelType.GAN: 100,
            ModelType.TRADITIONAL_ML: 10,
            ModelType.CLUSTERING: 5,
            ModelType.MARKOV_CHAIN: 3,
            ModelType.BAYESIAN: 10
        }
        
        min_epochs = min_epochs_by_type.get(model_type, 10)
        return max(min_epochs, base_epochs)
    
    def _estimate_training_time(self, epochs: int, config: TrainingConfig) -> float:
        """估算训练时间（分钟）"""
        # 基于模型类型和硬件性能估算每轮训练时间
        time_per_epoch = {
            ModelType.LSTM: 0.5,
            ModelType.TRANSFORMER: 1.0,
            ModelType.GAN: 1.5,
            ModelType.TRADITIONAL_ML: 0.1,
            ModelType.CLUSTERING: 0.05,
            ModelType.MARKOV_CHAIN: 0.02,
            ModelType.BAYESIAN: 0.08
        }
        
        base_time = time_per_epoch.get(config.model_type, 0.5)
        
        # 硬件性能调整
        hardware_multiplier = 1.0
        if self.benchmark_result.overall_score > 75:
            hardware_multiplier = 0.5  # 高性能硬件训练更快
        elif self.benchmark_result.overall_score < 25:
            hardware_multiplier = 2.0  # 低性能硬件训练更慢
        
        # 数据量调整
        data_multiplier = math.log10(max(100, config.data_size)) / 2.0
        
        total_time = epochs * base_time * hardware_multiplier * data_multiplier
        return max(0.1, total_time)
    
    def _calculate_confidence(self, config: TrainingConfig, hardware_factor: float, data_factor: float) -> float:
        """计算推荐置信度"""
        # 基于各种因素计算置信度
        confidence = 0.8  # 基础置信度
        
        # 硬件因子影响
        if 0.8 <= hardware_factor <= 1.2:
            confidence += 0.1  # 硬件适中，置信度高
        else:
            confidence -= 0.1  # 硬件极端，置信度低
        
        # 数据因子影响
        if 0.9 <= data_factor <= 1.3:
            confidence += 0.1  # 数据量适中，置信度高
        else:
            confidence -= 0.1  # 数据量极端，置信度低
        
        # 模型类型影响
        if config.model_type in [ModelType.LSTM, ModelType.TRANSFORMER]:
            confidence += 0.05  # 成熟的深度学习模型
        elif config.model_type == ModelType.GAN:
            confidence -= 0.05  # GAN训练不稳定
        
        return max(0.1, min(0.95, confidence))
    
    def _generate_reasoning(self, config: TrainingConfig, hardware_factor: float, 
                          data_factor: float, complexity_factor: float) -> str:
        """生成推理说明"""
        reasons = []
        
        # 硬件因素
        if hardware_factor > 1.2:
            reasons.append("高性能硬件支持更多训练轮数")
        elif hardware_factor < 0.8:
            reasons.append("硬件性能限制，减少训练轮数")
        
        # 数据因素
        if data_factor > 1.2:
            reasons.append(f"大数据量({config.data_size}样本)支持充分训练")
        elif data_factor < 0.8:
            reasons.append(f"小数据量({config.data_size}样本)防止过拟合")
        
        # 模型复杂度
        if complexity_factor > 1.3:
            reasons.append(f"{config.model_type.value}模型复杂度高，需要更多训练")
        elif complexity_factor < 0.7:
            reasons.append(f"{config.model_type.value}模型相对简单")
        
        # 性能模式
        if config.performance_mode == PerformanceMode.HIGH:
            reasons.append("高性能模式，追求最佳效果")
        elif config.performance_mode == PerformanceMode.LOW:
            reasons.append("低性能模式，平衡效率与效果")
        
        return "; ".join(reasons) if reasons else "基于标准配置推荐"
    
    def get_adaptive_early_stopping_config(self, config: TrainingConfig) -> Dict[str, Any]:
        """获取自适应早停配置"""
        base_patience = {
            ModelType.LSTM: 15,
            ModelType.TRANSFORMER: 10,
            ModelType.GAN: 20,
            ModelType.TRADITIONAL_ML: 5,
            ModelType.CLUSTERING: 3,
            ModelType.MARKOV_CHAIN: 2,
            ModelType.BAYESIAN: 5
        }
        
        patience = base_patience.get(config.model_type, 10)
        
        # 基于数据量调整耐心值
        if config.data_size > 1000:
            patience = int(patience * 1.5)
        elif config.data_size < 300:
            patience = max(3, int(patience * 0.7))
        
        return {
            'monitor': 'val_loss',
            'patience': patience,
            'min_delta': 0.001,
            'restore_best_weights': True,
            'verbose': 1
        }


class TrainingMonitor:
    """训练监控器"""

    def __init__(self):
        """初始化训练监控器"""
        self.training_history = {}
        self.performance_metrics = {}
        self.adjustment_suggestions = []

        logger_manager.info("训练监控器初始化完成")

    def monitor_training_progress(self, model_name: str, epoch: int, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        监控训练进度

        Args:
            model_name: 模型名称
            epoch: 当前轮数
            metrics: 训练指标

        Returns:
            监控结果和建议
        """
        if model_name not in self.training_history:
            self.training_history[model_name] = []

        # 记录训练历史
        self.training_history[model_name].append({
            'epoch': epoch,
            'metrics': metrics.copy(),
            'timestamp': time.time()
        })

        # 分析训练趋势
        analysis = self._analyze_training_trend(model_name)

        # 检测异常
        anomalies = self._detect_training_anomalies(model_name, metrics)

        # 生成调整建议
        suggestions = self._generate_adjustment_suggestions(model_name, analysis, anomalies)

        result = {
            'trend_analysis': analysis,
            'anomalies': anomalies,
            'suggestions': suggestions,
            'should_continue': not any(anomaly['severity'] == 'critical' for anomaly in anomalies)
        }

        logger_manager.debug(f"训练监控 - {model_name} 第{epoch}轮: {len(anomalies)}个异常, {len(suggestions)}个建议")

        return result

    def _analyze_training_trend(self, model_name: str) -> Dict[str, Any]:
        """分析训练趋势"""
        history = self.training_history[model_name]
        if len(history) < 3:
            return {'status': 'insufficient_data'}

        # 获取最近几轮的损失值
        recent_losses = [h['metrics'].get('loss', float('inf')) for h in history[-5:]]

        # 计算趋势
        if len(recent_losses) >= 3:
            trend = 'improving' if recent_losses[-1] < recent_losses[0] else 'degrading'
            improvement_rate = (recent_losses[0] - recent_losses[-1]) / recent_losses[0] if recent_losses[0] > 0 else 0
        else:
            trend = 'unknown'
            improvement_rate = 0

        # 检测收敛
        if len(recent_losses) >= 3:
            variance = np.var(recent_losses)
            is_converging = variance < 0.001  # 损失变化很小
        else:
            is_converging = False

        return {
            'status': 'analyzed',
            'trend': trend,
            'improvement_rate': improvement_rate,
            'is_converging': is_converging,
            'recent_losses': recent_losses
        }

    def _detect_training_anomalies(self, model_name: str, current_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """检测训练异常"""
        anomalies = []
        history = self.training_history[model_name]

        if len(history) < 2:
            return anomalies

        # 检测损失爆炸
        current_loss = current_metrics.get('loss', 0)
        if len(history) >= 2:
            prev_loss = history[-2]['metrics'].get('loss', 0)
            if current_loss > prev_loss * 2 and current_loss > 1.0:
                anomalies.append({
                    'type': 'loss_explosion',
                    'severity': 'critical',
                    'description': f"损失爆炸: {prev_loss:.4f} -> {current_loss:.4f}",
                    'suggestion': "降低学习率或检查数据"
                })

        # 检测梯度消失
        if 'gradient_norm' in current_metrics:
            grad_norm = current_metrics['gradient_norm']
            if grad_norm < 1e-6:
                anomalies.append({
                    'type': 'gradient_vanishing',
                    'severity': 'warning',
                    'description': f"梯度消失: 梯度范数 {grad_norm:.2e}",
                    'suggestion': "调整网络架构或初始化方法"
                })

        # 检测过拟合
        if 'val_loss' in current_metrics and 'loss' in current_metrics:
            train_loss = current_metrics['loss']
            val_loss = current_metrics['val_loss']
            if val_loss > train_loss * 1.5:
                anomalies.append({
                    'type': 'overfitting',
                    'severity': 'warning',
                    'description': f"过拟合迹象: 训练损失 {train_loss:.4f}, 验证损失 {val_loss:.4f}",
                    'suggestion': "增加正则化或减少模型复杂度"
                })

        return anomalies

    def _generate_adjustment_suggestions(self, model_name: str, analysis: Dict[str, Any],
                                       anomalies: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """生成调整建议"""
        suggestions = []

        # 基于趋势分析的建议
        if analysis.get('trend') == 'degrading':
            suggestions.append({
                'type': 'learning_rate',
                'action': 'decrease',
                'description': "训练趋势恶化，建议降低学习率"
            })

        if analysis.get('is_converging') and analysis.get('improvement_rate', 0) < 0.01:
            suggestions.append({
                'type': 'early_stopping',
                'action': 'consider',
                'description': "模型已收敛，可考虑提前停止"
            })

        # 基于异常的建议
        for anomaly in anomalies:
            if anomaly['type'] == 'loss_explosion':
                suggestions.append({
                    'type': 'learning_rate',
                    'action': 'decrease_significantly',
                    'description': "损失爆炸，立即大幅降低学习率"
                })
            elif anomaly['type'] == 'overfitting':
                suggestions.append({
                    'type': 'regularization',
                    'action': 'increase',
                    'description': "增加Dropout或L2正则化"
                })

        return suggestions


class AdaptiveTrainingAdjuster:
    """自适应训练调整器"""

    def __init__(self):
        """初始化自适应训练调整器"""
        self.adjustment_history = {}
        self.performance_tracker = {}

        logger_manager.info("自适应训练调整器初始化完成")

    def adjust_training_params(self, model_name: str, current_epoch: int,
                             monitor_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        自适应调整训练参数

        Args:
            model_name: 模型名称
            current_epoch: 当前轮数
            monitor_result: 监控结果

        Returns:
            调整后的参数
        """
        adjustments = {}

        # 处理建议
        for suggestion in monitor_result.get('suggestions', []):
            if suggestion['type'] == 'learning_rate':
                adjustments['learning_rate'] = self._adjust_learning_rate(
                    model_name, suggestion['action']
                )
            elif suggestion['type'] == 'regularization':
                adjustments['dropout_rate'] = self._adjust_dropout(
                    model_name, suggestion['action']
                )

        # 处理严重异常
        critical_anomalies = [a for a in monitor_result.get('anomalies', [])
                            if a['severity'] == 'critical']
        if critical_anomalies:
            adjustments['emergency_stop'] = True
            adjustments['reason'] = critical_anomalies[0]['description']

        # 记录调整历史
        if model_name not in self.adjustment_history:
            self.adjustment_history[model_name] = []

        self.adjustment_history[model_name].append({
            'epoch': current_epoch,
            'adjustments': adjustments.copy(),
            'reason': monitor_result.get('suggestions', [])
        })

        logger_manager.info(f"自适应调整 - {model_name}: {adjustments}")

        return adjustments

    def _adjust_learning_rate(self, model_name: str, action: str) -> float:
        """调整学习率"""
        # 获取当前学习率（这里简化处理）
        current_lr = 0.001  # 默认学习率

        if action == 'decrease':
            new_lr = current_lr * 0.8
        elif action == 'decrease_significantly':
            new_lr = current_lr * 0.5
        elif action == 'increase':
            new_lr = current_lr * 1.2
        else:
            new_lr = current_lr

        return max(1e-6, min(0.1, new_lr))

    def _adjust_dropout(self, model_name: str, action: str) -> float:
        """调整Dropout率"""
        current_dropout = 0.2  # 默认Dropout率

        if action == 'increase':
            new_dropout = min(0.8, current_dropout + 0.1)
        elif action == 'decrease':
            new_dropout = max(0.0, current_dropout - 0.1)
        else:
            new_dropout = current_dropout

        return new_dropout
