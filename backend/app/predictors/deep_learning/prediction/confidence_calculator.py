#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
置信度计算器模块
Confidence Calculator Module

提供多种置信度计算方法和不确定性量化功能。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

from core_modules import logger_manager
from ..utils.exceptions import PredictionException


class ConfidenceMethod(Enum):
    """置信度计算方法枚举"""
    ENTROPY = "entropy"
    VARIANCE = "variance"
    ENSEMBLE_AGREEMENT = "ensemble_agreement"
    PREDICTION_INTERVAL = "prediction_interval"
    BAYESIAN = "bayesian"
    MONTE_CARLO = "monte_carlo"
    TEMPERATURE_SCALING = "temperature_scaling"


@dataclass
class ConfidenceScore:
    """置信度分数"""
    method: ConfidenceMethod
    score: float
    uncertainty: float
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseConfidenceCalculator(ABC):
    """置信度计算器基类"""
    
    def __init__(self, method: ConfidenceMethod):
        """
        初始化置信度计算器
        
        Args:
            method: 置信度计算方法
        """
        self.method = method
        
    @abstractmethod
    def calculate(self, predictions: np.ndarray, **kwargs) -> np.ndarray:
        """
        计算置信度
        
        Args:
            predictions: 预测结果
            **kwargs: 其他参数
            
        Returns:
            置信度分数数组
        """
        pass
    
    def validate_input(self, predictions: np.ndarray) -> bool:
        """验证输入数据"""
        try:
            if not isinstance(predictions, np.ndarray):
                raise PredictionException("预测结果必须是numpy数组")
            
            if len(predictions) == 0:
                raise PredictionException("预测结果不能为空")
            
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                raise PredictionException("预测结果包含无效值")
            
            return True
            
        except Exception as e:
            logger_manager.error(f"输入验证失败: {e}")
            return False


class EntropyConfidenceCalculator(BaseConfidenceCalculator):
    """基于熵的置信度计算器"""
    
    def __init__(self):
        """初始化熵置信度计算器"""
        super().__init__(ConfidenceMethod.ENTROPY)
        logger_manager.debug("熵置信度计算器初始化完成")
    
    def calculate(self, predictions: np.ndarray, **kwargs) -> np.ndarray:
        """
        基于熵计算置信度
        
        Args:
            predictions: 预测概率分布 (N, C) 其中N是样本数，C是类别数
            **kwargs: 其他参数
            
        Returns:
            置信度分数数组
        """
        try:
            if not self.validate_input(predictions):
                raise PredictionException("输入验证失败")
            
            # 确保是概率分布
            if predictions.ndim == 1:
                # 单一预测值，转换为二分类概率
                probs = np.column_stack([1 - predictions, predictions])
            else:
                probs = predictions
            
            # 归一化确保是概率分布
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            
            # 避免log(0)
            probs = np.clip(probs, 1e-10, 1.0)
            
            # 计算熵
            entropy = -np.sum(probs * np.log(probs), axis=1)
            
            # 归一化熵到[0,1]范围，熵越小置信度越高
            max_entropy = np.log(probs.shape[1])
            normalized_entropy = entropy / max_entropy
            
            # 置信度 = 1 - 归一化熵
            confidence = 1.0 - normalized_entropy
            
            logger_manager.debug(f"熵置信度计算完成，平均置信度: {np.mean(confidence):.4f}")
            return confidence
            
        except Exception as e:
            logger_manager.error(f"熵置信度计算失败: {e}")
            raise PredictionException(f"熵置信度计算失败: {e}")


class VarianceConfidenceCalculator(BaseConfidenceCalculator):
    """基于方差的置信度计算器"""
    
    def __init__(self):
        """初始化方差置信度计算器"""
        super().__init__(ConfidenceMethod.VARIANCE)
        logger_manager.debug("方差置信度计算器初始化完成")
    
    def calculate(self, predictions: np.ndarray, **kwargs) -> np.ndarray:
        """
        基于方差计算置信度
        
        Args:
            predictions: 预测结果 (N, M) 其中N是样本数，M是预测次数或特征数
            **kwargs: 其他参数
            
        Returns:
            置信度分数数组
        """
        try:
            if not self.validate_input(predictions):
                raise PredictionException("输入验证失败")
            
            if predictions.ndim == 1:
                # 单一预测，无法计算方差
                return np.ones(len(predictions))
            
            # 计算每个样本的方差
            variances = np.var(predictions, axis=1)
            
            # 归一化方差到[0,1]范围
            if np.max(variances) > 0:
                normalized_variance = variances / np.max(variances)
            else:
                normalized_variance = variances
            
            # 置信度 = 1 - 归一化方差
            confidence = 1.0 - normalized_variance
            
            logger_manager.debug(f"方差置信度计算完成，平均置信度: {np.mean(confidence):.4f}")
            return confidence
            
        except Exception as e:
            logger_manager.error(f"方差置信度计算失败: {e}")
            raise PredictionException(f"方差置信度计算失败: {e}")


class EnsembleAgreementCalculator(BaseConfidenceCalculator):
    """基于集成一致性的置信度计算器"""
    
    def __init__(self):
        """初始化集成一致性置信度计算器"""
        super().__init__(ConfidenceMethod.ENSEMBLE_AGREEMENT)
        logger_manager.debug("集成一致性置信度计算器初始化完成")
    
    def calculate(self, predictions: np.ndarray, **kwargs) -> np.ndarray:
        """
        基于集成模型一致性计算置信度
        
        Args:
            predictions: 集成预测结果 (N, M) 其中N是样本数，M是模型数
            **kwargs: 其他参数
            
        Returns:
            置信度分数数组
        """
        try:
            if not self.validate_input(predictions):
                raise PredictionException("输入验证失败")
            
            if predictions.ndim == 1:
                # 单一预测，无法计算一致性
                return np.ones(len(predictions))
            
            # 计算每个样本的标准差（不一致性）
            std_devs = np.std(predictions, axis=1)
            
            # 计算相对标准差
            means = np.mean(predictions, axis=1)
            relative_std = np.where(means != 0, std_devs / np.abs(means), std_devs)
            
            # 归一化到[0,1]范围
            if np.max(relative_std) > 0:
                normalized_std = relative_std / np.max(relative_std)
            else:
                normalized_std = relative_std
            
            # 置信度 = 1 - 归一化标准差
            confidence = 1.0 - normalized_std
            
            logger_manager.debug(f"集成一致性置信度计算完成，平均置信度: {np.mean(confidence):.4f}")
            return confidence
            
        except Exception as e:
            logger_manager.error(f"集成一致性置信度计算失败: {e}")
            raise PredictionException(f"集成一致性置信度计算失败: {e}")


class PredictionIntervalCalculator(BaseConfidenceCalculator):
    """基于预测区间的置信度计算器"""
    
    def __init__(self):
        """初始化预测区间置信度计算器"""
        super().__init__(ConfidenceMethod.PREDICTION_INTERVAL)
        logger_manager.debug("预测区间置信度计算器初始化完成")
    
    def calculate(self, predictions: np.ndarray, **kwargs) -> np.ndarray:
        """
        基于预测区间计算置信度
        
        Args:
            predictions: 预测结果
            **kwargs: 其他参数，包括confidence_level
            
        Returns:
            置信度分数数组
        """
        try:
            if not self.validate_input(predictions):
                raise PredictionException("输入验证失败")
            
            confidence_level = kwargs.get('confidence_level', 0.95)
            
            if predictions.ndim == 1:
                # 单一预测，使用简单的置信度估计
                return np.full(len(predictions), confidence_level)
            
            # 计算预测区间
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bounds = np.percentile(predictions, lower_percentile, axis=1)
            upper_bounds = np.percentile(predictions, upper_percentile, axis=1)
            
            # 计算区间宽度
            interval_widths = upper_bounds - lower_bounds
            
            # 归一化区间宽度
            if np.max(interval_widths) > 0:
                normalized_widths = interval_widths / np.max(interval_widths)
            else:
                normalized_widths = interval_widths
            
            # 置信度与区间宽度成反比
            confidence = 1.0 - normalized_widths
            
            logger_manager.debug(f"预测区间置信度计算完成，平均置信度: {np.mean(confidence):.4f}")
            return confidence
            
        except Exception as e:
            logger_manager.error(f"预测区间置信度计算失败: {e}")
            raise PredictionException(f"预测区间置信度计算失败: {e}")


class MonteCarloConfidenceCalculator(BaseConfidenceCalculator):
    """基于蒙特卡洛的置信度计算器"""
    
    def __init__(self):
        """初始化蒙特卡洛置信度计算器"""
        super().__init__(ConfidenceMethod.MONTE_CARLO)
        logger_manager.debug("蒙特卡洛置信度计算器初始化完成")
    
    def calculate(self, predictions: np.ndarray, **kwargs) -> np.ndarray:
        """
        基于蒙特卡洛采样计算置信度
        
        Args:
            predictions: 蒙特卡洛采样预测结果 (N, M) 其中N是样本数，M是采样次数
            **kwargs: 其他参数
            
        Returns:
            置信度分数数组
        """
        try:
            if not self.validate_input(predictions):
                raise PredictionException("输入验证失败")
            
            if predictions.ndim == 1:
                # 单一预测，无法进行蒙特卡洛分析
                return np.ones(len(predictions))
            
            # 计算每个样本的统计量
            means = np.mean(predictions, axis=1)
            stds = np.std(predictions, axis=1)
            
            # 计算变异系数（相对标准差）
            cv = np.where(means != 0, stds / np.abs(means), stds)
            
            # 归一化变异系数
            if np.max(cv) > 0:
                normalized_cv = cv / np.max(cv)
            else:
                normalized_cv = cv
            
            # 置信度与变异系数成反比
            confidence = 1.0 - normalized_cv
            
            # 确保置信度在[0,1]范围内
            confidence = np.clip(confidence, 0.0, 1.0)
            
            logger_manager.debug(f"蒙特卡洛置信度计算完成，平均置信度: {np.mean(confidence):.4f}")
            return confidence
            
        except Exception as e:
            logger_manager.error(f"蒙特卡洛置信度计算失败: {e}")
            raise PredictionException(f"蒙特卡洛置信度计算失败: {e}")


class ConfidenceCalculator:
    """置信度计算器主类"""
    
    def __init__(self):
        """初始化置信度计算器"""
        self.calculators = {
            ConfidenceMethod.ENTROPY: EntropyConfidenceCalculator(),
            ConfidenceMethod.VARIANCE: VarianceConfidenceCalculator(),
            ConfidenceMethod.ENSEMBLE_AGREEMENT: EnsembleAgreementCalculator(),
            ConfidenceMethod.PREDICTION_INTERVAL: PredictionIntervalCalculator(),
            ConfidenceMethod.MONTE_CARLO: MonteCarloConfidenceCalculator()
        }
        
        logger_manager.info("置信度计算器初始化完成")
    
    def calculate_confidence(self, predictions: np.ndarray, 
                           method: ConfidenceMethod = ConfidenceMethod.ENTROPY,
                           **kwargs) -> ConfidenceScore:
        """
        计算置信度
        
        Args:
            predictions: 预测结果
            method: 置信度计算方法
            **kwargs: 其他参数
            
        Returns:
            置信度分数对象
        """
        try:
            if method not in self.calculators:
                raise PredictionException(f"不支持的置信度计算方法: {method}")
            
            calculator = self.calculators[method]
            confidence_scores = calculator.calculate(predictions, **kwargs)
            
            # 计算整体统计量
            mean_confidence = float(np.mean(confidence_scores))
            uncertainty = float(np.std(confidence_scores))
            
            # 创建置信度分数对象
            confidence_score = ConfidenceScore(
                method=method,
                score=mean_confidence,
                uncertainty=uncertainty,
                details={
                    'individual_scores': confidence_scores.tolist(),
                    'min_confidence': float(np.min(confidence_scores)),
                    'max_confidence': float(np.max(confidence_scores)),
                    'median_confidence': float(np.median(confidence_scores)),
                    'confidence_distribution': self._analyze_confidence_distribution(confidence_scores)
                },
                metadata={
                    'sample_count': len(confidence_scores),
                    'method_parameters': kwargs
                }
            )
            
            logger_manager.info(f"置信度计算完成，方法: {method.value}, 平均置信度: {mean_confidence:.4f}")
            return confidence_score
            
        except Exception as e:
            logger_manager.error(f"置信度计算失败: {e}")
            raise PredictionException(f"置信度计算失败: {e}")
    
    def calculate_multiple_confidence(self, predictions: np.ndarray,
                                    methods: List[ConfidenceMethod] = None,
                                    **kwargs) -> Dict[ConfidenceMethod, ConfidenceScore]:
        """
        使用多种方法计算置信度
        
        Args:
            predictions: 预测结果
            methods: 置信度计算方法列表
            **kwargs: 其他参数
            
        Returns:
            置信度分数字典
        """
        try:
            if methods is None:
                methods = [ConfidenceMethod.ENTROPY, ConfidenceMethod.VARIANCE]
            
            confidence_scores = {}
            
            for method in methods:
                try:
                    score = self.calculate_confidence(predictions, method, **kwargs)
                    confidence_scores[method] = score
                except Exception as e:
                    logger_manager.warning(f"方法 {method.value} 计算失败: {e}")
                    continue
            
            logger_manager.info(f"多方法置信度计算完成，成功方法数: {len(confidence_scores)}")
            return confidence_scores
            
        except Exception as e:
            logger_manager.error(f"多方法置信度计算失败: {e}")
            raise PredictionException(f"多方法置信度计算失败: {e}")
    
    def ensemble_confidence(self, confidence_scores: Dict[ConfidenceMethod, ConfidenceScore],
                          weights: Dict[ConfidenceMethod, float] = None) -> ConfidenceScore:
        """
        集成多种置信度计算结果
        
        Args:
            confidence_scores: 置信度分数字典
            weights: 权重字典
            
        Returns:
            集成置信度分数
        """
        try:
            if not confidence_scores:
                raise PredictionException("置信度分数字典为空")
            
            # 默认等权重
            if weights is None:
                weights = {method: 1.0 for method in confidence_scores.keys()}
            
            # 归一化权重
            total_weight = sum(weights.values())
            normalized_weights = {k: v / total_weight for k, v in weights.items()}
            
            # 计算加权平均置信度
            weighted_score = 0.0
            weighted_uncertainty = 0.0
            
            for method, score in confidence_scores.items():
                weight = normalized_weights.get(method, 0.0)
                weighted_score += score.score * weight
                weighted_uncertainty += score.uncertainty * weight
            
            # 创建集成置信度分数
            ensemble_score = ConfidenceScore(
                method=ConfidenceMethod.ENSEMBLE_AGREEMENT,  # 使用集成标识
                score=weighted_score,
                uncertainty=weighted_uncertainty,
                details={
                    'individual_methods': {method.value: score.score for method, score in confidence_scores.items()},
                    'weights': {method.value: weight for method, weight in normalized_weights.items()},
                    'method_count': len(confidence_scores)
                },
                metadata={
                    'ensemble_type': 'weighted_average',
                    'component_methods': [method.value for method in confidence_scores.keys()]
                }
            )
            
            logger_manager.info(f"集成置信度计算完成，最终置信度: {weighted_score:.4f}")
            return ensemble_score
            
        except Exception as e:
            logger_manager.error(f"集成置信度计算失败: {e}")
            raise PredictionException(f"集成置信度计算失败: {e}")
    
    def _analyze_confidence_distribution(self, confidence_scores: np.ndarray) -> Dict[str, float]:
        """分析置信度分布"""
        try:
            return {
                'very_high': float(np.sum(confidence_scores > 0.9) / len(confidence_scores)),
                'high': float(np.sum((confidence_scores > 0.7) & (confidence_scores <= 0.9)) / len(confidence_scores)),
                'medium': float(np.sum((confidence_scores > 0.5) & (confidence_scores <= 0.7)) / len(confidence_scores)),
                'low': float(np.sum(confidence_scores <= 0.5) / len(confidence_scores))
            }
        except Exception:
            return {}
    
    def get_available_methods(self) -> List[ConfidenceMethod]:
        """获取可用的置信度计算方法"""
        return list(self.calculators.keys())


# 全局置信度计算器实例
confidence_calculator = ConfidenceCalculator()


if __name__ == "__main__":
    # 测试置信度计算器功能
    print("🎯 测试置信度计算器功能...")
    
    try:
        import numpy as np
        
        # 创建测试数据
        # 模拟集成预测结果
        ensemble_predictions = np.random.random((100, 5))  # 100个样本，5个模型
        
        # 模拟概率分布
        prob_predictions = np.random.dirichlet([1, 1, 1], 100)  # 100个样本，3个类别
        
        calculator = ConfidenceCalculator()
        
        # 测试熵置信度
        entropy_score = calculator.calculate_confidence(
            prob_predictions, 
            ConfidenceMethod.ENTROPY
        )
        print(f"✅ 熵置信度计算成功，平均置信度: {entropy_score.score:.4f}")
        
        # 测试方差置信度
        variance_score = calculator.calculate_confidence(
            ensemble_predictions, 
            ConfidenceMethod.VARIANCE
        )
        print(f"✅ 方差置信度计算成功，平均置信度: {variance_score.score:.4f}")
        
        # 测试集成一致性置信度
        agreement_score = calculator.calculate_confidence(
            ensemble_predictions, 
            ConfidenceMethod.ENSEMBLE_AGREEMENT
        )
        print(f"✅ 集成一致性置信度计算成功，平均置信度: {agreement_score.score:.4f}")
        
        # 测试多方法置信度计算
        multi_scores = calculator.calculate_multiple_confidence(
            ensemble_predictions,
            methods=[ConfidenceMethod.VARIANCE, ConfidenceMethod.ENSEMBLE_AGREEMENT]
        )
        print(f"✅ 多方法置信度计算成功，方法数: {len(multi_scores)}")
        
        # 测试集成置信度
        ensemble_confidence = calculator.ensemble_confidence(multi_scores)
        print(f"✅ 集成置信度计算成功，最终置信度: {ensemble_confidence.score:.4f}")
        
        # 测试可用方法
        available_methods = calculator.get_available_methods()
        print(f"✅ 可用方法获取成功，方法数: {len(available_methods)}")
        
        print("✅ 置信度计算器功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("置信度计算器功能测试完成")
