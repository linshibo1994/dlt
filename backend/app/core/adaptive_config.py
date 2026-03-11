#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自适应算法配置常量模块
Adaptive Algorithm Configuration Constants

集中管理所有自适应学习算法的配置参数，避免魔法数字硬编码
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass(frozen=True)
class MultiArmedBanditConfig:
    """多臂老虎机算法配置"""

    # Epsilon-Greedy 参数
    epsilon: float = 0.1  # 探索概率
    epsilon_min: float = 0.01  # 最小探索概率
    epsilon_decay: float = 0.995  # 探索概率衰减因子

    # UCB1 参数
    ucb_c: float = 2.0  # UCB1 探索参数 (标准值为 sqrt(2) ≈ 1.414，2.0 更激进探索)

    # Thompson Sampling 参数
    thompson_alpha_init: float = 1.0  # Beta 分布初始 alpha
    thompson_beta_init: float = 1.0  # Beta 分布初始 beta
    thompson_reward_scale: float = 1.0  # 奖励值缩放因子（用于连续奖励）


@dataclass(frozen=True)
class AdaptiveLearningConfig:
    """自适应学习配置"""

    # 强化学习参数
    learning_rate: float = 0.1  # 学习率
    discount_factor: float = 0.95  # 折扣因子 (γ)
    exploration_rate: float = 0.2  # 初始探索率
    exploration_decay: float = 0.995  # 探索率衰减
    min_exploration_rate: float = 0.05  # 最小探索率

    # 性能窗口大小
    performance_window_size: int = 50  # 性能跟踪滑动窗口大小
    recent_scores_window: int = 20  # 最近得分窗口大小
    recent_performance_window: int = 10  # 最近性能窗口大小

    # 置信度范围
    confidence_min: float = 0.1  # 最小置信度
    confidence_max: float = 0.9  # 最大置信度

    # 预热阶段
    warmup_multiplier: int = 3  # 预热轮数 = 预测器数量 * 此乘数
    max_warmup_rounds: int = 20  # 最大预热轮数

    # 早停参数
    early_stopping_patience: int = 20  # 早停耐心值
    early_stopping_min_delta: float = 1e-6  # 最小变化阈值


@dataclass(frozen=True)
class RewardConfig:
    """奖励配置"""

    # 中奖等级对应得分
    prize_scores: Dict[str, float] = field(default_factory=lambda: {
        "一等奖": 1000.0,
        "二等奖": 500.0,
        "三等奖": 100.0,
        "四等奖": 50.0,
        "五等奖": 20.0,
        "六等奖": 10.0,
        "七等奖": 5.0,
        "八等奖": 3.0,
        "九等奖": 1.0,
        "未中奖": 0.0
    })

    # 奖励标准化参数
    reward_normalization_factor: float = 10.0  # 标准化因子
    max_normalized_reward: float = 1.0  # 最大标准化奖励

    # 置信度到奖励的映射系数
    confidence_to_score_factor: float = 10.0  # 置信度转分数因子


@dataclass(frozen=True)
class PredictorWeightConfig:
    """预测器权重配置"""

    # 基础置信度（按预测器类型）
    base_confidence: Dict[str, float] = field(default_factory=lambda: {
        'traditional_frequency': 0.5,
        'traditional_hot_cold': 0.5,
        'traditional_missing': 0.5,
        'advanced_markov': 0.6,
        'advanced_bayesian': 0.6,
        'advanced_ensemble': 0.7,
        'super_predictor': 0.8
    })

    # 权重调整范围
    weight_min: float = 0.1
    weight_max: float = 0.99

    # 性能因子范围
    performance_factor_min: float = 0.5
    performance_factor_max: float = 2.0


@dataclass(frozen=True)
class RegretTrackingConfig:
    """后悔值跟踪配置"""

    # 是否启用后悔值跟踪
    enabled: bool = True

    # 后悔值计算窗口
    window_size: int = 100

    # 后悔值报告阈值（累积后悔值超过此值时发出警告）
    warning_threshold: float = 50.0

    # 后悔值归一化因子
    normalization_factor: float = 1.0


@dataclass
class AdaptiveAlgorithmConfig:
    """自适应算法总配置"""

    bandit: MultiArmedBanditConfig = field(default_factory=MultiArmedBanditConfig)
    learning: AdaptiveLearningConfig = field(default_factory=AdaptiveLearningConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    predictor_weight: PredictorWeightConfig = field(default_factory=PredictorWeightConfig)
    regret_tracking: RegretTrackingConfig = field(default_factory=RegretTrackingConfig)

    def to_dict(self) -> Dict:
        """转换为字典格式（完整版本）"""
        return {
            'bandit': {
                'epsilon': self.bandit.epsilon,
                'epsilon_min': self.bandit.epsilon_min,
                'epsilon_decay': self.bandit.epsilon_decay,
                'ucb_c': self.bandit.ucb_c,
                'thompson_alpha_init': self.bandit.thompson_alpha_init,
                'thompson_beta_init': self.bandit.thompson_beta_init,
                'thompson_reward_scale': self.bandit.thompson_reward_scale,
            },
            'learning': {
                'learning_rate': self.learning.learning_rate,
                'discount_factor': self.learning.discount_factor,
                'exploration_rate': self.learning.exploration_rate,
                'exploration_decay': self.learning.exploration_decay,
                'min_exploration_rate': self.learning.min_exploration_rate,
                'performance_window_size': self.learning.performance_window_size,
                'recent_scores_window': self.learning.recent_scores_window,
                'recent_performance_window': self.learning.recent_performance_window,
                'confidence_min': self.learning.confidence_min,
                'confidence_max': self.learning.confidence_max,
                'warmup_multiplier': self.learning.warmup_multiplier,
                'max_warmup_rounds': self.learning.max_warmup_rounds,
                'early_stopping_patience': self.learning.early_stopping_patience,
                'early_stopping_min_delta': self.learning.early_stopping_min_delta,
            },
            'reward': {
                'prize_scores': dict(self.reward.prize_scores),
                'reward_normalization_factor': self.reward.reward_normalization_factor,
                'max_normalized_reward': self.reward.max_normalized_reward,
                'confidence_to_score_factor': self.reward.confidence_to_score_factor,
            },
            'predictor_weight': {
                'base_confidence': dict(self.predictor_weight.base_confidence),
                'weight_min': self.predictor_weight.weight_min,
                'weight_max': self.predictor_weight.weight_max,
                'performance_factor_min': self.predictor_weight.performance_factor_min,
                'performance_factor_max': self.predictor_weight.performance_factor_max,
            },
            'regret_tracking': {
                'enabled': self.regret_tracking.enabled,
                'window_size': self.regret_tracking.window_size,
                'warning_threshold': self.regret_tracking.warning_threshold,
                'normalization_factor': self.regret_tracking.normalization_factor,
            }
        }


# 全局默认配置实例
DEFAULT_ADAPTIVE_CONFIG = AdaptiveAlgorithmConfig()


def get_adaptive_config() -> AdaptiveAlgorithmConfig:
    """获取自适应算法配置实例"""
    return DEFAULT_ADAPTIVE_CONFIG


def create_custom_config(
    epsilon: Optional[float] = None,
    ucb_c: Optional[float] = None,
    learning_rate: Optional[float] = None,
    exploration_decay: Optional[float] = None,
    thompson_reward_scale: Optional[float] = None,
    performance_window_size: Optional[int] = None,
) -> AdaptiveAlgorithmConfig:
    """创建自定义配置

    Args:
        epsilon: Epsilon-Greedy 探索概率
        ucb_c: UCB1 探索参数
        learning_rate: 学习率
        exploration_decay: 探索率衰减
        thompson_reward_scale: Thompson 采样奖励缩放因子
        performance_window_size: 性能跟踪窗口大小

    Returns:
        自定义配置实例（从默认配置继承未指定的参数）
    """
    default = DEFAULT_ADAPTIVE_CONFIG

    # 创建完整的 bandit 配置（继承所有未指定的参数）
    bandit_config = MultiArmedBanditConfig(
        epsilon=epsilon if epsilon is not None else default.bandit.epsilon,
        epsilon_min=default.bandit.epsilon_min,
        epsilon_decay=default.bandit.epsilon_decay,
        ucb_c=ucb_c if ucb_c is not None else default.bandit.ucb_c,
        thompson_alpha_init=default.bandit.thompson_alpha_init,
        thompson_beta_init=default.bandit.thompson_beta_init,
        thompson_reward_scale=thompson_reward_scale if thompson_reward_scale is not None else default.bandit.thompson_reward_scale,
    )

    # 创建完整的 learning 配置（继承所有未指定的参数）
    learning_config = AdaptiveLearningConfig(
        learning_rate=learning_rate if learning_rate is not None else default.learning.learning_rate,
        discount_factor=default.learning.discount_factor,
        exploration_rate=default.learning.exploration_rate,
        exploration_decay=exploration_decay if exploration_decay is not None else default.learning.exploration_decay,
        min_exploration_rate=default.learning.min_exploration_rate,
        performance_window_size=performance_window_size if performance_window_size is not None else default.learning.performance_window_size,
        recent_scores_window=default.learning.recent_scores_window,
        recent_performance_window=default.learning.recent_performance_window,
        confidence_min=default.learning.confidence_min,
        confidence_max=default.learning.confidence_max,
        warmup_multiplier=default.learning.warmup_multiplier,
        max_warmup_rounds=default.learning.max_warmup_rounds,
        early_stopping_patience=default.learning.early_stopping_patience,
        early_stopping_min_delta=default.learning.early_stopping_min_delta,
    )

    return AdaptiveAlgorithmConfig(
        bandit=bandit_config,
        learning=learning_config,
        reward=default.reward,
        predictor_weight=default.predictor_weight,
        regret_tracking=default.regret_tracking,
    )


# 预定义配置方案
class ConfigPresets:
    """预定义配置方案"""

    @staticmethod
    def aggressive_exploration() -> AdaptiveAlgorithmConfig:
        """激进探索配置（适合初期学习）"""
        return AdaptiveAlgorithmConfig(
            bandit=MultiArmedBanditConfig(
                epsilon=0.3,
                ucb_c=3.0,
            ),
            learning=AdaptiveLearningConfig(
                exploration_rate=0.4,
                exploration_decay=0.99,
            ),
        )

    @staticmethod
    def conservative_exploitation() -> AdaptiveAlgorithmConfig:
        """保守利用配置（适合稳定期）"""
        return AdaptiveAlgorithmConfig(
            bandit=MultiArmedBanditConfig(
                epsilon=0.05,
                ucb_c=1.0,
            ),
            learning=AdaptiveLearningConfig(
                exploration_rate=0.1,
                exploration_decay=0.999,
            ),
        )

    @staticmethod
    def balanced() -> AdaptiveAlgorithmConfig:
        """平衡配置（默认）"""
        return DEFAULT_ADAPTIVE_CONFIG


if __name__ == "__main__":
    # 测试配置模块
    config = get_adaptive_config()
    print("默认自适应算法配置:")
    print(f"  Epsilon: {config.bandit.epsilon}")
    print(f"  UCB_C: {config.bandit.ucb_c}")
    print(f"  学习率: {config.learning.learning_rate}")
    print(f"  探索衰减: {config.learning.exploration_decay}")
    print(f"  最小探索率: {config.learning.min_exploration_rate}")

    # 测试预设配置
    aggressive = ConfigPresets.aggressive_exploration()
    print(f"\n激进探索配置 Epsilon: {aggressive.bandit.epsilon}")

    conservative = ConfigPresets.conservative_exploitation()
    print(f"保守利用配置 Epsilon: {conservative.bandit.epsilon}")
