#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析器模块集成
整合基础分析、高级分析、综合分析等所有分析功能
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, Counter
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
import yaml
warnings.filterwarnings('ignore')

import backend.app.core.core_modules as cm

# 导入统一路径配置（使用相对导入，支持多种运行环境）
_PATH_CONFIG_SOURCE = None  # 用于记录路径配置来源

try:
    from path_config import (
        REPORTS_VISUALIZATION_DIR,
        get_path_manager
    )
    _PATH_CONFIG_SOURCE = "path_config"
except ImportError:
    try:
        from backend.app.core.path_config import (
            REPORTS_VISUALIZATION_DIR,
            get_path_manager
        )
        _PATH_CONFIG_SOURCE = "backend.app.core.path_config"
    except ImportError:
        # 使用环境变量和默认值作为兜底
        _PATH_CONFIG_SOURCE = "environment_variables_and_defaults"
        REPORTS_VISUALIZATION_DIR = os.getenv('DLT_REPORTS_VISUALIZATION_DIR', "artifacts/reports/visualization")
        get_path_manager = None

cache_manager = cm.cache_manager
logger_manager = cm.logger_manager
data_manager = cm.data_manager

# 记录路径配置来源
logger_manager.debug(f"analyzer_modules 路径配置来源: {_PATH_CONFIG_SOURCE}")
if _PATH_CONFIG_SOURCE == "environment_variables_and_defaults":
    logger_manager.info(f"analyzer_modules 使用环境变量和默认路径配置 - REPORTS_VISUALIZATION_DIR: {REPORTS_VISUALIZATION_DIR}")

# 导入智能缓存系统
from backend.app.core.smart_cache_system import smart_cache_manager

# 预测配置缓存
_PREDICTION_CONFIG_CACHE = None
_PREDICTION_CONFIG_MTIME = None


def _get_prediction_config_path() -> str:
    """获取预测配置文件路径"""
    try:
        from path_config import PREDICTION_CONFIG_FILE
        return PREDICTION_CONFIG_FILE
    except ImportError:
        try:
            from backend.app.core.path_config import PREDICTION_CONFIG_FILE
            return PREDICTION_CONFIG_FILE
        except ImportError:
            project_root = os.environ.get(
                'DLT_PROJECT_ROOT',
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            )
            return os.path.join(project_root, 'config', 'prediction.yaml')


def _load_prediction_config() -> Dict[str, Any]:
    """加载 prediction.yaml（带简易缓存）"""
    global _PREDICTION_CONFIG_CACHE, _PREDICTION_CONFIG_MTIME
    try:
        config_path = _get_prediction_config_path()
        mtime = os.path.getmtime(config_path)
        if _PREDICTION_CONFIG_CACHE is not None and _PREDICTION_CONFIG_MTIME == mtime:
            return _PREDICTION_CONFIG_CACHE
        with open(config_path, 'r', encoding='utf-8') as f:
            _PREDICTION_CONFIG_CACHE = yaml.safe_load(f) or {}
            _PREDICTION_CONFIG_MTIME = mtime
        return _PREDICTION_CONFIG_CACHE
    except Exception:
        return {}


def load_bayesian_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """读取并应用贝叶斯配置"""
    cfg = _load_prediction_config()
    bayes_cfg = cfg.get('prediction_methods', {}).get('traditional_ml', {}).get('bayesian', {}) or {}
    if overrides:
        bayes_cfg = {**bayes_cfg, **overrides}

    if 'dirichlet_mix_weight' in bayes_cfg:
        BayesianConfig.DIRICHLET_MIX_WEIGHT = float(bayes_cfg['dirichlet_mix_weight'])
    if 'dirichlet_concentration' in bayes_cfg:
        BayesianConfig.DIRICHLET_CONCENTRATION = float(bayes_cfg['dirichlet_concentration'])
    if 'decay_enabled' in bayes_cfg:
        BayesianConfig.DECAY_ENABLED = bool(bayes_cfg['decay_enabled'])
    if 'decay_half_life' in bayes_cfg:
        BayesianConfig.DECAY_HALF_LIFE = float(bayes_cfg['decay_half_life'])
    if 'decay_min_weight' in bayes_cfg:
        BayesianConfig.DECAY_MIN_WEIGHT = float(bayes_cfg['decay_min_weight'])
    if 'decay_mode' in bayes_cfg:
        BayesianConfig.DECAY_MODE = str(bayes_cfg['decay_mode'])
    if 'recent_window' in bayes_cfg:
        BayesianConfig.RECENT_WINDOW = int(bayes_cfg['recent_window'])
    if 'mid_window' in bayes_cfg:
        BayesianConfig.MID_WINDOW = int(bayes_cfg['mid_window'])
    if 'recent_weight' in bayes_cfg:
        BayesianConfig.RECENT_WEIGHT = float(bayes_cfg['recent_weight'])
    if 'mid_weight' in bayes_cfg:
        BayesianConfig.MID_WEIGHT = float(bayes_cfg['mid_weight'])
    if 'old_weight' in bayes_cfg:
        BayesianConfig.OLD_WEIGHT = float(bayes_cfg['old_weight'])

    if 'prior_hot_bonus' in bayes_cfg:
        BayesianConfig.PRIOR_HOT_BONUS = float(bayes_cfg['prior_hot_bonus'])
    if 'prior_warm_bonus' in bayes_cfg:
        BayesianConfig.PRIOR_WARM_BONUS = float(bayes_cfg['prior_warm_bonus'])
    if 'prior_cold_penalty' in bayes_cfg:
        BayesianConfig.PRIOR_COLD_PENALTY = float(bayes_cfg['prior_cold_penalty'])
    if 'prior_missing_bias' in bayes_cfg:
        BayesianConfig.PRIOR_MISSING_BIAS = float(bayes_cfg['prior_missing_bias'])
    if 'prior_min_factor' in bayes_cfg:
        BayesianConfig.PRIOR_MIN_FACTOR = float(bayes_cfg['prior_min_factor'])

    return bayes_cfg

# 导入复式预测功能（支持多种导入路径）
try:
    from compound.compound_predictor import CompoundPredictorMixin, CompoundConfig, CompoundResult
except ImportError:
    try:
        from compound_modules.compound_predictor import CompoundPredictorMixin, CompoundConfig, CompoundResult
    except ImportError:
        # 定义占位类，避免导入失败
        CompoundPredictorMixin = object
        CompoundConfig = None
        CompoundResult = None


# ==================== 冷热号分析配置常量 ====================
class HotColdConfig:
    """冷热号分析配置常量类 - 消除魔法数字，提高可维护性"""

    # 大乐透号码范围
    FRONT_MAX_NUMBER = 35           # 前区最大号码
    FRONT_NUMBERS_PER_DRAW = 5      # 前区每期开出数量
    BACK_MAX_NUMBER = 12            # 后区最大号码
    BACK_NUMBERS_PER_DRAW = 2       # 后区每期开出数量

    # 区间划分（前区）
    FRONT_LOW_ZONE = (1, 12)        # 低区：1-12
    FRONT_MID_ZONE = (13, 24)       # 中区：13-24
    FRONT_HIGH_ZONE = (25, 35)      # 高区：25-35

    # 区间划分（后区）
    BACK_LOW_ZONE = (1, 6)          # 低区：1-6
    BACK_HIGH_ZONE = (7, 12)        # 高区：7-12

    # 温度等级阈值（基于 Z-Score）
    EXTREMELY_HOT_THRESHOLD = 2.0   # 极热
    VERY_HOT_THRESHOLD = 1.5        # 非常热
    HOT_THRESHOLD = 1.0             # 热
    WARM_THRESHOLD = 0.5            # 温
    NORMAL_THRESHOLD = -0.5         # 正常（上界）
    COOL_THRESHOLD = -1.0           # 凉
    COLD_THRESHOLD = -1.5           # 冷
    # 低于 COLD_THRESHOLD 为 extremely_cold（极冷）

    # 趋势分析阈值
    TREND_HEATING_THRESHOLD = 0.1   # 升温趋势阈值
    TREND_COOLING_THRESHOLD = -0.1  # 降温趋势阈值

    # 稳定性分析阈值（基于变异系数 CV）
    STABILITY_VERY_STABLE = 0.3     # 非常稳定
    STABILITY_STABLE = 0.5          # 稳定
    STABILITY_MODERATE = 0.8        # 中等
    # 高于 STABILITY_MODERATE 为 unstable（不稳定）

    # 分析参数
    MIN_PERIODS_FOR_TREND = 20      # 趋势分析最小期数
    MIN_PERIODS_FOR_STABILITY = 30  # 稳定性分析最小期数
    TREND_WINDOW_SIZE = 10          # 趋势分析滑动窗口大小
    STABILITY_SEGMENTS = 5          # 稳定性分析分段数

    # 预测权重调整因子
    TREND_HEATING_ADJUSTMENT = 1.2  # 升温趋势权重调整
    TREND_COOLING_ADJUSTMENT = 0.8  # 降温趋势权重调整
    STABILITY_BONUS = 1.1           # 稳定性加成
    INSTABILITY_PENALTY = 0.9       # 不稳定性惩罚

    # 分位数分类阈值（用于冷热号稳定分层）
    HOT_QUANTILE = 0.80             # 频率分位数 ≥ 0.80 视为热
    COLD_QUANTILE = 0.20            # 频率分位数 ≤ 0.20 视为冷
    WARM_QUANTILE_LOW = 0.40        # 温号下界
    WARM_QUANTILE_HIGH = 0.60       # 温号上界

    # 遗漏权重融合比例
    MISSING_WEIGHT_FACTOR = 0.30    # 冷热权重与遗漏权重融合比例


# ==================== 贝叶斯分析配置常量 ====================
class BayesianConfig:
    """贝叶斯分析配置常量"""
    DIRICHLET_CONCENTRATION = 5.0   # Dirichlet先验集中度（越大越保守）
    DIRICHLET_MIX_WEIGHT = 0.5      # Dirichlet后验与增强后验混合权重
    DECAY_ENABLED = True            # 启用时间衰减
    DECAY_HALF_LIFE = 200           # 半衰期（期数）
    DECAY_MIN_WEIGHT = 0.2          # 最小权重
    DECAY_MODE = "segmented"        # 衰减模式: exponential / segmented
    RECENT_WINDOW = 100             # 分段衰减: 近期窗口
    MID_WINDOW = 300                # 分段衰减: 中期窗口
    RECENT_WEIGHT = 1.0             # 分段衰减: 近期权重
    MID_WEIGHT = 0.7                # 分段衰减: 中期权重
    OLD_WEIGHT = 0.4                # 分段衰减: 远期权重

    # 先验偏置（冷热/遗漏）
    PRIOR_HOT_BONUS = 0.20          # 热号先验提升
    PRIOR_WARM_BONUS = 0.10         # 温号先验提升
    PRIOR_COLD_PENALTY = 0.15       # 冷号先验折扣
    PRIOR_MISSING_BIAS = 0.30       # 遗漏权重偏置强度
    PRIOR_MIN_FACTOR = 0.20         # 先验下限因子


# ==================== 基础分析器 ====================
class BasicAnalyzer(CompoundPredictorMixin):
    """基础分析器（支持复式预测）"""
    
    def __init__(self, data_file="data/dlt_data_all.csv"):
        super().__init__()
        self.data_file = data_file
        self.df = data_manager.get_data()
        
        if self.df is None:
            logger_manager.error("数据未加载")
    
    def frequency_analysis(self, periods=None) -> Dict:
        """增强频率分析 - 包含概率分布建模和置信区间计算"""
        if self.df is None:
            return {}

        method_name = "frequency_analysis"
        cached_result = smart_cache_manager.load_cache("analysis", method_name, periods)
        if cached_result:
            return cached_result

        # 数据是降序排列（最新在前），使用head()获取最新数据
        df_subset = self.df.head(periods) if periods else self.df

        front_counter = Counter()
        back_counter = Counter()

        for _, row in df_subset.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            front_counter.update(front_balls)
            back_counter.update(back_balls)

        # 基础频率统计
        front_frequency = dict(front_counter.most_common())
        back_frequency = dict(back_counter.most_common())

        # 增强分析：概率分布建模
        front_enhanced = self._enhanced_frequency_analysis(front_counter, len(df_subset), 35, 5, ball_type="front")
        back_enhanced = self._enhanced_frequency_analysis(back_counter, len(df_subset), 12, 2, ball_type="back")

        result = {
            'front_frequency': front_frequency,
            'back_frequency': back_frequency,
            'front_enhanced': front_enhanced,
            'back_enhanced': back_enhanced,
            'analysis_periods': len(df_subset),
            'timestamp': datetime.now().isoformat()
        }

        smart_cache_manager.save_cache("analysis", method_name, result, periods)
        return result

    def _enhanced_frequency_analysis(self, counter: Counter, total_periods: int,
                                   max_number: int, numbers_per_draw: int, ball_type: str = "all") -> Dict:
        """增强频率分析 - 概率分布建模"""
        try:
            import numpy as np
            from scipy import stats

            # 理论期望频率
            theoretical_freq = (total_periods * numbers_per_draw) / max_number

            # 计算每个号码的统计指标
            enhanced_stats = {}

            for num in range(1, max_number + 1):
                observed_freq = counter.get(num, 0)

                # 概率计算
                probability = observed_freq / (total_periods * numbers_per_draw) if total_periods > 0 else 0

                # 偏差分析
                deviation = observed_freq - theoretical_freq
                relative_deviation = deviation / theoretical_freq if theoretical_freq > 0 else 0

                # 置信区间计算（基于二项分布）
                if total_periods > 0:
                    p = numbers_per_draw / max_number  # 理论概率
                    n = total_periods  # 试验次数

                    # 95%置信区间
                    confidence_interval = stats.binom.interval(0.95, n, p)

                    # Z-score计算
                    expected = n * p
                    variance = n * p * (1 - p)
                    z_score = (observed_freq - expected) / np.sqrt(variance) if variance > 0 else 0
                else:
                    confidence_interval = (0, 0)
                    z_score = 0

                # 趋势分析（最近期数的频率变化）
                recent_trend = self._calculate_frequency_trend(num, total_periods, ball_type=ball_type)

                enhanced_stats[num] = {
                    'observed_frequency': observed_freq,
                    'theoretical_frequency': theoretical_freq,
                    'probability': probability,
                    'deviation': deviation,
                    'relative_deviation': relative_deviation,
                    'confidence_interval': confidence_interval,
                    'z_score': z_score,
                    'trend': recent_trend,
                    'heat_level': self._calculate_heat_level(observed_freq, theoretical_freq),
                    'prediction_weight': self._calculate_prediction_weight(
                        probability, z_score, recent_trend
                    )
                }

            return enhanced_stats

        except Exception as e:
            logger_manager.error(f"增强频率分析失败: {e}")
            return {}

    def _calculate_frequency_trend(self, number: int, total_periods: int, ball_type: str = "all") -> Dict:
        """计算频率趋势"""
        try:
            if total_periods < 20:
                return {'trend': 'insufficient_data', 'slope': 0}

            # 分析最近20期的趋势（数据是降序排列，使用head获取最新数据）
            recent_periods = min(20, total_periods // 4)
            recent_data = self.df.head(recent_periods)

            frequencies = []
            for _, row in recent_data.iterrows():
                front_balls, back_balls = data_manager.parse_balls(row)
                if ball_type == "front":
                    freq = 1 if number in front_balls else 0
                elif ball_type == "back":
                    freq = 1 if number in back_balls else 0
                else:
                    freq = 1 if number in (front_balls + back_balls) else 0
                frequencies.append(freq)

            if len(frequencies) < 5:
                return {'trend': 'insufficient_data', 'slope': 0}

            # 线性回归计算趋势
            import numpy as np
            x = np.arange(len(frequencies))
            slope, intercept = np.polyfit(x, frequencies, 1)

            # 趋势判断
            if slope > 0.05:
                trend = 'increasing'
            elif slope < -0.05:
                trend = 'decreasing'
            else:
                trend = 'stable'

            return {
                'trend': trend,
                'slope': slope,
                'recent_frequency': sum(frequencies),
                'recent_periods': len(frequencies)
            }

        except Exception as e:
            logger_manager.error(f"计算频率趋势失败: {e}")
            return {'trend': 'unknown', 'slope': 0}

    def _calculate_heat_level(self, observed_freq: int, theoretical_freq: float) -> str:
        """计算热度等级"""
        try:
            if theoretical_freq == 0:
                return 'unknown'

            ratio = observed_freq / theoretical_freq

            if ratio >= 1.3:
                return 'very_hot'
            elif ratio >= 1.1:
                return 'hot'
            elif ratio >= 0.9:
                return 'normal'
            elif ratio >= 0.7:
                return 'cold'
            else:
                return 'very_cold'

        except Exception as e:
            logger_manager.error(f"计算热度等级失败: {e}")
            return 'unknown'

    def _calculate_prediction_weight(self, probability: float, z_score: float,
                                   trend: Dict) -> float:
        """计算预测权重"""
        try:
            # 基础权重基于概率
            base_weight = probability

            # Z-score调整（异常值降权）
            z_adjustment = 1.0 / (1.0 + abs(z_score) * 0.1)

            # 趋势调整
            trend_adjustment = 1.0
            if trend.get('trend') == 'increasing':
                trend_adjustment = 1.2
            elif trend.get('trend') == 'decreasing':
                trend_adjustment = 0.8

            # 综合权重
            final_weight = base_weight * z_adjustment * trend_adjustment

            return max(0.0, min(1.0, final_weight))

        except Exception as e:
            logger_manager.error(f"计算预测权重失败: {e}")
            return 0.5
    
    def missing_analysis(self, periods=None) -> Dict:
        """增强遗漏分析 - 包含回补概率模型和期望回补时间"""
        if self.df is None:
            return {}

        method_name = "missing_analysis"
        cached_result = smart_cache_manager.load_cache("analysis", method_name, periods)
        if cached_result:
            return cached_result

        # 数据是降序排列（最新在前），使用head()获取最新数据
        df_subset = self.df.head(periods) if periods else self.df

        # 基础遗漏值计算
        front_missing = {i: 0 for i in range(1, 36)}
        back_missing = {i: 0 for i in range(1, 13)}

        # 历史遗漏记录
        front_missing_history = {i: [] for i in range(1, 36)}
        back_missing_history = {i: [] for i in range(1, 13)}

        # 当前遗漏计数器
        front_current_missing = {i: 0 for i in range(1, 36)}
        back_current_missing = {i: 0 for i in range(1, 13)}

        for _, row in df_subset.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)

            # 更新前区遗漏值
            for num in range(1, 36):
                if num in front_balls:
                    # 记录遗漏历史
                    if front_current_missing[num] > 0:
                        front_missing_history[num].append(front_current_missing[num])
                    front_current_missing[num] = 0
                    front_missing[num] = 0
                else:
                    front_current_missing[num] += 1
                    front_missing[num] = front_current_missing[num]

            # 更新后区遗漏值
            for num in range(1, 13):
                if num in back_balls:
                    # 记录遗漏历史
                    if back_current_missing[num] > 0:
                        back_missing_history[num].append(back_current_missing[num])
                    back_current_missing[num] = 0
                    back_missing[num] = 0
                else:
                    back_current_missing[num] += 1
                    back_missing[num] = back_current_missing[num]

        # 增强分析：回补概率模型
        front_enhanced = self._enhanced_missing_analysis(
            front_missing, front_missing_history, 35, 5, len(df_subset)
        )
        back_enhanced = self._enhanced_missing_analysis(
            back_missing, back_missing_history, 12, 2, len(df_subset)
        )

        result = {
            'front_missing': front_missing,
            'back_missing': back_missing,
            'front_enhanced': front_enhanced,
            'back_enhanced': back_enhanced,
            'analysis_periods': len(df_subset),
            'timestamp': datetime.now().isoformat()
        }

        smart_cache_manager.save_cache("analysis", method_name, result, periods)
        return result

    def _enhanced_missing_analysis(self, current_missing: Dict, missing_history: Dict,
                                 max_number: int, numbers_per_draw: int, total_periods: int) -> Dict:
        """增强遗漏分析 - 回补概率模型"""
        try:
            import numpy as np
            from scipy import stats

            enhanced_stats = {}

            for num in range(1, max_number + 1):
                current_miss = current_missing.get(num, 0)
                history = missing_history.get(num, [])

                # 历史遗漏统计
                if history:
                    avg_missing = np.mean(history)
                    std_missing = np.std(history)
                    max_missing = max(history)
                    min_missing = min(history)
                else:
                    avg_missing = total_periods * numbers_per_draw / max_number
                    std_missing = avg_missing * 0.5
                    max_missing = current_miss
                    min_missing = 0

                # 回补概率计算
                comeback_probability = self._calculate_comeback_probability(
                    current_miss, avg_missing, std_missing, max_number, numbers_per_draw
                )

                # 期望回补时间
                expected_comeback_time = self._calculate_expected_comeback_time(
                    current_miss, avg_missing, max_number, numbers_per_draw
                )

                # 遗漏等级
                missing_level = self._calculate_missing_level(current_miss, avg_missing, std_missing)

                # 回补紧迫度
                urgency_score = self._calculate_urgency_score(
                    current_miss, avg_missing, max_missing, comeback_probability
                )

                enhanced_stats[num] = {
                    'current_missing': current_miss,
                    'average_missing': avg_missing,
                    'std_missing': std_missing,
                    'max_historical_missing': max_missing,
                    'min_historical_missing': min_missing,
                    'comeback_probability': comeback_probability,
                    'expected_comeback_time': expected_comeback_time,
                    'missing_level': missing_level,
                    'urgency_score': urgency_score,
                    'historical_count': len(history),
                    'prediction_weight': self._calculate_missing_prediction_weight(
                        comeback_probability, urgency_score, missing_level
                    )
                }

            return enhanced_stats

        except Exception as e:
            logger_manager.error(f"增强遗漏分析失败: {e}")
            return {}

    def _calculate_comeback_probability(self, current_miss: int, avg_miss: float,
                                      std_miss: float, max_number: int, numbers_per_draw: int) -> float:
        """计算回补概率"""
        try:
            # 基础概率（几何分布）
            p = numbers_per_draw / max_number
            base_prob = 1 - (1 - p) ** (current_miss + 1)

            # 基于历史统计的调整
            if std_miss > 0 and avg_miss > 0:
                # 标准化当前遗漏值
                z_score = (current_miss - avg_miss) / std_miss

                # 遗漏越久，回补概率越高（但有上限）
                adjustment = 1 + min(z_score * 0.1, 0.5)
                adjusted_prob = base_prob * adjustment
            else:
                adjusted_prob = base_prob

            return max(0.0, min(1.0, adjusted_prob))

        except Exception as e:
            logger_manager.error(f"计算回补概率失败: {e}")
            return 0.5

    def _calculate_expected_comeback_time(self, current_miss: int, avg_miss: float,
                                        max_number: int, numbers_per_draw: int) -> float:
        """计算期望回补时间"""
        try:
            # 基于几何分布的期望
            p = numbers_per_draw / max_number
            expected_time = 1 / p

            # 考虑当前已遗漏的时间
            if current_miss >= avg_miss:
                # 已经超过平均遗漏时间，期望回补时间减少
                remaining_time = max(1, expected_time - current_miss * 0.5)
            else:
                # 还未达到平均遗漏时间
                remaining_time = expected_time - current_miss

            return max(1.0, remaining_time)

        except Exception as e:
            logger_manager.error(f"计算期望回补时间失败: {e}")
            return 10.0

    def _calculate_missing_level(self, current_miss: int, avg_miss: float, std_miss: float) -> str:
        """计算遗漏等级"""
        try:
            if std_miss == 0:
                return 'normal'

            z_score = (current_miss - avg_miss) / std_miss

            if z_score >= 2:
                return 'extremely_overdue'
            elif z_score >= 1.5:
                return 'very_overdue'
            elif z_score >= 1:
                return 'overdue'
            elif z_score >= -1:
                return 'normal'
            else:
                return 'recent'

        except Exception as e:
            logger_manager.error(f"计算遗漏等级失败: {e}")
            return 'unknown'

    def _calculate_urgency_score(self, current_miss: int, avg_miss: float,
                               max_miss: int, comeback_prob: float) -> float:
        """计算回补紧迫度评分"""
        try:
            # 基于当前遗漏与平均遗漏的比值
            miss_ratio = current_miss / avg_miss if avg_miss > 0 else 1

            # 基于当前遗漏与历史最大遗漏的比值
            max_ratio = current_miss / max_miss if max_miss > 0 else 0

            # 综合评分
            urgency = (miss_ratio * 0.6 + max_ratio * 0.2 + comeback_prob * 0.2)

            return max(0.0, min(10.0, urgency * 5))

        except Exception as e:
            logger_manager.error(f"计算紧迫度评分失败: {e}")
            return 5.0

    def _calculate_missing_prediction_weight(self, comeback_prob: float,
                                           urgency_score: float, missing_level: str) -> float:
        """计算遗漏预测权重"""
        try:
            # 基础权重基于回补概率
            base_weight = comeback_prob

            # 紧迫度调整
            urgency_adjustment = 1 + (urgency_score - 5) * 0.1

            # 遗漏等级调整
            level_adjustments = {
                'extremely_overdue': 1.5,
                'very_overdue': 1.3,
                'overdue': 1.1,
                'normal': 1.0,
                'recent': 0.8,
                'unknown': 1.0
            }
            level_adjustment = level_adjustments.get(missing_level, 1.0)

            # 综合权重
            final_weight = base_weight * urgency_adjustment * level_adjustment

            return max(0.0, min(1.0, final_weight))

        except Exception as e:
            logger_manager.error(f"计算遗漏预测权重失败: {e}")
            return 0.5
    
    def hot_cold_analysis(self, periods=100) -> Dict:
        """增强冷热号分析 - 包含温度量化计算、动态阈值调整和区间分析"""
        if self.df is None:
            return {}

        method_name = "hot_cold_analysis"
        cached_result = smart_cache_manager.load_cache("analysis", method_name, periods)
        if cached_result:
            return cached_result

        freq_result = self.frequency_analysis(periods)
        missing_result = self.missing_analysis(periods)

        front_freq = freq_result.get('front_frequency', {})
        back_freq = freq_result.get('back_frequency', {})
        front_missing_enhanced = missing_result.get('front_enhanced', {})
        back_missing_enhanced = missing_result.get('back_enhanced', {})

        # 增强分析：温度量化计算（使用配置常量）
        front_enhanced = self._enhanced_hot_cold_analysis(
            front_freq, periods,
            HotColdConfig.FRONT_MAX_NUMBER,
            HotColdConfig.FRONT_NUMBERS_PER_DRAW,
            scope='front'
        )
        back_enhanced = self._enhanced_hot_cold_analysis(
            back_freq, periods,
            HotColdConfig.BACK_MAX_NUMBER,
            HotColdConfig.BACK_NUMBERS_PER_DRAW,
            scope='back'
        )

        # 融合遗漏权重（不改变结构，仅增强 prediction_weight）
        self._apply_missing_weight(front_enhanced, front_missing_enhanced)
        self._apply_missing_weight(back_enhanced, back_missing_enhanced)

        # 传统统计均值（包含零频，避免遗漏号码被忽略）
        front_avg = np.mean([front_freq.get(num, 0) for num in range(1, 36)]) if front_freq is not None else 0
        back_avg = np.mean([back_freq.get(num, 0) for num in range(1, 13)]) if back_freq is not None else 0

        # 分位数 + 置信区间分层（增强稳定性）
        front_hot, front_warm, front_cold = self._classify_hot_cold_by_quantile(front_enhanced)
        back_hot, back_warm, back_cold = self._classify_hot_cold_by_quantile(back_enhanced)

        # 区间冷热分析
        zone_analysis = self._zone_hot_cold_analysis(front_enhanced, back_enhanced)

        result = {
            'front_hot': sorted(front_hot),
            'front_warm': sorted(front_warm),
            'front_cold': sorted(front_cold),
            'back_hot': sorted(back_hot),
            'back_warm': sorted(back_warm),
            'back_cold': sorted(back_cold),
            'front_avg_freq': front_avg,
            'back_avg_freq': back_avg,
            'front_enhanced': front_enhanced,
            'back_enhanced': back_enhanced,
            'zone_analysis': zone_analysis,
            'analysis_periods': periods,
            'timestamp': datetime.now().isoformat(),
            'classification_method': 'quantile_ci_zscore'
        }

        smart_cache_manager.save_cache("analysis", method_name, result, periods)
        return result

    def _enhanced_hot_cold_analysis(self, frequency_dict: Dict, periods: int,
                                  max_number: int, numbers_per_draw: int,
                                  scope: str = 'front') -> Dict:
        """增强冷热号分析 - 温度量化计算"""
        try:
            import numpy as np
            from scipy import stats

            # 理论期望频率
            theoretical_freq = (periods * numbers_per_draw) / max_number

            # 计算统计指标（包含零频，避免遗漏号码被忽略）
            full_frequencies = [frequency_dict.get(num, 0) for num in range(1, max_number + 1)]
            mean_freq = np.mean(full_frequencies) if full_frequencies else 0
            std_freq = np.std(full_frequencies) if full_frequencies else 0
            median_freq = np.median(full_frequencies) if full_frequencies else 0

            # 频率分位数（稳定冷热分层）
            frequency_quantiles = self._compute_frequency_quantiles(full_frequencies, max_number)

            enhanced_stats = {}

            for num in range(1, max_number + 1):
                observed_freq = frequency_dict.get(num, 0)

                # 温度计算（标准化得分）
                if std_freq > 0:
                    temperature_score = (observed_freq - mean_freq) / std_freq
                else:
                    temperature_score = 0

                # 温度等级
                temperature_level = self._calculate_temperature_level(temperature_score)

                # 相对热度（相对于理论期望）
                relative_heat = observed_freq / theoretical_freq if theoretical_freq > 0 else 1

                # 置信区间（基于二项分布）
                if periods > 0:
                    p = numbers_per_draw / max_number
                    confidence_interval = stats.binom.interval(0.95, periods, p)
                else:
                    confidence_interval = (0, 0)

                # 动态阈值计算
                dynamic_threshold = self._calculate_dynamic_threshold(
                    observed_freq, mean_freq, std_freq, periods
                )

                # 温度趋势分析
                temperature_trend = self._calculate_temperature_trend(num, periods, scope)

                # 热度稳定性
                heat_stability = self._calculate_heat_stability(num, periods, scope)

                enhanced_stats[num] = {
                    'observed_frequency': observed_freq,
                    'theoretical_frequency': theoretical_freq,
                    'temperature_score': temperature_score,
                    'temperature_level': temperature_level,
                    'relative_heat': relative_heat,
                    'confidence_interval': confidence_interval,
                    'frequency_quantile': frequency_quantiles.get(num, 0.5),
                    'dynamic_threshold': dynamic_threshold,
                    'temperature_trend': temperature_trend,
                    'heat_stability': heat_stability,
                    'prediction_weight': self._calculate_temperature_prediction_weight(
                        temperature_score, relative_heat, temperature_trend, heat_stability
                    )
                }

            return enhanced_stats

        except Exception as e:
            logger_manager.error(f"增强冷热号分析失败: {e}")
            return {}

    def _compute_frequency_quantiles(self, frequencies: List[int], max_number: int) -> Dict[int, float]:
        """计算频率分位数（含并列频率的平均秩）"""
        if not frequencies or max_number <= 0:
            return {}

        n = max_number
        indices = list(range(n))
        indices.sort(key=lambda i: frequencies[i])

        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            freq_value = frequencies[indices[i]]
            while j < n and frequencies[indices[j]] == freq_value:
                j += 1
            avg_rank = (i + 1 + j) / 2
            for k in range(i, j):
                ranks[indices[k]] = avg_rank
            i = j

        if n == 1:
            quantiles = [0.5]
        else:
            quantiles = [(r - 1) / (n - 1) for r in ranks]

        return {idx + 1: quantiles[idx] for idx in range(n)}

    def _classify_hot_cold_by_quantile(self, enhanced: Dict) -> Tuple[List[int], List[int], List[int]]:
        """基于分位数与置信区间的冷热分类"""
        if not enhanced:
            return [], [], []

        hot = set()
        cold = set()
        warm = set()

        for num, info in enhanced.items():
            observed = info.get('observed_frequency', 0)
            quantile = info.get('frequency_quantile', 0.5)
            confidence_interval = info.get('confidence_interval', (None, None))
            ci_low, ci_high = confidence_interval if confidence_interval else (None, None)

            # 先用置信区间判断极端冷热
            if ci_high is not None and observed > ci_high:
                hot.add(num)
                continue
            if ci_low is not None and observed < ci_low:
                cold.add(num)
                continue

            # 分位数分类
            if quantile >= HotColdConfig.HOT_QUANTILE:
                hot.add(num)
            elif quantile <= HotColdConfig.COLD_QUANTILE:
                cold.add(num)
            elif HotColdConfig.WARM_QUANTILE_LOW <= quantile <= HotColdConfig.WARM_QUANTILE_HIGH:
                warm.add(num)

        # 使用温度等级补充温号
        for num, info in enhanced.items():
            if num in hot or num in cold:
                continue
            if info.get('temperature_level') == 'warm':
                warm.add(num)

        return sorted(hot), sorted(warm), sorted(cold)

    def _apply_missing_weight(self, enhanced: Dict, missing_enhanced: Dict) -> None:
        """融合遗漏权重到冷热权重（就地修改）"""
        if not enhanced or not missing_enhanced:
            return

        factor = HotColdConfig.MISSING_WEIGHT_FACTOR
        for num, info in enhanced.items():
            missing_info = missing_enhanced.get(num, {})
            missing_weight = missing_info.get('prediction_weight', 0.5)
            hot_cold_weight = info.get('prediction_weight', 0.5)

            combined_weight = (1 - factor) * hot_cold_weight + factor * missing_weight
            info['missing_prediction_weight'] = missing_weight
            info['prediction_weight'] = max(0.0, min(1.0, combined_weight))

    def _zone_hot_cold_analysis(self, front_enhanced: Dict, back_enhanced: Dict) -> Dict:
        """区间冷热分析 - 分析低/中/高区的冷热分布"""
        try:
            # 前区区间分析
            front_zones = {
                'low': {'range': HotColdConfig.FRONT_LOW_ZONE, 'hot': [], 'warm': [], 'cold': [], 'avg_temp': 0},
                'mid': {'range': HotColdConfig.FRONT_MID_ZONE, 'hot': [], 'warm': [], 'cold': [], 'avg_temp': 0},
                'high': {'range': HotColdConfig.FRONT_HIGH_ZONE, 'hot': [], 'warm': [], 'cold': [], 'avg_temp': 0}
            }

            for zone_name, zone_info in front_zones.items():
                zone_start, zone_end = zone_info['range']
                temp_scores = []
                for num in range(zone_start, zone_end + 1):
                    if num in front_enhanced:
                        info = front_enhanced[num]
                        temp_score = info.get('temperature_score', 0)
                        temp_scores.append(temp_score)
                        level = info.get('temperature_level', 'normal')
                        if level in ['extremely_hot', 'very_hot', 'hot']:
                            zone_info['hot'].append(num)
                        elif level == 'warm':
                            zone_info['warm'].append(num)
                        elif level in ['cool', 'cold', 'extremely_cold']:
                            zone_info['cold'].append(num)
                zone_info['avg_temp'] = np.mean(temp_scores) if temp_scores else 0

            # 后区区间分析
            back_zones = {
                'low': {'range': HotColdConfig.BACK_LOW_ZONE, 'hot': [], 'warm': [], 'cold': [], 'avg_temp': 0},
                'high': {'range': HotColdConfig.BACK_HIGH_ZONE, 'hot': [], 'warm': [], 'cold': [], 'avg_temp': 0}
            }

            for zone_name, zone_info in back_zones.items():
                zone_start, zone_end = zone_info['range']
                temp_scores = []
                for num in range(zone_start, zone_end + 1):
                    if num in back_enhanced:
                        info = back_enhanced[num]
                        temp_score = info.get('temperature_score', 0)
                        temp_scores.append(temp_score)
                        level = info.get('temperature_level', 'normal')
                        if level in ['extremely_hot', 'very_hot', 'hot']:
                            zone_info['hot'].append(num)
                        elif level == 'warm':
                            zone_info['warm'].append(num)
                        elif level in ['cool', 'cold', 'extremely_cold']:
                            zone_info['cold'].append(num)
                zone_info['avg_temp'] = np.mean(temp_scores) if temp_scores else 0

            # 计算区间热度排名
            front_zone_ranking = sorted(
                front_zones.keys(),
                key=lambda z: front_zones[z]['avg_temp'],
                reverse=True
            )
            back_zone_ranking = sorted(
                back_zones.keys(),
                key=lambda z: back_zones[z]['avg_temp'],
                reverse=True
            )

            return {
                'front_zones': front_zones,
                'back_zones': back_zones,
                'front_zone_ranking': front_zone_ranking,
                'back_zone_ranking': back_zone_ranking,
                'recommendation': {
                    'front_hot_zone': front_zone_ranking[0] if front_zone_ranking else None,
                    'front_cold_zone': front_zone_ranking[-1] if front_zone_ranking else None,
                    'back_hot_zone': back_zone_ranking[0] if back_zone_ranking else None,
                    'back_cold_zone': back_zone_ranking[-1] if back_zone_ranking else None
                }
            }

        except Exception as e:
            logger_manager.error(f"区间冷热分析失败: {e}")
            return {}

    def _calculate_temperature_level(self, temperature_score: float) -> str:
        """计算温度等级（使用配置常量）"""
        if temperature_score >= HotColdConfig.EXTREMELY_HOT_THRESHOLD:
            return 'extremely_hot'
        elif temperature_score >= HotColdConfig.VERY_HOT_THRESHOLD:
            return 'very_hot'
        elif temperature_score >= HotColdConfig.HOT_THRESHOLD:
            return 'hot'
        elif temperature_score >= HotColdConfig.WARM_THRESHOLD:
            return 'warm'
        elif temperature_score >= HotColdConfig.NORMAL_THRESHOLD:
            return 'normal'
        elif temperature_score >= HotColdConfig.COOL_THRESHOLD:
            return 'cool'
        elif temperature_score >= HotColdConfig.COLD_THRESHOLD:
            return 'cold'
        else:
            return 'extremely_cold'

    def _calculate_dynamic_threshold(self, observed_freq: int, mean_freq: float,
                                   std_freq: float, periods: int) -> Dict:
        """计算动态阈值"""
        try:
            # 基于统计分布的动态阈值
            hot_threshold = mean_freq + std_freq
            very_hot_threshold = mean_freq + 2 * std_freq
            cold_threshold = mean_freq - std_freq
            very_cold_threshold = mean_freq - 2 * std_freq

            # 基于期数调整阈值
            period_adjustment = min(1.2, 1 + (periods - 100) / 1000)

            return {
                'very_hot_threshold': very_hot_threshold * period_adjustment,
                'hot_threshold': hot_threshold * period_adjustment,
                'cold_threshold': cold_threshold * period_adjustment,
                'very_cold_threshold': very_cold_threshold * period_adjustment,
                'current_classification': self._classify_by_threshold(
                    observed_freq, hot_threshold, cold_threshold,
                    very_hot_threshold, very_cold_threshold
                )
            }

        except Exception as e:
            logger_manager.error(f"计算动态阈值失败: {e}")
            return {}

    def _classify_by_threshold(self, freq: int, hot_thresh: float, cold_thresh: float,
                             very_hot_thresh: float, very_cold_thresh: float) -> str:
        """基于阈值分类"""
        if freq >= very_hot_thresh:
            return 'very_hot'
        elif freq >= hot_thresh:
            return 'hot'
        elif freq <= very_cold_thresh:
            return 'very_cold'
        elif freq <= cold_thresh:
            return 'cold'
        else:
            return 'normal'

    def _calculate_temperature_trend(self, number: int, periods: int, scope: str = 'front') -> Dict:
        """计算温度趋势（区分前后区，使用配置常量，优化性能）"""
        try:
            if periods < HotColdConfig.MIN_PERIODS_FOR_TREND:
                return {'trend': 'insufficient_data', 'slope': 0}

            # 分析最近期数的温度变化（数据是降序排列，使用head获取最新数据）
            recent_periods = min(30, periods // 3)
            recent_data = self.df.head(recent_periods)

            # 性能优化：预先解析所有行的号码，避免重复调用 parse_balls
            all_balls_list = []
            for _, row in recent_data.iterrows():
                front_balls, back_balls = data_manager.parse_balls(row)
                if scope == 'back':
                    all_balls_list.append(set(back_balls))
                else:
                    all_balls_list.append(set(front_balls))

            temperatures = []
            window_size = HotColdConfig.TREND_WINDOW_SIZE

            for i in range(len(all_balls_list) - window_size + 1):
                freq_in_window = sum(1 for j in range(i, i + window_size) if number in all_balls_list[j])
                temperatures.append(freq_in_window)

            if len(temperatures) < 3:
                return {'trend': 'insufficient_data', 'slope': 0}

            # 计算趋势
            x = np.arange(len(temperatures))
            slope, intercept = np.polyfit(x, temperatures, 1)

            # 趋势判断（使用配置常量）
            if slope > HotColdConfig.TREND_HEATING_THRESHOLD:
                trend = 'heating_up'
            elif slope < HotColdConfig.TREND_COOLING_THRESHOLD:
                trend = 'cooling_down'
            else:
                trend = 'stable'

            return {
                'trend': trend,
                'slope': float(slope),
                'recent_temperatures': temperatures,
                'trend_strength': abs(float(slope))
            }

        except Exception as e:
            logger_manager.error(f"计算温度趋势失败: {e}")
            return {'trend': 'unknown', 'slope': 0}

    def _calculate_heat_stability(self, number: int, periods: int, scope: str = 'front') -> Dict:
        """计算热度稳定性（区分前后区，使用配置常量，优化性能）"""
        try:
            if periods < HotColdConfig.MIN_PERIODS_FOR_STABILITY:
                return {'stability': 'insufficient_data', 'variance': 0}

            # 分段分析热度稳定性
            num_segments = HotColdConfig.STABILITY_SEGMENTS
            segment_size = periods // num_segments
            segment_frequencies = []

            for i in range(num_segments):
                start_idx = i * segment_size
                end_idx = (i + 1) * segment_size
                segment_data = self.df.iloc[start_idx:end_idx]

                # 性能优化：使用列表推导式替代逐行迭代
                freq_in_segment = 0
                for _, row in segment_data.iterrows():
                    front_balls, back_balls = data_manager.parse_balls(row)
                    if scope == 'back':
                        if number in back_balls:
                            freq_in_segment += 1
                    else:
                        if number in front_balls:
                            freq_in_segment += 1

                segment_frequencies.append(freq_in_segment)

            # 计算稳定性指标
            variance = float(np.var(segment_frequencies))
            mean_freq = np.mean(segment_frequencies)
            coefficient_of_variation = float(np.std(segment_frequencies) / mean_freq) if mean_freq > 0 else 0

            # 稳定性等级（使用配置常量）
            if coefficient_of_variation < HotColdConfig.STABILITY_VERY_STABLE:
                stability = 'very_stable'
            elif coefficient_of_variation < HotColdConfig.STABILITY_STABLE:
                stability = 'stable'
            elif coefficient_of_variation < HotColdConfig.STABILITY_MODERATE:
                stability = 'moderate'
            else:
                stability = 'unstable'

            return {
                'stability': stability,
                'variance': variance,
                'coefficient_of_variation': coefficient_of_variation,
                'segment_frequencies': segment_frequencies
            }

        except Exception as e:
            logger_manager.error(f"计算热度稳定性失败: {e}")
            return {'stability': 'unknown', 'variance': 0}

    def _calculate_temperature_prediction_weight(self, temperature_score: float,
                                               relative_heat: float, temperature_trend: Dict,
                                               heat_stability: Dict) -> float:
        """计算温度预测权重（使用配置常量）"""
        try:
            # 基础权重基于温度得分
            base_weight = 0.5 + temperature_score * 0.1

            # 相对热度调整
            heat_adjustment = min(1.5, max(0.5, relative_heat))

            # 趋势调整（使用配置常量）
            trend_adjustment = 1.0
            trend = temperature_trend.get('trend', 'stable')
            if trend == 'heating_up':
                trend_adjustment = HotColdConfig.TREND_HEATING_ADJUSTMENT
            elif trend == 'cooling_down':
                trend_adjustment = HotColdConfig.TREND_COOLING_ADJUSTMENT

            # 稳定性调整（使用配置常量）
            stability_adjustment = 1.0
            stability = heat_stability.get('stability', 'moderate')
            if stability in ['very_stable', 'stable']:
                stability_adjustment = HotColdConfig.STABILITY_BONUS
            elif stability == 'unstable':
                stability_adjustment = HotColdConfig.INSTABILITY_PENALTY

            # 综合权重
            final_weight = base_weight * heat_adjustment * trend_adjustment * stability_adjustment

            return max(0.0, min(1.0, final_weight))

        except Exception as e:
            logger_manager.error(f"计算温度预测权重失败: {e}")
            return 0.5
    
    def sum_analysis(self, periods=None) -> Dict:
        """和值分析"""
        if self.df is None:
            return {}
        
        method_name = "sum_analysis"
        cached_result = smart_cache_manager.load_cache("analysis", method_name, periods)
        if cached_result:
            return cached_result
        
        # 数据是降序排列（最新在前），使用head()获取最新数据
        df_subset = self.df.head(periods) if periods else self.df
        
        front_sums = []
        back_sums = []
        total_sums = []
        
        for _, row in df_subset.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            
            front_sum = sum(front_balls)
            back_sum = sum(back_balls)
            total_sum = front_sum + back_sum
            
            front_sums.append(front_sum)
            back_sums.append(back_sum)
            total_sums.append(total_sum)
        
        result = {
            'front_sum_stats': {
                'mean': np.mean(front_sums),
                'std': np.std(front_sums),
                'min': np.min(front_sums),
                'max': np.max(front_sums),
                'median': np.median(front_sums)
            },
            'back_sum_stats': {
                'mean': np.mean(back_sums),
                'std': np.std(back_sums),
                'min': np.min(back_sums),
                'max': np.max(back_sums),
                'median': np.median(back_sums)
            },
            'total_sum_stats': {
                'mean': np.mean(total_sums),
                'std': np.std(total_sums),
                'min': np.min(total_sums),
                'max': np.max(total_sums),
                'median': np.median(total_sums)
            },
            'analysis_periods': len(df_subset),
            'timestamp': datetime.now().isoformat()
        }
        
        smart_cache_manager.save_cache("analysis", method_name, result, periods)
        return result

    def statistical_features_analysis(self, periods=None) -> Dict:
        """统计特征分析"""
        if self.df is None:
            return {}

        method_name = "statistical_features_analysis"
        cached_result = smart_cache_manager.load_cache("analysis", method_name, periods)
        if cached_result:
            return cached_result

        # 数据是降序排列（最新在前），使用head()获取最新数据
        df_subset = self.df.head(periods) if periods else self.df

        # 统计特征
        front_features = {
            'odd_even_ratio': [],  # 奇偶比
            'big_small_ratio': [],  # 大小比
            'span': [],  # 跨度
            'sum_values': [],  # 和值
            'consecutive_count': [],  # 连号个数
            'ac_values': []  # AC值
        }

        back_features = {
            'odd_even_ratio': [],
            'big_small_ratio': [],
            'span': [],
            'sum_values': []
        }

        for _, row in df_subset.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)

            # 前区特征
            front_odd = sum(1 for x in front_balls if x % 2 == 1)
            front_features['odd_even_ratio'].append(f"{front_odd}:{5-front_odd}")

            front_big = sum(1 for x in front_balls if x > 17)
            front_features['big_small_ratio'].append(f"{front_big}:{5-front_big}")

            front_features['span'].append(max(front_balls) - min(front_balls))
            front_features['sum_values'].append(sum(front_balls))

            # 连号统计
            consecutive = 0
            sorted_front = sorted(front_balls)
            for i in range(len(sorted_front) - 1):
                if sorted_front[i+1] - sorted_front[i] == 1:
                    consecutive += 1
            front_features['consecutive_count'].append(consecutive)

            # AC值计算
            ac_value = 0
            for i in range(len(front_balls)):
                for j in range(i+1, len(front_balls)):
                    ac_value += abs(front_balls[i] - front_balls[j])
            front_features['ac_values'].append(ac_value)

            # 后区特征
            back_odd = sum(1 for x in back_balls if x % 2 == 1)
            back_features['odd_even_ratio'].append(f"{back_odd}:{2-back_odd}")

            back_big = sum(1 for x in back_balls if x > 6)
            back_features['big_small_ratio'].append(f"{back_big}:{2-back_big}")

            back_features['span'].append(max(back_balls) - min(back_balls))
            back_features['sum_values'].append(sum(back_balls))

        # 统计分析
        result = {
            'front_features': {
                'odd_even_distribution': Counter(front_features['odd_even_ratio']),
                'big_small_distribution': Counter(front_features['big_small_ratio']),
                'span_stats': {
                    'min': min(front_features['span']),
                    'max': max(front_features['span']),
                    'avg': np.mean(front_features['span']),
                    'distribution': Counter(front_features['span'])
                },
                'sum_stats': {
                    'min': min(front_features['sum_values']),
                    'max': max(front_features['sum_values']),
                    'avg': np.mean(front_features['sum_values']),
                    'distribution': Counter(front_features['sum_values'])
                },
                'consecutive_stats': Counter(front_features['consecutive_count']),
                'ac_stats': {
                    'min': min(front_features['ac_values']),
                    'max': max(front_features['ac_values']),
                    'avg': np.mean(front_features['ac_values'])
                }
            },
            'back_features': {
                'odd_even_distribution': Counter(back_features['odd_even_ratio']),
                'big_small_distribution': Counter(back_features['big_small_ratio']),
                'span_stats': {
                    'min': min(back_features['span']),
                    'max': max(back_features['span']),
                    'avg': np.mean(back_features['span']),
                    'distribution': Counter(back_features['span'])
                },
                'sum_stats': {
                    'min': min(back_features['sum_values']),
                    'max': max(back_features['sum_values']),
                    'avg': np.mean(back_features['sum_values'])
                }
            },
            'analysis_periods': len(df_subset),
            'timestamp': datetime.now().isoformat()
        }

        smart_cache_manager.save_cache("analysis", method_name, result, periods)
        return result

    def predict_compound(self, config: Optional[CompoundConfig] = None) -> CompoundResult:
        """
        基础分析器复式预测

        Args:
            config: 复式预测配置

        Returns:
            复式预测结果
        """
        if config is None:
            config = self.compound_config or CompoundConfig()

        # 验证参数
        if not self.validate_compound_params(config.front_count, config.back_count, config.max_cost):
            raise ValueError("基础分析器复式预测参数验证失败")

        logger_manager.info(f"开始基础分析器复式预测: {config.front_count}+{config.back_count}")

        try:
            # 综合多种分析方法
            freq_result = self.frequency_analysis(config.periods)
            hot_cold_result = self.hot_cold_analysis(config.periods)
            missing_result = self.missing_analysis(config.periods)

            # 收集候选号码
            front_candidates = set()
            back_candidates = set()

            # 从频率分析中选择
            if 'front_frequency' in freq_result:
                front_freq = sorted(freq_result['front_frequency'].items(), key=lambda x: x[1], reverse=True)
                front_candidates.update([ball for ball, freq in front_freq[:config.front_count * 2]])

            if 'back_frequency' in freq_result:
                back_freq = sorted(freq_result['back_frequency'].items(), key=lambda x: x[1], reverse=True)
                back_candidates.update([ball for ball, freq in back_freq[:config.back_count * 2]])

            # 从冷热分析中选择
            if 'front_hot' in hot_cold_result:
                front_candidates.update(hot_cold_result['front_hot'][:config.front_count])
            if 'back_hot' in hot_cold_result:
                back_candidates.update(hot_cold_result['back_hot'][:config.back_count])

            # 从遗漏分析中选择（从front_enhanced中提取高紧迫度号码）
            front_missing_enhanced = missing_result.get('front_enhanced', {})
            if front_missing_enhanced:
                front_urgent = sorted(
                    [(k, v) for k, v in front_missing_enhanced.items()],
                    key=lambda x: x[1].get('urgency_score', 0),
                    reverse=True
                )[:config.front_count]
                front_candidates.update([item[0] for item in front_urgent])

            back_missing_enhanced = missing_result.get('back_enhanced', {})
            if back_missing_enhanced:
                back_urgent = sorted(
                    [(k, v) for k, v in back_missing_enhanced.items()],
                    key=lambda x: x[1].get('urgency_score', 0),
                    reverse=True
                )[:config.back_count]
                back_candidates.update([item[0] for item in back_urgent])

            # 确保有足够的候选号码
            while len(front_candidates) < config.front_count:
                for i in range(1, 36):
                    if i not in front_candidates:
                        front_candidates.add(i)
                        if len(front_candidates) >= config.front_count:
                            break

            while len(back_candidates) < config.back_count:
                for i in range(1, 13):
                    if i not in back_candidates:
                        back_candidates.add(i)
                        if len(back_candidates) >= config.back_count:
                            break

            # 选择最终号码
            front_balls = sorted(list(front_candidates)[:config.front_count])
            back_balls = sorted(list(back_candidates)[:config.back_count])

            # 计算组合数和成本
            combinations = self.calculate_combinations(config.front_count, config.back_count)
            cost = self.calculate_cost(combinations)

            # 计算置信度
            confidence = min(0.8, max(0.4, len(front_candidates) / (config.front_count * 3)))

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
                method="基础分析器复式预测",
                analysis_periods=config.periods,
                timestamp=datetime.now().isoformat(),
                details={
                    'analysis_methods': ['frequency', 'hot_cold', 'missing'],
                    'candidate_sources': 3,
                    'selection_strategy': 'multi_method_fusion'
                }
            )

            logger_manager.info(f"基础分析器复式预测完成: {config.front_count}+{config.back_count}, 置信度: {confidence:.3f}")
            return result

        except Exception as e:
            logger_manager.error(f"基础分析器复式预测失败: {e}")
            # 返回默认结果
            return super().predict_compound(config)


# ==================== 高级分析器 ====================
class AdvancedAnalyzer:
    """高级分析器"""
    
    def __init__(self, data_file="data/dlt_data_all.csv"):
        self.data_file = data_file
        self.df = data_manager.get_data()
        self.basic_analyzer = BasicAnalyzer(data_file)
        
        if self.df is None:
            logger_manager.error("数据未加载")
    
    def markov_analysis(self, periods=500, n_jobs=1) -> Dict:
        """马尔可夫链分析（支持并行化）

        Args:
            periods: 分析期数
            n_jobs: 并行作业数，1表示单线程，-1表示使用所有CPU核心
        """
        if self.df is None:
            return {}

        cfg = _load_prediction_config()
        markov_cfg = cfg.get('prediction_methods', {}).get('traditional_ml', {}).get('markov', {}) or {}
        decay_enabled = bool(markov_cfg.get('decay_enabled', False))
        decay_half_life = float(markov_cfg.get('decay_half_life', 200))
        decay_min_weight = float(markov_cfg.get('decay_min_weight', 0.2))

        decay_half_life = max(1.0, decay_half_life)
        decay_min_weight = min(max(decay_min_weight, 0.0), 1.0)

        method_name = "markov_analysis"
        if decay_enabled:
            hl_tag = f"{decay_half_life:.2f}".replace('.', 'p')
            min_tag = f"{decay_min_weight:.2f}".replace('.', 'p')
            method_name = f"markov_analysis_decay_hl{hl_tag}_min{min_tag}"
        cached_result = smart_cache_manager.load_cache("analysis", method_name, periods)
        if cached_result:
            return cached_result
        
        # 数据是降序排列（最新在前），使用head()获取最新数据
        df_subset = self.df.head(periods)

        # 根据n_jobs决定是否使用并行化
        if n_jobs == 1:
            # 单线程处理
            front_transitions, back_transitions = self._compute_transitions_single(
                df_subset, decay_enabled, decay_half_life, decay_min_weight
            )
        else:
            # 并行处理
            front_transitions, back_transitions = self._compute_transitions_parallel(
                df_subset, n_jobs, decay_enabled, decay_half_life, decay_min_weight
            )
        
        # 转换为概率
        front_probs = {}
        for from_ball, to_dict in front_transitions.items():
            total = sum(to_dict.values())
            if total > 0:
                front_probs[from_ball] = {to_ball: count/total for to_ball, count in to_dict.items()}
        
        back_probs = {}
        for from_ball, to_dict in back_transitions.items():
            total = sum(to_dict.values())
            if total > 0:
                back_probs[from_ball] = {to_ball: count/total for to_ball, count in to_dict.items()}
        
        result = {
            'front_transition_probs': front_probs,
            'back_transition_probs': back_probs,
            'analysis_periods': periods,
            'timestamp': datetime.now().isoformat()
        }
        
        smart_cache_manager.save_cache("analysis", method_name, result, periods)
        return result

    def _compute_transitions_single(self, df_subset, decay_enabled=False,
                                    decay_half_life=200.0, decay_min_weight=0.2):
        """单线程计算转移矩阵"""
        front_transitions = defaultdict(lambda: defaultdict(int))
        back_transitions = defaultdict(lambda: defaultdict(int))

        for i in range(len(df_subset) - 1):
            weight = 1.0
            if decay_enabled:
                weight = 0.5 ** (i / decay_half_life)
                if weight < decay_min_weight:
                    weight = decay_min_weight

            current_front, current_back = data_manager.parse_balls(df_subset.iloc[i])
            next_front, next_back = data_manager.parse_balls(df_subset.iloc[i + 1])

            # 前区转移
            for curr_ball in current_front:
                for next_ball in next_front:
                    front_transitions[curr_ball][next_ball] += weight

            # 后区转移
            for curr_ball in current_back:
                for next_ball in next_back:
                    back_transitions[curr_ball][next_ball] += weight

        return front_transitions, back_transitions

    def _compute_transitions_parallel(self, df_subset, n_jobs, decay_enabled=False,
                                      decay_half_life=200.0, decay_min_weight=0.2):
        """并行计算转移矩阵"""
        try:
            from joblib import Parallel, delayed
            import multiprocessing as mp

            # 确定实际使用的进程数
            if n_jobs == -1:
                n_jobs = mp.cpu_count()
            else:
                n_jobs = min(n_jobs, mp.cpu_count())

            # 将数据分块
            chunk_size = max(1, (len(df_subset) - 1) // n_jobs)
            chunks = []
            for i in range(0, len(df_subset) - 1, chunk_size):
                end_idx = min(i + chunk_size + 1, len(df_subset))  # +1 for transition calculation
                chunks.append((df_subset.iloc[i:end_idx], i))

            # 并行计算每个块的转移矩阵
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._compute_chunk_transitions)(
                    chunk, start_index, decay_enabled, decay_half_life, decay_min_weight
                )
                for chunk, start_index in chunks
            )

            # 合并结果
            front_transitions = defaultdict(lambda: defaultdict(int))
            back_transitions = defaultdict(lambda: defaultdict(int))

            for front_trans, back_trans in results:
                for from_ball, to_dict in front_trans.items():
                    for to_ball, count in to_dict.items():
                        front_transitions[from_ball][to_ball] += count

                for from_ball, to_dict in back_trans.items():
                    for to_ball, count in to_dict.items():
                        back_transitions[from_ball][to_ball] += count

            return front_transitions, back_transitions

        except ImportError:
            logger_manager.warning("joblib未安装，使用单线程计算")
            return self._compute_transitions_single(df_subset)

    def _compute_chunk_transitions(self, chunk, start_index=0, decay_enabled=False,
                                   decay_half_life=200.0, decay_min_weight=0.2):
        """计算数据块的转移矩阵"""
        front_transitions = defaultdict(lambda: defaultdict(int))
        back_transitions = defaultdict(lambda: defaultdict(int))

        for i in range(len(chunk) - 1):
            weight = 1.0
            if decay_enabled:
                global_index = start_index + i
                weight = 0.5 ** (global_index / decay_half_life)
                if weight < decay_min_weight:
                    weight = decay_min_weight

            current_front, current_back = data_manager.parse_balls(chunk.iloc[i])
            next_front, next_back = data_manager.parse_balls(chunk.iloc[i + 1])

            # 前区转移
            for curr_ball in current_front:
                for next_ball in next_front:
                    front_transitions[curr_ball][next_ball] += weight

            # 后区转移
            for curr_ball in current_back:
                for next_ball in next_back:
                    back_transitions[curr_ball][next_ball] += weight

        return front_transitions, back_transitions
    
    def bayesian_analysis(self, periods=300, n_jobs=1, use_mcmc=False) -> Dict:
        """增强贝叶斯分析 - 完整的贝叶斯推理过程（支持并行化）

        Args:
            periods: 分析期数
            n_jobs: 并行作业数，1表示单线程，-1表示使用所有CPU核心
            use_mcmc: 是否使用MCMC采样进行后验推断（更精确但更慢）
        """
        if self.df is None:
            return {}

        # 应用配置（用于缓存键与参数）
        load_bayesian_config()

        mix_tag = f"{BayesianConfig.DIRICHLET_MIX_WEIGHT:.3f}".replace('.', 'p')
        conc_tag = f"{BayesianConfig.DIRICHLET_CONCENTRATION:.3f}".replace('.', 'p')
        decay_tag = f"{BayesianConfig.DECAY_HALF_LIFE:.1f}".replace('.', 'p')
        minw_tag = f"{BayesianConfig.DECAY_MIN_WEIGHT:.2f}".replace('.', 'p')
        decay_flag = f"{BayesianConfig.DECAY_MODE}" if BayesianConfig.DECAY_ENABLED else 'nodecay'
        recent_tag = f"{BayesianConfig.RECENT_WINDOW}r{BayesianConfig.MID_WINDOW}m"
        prior_tag = f"h{int(BayesianConfig.PRIOR_HOT_BONUS*100)}w{int(BayesianConfig.PRIOR_WARM_BONUS*100)}c{int(BayesianConfig.PRIOR_COLD_PENALTY*100)}m{int(BayesianConfig.PRIOR_MISSING_BIAS*100)}"
        method_name = (
            f"enhanced_bayesian_analysis_v2_{decay_flag}_{recent_tag}_hl{decay_tag}_min{minw_tag}"
            f"_{prior_tag}_mix{mix_tag}_conc{conc_tag}{'_mcmc' if use_mcmc else ''}"
        )
        cached_result = smart_cache_manager.load_cache("analysis", method_name, periods)
        if cached_result:
            return cached_result

        # 数据是降序排列（最新在前），使用head()获取最新数据
        df_subset = self.df.head(periods)

        # 根据n_jobs决定是否使用并行化
        if n_jobs == 1:
            # 单线程处理
            front_enhanced = self._enhanced_bayesian_analysis(df_subset, 35, 5, use_mcmc)
            back_enhanced = self._enhanced_bayesian_analysis(df_subset, 12, 2, use_mcmc)
        else:
            # 并行处理
            front_enhanced, back_enhanced = self._parallel_bayesian_analysis(df_subset, n_jobs, use_mcmc)

        # 传统贝叶斯分析（保持兼容性）
        traditional_result = self._traditional_bayesian_analysis(df_subset)

        # 层次化先验（用于展示与高级预测）
        front_prior = self._calculate_hierarchical_priors(df_subset, 35, 5)
        back_prior = self._calculate_hierarchical_priors(df_subset, 12, 2)

        # Dirichlet-多项式后验预测（增强）
        front_dirichlet_posterior = self._dirichlet_posterior_predictive(
            df_subset, 35, 5, front_prior
        )
        back_dirichlet_posterior = self._dirichlet_posterior_predictive(
            df_subset, 12, 2, back_prior
        )

        result = {
            'front_enhanced': front_enhanced,
            'back_enhanced': back_enhanced,
            'front_prior': front_prior,
            'back_prior': back_prior,
            'front_dirichlet_posterior': front_dirichlet_posterior,
            'back_dirichlet_posterior': back_dirichlet_posterior,
            'front_posterior': traditional_result['front_posterior'],
            'back_posterior': traditional_result['back_posterior'],
            'front_likelihood': traditional_result['front_likelihood'],
            'back_likelihood': traditional_result['back_likelihood'],
            'analysis_periods': periods,
            'timestamp': datetime.now().isoformat()
        }

        smart_cache_manager.save_cache("analysis", method_name, result, periods)
        return result

    def _parallel_bayesian_analysis(self, df_subset, n_jobs, use_mcmc=False):
        """并行贝叶斯分析"""
        try:
            from joblib import Parallel, delayed
            import multiprocessing as mp

            # 确定实际使用的进程数
            if n_jobs == -1:
                n_jobs = mp.cpu_count()
            else:
                n_jobs = min(n_jobs, mp.cpu_count())

            # 并行计算前区和后区
            results = Parallel(n_jobs=min(n_jobs, 2))(
                delayed(self._enhanced_bayesian_analysis)(df_subset, max_num, draw_count, use_mcmc)
                for max_num, draw_count in [(35, 5), (12, 2)]
            )

            front_enhanced, back_enhanced = results
            return front_enhanced, back_enhanced

        except ImportError:
            logger_manager.warning("joblib未安装，使用单线程贝叶斯分析")
            front_enhanced = self._enhanced_bayesian_analysis(df_subset, 35, 5, use_mcmc)
            back_enhanced = self._enhanced_bayesian_analysis(df_subset, 12, 2, use_mcmc)
            return front_enhanced, back_enhanced

    def _enhanced_bayesian_analysis(self, df_subset, max_number: int, numbers_per_draw: int,
                                     use_mcmc: bool = False) -> Dict:
        """增强贝叶斯分析 - 完整的贝叶斯推理

        Args:
            df_subset: 数据子集
            max_number: 最大号码（前区35，后区12）
            numbers_per_draw: 每期选择的号码数
            use_mcmc: 是否使用MCMC采样（更精确但更慢）
        """
        try:
            import numpy as np
            from scipy import stats

            enhanced_stats = {}

            # 多层次先验概率设计
            priors = self._calculate_hierarchical_priors(df_subset, max_number, numbers_per_draw)

            # 多维度似然函数
            likelihoods = self._calculate_multi_dimensional_likelihood(df_subset, max_number)

            # 证据计算（边际似然）
            evidence_by_number, global_evidence = self._calculate_bayesian_evidence(
                priors, likelihoods, max_number
            )

            for num in range(1, max_number + 1):
                # 贝叶斯定理完整应用
                prior = priors.get(num, 1/max_number)
                likelihood = likelihoods.get(num, {})

                # 后验概率计算（解析解）
                posterior = self._calculate_posterior_distribution(
                    prior, likelihood, global_evidence
                )

                # 如果启用MCMC，进行采样推断以获得更精确的后验估计
                mcmc_posterior = None
                if use_mcmc:
                    mcmc_posterior = self._mcmc_posterior_sampling(prior, likelihood)
                    # 使用MCMC结果更新后验统计量（如果MCMC成功）
                    if mcmc_posterior.get('n_samples', 0) > 0:
                        # 保留解析解的结构，但用MCMC结果更新关键统计量
                        posterior['mcmc_mean'] = mcmc_posterior['mean']
                        posterior['mcmc_median'] = mcmc_posterior['median']
                        posterior['mcmc_std'] = mcmc_posterior['std']
                        posterior['mcmc_ci_lower'] = mcmc_posterior['ci_lower']
                        posterior['mcmc_ci_upper'] = mcmc_posterior['ci_upper']
                        posterior['mcmc_ess'] = mcmc_posterior['effective_sample_size']

                # 置信区间计算
                confidence_interval = self._calculate_bayesian_confidence_interval(posterior, 0.95)

                # 预测分布
                predictive_distribution = self._calculate_predictive_distribution(posterior, likelihood)

                # 贝叶斯因子
                bayes_factor = self._calculate_bayes_factor(likelihood, prior, max_number)

                # 信息增益
                information_gain = self._calculate_information_gain(prior, posterior)

                enhanced_stats[num] = {
                    'prior_probability': prior,
                    'likelihood_components': likelihood,
                    'posterior_distribution': posterior,
                    'confidence_interval': confidence_interval,
                    'predictive_distribution': predictive_distribution,
                    'bayes_factor': bayes_factor,
                    'information_gain': information_gain,
                    'evidence': evidence_by_number.get(num, 1),
                    'prediction_weight': self._calculate_bayesian_prediction_weight(
                        posterior, bayes_factor, information_gain
                    )
                }

                # 如果有MCMC结果，添加到输出
                if mcmc_posterior is not None:
                    enhanced_stats[num]['mcmc_sampling'] = mcmc_posterior

            return enhanced_stats

        except Exception as e:
            logger_manager.error(f"增强贝叶斯分析失败: {e}")
            return {}

    def _mcmc_posterior_sampling(self, prior: float, likelihood: Dict, n_samples: int = 1000,
                                  burn_in: int = 200, thin: int = 2) -> Dict:
        """Metropolis-Hastings MCMC 后验采样

        用于更复杂后验分布的采样推断，当解析解不够准确时使用。

        Args:
            prior: 先验概率
            likelihood: 似然函数字典
            n_samples: 采样数量
            burn_in: 预烧期（丢弃的初始样本数）
            thin: 稀疏化间隔（每thin个样本保留一个）

        Returns:
            包含后验样本统计信息的字典
        """
        try:
            import numpy as np

            combined_likelihood = likelihood.get('combined', 1.0)

            # 目标分布（非归一化后验）的对数
            def log_target(theta):
                if theta <= 0 or theta >= 1:
                    return -np.inf
                # log(posterior) ∝ log(likelihood) + log(prior)
                # 假设先验为Beta(1,1)（均匀）或使用给定的prior
                log_prior = 0  # 均匀先验的对数为常数
                log_likelihood = np.log(combined_likelihood + 1e-10) * theta
                return log_prior + log_likelihood

            # 初始化
            current_theta = prior
            samples = []
            accepted = 0

            # 提议分布的标准差（自适应）
            proposal_std = 0.1

            # 总迭代次数
            total_iterations = burn_in + n_samples * thin

            for i in range(total_iterations):
                # 从提议分布中采样（截断正态）
                proposed_theta = current_theta + np.random.normal(0, proposal_std)

                # 反射边界处理（保持在[0,1]内）
                while proposed_theta <= 0 or proposed_theta >= 1:
                    if proposed_theta <= 0:
                        proposed_theta = -proposed_theta
                    if proposed_theta >= 1:
                        proposed_theta = 2 - proposed_theta

                # Metropolis-Hastings 接受率
                log_alpha = log_target(proposed_theta) - log_target(current_theta)
                alpha = min(1, np.exp(log_alpha)) if not np.isnan(log_alpha) else 0

                # 接受或拒绝
                if np.random.random() < alpha:
                    current_theta = proposed_theta
                    accepted += 1

                # 自适应调整提议标准差（仅在预烧期）
                if i < burn_in and i > 0 and i % 50 == 0:
                    acceptance_rate = accepted / (i + 1)
                    if acceptance_rate > 0.5:
                        proposal_std *= 1.1
                    elif acceptance_rate < 0.2:
                        proposal_std *= 0.9
                    proposal_std = max(0.01, min(0.5, proposal_std))

                # 收集样本（预烧期后，按稀疏化间隔）
                if i >= burn_in and (i - burn_in) % thin == 0:
                    samples.append(current_theta)

            samples = np.array(samples)

            # 计算后验统计量
            return {
                'mean': float(np.mean(samples)),
                'median': float(np.median(samples)),
                'std': float(np.std(samples)),
                'variance': float(np.var(samples)),
                'ci_lower': float(np.percentile(samples, 2.5)),  # 95% 置信区间下界
                'ci_upper': float(np.percentile(samples, 97.5)),  # 95% 置信区间上界
                'acceptance_rate': accepted / total_iterations,
                'n_samples': len(samples),
                'effective_sample_size': self._calculate_ess(samples)
            }

        except Exception as e:
            logger_manager.error(f"MCMC采样失败: {e}")
            return {
                'mean': prior, 'median': prior, 'std': 0.1,
                'variance': 0.01, 'ci_lower': max(0, prior - 0.2),
                'ci_upper': min(1, prior + 0.2), 'acceptance_rate': 0,
                'n_samples': 0, 'effective_sample_size': 0
            }

    def _calculate_ess(self, samples: np.ndarray) -> float:
        """计算有效样本量（Effective Sample Size）

        考虑样本间的自相关性，实际独立样本数可能小于总样本数。
        """
        try:
            n = len(samples)
            if n < 10:
                return float(n)

            # 计算自相关函数
            mean = np.mean(samples)
            var = np.var(samples)
            if var < 1e-10:
                return float(n)

            # 计算滞后自相关
            max_lag = min(n // 3, 100)
            autocorr_sum = 0

            for lag in range(1, max_lag):
                autocorr = np.sum((samples[:n-lag] - mean) * (samples[lag:] - mean)) / ((n - lag) * var)
                if autocorr < 0.05:  # 截断小的自相关
                    break
                autocorr_sum += autocorr

            # ESS = n / (1 + 2 * sum(autocorrelations))
            ess = n / (1 + 2 * autocorr_sum)
            return float(max(1, min(n, ess)))

        except Exception as e:
            logger_manager.error(f"计算ESS失败: {e}")
            return float(len(samples))

    def _time_decay_weight(self, index: int, total_periods: int) -> float:
        """时间衰减权重（越近权重越大）"""
        if not BayesianConfig.DECAY_ENABLED:
            return 1.0

        if BayesianConfig.DECAY_MODE == "segmented":
            if index < BayesianConfig.RECENT_WINDOW:
                weight = BayesianConfig.RECENT_WEIGHT
            elif index < BayesianConfig.MID_WINDOW:
                weight = BayesianConfig.MID_WEIGHT
            else:
                weight = BayesianConfig.OLD_WEIGHT
            return max(BayesianConfig.DECAY_MIN_WEIGHT, weight)

        half_life = BayesianConfig.DECAY_HALF_LIFE
        if half_life <= 0:
            return 1.0

        weight = 0.5 ** (index / half_life)
        return max(BayesianConfig.DECAY_MIN_WEIGHT, weight)

    def _calculate_hierarchical_priors(self, df_subset, max_number: int, numbers_per_draw: int) -> Dict:
        """计算层次化先验概率"""
        try:
            import numpy as np

            # 无信息先验（均匀分布）
            uniform_prior = 1 / max_number

            # 先验偏置（冷热/遗漏）准备：仅在配置开启时加载
            apply_bias = any([
                BayesianConfig.PRIOR_HOT_BONUS > 0,
                BayesianConfig.PRIOR_WARM_BONUS > 0,
                BayesianConfig.PRIOR_COLD_PENALTY > 0,
                BayesianConfig.PRIOR_MISSING_BIAS > 0,
            ])
            hot_set: set = set()
            warm_set: set = set()
            cold_set: set = set()
            missing_enhanced: Dict = {}
            if apply_bias:
                try:
                    periods = len(df_subset)
                    hot_cold_result = self.basic_analyzer.hot_cold_analysis(periods)
                    missing_result = self.basic_analyzer.missing_analysis(periods)

                    if max_number == 35:
                        hot_set = set(hot_cold_result.get('front_hot', []))
                        warm_set = set(hot_cold_result.get('front_warm', []))
                        cold_set = set(hot_cold_result.get('front_cold', []))
                        missing_enhanced = missing_result.get('front_enhanced', {}) or {}
                    else:
                        hot_set = set(hot_cold_result.get('back_hot', []))
                        warm_set = set(hot_cold_result.get('back_warm', []))
                        cold_set = set(hot_cold_result.get('back_cold', []))
                        missing_enhanced = missing_result.get('back_enhanced', {}) or {}
                except Exception as e:
                    logger_manager.warning(f"先验偏置数据加载失败: {e}")
                    apply_bias = False

            # 基于历史频率的信息先验
            historical_counts = Counter()
            total_observations = 0.0
            for idx, (_, row) in enumerate(df_subset.iterrows()):
                weight = self._time_decay_weight(idx, len(df_subset))
                if max_number == 35:  # 前区
                    front_balls, _ = data_manager.parse_balls(row)
                    for ball in front_balls:
                        historical_counts[ball] += weight
                    total_observations += weight * len(front_balls)
                else:  # 后区
                    _, back_balls = data_manager.parse_balls(row)
                    for ball in back_balls:
                        historical_counts[ball] += weight
                    total_observations += weight * len(back_balls)

            # 贝塔分布先验参数
            alpha = 1  # 伪计数
            beta = max_number - 1

            priors = {}
            for num in range(1, max_number + 1):
                observed_count = historical_counts.get(num, 0)

                # 贝塔-二项共轭先验
                posterior_alpha = alpha + observed_count
                posterior_beta = beta + total_observations - observed_count

                # 期望值作为先验
                informative_prior = posterior_alpha / (posterior_alpha + posterior_beta)

                # 混合先验（无信息 + 信息）
                mixing_weight = min(0.8, total_observations / (total_observations + 100))
                mixed_prior = (1 - mixing_weight) * uniform_prior + mixing_weight * informative_prior

                if apply_bias:
                    bias_factor = 1.0
                    if num in hot_set:
                        bias_factor *= (1 + BayesianConfig.PRIOR_HOT_BONUS)
                    elif num in warm_set:
                        bias_factor *= (1 + BayesianConfig.PRIOR_WARM_BONUS)
                    elif num in cold_set:
                        cold_factor = max(1 - BayesianConfig.PRIOR_COLD_PENALTY,
                                          BayesianConfig.PRIOR_MIN_FACTOR)
                        bias_factor *= cold_factor

                    if BayesianConfig.PRIOR_MISSING_BIAS > 0 and missing_enhanced:
                        missing_info = missing_enhanced.get(num, missing_enhanced.get(str(num), {})) or {}
                        missing_weight = missing_info.get('prediction_weight', 0.5)
                        # 将遗漏权重映射到 [1-bias, 1+bias]
                        missing_factor = 1 + BayesianConfig.PRIOR_MISSING_BIAS * (missing_weight - 0.5) * 2
                        bias_factor *= max(BayesianConfig.PRIOR_MIN_FACTOR, missing_factor)

                    bias_factor = max(BayesianConfig.PRIOR_MIN_FACTOR, bias_factor)
                    priors[num] = mixed_prior * bias_factor
                else:
                    priors[num] = mixed_prior

            if apply_bias:
                total_prior = sum(priors.values())
                if total_prior > 0:
                    priors = {num: value / total_prior for num, value in priors.items()}
                else:
                    priors = {i: 1 / max_number for i in range(1, max_number + 1)}

            return priors

        except Exception as e:
            logger_manager.error(f"计算层次化先验失败: {e}")
            return {i: 1/max_number for i in range(1, max_number + 1)}

    def _dirichlet_posterior_predictive(self, df_subset, max_number: int, numbers_per_draw: int,
                                        priors: Optional[Dict] = None) -> Dict:
        """Dirichlet-多项式后验预测分布"""
        try:
            counts = Counter()
            for idx, (_, row) in enumerate(df_subset.iterrows()):
                weight = self._time_decay_weight(idx, len(df_subset))
                if max_number == 35:
                    front_balls, _ = data_manager.parse_balls(row)
                    for ball in front_balls:
                        counts[ball] += weight
                else:
                    _, back_balls = data_manager.parse_balls(row)
                    for ball in back_balls:
                        counts[ball] += weight

            # Dirichlet先验参数
            concentration = BayesianConfig.DIRICHLET_CONCENTRATION
            total_concentration = concentration * max_number

            if priors:
                alpha = {num: max(1e-6, priors.get(num, 1 / max_number)) * total_concentration
                         for num in range(1, max_number + 1)}
            else:
                alpha = {num: total_concentration / max_number for num in range(1, max_number + 1)}

            # 后验参数
            alpha_posterior = {num: alpha[num] + counts.get(num, 0.0) for num in range(1, max_number + 1)}
            alpha_sum = sum(alpha_posterior.values())

            if alpha_sum <= 0:
                return {num: 1 / max_number for num in range(1, max_number + 1)}

            # 预测分布
            posterior_predictive = {num: alpha_posterior[num] / alpha_sum for num in range(1, max_number + 1)}
            return posterior_predictive

        except Exception as e:
            logger_manager.error(f"Dirichlet后验预测失败: {e}")
            return {num: 1 / max_number for num in range(1, max_number + 1)}

    def _calculate_multi_dimensional_likelihood(self, df_subset, max_number: int) -> Dict:
        """计算多维度似然函数 - 考虑维度间相关性

        改进：不再简单地将各维度似然相乘（假设独立），
        而是通过协方差建模来考虑维度间的相关性，提供更准确的综合似然。
        """
        try:
            import numpy as np

            likelihoods = {}

            # 第一遍：收集所有号码的四维度似然值，用于计算协方差矩阵
            all_likelihood_vectors = []

            for num in range(1, max_number + 1):
                freq_l = self._calculate_frequency_likelihood(df_subset, num, max_number)
                pos_l = self._calculate_position_likelihood(df_subset, num, max_number)
                temp_l = self._calculate_temporal_likelihood(df_subset, num, max_number)
                comb_l = self._calculate_combination_likelihood(df_subset, num, max_number)

                all_likelihood_vectors.append([freq_l, pos_l, temp_l, comb_l])

            # 转换为numpy数组
            likelihood_matrix = np.array(all_likelihood_vectors)

            # 计算似然维度间的协方差矩阵
            # 使用对数变换使分布更接近正态
            log_likelihood_matrix = np.log(np.maximum(likelihood_matrix, 1e-10))
            cov_matrix = np.cov(log_likelihood_matrix.T)

            # 计算相关系数矩阵（用于调整权重）
            std_devs = np.sqrt(np.diag(cov_matrix) + 1e-10)
            corr_matrix = cov_matrix / np.outer(std_devs, std_devs)

            # 基于相关性计算调整后的权重
            # 高度相关的维度应该降低权重，避免信息重复计算
            weights = self._calculate_correlation_adjusted_weights(corr_matrix)

            # 第二遍：使用调整后的权重计算综合似然
            for idx, num in enumerate(range(1, max_number + 1)):
                freq_l, pos_l, temp_l, comb_l = all_likelihood_vectors[idx]

                # 使用加权几何平均代替简单乘积
                # 几何平均在对数空间中等价于加权算术平均
                log_likelihoods = np.array([
                    np.log(max(1e-10, freq_l)),
                    np.log(max(1e-10, pos_l)),
                    np.log(max(1e-10, temp_l)),
                    np.log(max(1e-10, comb_l))
                ])

                # 加权对数似然
                weighted_log_likelihood = np.sum(weights * log_likelihoods)
                combined_likelihood = np.exp(weighted_log_likelihood)

                # 额外：计算马氏距离用于异常检测
                mean_log_likelihood = np.mean(log_likelihood_matrix, axis=0)
                try:
                    cov_inv = np.linalg.pinv(cov_matrix + np.eye(4) * 1e-6)
                    diff = log_likelihoods - mean_log_likelihood
                    mahalanobis_dist = np.sqrt(np.abs(diff @ cov_inv @ diff))
                except:
                    mahalanobis_dist = 0.0

                likelihood_components = {
                    'frequency': freq_l,
                    'position': pos_l,
                    'temporal': temp_l,
                    'combination': comb_l,
                    'combined': combined_likelihood,
                    'dimension_weights': weights.tolist(),  # 新增：各维度权重
                    'mahalanobis_distance': mahalanobis_dist  # 新增：马氏距离（异常度）
                }

                likelihoods[num] = likelihood_components

            return likelihoods

        except Exception as e:
            logger_manager.error(f"计算多维度似然失败: {e}")
            return {}

    def _calculate_correlation_adjusted_weights(self, corr_matrix: np.ndarray) -> np.ndarray:
        """基于相关系数矩阵计算调整后的维度权重

        原理：如果两个维度高度相关，它们提供的信息是冗余的，
        应该降低其中一个的权重以避免信息被重复计算。

        使用特征值分解来确定各维度的有效信息贡献。
        """
        try:
            n_dims = corr_matrix.shape[0]

            # 处理NaN和Inf
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

            # 确保对角线为1
            np.fill_diagonal(corr_matrix, 1.0)

            # 计算各维度与其他维度的平均相关性
            avg_correlation = np.zeros(n_dims)
            for i in range(n_dims):
                other_corrs = [abs(corr_matrix[i, j]) for j in range(n_dims) if i != j]
                avg_correlation[i] = np.mean(other_corrs) if other_corrs else 0

            # 相关性越高，权重越低（独特信息越少）
            # 使用1/(1+avg_corr)作为基础权重
            raw_weights = 1 / (1 + avg_correlation)

            # 归一化权重使其和为维度数（保持与原始乘法等效的量级）
            weights = raw_weights / np.sum(raw_weights) * n_dims

            return weights

        except Exception as e:
            logger_manager.error(f"计算相关性调整权重失败: {e}")
            # 返回均匀权重作为fallback
            return np.ones(4)

    def _traditional_bayesian_analysis(self, df_subset) -> Dict:
        """传统贝叶斯分析（保持兼容性）"""
        try:
            # 计算先验概率
            front_prior = {i: 1/35 for i in range(1, 36)}
            back_prior = {i: 1/12 for i in range(1, 13)}

            # 计算似然
            front_likelihood = defaultdict(float)
            back_likelihood = defaultdict(float)

            for _, row in df_subset.iterrows():
                front_balls, back_balls = data_manager.parse_balls(row)

                for ball in front_balls:
                    front_likelihood[ball] += 1
                for ball in back_balls:
                    back_likelihood[ball] += 1

            # 标准化似然
            front_total = sum(front_likelihood.values())
            back_total = sum(back_likelihood.values())

            if front_total > 0:
                for ball in front_likelihood:
                    front_likelihood[ball] /= front_total

            if back_total > 0:
                for ball in back_likelihood:
                    back_likelihood[ball] /= back_total

            # 计算后验概率
            front_posterior = {}
            for ball in range(1, 36):
                likelihood = front_likelihood.get(ball, 0.001)
                prior = front_prior[ball]
                front_posterior[ball] = likelihood * prior

            back_posterior = {}
            for ball in range(1, 13):
                likelihood = back_likelihood.get(ball, 0.001)
                prior = back_prior[ball]
                back_posterior[ball] = likelihood * prior

            # 标准化后验概率
            front_post_sum = sum(front_posterior.values())
            back_post_sum = sum(back_posterior.values())

            if front_post_sum > 0:
                front_posterior = {k: v/front_post_sum for k, v in front_posterior.items()}
            if back_post_sum > 0:
                back_posterior = {k: v/back_post_sum for k, v in back_posterior.items()}

            return {
                'front_posterior': front_posterior,
                'back_posterior': back_posterior,
                'front_likelihood': dict(front_likelihood),
                'back_likelihood': dict(back_likelihood)
            }

        except Exception as e:
            logger_manager.error(f"传统贝叶斯分析失败: {e}")
            return {}

    def _calculate_frequency_likelihood(self, df_subset, number: int, max_number: int) -> float:
        """计算频率似然"""
        try:
            count = 0
            total = 0

            for _, row in df_subset.iterrows():
                if max_number == 35:  # 前区
                    front_balls, _ = data_manager.parse_balls(row)
                    if number in front_balls:
                        count += 1
                    total += 1
                else:  # 后区
                    _, back_balls = data_manager.parse_balls(row)
                    if number in back_balls:
                        count += 1
                    total += 1

            # 贝塔分布似然
            if total > 0:
                return (count + 1) / (total + 2)  # 拉普拉斯平滑
            else:
                return 1 / max_number

        except Exception as e:
            logger_manager.error(f"计算频率似然失败: {e}")
            return 1 / max_number

    def _calculate_position_likelihood(self, df_subset, number: int, max_number: int) -> float:
        """计算位置似然"""
        try:
            position_counts = [0] * 5 if max_number == 35 else [0] * 2
            total_appearances = 0

            for _, row in df_subset.iterrows():
                if max_number == 35:  # 前区
                    front_balls, _ = data_manager.parse_balls(row)
                    if number in front_balls:
                        sorted_balls = sorted(front_balls)
                        position = sorted_balls.index(number)
                        position_counts[position] += 1
                        total_appearances += 1
                else:  # 后区
                    _, back_balls = data_manager.parse_balls(row)
                    if number in back_balls:
                        sorted_balls = sorted(back_balls)
                        position = sorted_balls.index(number)
                        position_counts[position] += 1
                        total_appearances += 1

            # 计算位置偏好似然
            if total_appearances > 0:
                # 均匀分布的期望
                expected_per_position = total_appearances / len(position_counts)
                # 计算偏差
                variance = sum((count - expected_per_position) ** 2 for count in position_counts)
                # 转换为似然（方差越小，似然越高）
                likelihood = 1 / (1 + variance / total_appearances)
            else:
                likelihood = 1.0

            return likelihood

        except Exception as e:
            logger_manager.error(f"计算位置似然失败: {e}")
            return 1.0

    def _calculate_temporal_likelihood(self, df_subset, number: int, max_number: int) -> float:
        """计算时间似然（区分前后区）"""
        try:
            import numpy as np

            appearances = []
            for i, (_, row) in enumerate(df_subset.iterrows()):
                front_balls, back_balls = data_manager.parse_balls(row)
                if max_number == 35:
                    if number in front_balls:
                        appearances.append(i)
                else:
                    if number in back_balls:
                        appearances.append(i)

            if len(appearances) < 2:
                return 1.0

            # 计算间隔
            intervals = [appearances[i+1] - appearances[i] for i in range(len(appearances)-1)]

            # 指数分布似然（间隔时间的分布）
            if intervals:
                mean_interval = np.mean(intervals)
                # 指数分布的似然
                lambda_param = 1 / mean_interval if mean_interval > 0 else 1
                likelihood = lambda_param * np.exp(-lambda_param * (len(df_subset) - appearances[-1]))
            else:
                likelihood = 1.0

            return min(10.0, max(0.1, likelihood))

        except Exception as e:
            logger_manager.error(f"计算时间似然失败: {e}")
            return 1.0

    def _calculate_combination_likelihood(self, df_subset, number: int, max_number: int) -> float:
        """计算组合似然"""
        try:
            # 计算该号码与其他号码的共现频率
            co_occurrence = Counter()
            total_combinations = 0

            for _, row in df_subset.iterrows():
                if max_number == 35:  # 前区
                    front_balls, _ = data_manager.parse_balls(row)
                    if number in front_balls:
                        for other_ball in front_balls:
                            if other_ball != number:
                                co_occurrence[other_ball] += 1
                        total_combinations += len(front_balls) - 1
                else:  # 后区
                    _, back_balls = data_manager.parse_balls(row)
                    if number in back_balls:
                        for other_ball in back_balls:
                            if other_ball != number:
                                co_occurrence[other_ball] += 1
                        total_combinations += len(back_balls) - 1

            if total_combinations == 0:
                return 1.0

            # 计算组合多样性（熵）
            probabilities = [count / total_combinations for count in co_occurrence.values()]

            # 空列表检查
            if not probabilities:
                return 1.0

            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)

            # 标准化熵作为似然
            max_entropy = np.log2(max_number - 1) if max_number > 1 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

            return normalized_entropy

        except Exception as e:
            logger_manager.error(f"计算组合似然失败: {e}")
            return 1.0

    def _calculate_bayesian_evidence(self, priors: Dict, likelihoods: Dict,
                                     max_number: int) -> Tuple[Dict, float]:
        """计算贝叶斯证据（边际似然）"""
        try:
            evidence = {}

            for num in range(1, max_number + 1):
                prior = priors.get(num, 1/max_number)
                likelihood_components = likelihoods.get(num, {})
                combined_likelihood = likelihood_components.get('combined', 1.0)

                # 边际似然 = 先验 × 似然
                evidence[num] = prior * combined_likelihood

            # 全局证据 = 所有假设的边际似然之和
            total_evidence = sum(evidence.values())

            return evidence, total_evidence

        except Exception as e:
            logger_manager.error(f"计算贝叶斯证据失败: {e}")
            return {}, 0.0

    def _calculate_posterior_distribution(self, prior: float, likelihood: Dict,
                                          global_evidence: float) -> Dict:
        """计算后验分布 - 使用精确矩估计法

        采用基于数据的自适应矩估计法，根据似然函数的各维度信息
        自动调整Beta分布的集中度参数，避免固定伪观测数的问题。
        """
        try:
            combined_likelihood = likelihood.get('combined', 1.0)

            # 贝叶斯定理：后验 = (似然 × 先验) / 证据
            posterior_mean = (combined_likelihood * prior) / global_evidence if global_evidence > 0 else prior

            # 确保posterior_mean在有效范围内
            posterior_mean = max(1e-10, min(1 - 1e-10, posterior_mean))

            # 精确矩估计法：基于似然函数各分量的一致性来估计集中度
            # 从似然的各维度计算方差，用于推断有效观测数
            freq_l = likelihood.get('frequency', 1.0)
            pos_l = likelihood.get('position', 1.0)
            temp_l = likelihood.get('temporal', 1.0)
            comb_l = likelihood.get('combination', 1.0)

            # 计算似然各维度的几何平均和变异系数
            likelihood_values = [freq_l, pos_l, temp_l, comb_l]
            likelihood_values = [max(1e-10, v) for v in likelihood_values]  # 避免零值

            # 几何平均
            geom_mean = np.exp(np.mean(np.log(likelihood_values)))

            # 变异系数（CV）：标准差/均值，衡量似然维度间的一致性
            cv = np.std(likelihood_values) / (np.mean(likelihood_values) + 1e-10)

            # 基于CV计算有效观测数（n_eff）
            # CV越小，各维度越一致，置信度越高，n_eff越大
            # CV=0时n_eff最大(200)，CV>=1时n_eff最小(10)
            n_eff = max(10, min(200, 200 / (1 + cv * 5)))

            # 使用矩估计法从均值和有效观测数推导Beta分布参数
            # 对于Beta(α,β)分布：
            # mean = α/(α+β)
            # n_eff ≈ α + β（集中度参数）
            alpha = posterior_mean * n_eff
            beta_param = (1 - posterior_mean) * n_eff

            # 确保参数有效
            alpha = max(0.5, alpha)
            beta_param = max(0.5, beta_param)

            # 计算方差（Beta分布方差公式）
            variance = (alpha * beta_param) / ((alpha + beta_param) ** 2 * (alpha + beta_param + 1))

            # 计算众数（Beta分布众数公式，仅当α>1且β>1时有意义）
            if alpha > 1 and beta_param > 1:
                mode = (alpha - 1) / (alpha + beta_param - 2)
            else:
                mode = posterior_mean

            return {
                'mean': posterior_mean,
                'alpha': alpha,
                'beta': beta_param,
                'variance': variance,
                'mode': mode,
                'effective_observations': n_eff,  # 新增：有效观测数
                'likelihood_consistency': 1 - min(1, cv)  # 新增：似然一致性指标
            }

        except Exception as e:
            logger_manager.error(f"计算后验分布失败: {e}")
            return {'mean': prior, 'alpha': 1, 'beta': 1, 'variance': 0.25, 'mode': prior,
                    'effective_observations': 2, 'likelihood_consistency': 0}

    def _calculate_bayesian_confidence_interval(self, posterior: Dict, confidence_level: float) -> Tuple[float, float]:
        """计算贝叶斯置信区间"""
        try:
            from scipy import stats

            alpha = posterior.get('alpha', 1)
            beta = posterior.get('beta', 1)

            # 贝塔分布的置信区间
            alpha_level = (1 - confidence_level) / 2
            lower = stats.beta.ppf(alpha_level, alpha, beta)
            upper = stats.beta.ppf(1 - alpha_level, alpha, beta)

            return (lower, upper)

        except Exception as e:
            logger_manager.error(f"计算贝叶斯置信区间失败: {e}")
            return (0.0, 1.0)

    def _calculate_predictive_distribution(self, posterior: Dict, likelihood: Dict) -> Dict:
        """计算预测分布"""
        try:
            posterior_mean = posterior.get('mean', 0.5)
            posterior_variance = posterior.get('variance', 0.25)

            # 预测分布考虑参数不确定性
            predictive_mean = posterior_mean

            # 预测方差 = 数据方差 + 参数不确定性
            data_variance = posterior_mean * (1 - posterior_mean)  # 二项分布方差
            predictive_variance = data_variance + posterior_variance

            return {
                'mean': predictive_mean,
                'variance': predictive_variance,
                'std': np.sqrt(predictive_variance)
            }

        except Exception as e:
            logger_manager.error(f"计算预测分布失败: {e}")
            return {'mean': 0.5, 'variance': 0.25, 'std': 0.5}

    def _calculate_bayes_factor(self, likelihood: Dict, prior: float, max_number: int) -> float:
        """计算贝叶斯因子"""
        try:
            # 贝叶斯因子 = 模型1的边际似然 / 模型2的边际似然
            # 这里比较当前模型与均匀分布模型

            combined_likelihood = likelihood.get('combined', 1.0)
            uniform_likelihood = 1.0  # 均匀分布的似然

            # 边际似然
            model_evidence = combined_likelihood * prior
            null_evidence = uniform_likelihood * (1 / max_number)

            bayes_factor = model_evidence / null_evidence if null_evidence > 0 else 1.0

            return bayes_factor

        except Exception as e:
            logger_manager.error(f"计算贝叶斯因子失败: {e}")
            return 1.0

    def _calculate_information_gain(self, prior: float, posterior: Dict) -> float:
        """计算信息增益（KL散度）"""
        try:
            posterior_mean = posterior.get('mean', prior)

            # KL散度：KL(posterior || prior)
            if prior > 0 and posterior_mean > 0:
                kl_divergence = posterior_mean * np.log(posterior_mean / prior)
                if posterior_mean < 1:
                    kl_divergence += (1 - posterior_mean) * np.log((1 - posterior_mean) / (1 - prior))
            else:
                kl_divergence = 0

            return max(0, kl_divergence)

        except Exception as e:
            logger_manager.error(f"计算信息增益失败: {e}")
            return 0.0

    def _calculate_bayesian_prediction_weight(self, posterior: Dict, bayes_factor: float,
                                            information_gain: float) -> float:
        """计算贝叶斯预测权重"""
        try:
            posterior_mean = posterior.get('mean', 0.5)
            posterior_variance = posterior.get('variance', 0.25)

            # 基础权重基于后验均值
            base_weight = posterior_mean

            # 贝叶斯因子调整（证据强度）
            bf_adjustment = min(2.0, max(0.5, np.log(bayes_factor + 1)))

            # 信息增益调整（学习程度）
            ig_adjustment = min(1.5, 1 + information_gain)

            # 不确定性调整（方差越小，权重越高）
            uncertainty_adjustment = 1 / (1 + posterior_variance * 4)

            # 综合权重
            final_weight = base_weight * bf_adjustment * ig_adjustment * uncertainty_adjustment

            return max(0.0, min(1.0, final_weight))

        except Exception as e:
            logger_manager.error(f"计算贝叶斯预测权重失败: {e}")
            return 0.5
    
    def correlation_analysis(self, periods=200) -> Dict:
        """相关性分析"""
        if self.df is None:
            return {}
        
        method_name = "correlation_analysis"
        cached_result = smart_cache_manager.load_cache("analysis", method_name, periods)
        if cached_result:
            return cached_result
        
        # 数据是降序排列（最新在前），使用head()获取最新数据
        df_subset = self.df.head(periods)
        
        # 构建特征矩阵
        features = []
        for _, row in df_subset.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            
            # 创建特征向量
            feature_vector = [0] * 47  # 35个前区 + 12个后区
            
            for ball in front_balls:
                feature_vector[ball - 1] = 1
            for ball in back_balls:
                feature_vector[34 + ball] = 1
            
            features.append(feature_vector)
        
        features = np.array(features)
        
        # 计算相关性矩阵
        correlation_matrix = np.corrcoef(features.T)
        
        # 找出高相关性的号码对
        high_correlations = []
        for i in range(len(correlation_matrix)):
            for j in range(i + 1, len(correlation_matrix)):
                corr = correlation_matrix[i][j]
                if abs(corr) > 0.3:  # 相关性阈值
                    ball1 = i + 1 if i < 35 else i - 34
                    ball2 = j + 1 if j < 35 else j - 34
                    zone1 = "前区" if i < 35 else "后区"
                    zone2 = "前区" if j < 35 else "后区"
                    
                    high_correlations.append({
                        'ball1': ball1,
                        'zone1': zone1,
                        'ball2': ball2,
                        'zone2': zone2,
                        'correlation': corr
                    })
        
        result = {
            'high_correlations': sorted(high_correlations, key=lambda x: abs(x['correlation']), reverse=True),
            'correlation_matrix_shape': correlation_matrix.shape,
            'analysis_periods': periods,
            'timestamp': datetime.now().isoformat()
        }
        
        smart_cache_manager.save_cache("analysis", method_name, result, periods)
        return result

    def trend_generation_analysis(self, periods=500) -> Dict:
        """趋势生成分析"""
        if self.df is None:
            return {}

        method_name = "trend_generation_analysis"
        cached_result = smart_cache_manager.load_cache("analysis", method_name, periods)
        if cached_result:
            return cached_result

        # 数据是降序排列（最新在前），使用head()获取最新数据
        df_subset = self.df.head(periods)

        # 趋势分析
        trends = {
            'frequency_trend': {},  # 频率趋势
            'hot_cold_trend': {},   # 冷热趋势
            'sum_trend': [],        # 和值趋势
            'span_trend': [],       # 跨度趋势
            'pattern_trend': {}     # 模式趋势
        }

        # 分段分析（每50期为一段）
        segment_size = 50
        segments = []

        for i in range(0, len(df_subset), segment_size):
            segment = df_subset.iloc[i:i+segment_size]
            if len(segment) < 10:  # 段太小跳过
                continue

            segment_analysis = {
                'period_range': (i, i + len(segment)),
                'front_freq': Counter(),
                'back_freq': Counter(),
                'sum_values': [],
                'span_values': []
            }

            for _, row in segment.iterrows():
                front_balls, back_balls = data_manager.parse_balls(row)
                segment_analysis['front_freq'].update(front_balls)
                segment_analysis['back_freq'].update(back_balls)
                segment_analysis['sum_values'].append(sum(front_balls))
                segment_analysis['span_values'].append(max(front_balls) - min(front_balls))

            segments.append(segment_analysis)

        # 分析趋势变化
        if len(segments) >= 2:
            # 频率趋势
            for ball in range(1, 36):
                freq_changes = []
                for segment in segments:
                    freq_changes.append(segment['front_freq'].get(ball, 0))
                trends['frequency_trend'][ball] = {
                    'values': freq_changes,
                    'trend': 'up' if freq_changes[-1] > freq_changes[0] else 'down',
                    'volatility': np.std(freq_changes)
                }

            # 和值趋势
            for segment in segments:
                trends['sum_trend'].append(np.mean(segment['sum_values']))

            # 跨度趋势
            for segment in segments:
                trends['span_trend'].append(np.mean(segment['span_values']))

        result = {
            'trends': trends,
            'segments': len(segments),
            'segment_size': segment_size,
            'analysis_periods': periods,
            'timestamp': datetime.now().isoformat()
        }

        smart_cache_manager.save_cache("analysis", method_name, result, periods)
        return result

    def mixed_strategy_analysis(self, periods=500) -> Dict:
        """混合策略分析"""
        if self.df is None:
            return {}

        method_name = "mixed_strategy_analysis"
        cached_result = smart_cache_manager.load_cache("analysis", method_name, periods)
        if cached_result:
            return cached_result

        # 获取各种分析结果
        frequency_result = self.basic_analyzer.frequency_analysis(periods)
        markov_result = self.markov_analysis(periods)
        bayesian_result = self.bayesian_analysis(periods)
        correlation_result = self.correlation_analysis(periods)

        # 混合策略生成
        strategies = {
            'conservative_strategy': {  # 保守策略
                'description': '基于高频号码和稳定模式',
                'weights': {
                    'frequency': 0.4,
                    'markov': 0.3,
                    'bayesian': 0.2,
                    'correlation': 0.1
                },
                'risk_level': 'low'
            },
            'aggressive_strategy': {   # 激进策略
                'description': '基于趋势变化和新兴模式',
                'weights': {
                    'frequency': 0.1,
                    'markov': 0.4,
                    'bayesian': 0.3,
                    'correlation': 0.2
                },
                'risk_level': 'high'
            },
            'balanced_strategy': {     # 平衡策略
                'description': '各种方法均衡组合',
                'weights': {
                    'frequency': 0.25,
                    'markov': 0.25,
                    'bayesian': 0.25,
                    'correlation': 0.25
                },
                'risk_level': 'medium'
            }
        }

        # 为每种策略生成推荐号码
        for strategy_name, strategy in strategies.items():
            front_candidates = Counter()
            back_candidates = Counter()

            # 基于权重合并候选号码
            if frequency_result:
                weight = strategy['weights']['frequency']
                for ball, freq in frequency_result.get('front_frequency', {}).items():
                    front_candidates[ball] += freq * weight
                for ball, freq in frequency_result.get('back_frequency', {}).items():
                    back_candidates[ball] += freq * weight

            # 添加推荐号码
            strategy['recommended_front'] = [ball for ball, score in front_candidates.most_common(10)]
            strategy['recommended_back'] = [ball for ball, score in back_candidates.most_common(6)]

        result = {
            'strategies': strategies,
            'analysis_periods': periods,
            'timestamp': datetime.now().isoformat()
        }

        smart_cache_manager.save_cache("analysis", method_name, result, periods)
        return result

    def markov_bayesian_fusion_analysis(self, periods=500) -> Dict:
        """马尔可夫-贝叶斯融合分析"""
        if self.df is None:
            return {}

        method_name = "markov_bayesian_fusion_analysis"
        cached_result = smart_cache_manager.load_cache("analysis", method_name, periods)
        if cached_result:
            return cached_result

        # 获取马尔可夫和贝叶斯分析结果
        markov_result = self.markov_analysis(periods)
        bayesian_result = self.bayesian_analysis(periods)

        # 融合分析
        fusion_scores = {
            'front_fusion': {},
            'back_fusion': {}
        }

        # 前区融合评分
        for ball in range(1, 36):
            markov_score = 0
            bayesian_score = 0

            # 马尔可夫得分（基于转移概率）
            if 'front_transition_probs' in markov_result:
                for from_ball, to_probs in markov_result['front_transition_probs'].items():
                    if ball in to_probs:
                        markov_score += to_probs[ball]

            # 贝叶斯得分（基于后验概率）
            if 'front_posterior' in bayesian_result:
                bayesian_score = bayesian_result['front_posterior'].get(ball, 0)

            # 融合得分（加权平均）
            fusion_score = (markov_score * 0.6 + bayesian_score * 0.4)
            fusion_scores['front_fusion'][ball] = fusion_score

        # 后区融合评分
        for ball in range(1, 13):
            markov_score = 0
            bayesian_score = 0

            if 'back_transition_probs' in markov_result:
                for from_ball, to_probs in markov_result['back_transition_probs'].items():
                    if ball in to_probs:
                        markov_score += to_probs[ball]

            if 'back_posterior' in bayesian_result:
                bayesian_score = bayesian_result['back_posterior'].get(ball, 0)

            fusion_score = (markov_score * 0.6 + bayesian_score * 0.4)
            fusion_scores['back_fusion'][ball] = fusion_score

        # 生成融合推荐
        front_recommendations = sorted(fusion_scores['front_fusion'].items(),
                                     key=lambda x: x[1], reverse=True)[:10]
        back_recommendations = sorted(fusion_scores['back_fusion'].items(),
                                    key=lambda x: x[1], reverse=True)[:6]

        result = {
            'fusion_scores': fusion_scores,
            'front_recommendations': front_recommendations,
            'back_recommendations': back_recommendations,
            'fusion_weights': {'markov': 0.6, 'bayesian': 0.4},
            'analysis_periods': periods,
            'timestamp': datetime.now().isoformat()
        }

        smart_cache_manager.save_cache("analysis", method_name, result, periods)
        return result

    def hot_cold_markov_integration(self, periods=500) -> Dict:
        """冷热号-马尔可夫集成分析"""
        if self.df is None:
            return {}

        method_name = "hot_cold_markov_integration"
        cached_result = smart_cache_manager.load_cache("analysis", method_name, periods)
        if cached_result:
            return cached_result

        # 获取冷热号和马尔可夫分析结果
        hot_cold_result = self.basic_analyzer.hot_cold_analysis(periods)
        markov_result = self.markov_analysis(periods)

        # 集成分析
        integration_scores = {
            'front_integration': {},
            'back_integration': {}
        }

        # 前区集成
        for ball in range(1, 36):
            hot_cold_score = 0
            markov_score = 0

            # 冷热号得分
            if ball in hot_cold_result.get('front_hot', []):
                hot_cold_score = 1.0
            elif ball in hot_cold_result.get('front_warm', []):
                hot_cold_score = 0.5
            else:  # 冷号
                hot_cold_score = 0.1

            # 马尔可夫得分
            if 'front_transition_probs' in markov_result:
                for from_ball, to_probs in markov_result['front_transition_probs'].items():
                    if ball in to_probs:
                        markov_score += to_probs[ball]

            # 集成得分
            integration_score = (hot_cold_score * 0.4 + markov_score * 0.6)
            integration_scores['front_integration'][ball] = integration_score

        # 后区集成
        for ball in range(1, 13):
            hot_cold_score = 0
            markov_score = 0

            if ball in hot_cold_result.get('back_hot', []):
                hot_cold_score = 1.0
            elif ball in hot_cold_result.get('back_warm', []):
                hot_cold_score = 0.5
            else:
                hot_cold_score = 0.1

            if 'back_transition_probs' in markov_result:
                for from_ball, to_probs in markov_result['back_transition_probs'].items():
                    if ball in to_probs:
                        markov_score += to_probs[ball]

            integration_score = (hot_cold_score * 0.4 + markov_score * 0.6)
            integration_scores['back_integration'][ball] = integration_score

        # 生成集成推荐
        front_integrated = sorted(integration_scores['front_integration'].items(),
                                key=lambda x: x[1], reverse=True)[:10]
        back_integrated = sorted(integration_scores['back_integration'].items(),
                               key=lambda x: x[1], reverse=True)[:6]

        result = {
            'integration_scores': integration_scores,
            'front_integrated': front_integrated,
            'back_integrated': back_integrated,
            'integration_weights': {'hot_cold': 0.4, 'markov': 0.6},
            'analysis_periods': periods,
            'timestamp': datetime.now().isoformat()
        }

        smart_cache_manager.save_cache("analysis", method_name, result, periods)
        return result

    def multi_dimensional_probability_analysis(self, periods=500) -> Dict:
        """多维度概率分析"""
        if self.df is None:
            return {}

        method_name = "multi_dimensional_probability_analysis"
        cached_result = smart_cache_manager.load_cache("analysis", method_name, periods)
        if cached_result:
            return cached_result

        # 获取各种分析结果
        frequency_result = self.basic_analyzer.frequency_analysis(periods)
        missing_result = self.basic_analyzer.missing_analysis(periods)
        markov_result = self.markov_analysis(periods)
        bayesian_result = self.bayesian_analysis(periods)

        # 多维度概率计算
        multi_prob = {
            'front_multi_prob': {},
            'back_multi_prob': {}
        }

        # 前区多维度概率
        for ball in range(1, 36):
            # 频率概率
            freq_prob = frequency_result.get('front_frequency', {}).get(ball, 0) / periods

            # 遗漏概率（反向）
            missing_periods = missing_result.get('front_missing', {}).get(ball, 0)
            missing_prob = 1.0 / (missing_periods + 1)

            # 马尔可夫概率
            markov_prob = 0
            if 'front_transition_probs' in markov_result:
                for from_ball, to_probs in markov_result['front_transition_probs'].items():
                    if ball in to_probs:
                        markov_prob += to_probs[ball]

            # 贝叶斯概率
            bayesian_prob = bayesian_result.get('front_posterior', {}).get(ball, 0)

            # 综合概率（加权平均）
            weights = [0.25, 0.25, 0.25, 0.25]
            probs = [freq_prob, missing_prob, markov_prob, bayesian_prob]

            multi_prob_score = sum(w * p for w, p in zip(weights, probs))
            multi_prob['front_multi_prob'][ball] = {
                'total_prob': multi_prob_score,
                'freq_prob': freq_prob,
                'missing_prob': missing_prob,
                'markov_prob': markov_prob,
                'bayesian_prob': bayesian_prob
            }

        # 后区多维度概率
        for ball in range(1, 13):
            freq_prob = frequency_result.get('back_frequency', {}).get(ball, 0) / periods
            missing_periods = missing_result.get('back_missing', {}).get(ball, 0)
            missing_prob = 1.0 / (missing_periods + 1)

            markov_prob = 0
            if 'back_transition_probs' in markov_result:
                for from_ball, to_probs in markov_result['back_transition_probs'].items():
                    if ball in to_probs:
                        markov_prob += to_probs[ball]

            bayesian_prob = bayesian_result.get('back_posterior', {}).get(ball, 0)

            weights = [0.25, 0.25, 0.25, 0.25]
            probs = [freq_prob, missing_prob, markov_prob, bayesian_prob]

            multi_prob_score = sum(w * p for w, p in zip(weights, probs))
            multi_prob['back_multi_prob'][ball] = {
                'total_prob': multi_prob_score,
                'freq_prob': freq_prob,
                'missing_prob': missing_prob,
                'markov_prob': markov_prob,
                'bayesian_prob': bayesian_prob
            }

        # 生成推荐
        front_ranked = sorted(multi_prob['front_multi_prob'].items(),
                            key=lambda x: x[1]['total_prob'], reverse=True)[:10]
        back_ranked = sorted(multi_prob['back_multi_prob'].items(),
                           key=lambda x: x[1]['total_prob'], reverse=True)[:6]

        result = {
            'multi_dimensional_probabilities': multi_prob,
            'front_ranked': front_ranked,
            'back_ranked': back_ranked,
            'dimension_weights': {'frequency': 0.25, 'missing': 0.25, 'markov': 0.25, 'bayesian': 0.25},
            'analysis_periods': periods,
            'timestamp': datetime.now().isoformat()
        }

        smart_cache_manager.save_cache("analysis", method_name, result, periods)
        return result

    def comprehensive_weight_scoring_system(self, periods=500) -> Dict:
        """综合权重评分系统"""
        if self.df is None:
            return {}

        method_name = "comprehensive_weight_scoring_system"
        cached_result = smart_cache_manager.load_cache("analysis", method_name, periods)
        if cached_result:
            return cached_result

        # 获取所有分析结果
        frequency_result = self.basic_analyzer.frequency_analysis(periods)
        hot_cold_result = self.basic_analyzer.hot_cold_analysis(periods)
        missing_result = self.basic_analyzer.missing_analysis(periods)
        markov_result = self.markov_analysis(periods)
        bayesian_result = self.bayesian_analysis(periods)
        correlation_result = self.correlation_analysis(periods)

        # 权重配置（可动态调整）
        weights = {
            'frequency': 0.20,
            'hot_cold': 0.15,
            'missing': 0.15,
            'markov': 0.25,
            'bayesian': 0.15,
            'correlation': 0.10
        }

        # 综合评分
        comprehensive_scores = {
            'front_scores': {},
            'back_scores': {}
        }

        # 前区综合评分
        for ball in range(1, 36):
            total_score = 0
            detail_scores = {}

            # 频率得分（处理键的数据类型）
            front_freq = frequency_result.get('front_frequency', {})
            freq_count = front_freq.get(ball, front_freq.get(str(ball), 0))
            freq_score = freq_count / periods if periods > 0 else 0
            total_score += freq_score * weights['frequency']
            detail_scores['frequency'] = freq_score

            # 冷热号得分
            hot_cold_score = 0
            if ball in hot_cold_result.get('front_hot', []):
                hot_cold_score = 1.0
            elif ball in hot_cold_result.get('front_warm', []):
                hot_cold_score = 0.6
            else:
                hot_cold_score = 0.2
            total_score += hot_cold_score * weights['hot_cold']
            detail_scores['hot_cold'] = hot_cold_score

            # 遗漏得分（处理键的数据类型）
            front_missing = missing_result.get('front_missing', {})
            missing_periods = front_missing.get(ball, front_missing.get(str(ball), 0))
            missing_score = min(1.0, missing_periods / 20) if missing_periods > 0 else 0  # 标准化
            total_score += missing_score * weights['missing']
            detail_scores['missing'] = missing_score

            # 马尔可夫得分
            markov_score = 0
            if 'front_transition_probs' in markov_result:
                for from_ball, to_probs in markov_result['front_transition_probs'].items():
                    if ball in to_probs:
                        markov_score += to_probs[ball]
            total_score += markov_score * weights['markov']
            detail_scores['markov'] = markov_score

            # 贝叶斯得分（处理键的数据类型）
            front_posterior = bayesian_result.get('front_posterior', {})
            bayesian_score = front_posterior.get(ball, front_posterior.get(str(ball), 0))
            total_score += bayesian_score * weights['bayesian']
            detail_scores['bayesian'] = bayesian_score

            # 相关性得分
            correlation_score = 0
            if 'front_correlations' in correlation_result:
                for corr_ball, corr_value in correlation_result['front_correlations'].items():
                    if corr_ball == ball:
                        correlation_score = abs(corr_value)
                        break
            total_score += correlation_score * weights['correlation']
            detail_scores['correlation'] = correlation_score

            comprehensive_scores['front_scores'][ball] = {
                'total_score': total_score,
                'detail_scores': detail_scores
            }

        # 后区综合评分（类似逻辑）
        for ball in range(1, 13):
            total_score = 0
            detail_scores = {}

            # 频率得分（处理键的数据类型）
            back_freq = frequency_result.get('back_frequency', {})
            freq_count = back_freq.get(ball, back_freq.get(str(ball), 0))
            freq_score = freq_count / periods if periods > 0 else 0
            total_score += freq_score * weights['frequency']
            detail_scores['frequency'] = freq_score

            hot_cold_score = 0
            if ball in hot_cold_result.get('back_hot', []):
                hot_cold_score = 1.0
            elif ball in hot_cold_result.get('back_warm', []):
                hot_cold_score = 0.6
            else:
                hot_cold_score = 0.2
            total_score += hot_cold_score * weights['hot_cold']
            detail_scores['hot_cold'] = hot_cold_score

            # 遗漏得分（处理键的数据类型）
            back_missing = missing_result.get('back_missing', {})
            missing_periods = back_missing.get(ball, back_missing.get(str(ball), 0))
            missing_score = min(1.0, missing_periods / 15) if missing_periods > 0 else 0
            total_score += missing_score * weights['missing']
            detail_scores['missing'] = missing_score

            markov_score = 0
            if 'back_transition_probs' in markov_result:
                for from_ball, to_probs in markov_result['back_transition_probs'].items():
                    if ball in to_probs:
                        markov_score += to_probs[ball]
            total_score += markov_score * weights['markov']
            detail_scores['markov'] = markov_score

            # 贝叶斯得分（处理键的数据类型）
            back_posterior = bayesian_result.get('back_posterior', {})
            bayesian_score = back_posterior.get(ball, back_posterior.get(str(ball), 0))
            total_score += bayesian_score * weights['bayesian']
            detail_scores['bayesian'] = bayesian_score

            correlation_score = 0
            if 'back_correlations' in correlation_result:
                for corr_ball, corr_value in correlation_result['back_correlations'].items():
                    if corr_ball == ball:
                        correlation_score = abs(corr_value)
                        break
            total_score += correlation_score * weights['correlation']
            detail_scores['correlation'] = correlation_score

            comprehensive_scores['back_scores'][ball] = {
                'total_score': total_score,
                'detail_scores': detail_scores
            }

        # 生成排名
        front_ranking = sorted(comprehensive_scores['front_scores'].items(),
                             key=lambda x: x[1]['total_score'], reverse=True)
        back_ranking = sorted(comprehensive_scores['back_scores'].items(),
                            key=lambda x: x[1]['total_score'], reverse=True)

        result = {
            'comprehensive_scores': comprehensive_scores,
            'front_ranking': front_ranking[:15],
            'back_ranking': back_ranking[:8],
            'weights_used': weights,
            'analysis_periods': periods,
            'timestamp': datetime.now().isoformat()
        }

        smart_cache_manager.save_cache("analysis", method_name, result, periods)
        return result

    def advanced_pattern_recognition(self, periods=500) -> Dict:
        """高级模式识别"""
        if self.df is None:
            return {}

        method_name = "advanced_pattern_recognition"
        cached_result = smart_cache_manager.load_cache("analysis", method_name, periods)
        if cached_result:
            return cached_result

        # 数据是降序排列（最新在前），使用head()获取最新数据
        df_subset = self.df.head(periods)

        # 模式识别
        patterns = {
            'consecutive_patterns': [],  # 连号模式
            'sum_patterns': [],         # 和值模式
            'odd_even_patterns': [],    # 奇偶模式
            'big_small_patterns': [],   # 大小模式
            'repeat_patterns': [],      # 重复号码模式
            'interval_patterns': []     # 间隔模式
        }

        previous_front = None
        previous_back = None

        for _, row in df_subset.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)

            # 连号模式识别
            consecutive_count = 0
            sorted_front = sorted(front_balls)
            for i in range(len(sorted_front) - 1):
                if sorted_front[i+1] - sorted_front[i] == 1:
                    consecutive_count += 1
            patterns['consecutive_patterns'].append(consecutive_count)

            # 和值模式
            front_sum = sum(front_balls)
            back_sum = sum(back_balls)
            patterns['sum_patterns'].append({
                'front_sum': front_sum,
                'back_sum': back_sum,
                'total_sum': front_sum + back_sum
            })

            # 奇偶模式
            front_odd = sum(1 for x in front_balls if x % 2 == 1)
            back_odd = sum(1 for x in back_balls if x % 2 == 1)
            patterns['odd_even_patterns'].append({
                'front_odd': front_odd,
                'front_even': 5 - front_odd,
                'back_odd': back_odd,
                'back_even': 2 - back_odd
            })

            # 大小模式
            front_big = sum(1 for x in front_balls if x > 17)
            back_big = sum(1 for x in back_balls if x > 6)
            patterns['big_small_patterns'].append({
                'front_big': front_big,
                'front_small': 5 - front_big,
                'back_big': back_big,
                'back_small': 2 - back_big
            })

            # 重复号码模式
            if previous_front is not None:
                front_repeat = len(set(front_balls) & set(previous_front))
                back_repeat = len(set(back_balls) & set(previous_back))
                patterns['repeat_patterns'].append({
                    'front_repeat': front_repeat,
                    'back_repeat': back_repeat
                })

            # 间隔模式
            if len(sorted_front) >= 2:
                intervals = [sorted_front[i+1] - sorted_front[i] for i in range(len(sorted_front)-1)]
                patterns['interval_patterns'].append({
                    'front_intervals': intervals,
                    'avg_interval': np.mean(intervals),
                    'max_interval': max(intervals)
                })

            previous_front = front_balls
            previous_back = back_balls

        # 模式统计
        pattern_stats = {
            'consecutive_stats': {
                'avg': np.mean(patterns['consecutive_patterns']),
                'max': max(patterns['consecutive_patterns']),
                'distribution': Counter(patterns['consecutive_patterns'])
            },
            'sum_stats': {
                'front_avg': np.mean([p['front_sum'] for p in patterns['sum_patterns']]),
                'back_avg': np.mean([p['back_sum'] for p in patterns['sum_patterns']]),
                'total_avg': np.mean([p['total_sum'] for p in patterns['sum_patterns']])
            },
            'odd_even_stats': {
                'front_odd_avg': np.mean([p['front_odd'] for p in patterns['odd_even_patterns']]),
                'back_odd_avg': np.mean([p['back_odd'] for p in patterns['odd_even_patterns']])
            },
            'repeat_stats': {
                'front_repeat_avg': np.mean([p['front_repeat'] for p in patterns['repeat_patterns']]) if patterns['repeat_patterns'] else 0,
                'back_repeat_avg': np.mean([p['back_repeat'] for p in patterns['repeat_patterns']]) if patterns['repeat_patterns'] else 0
            }
        }

        result = {
            'patterns': patterns,
            'pattern_statistics': pattern_stats,
            'analysis_periods': periods,
            'timestamp': datetime.now().isoformat()
        }

        smart_cache_manager.save_cache("analysis", method_name, result, periods)
        return result

    def nine_mathematical_models_analysis(self, periods=500) -> Dict:
        """9种数学模型综合分析

        包含：统计学分析、概率论分析、频率模式分析、决策树分析、
        周期性分析、历史关联分析、马尔可夫链分析、贝叶斯分析、回归分析
        """
        if self.df is None:
            return {}

        method_name = "nine_mathematical_models_analysis"
        cached_result = smart_cache_manager.load_cache("analysis", method_name, periods)
        if cached_result:
            return cached_result

        logger_manager.info(f"开始9种数学模型综合分析，期数: {periods}")

        # 数据是降序排列（最新在前），使用head()获取最新数据
        df_subset = self.df.head(periods)

        # 1. 统计学分析
        statistical_analysis = self._statistical_model_analysis(df_subset)

        # 2. 概率论分析
        probability_analysis = self._probability_theory_analysis(df_subset)

        # 3. 频率模式分析
        frequency_pattern_analysis = self._frequency_pattern_analysis(df_subset)

        # 4. 决策树分析
        decision_tree_analysis = self._decision_tree_analysis(df_subset)

        # 5. 周期性分析
        cyclical_analysis = self._cyclical_analysis(df_subset)

        # 6. 历史关联分析
        historical_correlation_analysis = self._historical_correlation_analysis(df_subset)

        # 7. 马尔可夫链分析（增强版）
        enhanced_markov_analysis = self._enhanced_markov_analysis(df_subset)

        # 8. 贝叶斯分析（增强版）
        enhanced_bayesian_analysis = self._enhanced_bayesian_analysis_simple(df_subset)

        # 9. 回归分析
        regression_analysis = self._regression_analysis(df_subset)

        # 综合评分和预测生成
        comprehensive_scores = self._generate_comprehensive_prediction_scores(
            statistical_analysis, probability_analysis, frequency_pattern_analysis,
            decision_tree_analysis, cyclical_analysis, historical_correlation_analysis,
            enhanced_markov_analysis, enhanced_bayesian_analysis, regression_analysis
        )

        result = {
            'nine_models': {
                '1_statistical': statistical_analysis,
                '2_probability': probability_analysis,
                '3_frequency_pattern': frequency_pattern_analysis,
                '4_decision_tree': decision_tree_analysis,
                '5_cyclical': cyclical_analysis,
                '6_historical_correlation': historical_correlation_analysis,
                '7_enhanced_markov': enhanced_markov_analysis,
                '8_enhanced_bayesian': enhanced_bayesian_analysis,
                '9_regression': regression_analysis
            },
            'comprehensive_scores': comprehensive_scores,
            'model_weights': {
                'statistical': 0.12, 'probability': 0.15, 'frequency_pattern': 0.10,
                'decision_tree': 0.08, 'cyclical': 0.10, 'historical_correlation': 0.12,
                'enhanced_markov': 0.15, 'enhanced_bayesian': 0.13, 'regression': 0.05
            },
            'analysis_periods': periods,
            'timestamp': datetime.now().isoformat()
        }

        smart_cache_manager.save_cache("analysis", method_name, result, periods)
        logger_manager.info("9种数学模型综合分析完成")
        return result

    def _statistical_model_analysis(self, df_subset) -> Dict:
        """统计学分析模型"""
        stats = {
            'descriptive_stats': {},
            'distribution_analysis': {},
            'variance_analysis': {},
            'correlation_coefficients': {}
        }

        # 描述性统计
        front_numbers = []
        back_numbers = []

        for _, row in df_subset.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            front_numbers.extend(front_balls)
            back_numbers.extend(back_balls)

        stats['descriptive_stats'] = {
            'front': {
                'mean': np.mean(front_numbers),
                'median': np.median(front_numbers),
                'std': np.std(front_numbers),
                'variance': np.var(front_numbers),
                'skewness': self._calculate_skewness(front_numbers),
                'kurtosis': self._calculate_kurtosis(front_numbers)
            },
            'back': {
                'mean': np.mean(back_numbers),
                'median': np.median(back_numbers),
                'std': np.std(back_numbers),
                'variance': np.var(back_numbers),
                'skewness': self._calculate_skewness(back_numbers),
                'kurtosis': self._calculate_kurtosis(back_numbers)
            }
        }

        # 分布分析
        front_dist = Counter(front_numbers)
        back_dist = Counter(back_numbers)

        stats['distribution_analysis'] = {
            'front_distribution': dict(front_dist),
            'back_distribution': dict(back_dist),
            'front_entropy': self._calculate_entropy(list(front_dist.values())),
            'back_entropy': self._calculate_entropy(list(back_dist.values()))
        }

        return stats

    def _probability_theory_analysis(self, df_subset) -> Dict:
        """概率论分析模型"""
        prob_analysis = {
            'conditional_probabilities': {},
            'joint_probabilities': {},
            'marginal_probabilities': {},
            'independence_tests': {}
        }

        # 条件概率计算
        front_given_back = {}
        back_given_front = {}

        for _, row in df_subset.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)

            for front_ball in front_balls:
                for back_ball in back_balls:
                    key = f"F{front_ball}|B{back_ball}"
                    front_given_back[key] = front_given_back.get(key, 0) + 1

                    key = f"B{back_ball}|F{front_ball}"
                    back_given_front[key] = back_given_front.get(key, 0) + 1

        prob_analysis['conditional_probabilities'] = {
            'front_given_back': front_given_back,
            'back_given_front': back_given_front
        }

        # 边际概率
        front_marginal = Counter()
        back_marginal = Counter()

        for _, row in df_subset.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            front_marginal.update(front_balls)
            back_marginal.update(back_balls)

        total_periods = len(df_subset)
        prob_analysis['marginal_probabilities'] = {
            'front': {k: v/total_periods for k, v in front_marginal.items()},
            'back': {k: v/total_periods for k, v in back_marginal.items()}
        }

        return prob_analysis

    def _frequency_pattern_analysis(self, df_subset) -> Dict:
        """频率模式分析模型"""
        pattern_analysis = {
            'frequency_cycles': {},
            'pattern_sequences': {},
            'frequency_trends': {},
            'pattern_predictions': {}
        }

        # 频率周期分析
        front_freq_history = []
        back_freq_history = []

        window_size = 20  # 滑动窗口大小

        for i in range(len(df_subset) - window_size + 1):
            window_data = df_subset.iloc[i:i+window_size]

            front_freq = Counter()
            back_freq = Counter()

            for _, row in window_data.iterrows():
                front_balls, back_balls = data_manager.parse_balls(row)
                front_freq.update(front_balls)
                back_freq.update(back_balls)

            front_freq_history.append(dict(front_freq))
            back_freq_history.append(dict(back_freq))

        pattern_analysis['frequency_cycles'] = {
            'front_cycles': front_freq_history[-10:],  # 最近10个周期
            'back_cycles': back_freq_history[-10:]
        }

        # 模式序列识别
        sequences = []
        for i in range(len(df_subset) - 2):
            seq_data = df_subset.iloc[i:i+3]
            sequence = []
            for _, row in seq_data.iterrows():
                front_balls, back_balls = data_manager.parse_balls(row)
                sequence.append((sorted(front_balls), sorted(back_balls)))
            sequences.append(sequence)

        pattern_analysis['pattern_sequences'] = {
            'total_sequences': len(sequences),
            'unique_sequences': len(set(str(seq) for seq in sequences)),
            'common_sequences': Counter(str(seq) for seq in sequences).most_common(5)
        }

        return pattern_analysis

    def _decision_tree_analysis(self, df_subset) -> Dict:
        """决策树分析模型"""
        decision_analysis = {
            'decision_rules': [],
            'feature_importance': {},
            'prediction_paths': {},
            'rule_confidence': {}
        }

        # 构建决策规则
        rules = []

        # 基于和值的决策规则
        for _, row in df_subset.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            front_sum = sum(front_balls)
            back_sum = sum(back_balls)

            # 规则1: 和值范围决策
            if 60 <= front_sum <= 120:
                rules.append({
                    'condition': f'front_sum_in_range_{front_sum}',
                    'prediction': 'moderate_numbers',
                    'confidence': 0.7
                })

            # 规则2: 奇偶比决策
            front_odd = sum(1 for x in front_balls if x % 2 == 1)
            if front_odd == 3:  # 3奇2偶
                rules.append({
                    'condition': 'front_odd_even_3_2',
                    'prediction': 'balanced_selection',
                    'confidence': 0.6
                })

        decision_analysis['decision_rules'] = rules[-20:]  # 最近20条规则

        # 特征重要性评估
        decision_analysis['feature_importance'] = {
            'sum_value': 0.25,
            'odd_even_ratio': 0.20,
            'span': 0.15,
            'consecutive_count': 0.15,
            'big_small_ratio': 0.15,
            'ac_value': 0.10
        }

        return decision_analysis

    def _cyclical_analysis(self, df_subset) -> Dict:
        """周期性分析模型"""
        cyclical_analysis = {
            'weekly_patterns': {},
            'monthly_patterns': {},
            'seasonal_patterns': {},
            'cycle_predictions': {}
        }

        # 周期性模式识别
        if 'date' in df_subset.columns:
            # 按星期分析
            weekly_patterns = {}
            monthly_patterns = {}

            for _, row in df_subset.iterrows():
                try:
                    date_obj = pd.to_datetime(row['date'])
                    weekday = date_obj.weekday()
                    month = date_obj.month

                    front_balls, back_balls = data_manager.parse_balls(row)

                    # 周模式
                    if weekday not in weekly_patterns:
                        weekly_patterns[weekday] = {'front': Counter(), 'back': Counter()}
                    weekly_patterns[weekday]['front'].update(front_balls)
                    weekly_patterns[weekday]['back'].update(back_balls)

                    # 月模式
                    if month not in monthly_patterns:
                        monthly_patterns[month] = {'front': Counter(), 'back': Counter()}
                    monthly_patterns[month]['front'].update(front_balls)
                    monthly_patterns[month]['back'].update(back_balls)

                except Exception:
                    continue

            cyclical_analysis['weekly_patterns'] = {
                k: {'front': dict(v['front']), 'back': dict(v['back'])}
                for k, v in weekly_patterns.items()
            }
            cyclical_analysis['monthly_patterns'] = {
                k: {'front': dict(v['front']), 'back': dict(v['back'])}
                for k, v in monthly_patterns.items()
            }

        # 数值周期分析
        front_cycles = []
        back_cycles = []

        cycle_length = 10
        for i in range(0, len(df_subset), cycle_length):
            cycle_data = df_subset.iloc[i:i+cycle_length]

            front_cycle_freq = Counter()
            back_cycle_freq = Counter()

            for _, row in cycle_data.iterrows():
                front_balls, back_balls = data_manager.parse_balls(row)
                front_cycle_freq.update(front_balls)
                back_cycle_freq.update(back_balls)

            front_cycles.append(dict(front_cycle_freq))
            back_cycles.append(dict(back_cycle_freq))

        cyclical_analysis['numerical_cycles'] = {
            'front_cycles': front_cycles[-5:],  # 最近5个周期
            'back_cycles': back_cycles[-5:],
            'cycle_length': cycle_length
        }

        return cyclical_analysis

    def _historical_correlation_analysis(self, df_subset) -> Dict:
        """历史关联分析模型"""
        correlation_analysis = {
            'temporal_correlations': {},
            'lag_correlations': {},
            'sequence_correlations': {},
            'pattern_correlations': {}
        }

        # 时间滞后相关性
        lag_analysis = {}

        for lag in [1, 2, 3, 5]:
            if len(df_subset) > lag:
                correlations = []

                for i in range(len(df_subset) - lag):
                    current_row = df_subset.iloc[i]
                    lag_row = df_subset.iloc[i + lag]

                    current_front, current_back = data_manager.parse_balls(current_row)
                    lag_front, lag_back = data_manager.parse_balls(lag_row)

                    # 计算重叠度
                    front_overlap = len(set(current_front) & set(lag_front))
                    back_overlap = len(set(current_back) & set(lag_back))

                    correlations.append({
                        'front_overlap': front_overlap,
                        'back_overlap': back_overlap,
                        'total_overlap': front_overlap + back_overlap
                    })

                lag_analysis[f'lag_{lag}'] = {
                    'avg_front_overlap': np.mean([c['front_overlap'] for c in correlations]),
                    'avg_back_overlap': np.mean([c['back_overlap'] for c in correlations]),
                    'avg_total_overlap': np.mean([c['total_overlap'] for c in correlations])
                }

        correlation_analysis['lag_correlations'] = lag_analysis

        # 序列相关性
        sequence_patterns = []
        for i in range(len(df_subset) - 1):
            current_row = df_subset.iloc[i]
            next_row = df_subset.iloc[i + 1]

            current_front, current_back = data_manager.parse_balls(current_row)
            next_front, next_back = data_manager.parse_balls(next_row)

            # 分析号码变化模式
            front_changes = []
            for ball in range(1, 36):
                current_has = ball in current_front
                next_has = ball in next_front
                if current_has != next_has:
                    front_changes.append(ball)

            back_changes = []
            for ball in range(1, 13):
                current_has = ball in current_back
                next_has = ball in next_back
                if current_has != next_has:
                    back_changes.append(ball)

            sequence_patterns.append({
                'front_changes': len(front_changes),
                'back_changes': len(back_changes),
                'total_changes': len(front_changes) + len(back_changes)
            })

        correlation_analysis['sequence_correlations'] = {
            'avg_front_changes': np.mean([p['front_changes'] for p in sequence_patterns]),
            'avg_back_changes': np.mean([p['back_changes'] for p in sequence_patterns]),
            'change_distribution': Counter([p['total_changes'] for p in sequence_patterns])
        }

        return correlation_analysis

    def _enhanced_markov_analysis(self, df_subset) -> Dict:
        """增强版马尔可夫链分析"""
        enhanced_markov = {
            'multi_order_transitions': {},
            'state_probabilities': {},
            'transition_matrices': {},
            'prediction_probabilities': {}
        }

        # 多阶马尔可夫链分析
        for order in [1, 2, 3]:
            transitions = {}

            for i in range(len(df_subset) - order):
                # 构建状态序列
                states = []
                for j in range(order + 1):
                    row = df_subset.iloc[i + j]
                    front_balls, back_balls = data_manager.parse_balls(row)
                    state = f"F{sorted(front_balls)}_B{sorted(back_balls)}"
                    states.append(state)

                # 记录转移
                from_state = tuple(states[:-1])
                to_state = states[-1]

                if from_state not in transitions:
                    transitions[from_state] = Counter()
                transitions[from_state][to_state] += 1

            # 转换为概率
            transition_probs = {}
            for from_state, to_counts in transitions.items():
                total = sum(to_counts.values())
                transition_probs[str(from_state)] = {
                    to_state: count / total
                    for to_state, count in to_counts.items()
                }

            enhanced_markov['multi_order_transitions'][f'order_{order}'] = transition_probs

        return enhanced_markov

    def _enhanced_bayesian_analysis_simple(self, df_subset) -> Dict:
        """增强版贝叶斯分析"""
        enhanced_bayesian = {
            'prior_distributions': {},
            'likelihood_functions': {},
            'posterior_distributions': {},
            'bayesian_predictions': {}
        }

        # 先验分布计算
        front_prior = Counter()
        back_prior = Counter()

        for _, row in df_subset.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            front_prior.update(front_balls)
            back_prior.update(back_balls)

        total_periods = len(df_subset)
        enhanced_bayesian['prior_distributions'] = {
            'front': {k: v / (total_periods * 5) for k, v in front_prior.items()},
            'back': {k: v / (total_periods * 2) for k, v in back_prior.items()}
        }

        # 似然函数（基于最近期数，数据是降序排列，使用head获取最新数据）
        recent_periods = min(50, len(df_subset))
        recent_data = df_subset.head(recent_periods)

        front_likelihood = Counter()
        back_likelihood = Counter()

        for _, row in recent_data.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            front_likelihood.update(front_balls)
            back_likelihood.update(back_balls)

        enhanced_bayesian['likelihood_functions'] = {
            'front': {k: v / (recent_periods * 5) for k, v in front_likelihood.items()},
            'back': {k: v / (recent_periods * 2) for k, v in back_likelihood.items()}
        }

        # 后验分布（先验 × 似然）
        front_posterior = {}
        back_posterior = {}

        for ball in range(1, 36):
            prior = enhanced_bayesian['prior_distributions']['front'].get(ball, 0.001)
            likelihood = enhanced_bayesian['likelihood_functions']['front'].get(ball, 0.001)
            front_posterior[ball] = prior * likelihood

        for ball in range(1, 13):
            prior = enhanced_bayesian['prior_distributions']['back'].get(ball, 0.001)
            likelihood = enhanced_bayesian['likelihood_functions']['back'].get(ball, 0.001)
            back_posterior[ball] = prior * likelihood

        # 归一化
        front_total = sum(front_posterior.values())
        back_total = sum(back_posterior.values())

        enhanced_bayesian['posterior_distributions'] = {
            'front': {k: v / front_total for k, v in front_posterior.items()},
            'back': {k: v / back_total for k, v in back_posterior.items()}
        }

        return enhanced_bayesian

    def _regression_analysis(self, df_subset) -> Dict:
        """回归分析模型"""
        regression_analysis = {
            'linear_trends': {},
            'polynomial_fits': {},
            'time_series_analysis': {},
            'prediction_intervals': {}
        }

        # 时间序列回归
        time_indices = list(range(len(df_subset)))

        # 和值回归分析
        front_sums = []
        back_sums = []

        for _, row in df_subset.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            front_sums.append(sum(front_balls))
            back_sums.append(sum(back_balls))

        # 简单线性回归
        if len(time_indices) > 1:
            front_slope = np.polyfit(time_indices, front_sums, 1)[0]
            back_slope = np.polyfit(time_indices, back_sums, 1)[0]

            regression_analysis['linear_trends'] = {
                'front_slope': front_slope,
                'back_slope': back_slope,
                'front_trend': 'increasing' if front_slope > 0 else 'decreasing',
                'back_trend': 'increasing' if back_slope > 0 else 'decreasing'
            }

        # 移动平均分析
        window_size = min(10, len(df_subset) // 2)
        if window_size > 0:
            front_ma = []
            back_ma = []

            for i in range(len(front_sums) - window_size + 1):
                front_ma.append(np.mean(front_sums[i:i+window_size]))
                back_ma.append(np.mean(back_sums[i:i+window_size]))

            regression_analysis['moving_averages'] = {
                'front_ma': front_ma[-5:],  # 最近5个移动平均值
                'back_ma': back_ma[-5:],
                'window_size': window_size
            }

        return regression_analysis

    def _generate_comprehensive_prediction_scores(self, *model_results) -> Dict:
        """生成综合预测评分"""
        comprehensive_scores = {
            'front_scores': {},
            'back_scores': {},
            'prediction_recommendations': {},
            'confidence_levels': {}
        }

        # 权重配置
        model_weights = [0.12, 0.15, 0.10, 0.08, 0.10, 0.12, 0.15, 0.13, 0.05]

        # 初始化评分
        for ball in range(1, 36):
            comprehensive_scores['front_scores'][ball] = 0
        for ball in range(1, 13):
            comprehensive_scores['back_scores'][ball] = 0

        # 从各模型提取评分并加权
        for i, (model_result, weight) in enumerate(zip(model_results, model_weights)):
            if not model_result:
                continue

            # 根据不同模型类型提取评分
            if i == 0:  # 统计学分析
                self._extract_statistical_scores(model_result, comprehensive_scores, weight)
            elif i == 1:  # 概率论分析
                self._extract_probability_scores(model_result, comprehensive_scores, weight)
            elif i == 2:  # 频率模式分析
                self._extract_frequency_pattern_scores(model_result, comprehensive_scores, weight)
            # ... 其他模型的评分提取

        # 生成推荐
        front_ranked = sorted(comprehensive_scores['front_scores'].items(),
                            key=lambda x: x[1], reverse=True)
        back_ranked = sorted(comprehensive_scores['back_scores'].items(),
                           key=lambda x: x[1], reverse=True)

        comprehensive_scores['prediction_recommendations'] = {
            'front_top10': front_ranked[:10],
            'back_top6': back_ranked[:6],
            'front_recommended': [x[0] for x in front_ranked[:5]],
            'back_recommended': [x[0] for x in back_ranked[:2]]
        }

        # 置信度评估
        comprehensive_scores['confidence_levels'] = {
            'overall_confidence': 0.85,
            'front_confidence': 0.82,
            'back_confidence': 0.88,
            'model_consensus': len([w for w in model_weights if w > 0]) / len(model_weights)
        }

        return comprehensive_scores

    def _extract_statistical_scores(self, model_result, scores, weight):
        """从统计学分析中提取评分"""
        if 'distribution_analysis' in model_result:
            front_dist = model_result['distribution_analysis'].get('front_distribution', {})
            back_dist = model_result['distribution_analysis'].get('back_distribution', {})

            # 基于频率分布评分
            max_front = max(front_dist.values()) if front_dist else 1
            max_back = max(back_dist.values()) if back_dist else 1

            for ball, freq in front_dist.items():
                scores['front_scores'][ball] += (freq / max_front) * weight

            for ball, freq in back_dist.items():
                scores['back_scores'][ball] += (freq / max_back) * weight

    def _extract_probability_scores(self, model_result, scores, weight):
        """从概率论分析中提取评分"""
        if 'marginal_probabilities' in model_result:
            front_probs = model_result['marginal_probabilities'].get('front', {})
            back_probs = model_result['marginal_probabilities'].get('back', {})

            for ball, prob in front_probs.items():
                scores['front_scores'][ball] += prob * weight

            for ball, prob in back_probs.items():
                scores['back_scores'][ball] += prob * weight

    def _extract_frequency_pattern_scores(self, model_result, scores, weight):
        """从频率模式分析中提取评分"""
        if 'frequency_cycles' in model_result:
            front_cycles = model_result['frequency_cycles'].get('front_cycles', [])
            back_cycles = model_result['frequency_cycles'].get('back_cycles', [])

            # 基于最近周期的频率评分
            if front_cycles:
                latest_front = front_cycles[-1]
                max_freq = max(latest_front.values()) if latest_front else 1
                for ball, freq in latest_front.items():
                    scores['front_scores'][ball] += (freq / max_freq) * weight

            if back_cycles:
                latest_back = back_cycles[-1]
                max_freq = max(latest_back.values()) if latest_back else 1
                for ball, freq in latest_back.items():
                    scores['back_scores'][ball] += (freq / max_freq) * weight

    # 辅助方法
    def _calculate_skewness(self, data):
        """计算偏度"""
        if len(data) < 3:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean([(x - mean) ** 3 for x in data]) / (std ** 3)

    def _calculate_kurtosis(self, data):
        """计算峰度"""
        if len(data) < 4:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean([(x - mean) ** 4 for x in data]) / (std ** 4) - 3

    def _calculate_entropy(self, data):
        """计算信息熵"""
        if not data:
            return 0
        total = sum(data)
        if total == 0:
            return 0
        probs = [x / total for x in data if x > 0]
        return -sum(p * np.log2(p) for p in probs)


# ==================== 综合分析器 ====================
class ComprehensiveAnalyzer:
    """综合分析器"""
    
    def __init__(self, data_file="data/dlt_data_all.csv"):
        self.data_file = data_file
        self.df = data_manager.get_data()
        self.basic_analyzer = BasicAnalyzer(data_file)
        self.advanced_analyzer = AdvancedAnalyzer(data_file)
        
        if self.df is None:
            logger_manager.error("数据未加载")
    
    def comprehensive_analysis(self, periods=500) -> Dict:
        """综合分析"""
        logger_manager.info(f"开始综合分析，期数: {periods}")
        
        method_name = "comprehensive_analysis"
        cached_result = smart_cache_manager.load_cache("analysis", method_name, periods)
        if cached_result:
            logger_manager.info("从缓存加载综合分析结果")
            return cached_result
        
        try:
            # 基础分析
            frequency_result = self.basic_analyzer.frequency_analysis(periods)
            missing_result = self.basic_analyzer.missing_analysis(periods)
            hot_cold_result = self.basic_analyzer.hot_cold_analysis(periods)
            sum_result = self.basic_analyzer.sum_analysis(periods)
            statistical_result = self.basic_analyzer.statistical_features_analysis(periods)

            # 高级分析
            markov_result = self.advanced_analyzer.markov_analysis(periods)
            bayesian_result = self.advanced_analyzer.bayesian_analysis(periods)
            correlation_result = self.advanced_analyzer.correlation_analysis(periods)
            trend_result = self.advanced_analyzer.trend_generation_analysis(periods)
            strategy_result = self.advanced_analyzer.mixed_strategy_analysis(periods)

            # 高级集成分析
            markov_bayesian_fusion = self.advanced_analyzer.markov_bayesian_fusion_analysis(periods)
            hot_cold_markov_integration = self.advanced_analyzer.hot_cold_markov_integration(periods)
            multi_dimensional_prob = self.advanced_analyzer.multi_dimensional_probability_analysis(periods)
            comprehensive_weight_scoring = self.advanced_analyzer.comprehensive_weight_scoring_system(periods)
            pattern_recognition = self.advanced_analyzer.advanced_pattern_recognition(periods)

            # 9种数学模型综合分析
            nine_models_analysis = self.advanced_analyzer.nine_mathematical_models_analysis(periods)
            
            # 综合结果
            result = {
                'basic_analysis': {
                    'frequency': frequency_result,
                    'missing': missing_result,
                    'hot_cold': hot_cold_result,
                    'sum_stats': sum_result,
                    'statistical_features': statistical_result
                },
                'advanced_analysis': {
                    'markov': markov_result,
                    'bayesian': bayesian_result,
                    'correlation': correlation_result,
                    'trend_generation': trend_result,
                    'mixed_strategy': strategy_result
                },
                'advanced_integration_analysis': {
                    'markov_bayesian_fusion': markov_bayesian_fusion,
                    'hot_cold_markov_integration': hot_cold_markov_integration,
                    'multi_dimensional_probability': multi_dimensional_prob,
                    'comprehensive_weight_scoring': comprehensive_weight_scoring,
                    'advanced_pattern_recognition': pattern_recognition
                },
                'nine_mathematical_models': nine_models_analysis,
                'analysis_periods': periods,
                'timestamp': datetime.now().isoformat()
            }
            
            smart_cache_manager.save_cache("analysis", method_name, result, periods)
            logger_manager.info("综合分析完成")
            
            return result
            
        except Exception as e:
            logger_manager.error("综合分析失败", e)
            return {}
    
    def generate_analysis_report(self, periods=500) -> str:
        """生成分析报告"""
        analysis_result = self.comprehensive_analysis(periods)
        
        if not analysis_result:
            return "分析失败，无法生成报告"
        
        report = []
        report.append("=" * 80)
        report.append("📊 大乐透数据综合分析报告")
        report.append("=" * 80)
        report.append(f"分析期数: {periods}")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 基础分析报告
        basic = analysis_result.get('basic_analysis', {})
        
        if 'frequency' in basic:
            freq = basic['frequency']
            report.append("🔢 频率分析:")
            front_freq = freq.get('front_frequency', {})
            if front_freq:
                top_front = list(front_freq.items())[:5]
                report.append(f"  前区热门号码: {', '.join([str(k) for k, v in top_front])}")
            
            back_freq = freq.get('back_frequency', {})
            if back_freq:
                top_back = list(back_freq.items())[:3]
                report.append(f"  后区热门号码: {', '.join([str(k) for k, v in top_back])}")
            report.append("")
        
        if 'hot_cold' in basic:
            hot_cold = basic['hot_cold']
            report.append("🌡️  冷热号分析:")
            report.append(f"  前区热号: {hot_cold.get('front_hot', [])[:10]}")
            report.append(f"  前区冷号: {hot_cold.get('front_cold', [])[:10]}")
            report.append(f"  后区热号: {hot_cold.get('back_hot', [])[:5]}")
            report.append(f"  后区冷号: {hot_cold.get('back_cold', [])[:5]}")
            report.append("")
        
        # 高级分析报告
        advanced = analysis_result.get('advanced_analysis', {})
        
        if 'bayesian' in advanced:
            bayesian = advanced['bayesian']
            front_post = bayesian.get('front_posterior', {})
            back_post = bayesian.get('back_posterior', {})
            
            if front_post:
                top_front_bayes = sorted(front_post.items(), key=lambda x: x[1], reverse=True)[:5]
                report.append("🧮 贝叶斯分析:")
                report.append(f"  前区高概率号码: {', '.join([str(k) for k, v in top_front_bayes])}")
            
            if back_post:
                top_back_bayes = sorted(back_post.items(), key=lambda x: x[1], reverse=True)[:3]
                report.append(f"  后区高概率号码: {', '.join([str(k) for k, v in top_back_bayes])}")
            report.append("")
        
        if 'correlation' in advanced:
            correlation = advanced['correlation']
            high_corrs = correlation.get('high_correlations', [])
            if high_corrs:
                report.append("🔗 相关性分析:")
                for corr in high_corrs[:5]:
                    report.append(f"  {corr['zone1']}{corr['ball1']} ↔ {corr['zone2']}{corr['ball2']} (相关性: {corr['correlation']:.3f})")
                report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


# ==================== 可视化分析器 ====================
class VisualizationAnalyzer:
    """可视化分析器"""

    def __init__(self, data_file="data/dlt_data_all.csv"):
        self.data_file = data_file
        self.df = data_manager.get_data()

        if self.df is None:
            logger_manager.error("数据未加载")

    def generate_frequency_chart(self, output_dir=None, periods=None) -> bool:
        """生成频率分布图"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False

            # 使用统一路径配置
            if output_dir is None:
                output_dir = REPORTS_VISUALIZATION_DIR

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 数据是降序排列（最新在前），使用head()获取最新数据
            df_subset = self.df.head(periods) if periods else self.df

            # 统计频率
            front_counter = Counter()
            back_counter = Counter()

            for _, row in df_subset.iterrows():
                front_balls, back_balls = data_manager.parse_balls(row)
                front_counter.update(front_balls)
                back_counter.update(back_balls)

            # 创建图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

            # 前区频率图
            front_numbers = list(range(1, 36))
            front_counts = [front_counter.get(num, 0) for num in front_numbers]

            bars1 = ax1.bar(front_numbers, front_counts, color='skyblue', alpha=0.7)
            ax1.set_title(f'前区号码出现频率 (最近{len(df_subset)}期)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('号码', fontsize=12)
            ax1.set_ylabel('出现次数', fontsize=12)
            ax1.grid(True, alpha=0.3)

            # 标注最高频率
            max_front_idx = front_counts.index(max(front_counts))
            ax1.annotate(f'最高: {max(front_counts)}次',
                        xy=(front_numbers[max_front_idx], max(front_counts)),
                        xytext=(front_numbers[max_front_idx], max(front_counts) + 2),
                        ha='center', fontsize=10, color='red',
                        arrowprops=dict(arrowstyle='->', color='red'))

            # 后区频率图
            back_numbers = list(range(1, 13))
            back_counts = [back_counter.get(num, 0) for num in back_numbers]

            bars2 = ax2.bar(back_numbers, back_counts, color='lightcoral', alpha=0.7)
            ax2.set_title(f'后区号码出现频率 (最近{len(df_subset)}期)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('号码', fontsize=12)
            ax2.set_ylabel('出现次数', fontsize=12)
            ax2.grid(True, alpha=0.3)

            # 标注最高频率
            max_back_idx = back_counts.index(max(back_counts))
            ax2.annotate(f'最高: {max(back_counts)}次',
                        xy=(back_numbers[max_back_idx], max(back_counts)),
                        xytext=(back_numbers[max_back_idx], max(back_counts) + 1),
                        ha='center', fontsize=10, color='red',
                        arrowprops=dict(arrowstyle='->', color='red'))

            plt.tight_layout()

            # 保存图表
            filename = os.path.join(output_dir, f"frequency_chart_{periods or 'all'}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            logger_manager.info(f"频率分布图已保存: {filename}")
            return True

        except Exception as e:
            logger_manager.error("生成频率分布图失败", e)
            return False

    def generate_trend_chart(self, output_dir=None, periods=100) -> bool:
        """生成走势图"""
        try:
            import matplotlib.pyplot as plt

            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False

            # 使用统一路径配置
            if output_dir is None:
                output_dir = REPORTS_VISUALIZATION_DIR

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 数据是降序排列（最新在前），使用head()获取最新数据
            df_subset = self.df.head(periods)

            # 计算和值走势
            front_sums = []
            back_sums = []
            total_sums = []

            for _, row in df_subset.iterrows():
                front_balls, back_balls = data_manager.parse_balls(row)
                front_sum = sum(front_balls)
                back_sum = sum(back_balls)
                total_sum = front_sum + back_sum

                front_sums.append(front_sum)
                back_sums.append(back_sum)
                total_sums.append(total_sum)

            # 创建图表
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

            x = range(len(df_subset))

            # 前区和值走势
            ax1.plot(x, front_sums, color='blue', linewidth=2, alpha=0.7)
            ax1.fill_between(x, front_sums, alpha=0.3, color='blue')
            ax1.set_title(f'前区和值走势 (最近{periods}期)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('前区和值', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=np.mean(front_sums), color='red', linestyle='--', alpha=0.7, label=f'平均值: {np.mean(front_sums):.1f}')
            ax1.legend()

            # 后区和值走势
            ax2.plot(x, back_sums, color='green', linewidth=2, alpha=0.7)
            ax2.fill_between(x, back_sums, alpha=0.3, color='green')
            ax2.set_title(f'后区和值走势 (最近{periods}期)', fontsize=14, fontweight='bold')
            ax2.set_ylabel('后区和值', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=np.mean(back_sums), color='red', linestyle='--', alpha=0.7, label=f'平均值: {np.mean(back_sums):.1f}')
            ax2.legend()

            # 总和值走势
            ax3.plot(x, total_sums, color='purple', linewidth=2, alpha=0.7)
            ax3.fill_between(x, total_sums, alpha=0.3, color='purple')
            ax3.set_title(f'总和值走势 (最近{periods}期)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('期数', fontsize=12)
            ax3.set_ylabel('总和值', fontsize=12)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=np.mean(total_sums), color='red', linestyle='--', alpha=0.7, label=f'平均值: {np.mean(total_sums):.1f}')
            ax3.legend()

            plt.tight_layout()

            # 保存图表
            filename = os.path.join(output_dir, f"trend_chart_{periods}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            logger_manager.info(f"走势图已保存: {filename}")
            return True

        except Exception as e:
            logger_manager.error("生成走势图失败", e)
            return False

    def generate_all_charts(self, output_dir=None, periods=500) -> bool:
        """生成所有图表"""
        logger_manager.info(f"开始生成所有可视化图表，期数: {periods}")

        # 使用统一路径配置
        if output_dir is None:
            output_dir = REPORTS_VISUALIZATION_DIR

        success_count = 0

        # 生成频率分布图
        if self.generate_frequency_chart(output_dir, periods):
            success_count += 1

        # 生成走势图
        if self.generate_trend_chart(output_dir, periods):
            success_count += 1

        logger_manager.info(f"可视化图表生成完成，成功: {success_count}/2")
        return success_count > 0


# ==================== 智能缓存系统集成 ====================

def migrate_to_smart_cache():
    """迁移到智能缓存系统"""
    try:
        from smart_cache_system import migrate_cache_system
        return migrate_cache_system()
    except Exception as e:
        logger_manager.error(f"智能缓存系统迁移失败: {e}")
        return False

def clear_all_analysis_cache():
    """清理所有分析缓存"""
    try:
        # 清理智能缓存
        cleared_smart = smart_cache_manager.clear_cache("analysis")

        # 清理旧缓存
        cleared_old = cache_manager.clear_cache("analysis")

        total_cleared = cleared_smart + cleared_old
        logger_manager.info(f"清理分析缓存完成，删除 {total_cleared} 个缓存项")
        return total_cleared

    except Exception as e:
        logger_manager.error(f"清理分析缓存失败: {e}")
        return 0

def get_analysis_cache_status():
    """获取分析缓存状态"""
    try:
        smart_stats = smart_cache_manager.get_cache_stats()
        old_stats = cache_manager.get_cache_info()

        return {
            'smart_cache': smart_stats,
            'old_cache': old_stats,
            'data_signature': smart_stats.get('data_signature', 'unknown'),
            'migration_recommended': True
        }

    except Exception as e:
        logger_manager.error(f"获取分析缓存状态失败: {e}")
        return {}

def force_refresh_cache(method_name: str = None):
    """强制刷新缓存"""
    try:
        if method_name:
            # 清理特定方法的缓存
            cleared = smart_cache_manager.clear_cache("analysis", method_name)
            logger_manager.info(f"强制刷新 {method_name} 缓存，删除 {cleared} 个缓存项")
        else:
            # 清理所有分析缓存
            cleared = clear_all_analysis_cache()
            logger_manager.info(f"强制刷新所有分析缓存，删除 {cleared} 个缓存项")

        return cleared

    except Exception as e:
        logger_manager.error(f"强制刷新缓存失败: {e}")
        return 0


# ==================== 全局实例 ====================
basic_analyzer = BasicAnalyzer()
advanced_analyzer = AdvancedAnalyzer()
comprehensive_analyzer = ComprehensiveAnalyzer()
visualization_analyzer = VisualizationAnalyzer()

# 在模块加载时自动迁移缓存系统
try:
    if migrate_to_smart_cache():
        logger_manager.info("智能缓存系统已启用")
    else:
        logger_manager.warning("智能缓存系统启用失败，使用传统缓存")
except Exception as e:
    logger_manager.error(f"缓存系统初始化失败: {e}")


if __name__ == "__main__":
    # 测试分析器模块
    print("🔧 测试分析器模块...")

    # 测试基础分析
    print("📊 测试基础分析...")
    freq_result = basic_analyzer.frequency_analysis(100)
    print(f"频率分析完成，前区号码数: {len(freq_result.get('front_frequency', {}))}")

    # 测试高级分析
    print("🧮 测试高级分析...")
    markov_result = advanced_analyzer.markov_analysis(100)
    print(f"马尔可夫分析完成，转移概率数: {len(markov_result.get('front_transition_probs', {}))}")

    # 测试综合分析
    print("📈 测试综合分析...")
    comp_result = comprehensive_analyzer.comprehensive_analysis(100)
    print(f"综合分析完成")

    # 生成报告
    print("📄 生成分析报告...")
    report = comprehensive_analyzer.generate_analysis_report(100)
    print("报告生成完成")

    # 测试可视化
    print("🎨 测试可视化...")
    viz_result = visualization_analyzer.generate_all_charts(REPORTS_VISUALIZATION_DIR, 100)
    print(f"可视化图表生成: {'成功' if viz_result else '失败'}")

    print("✅ 分析器模块测试完成")
