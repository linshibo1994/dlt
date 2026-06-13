#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强马尔可夫链模块
提供多阶马尔可夫链和自适应马尔可夫链预测

优化内容:
- 单球级别状态建模，解决状态空间爆炸问题
- 拉普拉斯平滑，消除零概率转移
- 时间衰减加权，近期数据权重更高
- 多步转移概率融合（1阶+2阶+全局先验）
- 共现关系矩阵，考虑号码间关联
- 区间分布约束，确保号码分布合理
"""

import os
import sys
import time
import random
import math
import ast
import numpy as np
import pandas as pd
import yaml
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict, Counter
from datetime import datetime
from itertools import combinations

# 尝试导入核心模块
try:
    from core_modules import logger_manager, data_manager, cache_manager
    from smart_cache_system import smart_cache_manager
except ImportError:
    # 如果在不同目录运行，添加父目录到路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core_modules import logger_manager, data_manager, cache_manager
    from smart_cache_system import smart_cache_manager


def _load_markov_config() -> Dict:
    """加载马尔可夫链配置"""
    try:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'config', 'prediction.yaml'
        )
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('prediction_methods', {}).get('traditional_ml', {}).get('markov', {})
    except Exception:
        # 返回默认配置
        return {
            'smoothing_alpha': 0.05,
            'global_mix': 0.15,
            'decay_enabled': True,
            'decay_half_life': 200,
            'decay_min_weight': 0.2,
            'posterior_mix': 0.15,
        }


class EnhancedMarkovAnalyzer:
    """增强马尔可夫链分析器"""

    def __init__(self):
        self.df = data_manager.get_data()
        if self.df is None:
            logger_manager.error("数据未加载")
        self._config = _load_markov_config()

    def multi_order_markov_analysis(self, periods=500, max_order=3) -> Dict:
        """多阶马尔可夫链分析

        Args:
            periods: 分析期数
            max_order: 最大马尔可夫链阶数 (1-3)

        Returns:
            Dict: 包含各阶马尔可夫链分析结果的字典
        """
        if self.df is None:
            return {}

        # 检查参数有效性
        max_order = min(max(1, max_order), 3)  # 限制在1-3之间

        method_name = "multi_order_markov_analysis_v3"
        cached_result = smart_cache_manager.load_cache("analysis", method_name, periods, max_order=max_order)
        if cached_result:
            return self._restore_cached_result(cached_result)

        df_subset = self.df.head(periods)

        result = {
            'orders': {},
            'analysis_periods': periods,
            'max_order': max_order
        }

        # 对每个阶数进行分析
        for order in range(1, max_order + 1):
            order_result = self._analyze_nth_order_markov(df_subset, order)
            result['orders'][order] = order_result

        # 缓存前去除 _raw 键（包含 tuple/int 键，无法 JSON 序列化）
        cache_result = {
            'orders': {},
            'analysis_periods': periods,
            'max_order': max_order
        }
        for o, o_result in result['orders'].items():
            cache_result['orders'][o] = {k: v for k, v in o_result.items() if not k.endswith('_raw')}

        smart_cache_manager.save_cache("analysis", method_name, cache_result, periods, max_order=max_order)
        return result

    def _restore_cached_result(self, cached_result: Dict) -> Dict:
        """恢复 JSON 缓存中的整数键和 tuple 条件键"""
        if not isinstance(cached_result, dict):
            return cached_result

        restored = dict(cached_result)
        orders = restored.get('orders', {})
        if not isinstance(orders, dict):
            return restored

        restored_orders = {}
        for order_key, order_result in orders.items():
            try:
                order_int = int(order_key)
            except (TypeError, ValueError):
                order_int = order_key

            if not isinstance(order_result, dict):
                restored_orders[order_int] = order_result
                continue

            normalized = dict(order_result)
            normalized['front_ball_probs_raw'] = self._int_nested_dict(
                normalized.get('front_ball_probs_raw') or normalized.get('front_ball_probs')
            )
            normalized['back_ball_probs_raw'] = self._int_nested_dict(
                normalized.get('back_ball_probs_raw') or normalized.get('back_ball_probs')
            )
            normalized['front_pair_probs_raw'] = self._tuple_nested_dict(
                normalized.get('front_pair_probs_raw') or normalized.get('front_pair_probs')
            )
            normalized['back_pair_probs_raw'] = self._tuple_nested_dict(
                normalized.get('back_pair_probs_raw') or normalized.get('back_pair_probs')
            )
            normalized['front_triple_probs_raw'] = self._tuple_nested_dict(
                normalized.get('front_triple_probs_raw') or normalized.get('front_triple_probs')
            )
            normalized['back_triple_probs_raw'] = self._tuple_nested_dict(
                normalized.get('back_triple_probs_raw') or normalized.get('back_triple_probs')
            )
            normalized['front_cooccurrence_raw'] = self._int_nested_dict(
                normalized.get('front_cooccurrence_raw') or normalized.get('front_cooccurrence')
            )
            normalized['back_cooccurrence_raw'] = self._int_nested_dict(
                normalized.get('back_cooccurrence_raw') or normalized.get('back_cooccurrence')
            )
            normalized['front_freq_probs_raw'] = self._int_float_dict(
                normalized.get('front_freq_probs_raw') or normalized.get('front_freq_probs')
            )
            normalized['back_freq_probs_raw'] = self._int_float_dict(
                normalized.get('back_freq_probs_raw') or normalized.get('back_freq_probs')
            )
            restored_orders[order_int] = normalized

        restored['orders'] = restored_orders
        return restored

    @staticmethod
    def _int_float_dict(data: Dict) -> Dict[int, float]:
        result = {}
        if not isinstance(data, dict):
            return result
        for key, value in data.items():
            try:
                result[int(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return result

    @classmethod
    def _int_nested_dict(cls, data: Dict) -> Dict[int, Dict[int, float]]:
        result = {}
        if not isinstance(data, dict):
            return result
        for key, value in data.items():
            if not isinstance(value, dict):
                continue
            try:
                result[int(key)] = cls._int_float_dict(value)
            except (TypeError, ValueError):
                continue
        return result

    @staticmethod
    def _parse_tuple_key(key) -> Optional[Tuple[int, ...]]:
        if isinstance(key, tuple):
            return tuple(int(item) for item in key)
        if isinstance(key, list):
            return tuple(int(item) for item in key)
        try:
            parsed = ast.literal_eval(str(key))
        except (SyntaxError, ValueError):
            return None
        if isinstance(parsed, (tuple, list)):
            try:
                return tuple(int(item) for item in parsed)
            except (TypeError, ValueError):
                return None
        return None

    @classmethod
    def _tuple_nested_dict(cls, data: Dict) -> Dict[Tuple[int, ...], Dict[int, float]]:
        result = {}
        if not isinstance(data, dict):
            return result
        for key, value in data.items():
            tuple_key = cls._parse_tuple_key(key)
            if tuple_key is None or not isinstance(value, dict):
                continue
            result[tuple_key] = cls._int_float_dict(value)
        return result

    def _analyze_nth_order_markov(self, df_subset, order=1) -> Dict:
        """分析n阶马尔可夫链 - 使用单球级别状态建模

        Args:
            df_subset: 数据子集
            order: 马尔可夫链阶数

        Returns:
            Dict: n阶马尔可夫链分析结果，包含单球转移矩阵和共现矩阵
        """
        alpha = self._config.get('smoothing_alpha', 0.05)
        decay_enabled = self._config.get('decay_enabled', True)
        decay_half_life = self._config.get('decay_half_life', 200)
        decay_min_weight = self._config.get('decay_min_weight', 0.2)

        num_front_balls = 35
        num_back_balls = 12

        # 解析所有期的号码
        all_fronts = []
        all_backs = []
        for i in range(len(df_subset)):
            front, back = data_manager.parse_balls(df_subset.iloc[i])
            all_fronts.append(front)
            all_backs.append(back)

        # 反转列表，使 index=0 为最早数据，index[-1] 为最新数据
        # 这样 all_data[i] -> all_data[i+1] 表示从过去到未来的转移方向
        all_fronts = list(reversed(all_fronts))
        all_backs = list(reversed(all_backs))
        data_len = len(all_fronts)

        # --- 单球转移矩阵（所有阶数都构建，用于融合） ---
        front_ball_trans = defaultdict(lambda: defaultdict(float))
        back_ball_trans = defaultdict(lambda: defaultdict(float))

        for i in range(data_len - 1):
            # 衰减权重: data_len-2-i 使得最新数据(靠近末尾)权重最高
            w = self._decay_weight(data_len - 2 - i, decay_enabled, decay_half_life, decay_min_weight)
            for a in all_fronts[i]:
                for b in all_fronts[i + 1]:
                    front_ball_trans[a][b] += w
            for a in all_backs[i]:
                for b in all_backs[i + 1]:
                    back_ball_trans[a][b] += w

        # --- 二阶球对转移矩阵（order >= 2 时构建） ---
        front_pair_trans = defaultdict(lambda: defaultdict(float))
        back_pair_trans = defaultdict(lambda: defaultdict(float))
        front_triple_trans = defaultdict(lambda: defaultdict(float))
        back_triple_trans = defaultdict(lambda: defaultdict(float))

        if order >= 2:
            for i in range(data_len - 2):
                w = self._decay_weight(data_len - 3 - i, decay_enabled, decay_half_life, decay_min_weight)
                # 前区: 第i期(较早)和第i+1期(较新)的球对 -> 第i+2期(最新)的每个球
                # 保留时序方向: pair_key = (较早期球, 较新期球)
                for a in all_fronts[i]:
                    for b in all_fronts[i + 1]:
                        pair_key = (a, b)
                        for c in all_fronts[i + 2]:
                            front_pair_trans[pair_key][c] += w

                for a in all_backs[i]:
                    for b in all_backs[i + 1]:
                        pair_key = (a, b)
                        for c in all_backs[i + 2]:
                            back_pair_trans[pair_key][c] += w

        # --- 三阶球组三元转移矩阵（order >= 3 时构建） ---
        if order >= 3:
            for i in range(data_len - 3):
                w = self._decay_weight(data_len - 4 - i, decay_enabled, decay_half_life, decay_min_weight)
                # triple_key = (更早期球, 中间期球, 最近期球)
                for a in all_fronts[i]:
                    for b in all_fronts[i + 1]:
                        for c in all_fronts[i + 2]:
                            triple_key = (a, b, c)
                            for d in all_fronts[i + 3]:
                                front_triple_trans[triple_key][d] += w

                for a in all_backs[i]:
                    for b in all_backs[i + 1]:
                        for c in all_backs[i + 2]:
                            triple_key = (a, b, c)
                            for d in all_backs[i + 3]:
                                back_triple_trans[triple_key][d] += w

        # --- 共现矩阵 ---
        front_cooccur = defaultdict(lambda: defaultdict(float))
        back_cooccur = defaultdict(lambda: defaultdict(float))

        for i in range(data_len):
            w = self._decay_weight(data_len - 1 - i, decay_enabled, decay_half_life, decay_min_weight)
            for a, b in combinations(all_fronts[i], 2):
                front_cooccur[a][b] += w
                front_cooccur[b][a] += w
            for a, b in combinations(all_backs[i], 2):
                back_cooccur[a][b] += w
                back_cooccur[b][a] += w

        # --- 全局频率统计 ---
        front_freq = defaultdict(float)
        back_freq = defaultdict(float)
        for i in range(data_len):
            w = self._decay_weight(data_len - 1 - i, decay_enabled, decay_half_life, decay_min_weight)
            for ball in all_fronts[i]:
                front_freq[ball] += w
            for ball in all_backs[i]:
                back_freq[ball] += w

        # --- 转换为概率（带拉普拉斯平滑） ---
        front_ball_probs = self._transitions_to_probs(front_ball_trans, alpha, num_front_balls)
        back_ball_probs = self._transitions_to_probs(back_ball_trans, alpha, num_back_balls)

        front_pair_probs_raw = self._transitions_to_probs(front_pair_trans, alpha, num_front_balls)
        back_pair_probs_raw = self._transitions_to_probs(back_pair_trans, alpha, num_back_balls)
        front_triple_probs_raw = self._transitions_to_probs(front_triple_trans, alpha, num_front_balls)
        back_triple_probs_raw = self._transitions_to_probs(back_triple_trans, alpha, num_back_balls)

        # 将 tuple 键转为字符串以支持 JSON 缓存序列化，运行时使用 _pair_probs_raw
        front_pair_probs = {str(k): {str(bk): bv for bk, bv in v.items()} for k, v in front_pair_probs_raw.items()}
        back_pair_probs = {str(k): {str(bk): bv for bk, bv in v.items()} for k, v in back_pair_probs_raw.items()}
        front_triple_probs = {str(k): {str(bk): bv for bk, bv in v.items()} for k, v in front_triple_probs_raw.items()}
        back_triple_probs = {str(k): {str(bk): bv for bk, bv in v.items()} for k, v in back_triple_probs_raw.items()}

        # 单球转移概率也转换键为字符串（缓存序列化需要）
        front_ball_probs_ser = {str(k): {str(bk): bv for bk, bv in v.items()} for k, v in front_ball_probs.items()}
        back_ball_probs_ser = {str(k): {str(bk): bv for bk, bv in v.items()} for k, v in back_ball_probs.items()}

        # 共现矩阵也转换键为字符串
        front_cooccur_serializable = {str(k): {str(ck): cv for ck, cv in v.items()} for k, v in front_cooccur.items()}
        back_cooccur_serializable = {str(k): {str(ck): cv for ck, cv in v.items()} for k, v in back_cooccur.items()}

        # 全局频率归一化（序列化版本键转为字符串，raw版本保留int键）
        front_total = sum(front_freq.values()) or 1.0
        back_total = sum(back_freq.values()) or 1.0
        front_freq_probs = {str(b): c / front_total for b, c in front_freq.items()}
        back_freq_probs = {str(b): c / back_total for b, c in back_freq.items()}
        front_freq_probs_raw = {b: c / front_total for b, c in front_freq.items()}
        back_freq_probs_raw = {b: c / back_total for b, c in back_freq.items()}
        back_freq_probs = {str(b): c / back_total for b, c in back_freq.items()}

        # --- 兼容旧格式的转移概率（字符串键） ---
        # 保持原有格式供其他方法（如1阶、3阶）使用
        front_transitions_compat = defaultdict(lambda: defaultdict(int))
        back_transitions_compat = defaultdict(lambda: defaultdict(int))

        for i in range(len(df_subset) - order):
            condition_front = []
            condition_back = []
            for j in range(order):
                condition_front.extend(all_fronts[i + j])
                condition_back.extend(all_backs[i + j])

            next_front = all_fronts[i + order]
            next_back = all_backs[i + order]

            condition_front_tuple = tuple(sorted(condition_front))
            condition_back_tuple = tuple(sorted(condition_back))

            for next_ball in next_front:
                front_transitions_compat[condition_front_tuple][next_ball] += 1
            for next_ball in next_back:
                back_transitions_compat[condition_back_tuple][next_ball] += 1

        front_probs_compat = {}
        for condition, to_dict in front_transitions_compat.items():
            total = sum(to_dict.values())
            if total > 0:
                front_probs_compat[str(condition)] = {
                    to_ball: count / total for to_ball, count in to_dict.items()
                }

        back_probs_compat = {}
        for condition, to_dict in back_transitions_compat.items():
            total = sum(to_dict.values())
            if total > 0:
                back_probs_compat[str(condition)] = {
                    to_ball: count / total for to_ball, count in to_dict.items()
                }

        front_stats = self._calculate_transition_stats(front_transitions_compat)
        back_stats = self._calculate_transition_stats(back_transitions_compat)

        return {
            # 兼容旧格式
            'front_transition_probs': front_probs_compat,
            'back_transition_probs': back_probs_compat,
            'front_stats': front_stats,
            'back_stats': back_stats,
            'order': order,
            # 新增: 单球转移概率（序列化版本用于缓存，raw版本用于预测）
            'front_ball_probs': front_ball_probs_ser,
            'back_ball_probs': back_ball_probs_ser,
            'front_ball_probs_raw': front_ball_probs,
            'back_ball_probs_raw': back_ball_probs,
            # 新增: 球对转移概率（二阶，tuple键原始版本供预测使用）
            'front_pair_probs': front_pair_probs,
            'back_pair_probs': back_pair_probs,
            'front_pair_probs_raw': front_pair_probs_raw,
            'back_pair_probs_raw': back_pair_probs_raw,
            # 新增: 球组三元转移概率（三阶，tuple键原始版本供预测使用）
            'front_triple_probs': front_triple_probs,
            'back_triple_probs': back_triple_probs,
            'front_triple_probs_raw': front_triple_probs_raw,
            'back_triple_probs_raw': back_triple_probs_raw,
            # 新增: 共现矩阵（原始dict版本供预测使用）
            'front_cooccurrence': front_cooccur_serializable,
            'back_cooccurrence': back_cooccur_serializable,
            'front_cooccurrence_raw': dict(front_cooccur),
            'back_cooccurrence_raw': dict(back_cooccur),
            # 新增: 全局频率
            'front_freq_probs': front_freq_probs,
            'back_freq_probs': back_freq_probs,
            'front_freq_probs_raw': front_freq_probs_raw,
            'back_freq_probs_raw': back_freq_probs_raw,
        }

    @staticmethod
    def _decay_weight(index: int, enabled: bool, half_life: float, min_weight: float) -> float:
        """计算时间衰减权重

        index=0 表示最新数据（权重最高），index 越大表示越早的数据
        """
        if not enabled:
            return 1.0
        return max(min_weight, 0.5 ** (index / max(half_life, 1.0)))

    @staticmethod
    def _transitions_to_probs(transitions: dict, alpha: float, num_possible: int) -> Dict:
        """将转移计数转换为带拉普拉斯平滑的概率分布

        prob(ball|state) = (count + alpha) / (total + alpha * num_possible)
        """
        probs = {}
        for state, to_dict in transitions.items():
            total = sum(to_dict.values())
            denominator = total + alpha * num_possible
            if denominator > 0:
                state_probs = {}
                for ball in range(1, num_possible + 1):
                    count = to_dict.get(ball, 0)
                    state_probs[ball] = (count + alpha) / denominator
                probs[state] = state_probs
        return probs

    def _calculate_transition_stats(self, transitions):
        """计算转移矩阵的统计信息"""
        stats = {
            'total_transitions': 0,
            'unique_states': len(transitions),
            'avg_transitions_per_state': 0,
            'max_probability': 0,
            'min_probability': 1.0 if transitions else 0.0
        }

        for from_state, to_dict in transitions.items():
            total = sum(to_dict.values())
            stats['total_transitions'] += total

            if total > 0:
                max_prob = max(to_dict.values()) / total
                min_prob = min(to_dict.values()) / total

                stats['max_probability'] = max(stats['max_probability'], max_prob)
                stats['min_probability'] = min(stats['min_probability'], min_prob)

        if stats['unique_states'] > 0:
            stats['avg_transitions_per_state'] = stats['total_transitions'] / stats['unique_states']

        return stats


class EnhancedMarkovPredictor:
    """增强马尔可夫链预测器"""

    def __init__(self):
        self.df = data_manager.get_data()
        self.analyzer = EnhancedMarkovAnalyzer()
        if self.df is None:
            logger_manager.error("数据未加载")
        self._config = _load_markov_config()

    def multi_order_markov_predict(self, count=1, periods=500, order=1) -> List[Tuple[List[int], List[int]]]:
        """多阶马尔可夫链预测

        使用单球级别建模，按阶数融合最近1-3期条件状态、全局先验和共现约束。

        Args:
            count: 预测注数
            periods: 分析期数
            order: 马尔可夫链阶数 (1-3)

        Returns:
            List[Tuple[List[int], List[int]]]: 预测结果列表
        """
        order = min(max(1, order), 3)

        markov_result = self.analyzer.multi_order_markov_analysis(periods, max_order=order)

        if not markov_result or 'orders' not in markov_result or order not in markov_result['orders']:
            logger_manager.error(f"{order}阶马尔可夫分析结果不可用")
            return []

        order_result = markov_result['orders'][order]

        if order == 1:
            return self._predict_1st_order_optimized(count, periods, order_result)

        if order == 2:
            return self._predict_2nd_order_optimized(count, periods, order_result, markov_result)

        if order == 3:
            return self._predict_3rd_order_optimized(count, periods, order_result, markov_result)

        return self._predict_legacy(count, periods, order, order_result)

    def _recent_periods(self, period_count: int) -> Tuple[List[List[int]], List[List[int]]]:
        """获取最近 period_count 期号码，返回顺序为最新 -> 更早"""
        recent_fronts = []
        recent_backs = []

        try:
            usable_count = min(period_count, len(self.df) if self.df is not None else 0)
            for i in range(usable_count):
                front, back = data_manager.parse_balls(self.df.iloc[i])
                recent_fronts.append([int(ball) for ball in front])
                recent_backs.append([int(ball) for ball in back])
        except Exception as e:
            logger_manager.debug(f"获取最近期号码失败: {e}")

        default_fronts = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
        ]
        default_backs = [
            [1, 2],
            [3, 4],
            [5, 6],
        ]

        while len(recent_fronts) < period_count:
            recent_fronts.append(default_fronts[len(recent_fronts) % len(default_fronts)])
        while len(recent_backs) < period_count:
            recent_backs.append(default_backs[len(recent_backs) % len(default_backs)])

        return recent_fronts[:period_count], recent_backs[:period_count]

    def _predict_1st_order_optimized(self, count: int, periods: int,
                                      order_result: Dict) -> List[Tuple[List[int], List[int]]]:
        """一阶马尔可夫预测：融合最近一期单球转移和全局频率先验"""
        posterior_mix = self._config.get('posterior_mix', 0.15)

        front_ball_probs = order_result.get('front_ball_probs_raw', order_result.get('front_ball_probs', {}))
        back_ball_probs = order_result.get('back_ball_probs_raw', order_result.get('back_ball_probs', {}))
        front_cooccur = order_result.get('front_cooccurrence_raw', order_result.get('front_cooccurrence', {}))
        back_cooccur = order_result.get('back_cooccurrence_raw', order_result.get('back_cooccurrence', {}))
        front_freq = order_result.get('front_freq_probs_raw', order_result.get('front_freq_probs', {}))
        back_freq = order_result.get('back_freq_probs_raw', order_result.get('back_freq_probs', {}))

        latest_front, latest_back = self._recent_periods(1)

        predictions = []
        for i in range(count):
            strategy_seed = int(time.time() * 1000000) + i * 1000

            front_dist = self._build_fused_distribution(
                latest_front, front_ball_probs, {}, front_freq, front_cooccur, 35, posterior_mix
            )
            back_dist = self._build_fused_distribution(
                latest_back, back_ball_probs, {}, back_freq, back_cooccur, 12, posterior_mix
            )

            front_balls = self._select_balls_with_strategy(
                front_dist, 5, 35, i, strategy_seed, front_cooccur, apply_zone_constraint=True
            )
            back_balls = self._select_balls_with_strategy(
                back_dist, 2, 12, i, strategy_seed + 500, back_cooccur, apply_zone_constraint=False
            )

            predictions.append((sorted(front_balls), sorted(back_balls)))

        return predictions

    def _predict_2nd_order_optimized(self, count: int, periods: int,
                                      order_result: Dict, full_result: Dict) -> List[Tuple[List[int], List[int]]]:
        """二阶马尔可夫优化预测

        融合一步转移+二阶球对转移+全局频率先验，并加入共现加成和区间约束
        """
        posterior_mix = self._config.get('posterior_mix', 0.15)

        # 获取分析数据（优先使用 raw 版本，即原始 tuple/int 键）
        front_ball_probs = order_result.get('front_ball_probs_raw', order_result.get('front_ball_probs', {}))
        back_ball_probs = order_result.get('back_ball_probs_raw', order_result.get('back_ball_probs', {}))
        front_pair_probs = order_result.get('front_pair_probs_raw', order_result.get('front_pair_probs', {}))
        back_pair_probs = order_result.get('back_pair_probs_raw', order_result.get('back_pair_probs', {}))
        front_cooccur = order_result.get('front_cooccurrence_raw', order_result.get('front_cooccurrence', {}))
        back_cooccur = order_result.get('back_cooccurrence_raw', order_result.get('back_cooccurrence', {}))
        front_freq = order_result.get('front_freq_probs_raw', order_result.get('front_freq_probs', {}))
        back_freq = order_result.get('back_freq_probs_raw', order_result.get('back_freq_probs', {}))

        # 同时获取一阶分析结果用于融合
        order1_result = full_result.get('orders', {}).get(1, {})
        front_ball_probs_1st = order1_result.get('front_ball_probs_raw', order1_result.get('front_ball_probs', front_ball_probs))
        back_ball_probs_1st = order1_result.get('back_ball_probs_raw', order1_result.get('back_ball_probs', back_ball_probs))

        recent_fronts, recent_backs = self._recent_periods(2)

        predictions = []
        for i in range(count):
            strategy_seed = int(time.time() * 1000000) + i * 1000

            # 构建前区融合概率分布
            front_dist = self._build_fused_distribution(
                recent_fronts, front_ball_probs_1st, front_pair_probs,
                front_freq, front_cooccur, 35, posterior_mix
            )

            # 构建后区融合概率分布
            back_dist = self._build_fused_distribution(
                recent_backs, back_ball_probs_1st, back_pair_probs,
                back_freq, back_cooccur, 12, posterior_mix
            )

            # 根据策略选择号码
            front_balls = self._select_balls_with_strategy(
                front_dist, 5, 35, i, strategy_seed, front_cooccur, apply_zone_constraint=True
            )
            back_balls = self._select_balls_with_strategy(
                back_dist, 2, 12, i, strategy_seed + 500, back_cooccur, apply_zone_constraint=False
            )

            predictions.append((sorted(front_balls), sorted(back_balls)))

        return predictions

    def _predict_3rd_order_optimized(self, count: int, periods: int,
                                      order_result: Dict, full_result: Dict) -> List[Tuple[List[int], List[int]]]:
        """三阶马尔可夫预测：融合一阶、二阶、三阶条件概率和全局先验"""
        posterior_mix = self._config.get('posterior_mix', 0.15)

        order1_result = full_result.get('orders', {}).get(1, {})
        order2_result = full_result.get('orders', {}).get(2, {})

        front_ball_probs = order1_result.get('front_ball_probs_raw', order_result.get('front_ball_probs_raw', {}))
        back_ball_probs = order1_result.get('back_ball_probs_raw', order_result.get('back_ball_probs_raw', {}))
        front_pair_probs = order2_result.get('front_pair_probs_raw', order_result.get('front_pair_probs_raw', {}))
        back_pair_probs = order2_result.get('back_pair_probs_raw', order_result.get('back_pair_probs_raw', {}))
        front_triple_probs = order_result.get('front_triple_probs_raw', order_result.get('front_triple_probs', {}))
        back_triple_probs = order_result.get('back_triple_probs_raw', order_result.get('back_triple_probs', {}))
        front_cooccur = order_result.get('front_cooccurrence_raw', order_result.get('front_cooccurrence', {}))
        back_cooccur = order_result.get('back_cooccurrence_raw', order_result.get('back_cooccurrence', {}))
        front_freq = order_result.get('front_freq_probs_raw', order_result.get('front_freq_probs', {}))
        back_freq = order_result.get('back_freq_probs_raw', order_result.get('back_freq_probs', {}))

        recent_fronts, recent_backs = self._recent_periods(3)

        predictions = []
        for i in range(count):
            strategy_seed = int(time.time() * 1000000) + i * 1000

            front_dist = self._build_third_order_fused_distribution(
                recent_fronts, front_ball_probs, front_pair_probs, front_triple_probs,
                front_freq, 35, posterior_mix
            )
            back_dist = self._build_third_order_fused_distribution(
                recent_backs, back_ball_probs, back_pair_probs, back_triple_probs,
                back_freq, 12, posterior_mix
            )

            front_balls = self._select_balls_with_strategy(
                front_dist, 5, 35, i, strategy_seed, front_cooccur, apply_zone_constraint=True
            )
            back_balls = self._select_balls_with_strategy(
                back_dist, 2, 12, i, strategy_seed + 500, back_cooccur, apply_zone_constraint=False
            )

            predictions.append((sorted(front_balls), sorted(back_balls)))

        return predictions

    def _build_fused_distribution(self, recent_periods: List[List[int]],
                                   ball_probs_1st: Dict, pair_probs: Dict,
                                   freq_probs: Dict, cooccur: Dict,
                                   max_ball: int, posterior_mix: float) -> Dict[int, float]:
        """构建多步融合概率分布

        融合权重:
        - 一步转移概率（最近1期 -> 下期）: 0.40
        - 二阶球对转移概率（最近2期球对 -> 下期）: 0.45
        - 全局频率先验: posterior_mix (0.15)
        """
        w_1st = 0.40
        w_2nd = 0.45
        w_prior = posterior_mix

        # 归一化权重
        w_total = w_1st + w_2nd + w_prior
        w_1st /= w_total
        w_2nd /= w_total
        w_prior /= w_total

        dist = defaultdict(float)

        # 1. 一步转移概率: 基于最近1期的每个球
        last_period = recent_periods[0] if recent_periods else []
        step1_dist = defaultdict(float)
        for ball in last_period:
            row = ball_probs_1st.get(ball) or ball_probs_1st.get(str(ball)) or {}
            for target, prob in row.items():
                try:
                    target_int = int(target)
                    if 1 <= target_int <= max_ball:
                        step1_dist[target_int] += float(prob)
                except (TypeError, ValueError):
                    continue
        # 归一化
        s1_total = sum(step1_dist.values()) or 1.0
        for ball in range(1, max_ball + 1):
            dist[ball] += w_1st * (step1_dist.get(ball, 0) / s1_total)

        # 2. 二阶球对转移概率: 基于最近2期的跨期球对
        # 保留时序方向: pair_key = (较早期球, 较新期球)
        step2_dist = defaultdict(float)
        if len(recent_periods) >= 2:
            for a in recent_periods[1]:  # 较早一期
                for b in recent_periods[0]:  # 最近一期
                    pair_key = (a, b)
                    row = pair_probs.get(pair_key) or pair_probs.get(str(pair_key)) or {}
                    for target, prob in row.items():
                        try:
                            target_int = int(target)
                            if 1 <= target_int <= max_ball:
                                step2_dist[target_int] += float(prob)
                        except (TypeError, ValueError):
                            continue
        s2_total = sum(step2_dist.values()) or 1.0
        for ball in range(1, max_ball + 1):
            dist[ball] += w_2nd * (step2_dist.get(ball, 0) / s2_total)

        # 3. 全局频率先验
        freq_probs = self._normalize_flat_distribution(freq_probs, max_ball)
        f_total = sum(freq_probs.values()) or 1.0
        for ball in range(1, max_ball + 1):
            dist[ball] += w_prior * (freq_probs.get(ball, 0) / f_total)

        # 最终归一化
        total = sum(dist.values()) or 1.0
        return {b: p / total for b, p in dist.items()}

    def _build_third_order_fused_distribution(self, recent_periods: List[List[int]],
                                               ball_probs_1st: Dict, pair_probs: Dict,
                                               triple_probs: Dict, freq_probs: Dict,
                                               max_ball: int, posterior_mix: float) -> Dict[int, float]:
        """构建三阶融合概率分布"""
        weights = {
            'first': 0.25,
            'second': 0.30,
            'third': 0.30,
            'prior': posterior_mix,
        }
        total_weight = sum(weights.values()) or 1.0
        weights = {key: value / total_weight for key, value in weights.items()}

        dist = defaultdict(float)

        step1_dist = defaultdict(float)
        for ball in (recent_periods[0] if recent_periods else []):
            row = ball_probs_1st.get(ball) or ball_probs_1st.get(str(ball)) or {}
            for target, prob in row.items():
                try:
                    target_int = int(target)
                    if 1 <= target_int <= max_ball:
                        step1_dist[target_int] += float(prob)
                except (TypeError, ValueError):
                    continue
        self._add_weighted_distribution(dist, step1_dist, max_ball, weights['first'])

        step2_dist = defaultdict(float)
        if len(recent_periods) >= 2:
            for a in recent_periods[1]:
                for b in recent_periods[0]:
                    pair_key = (a, b)
                    row = pair_probs.get(pair_key) or pair_probs.get(str(pair_key)) or {}
                    for target, prob in row.items():
                        try:
                            target_int = int(target)
                            if 1 <= target_int <= max_ball:
                                step2_dist[target_int] += float(prob)
                        except (TypeError, ValueError):
                            continue
        self._add_weighted_distribution(dist, step2_dist, max_ball, weights['second'])

        step3_dist = defaultdict(float)
        if len(recent_periods) >= 3:
            for a in recent_periods[2]:
                for b in recent_periods[1]:
                    for c in recent_periods[0]:
                        triple_key = (a, b, c)
                        row = triple_probs.get(triple_key) or triple_probs.get(str(triple_key)) or {}
                        for target, prob in row.items():
                            try:
                                target_int = int(target)
                                if 1 <= target_int <= max_ball:
                                    step3_dist[target_int] += float(prob)
                            except (TypeError, ValueError):
                                continue
        self._add_weighted_distribution(dist, step3_dist, max_ball, weights['third'])

        freq_dist = self._normalize_flat_distribution(freq_probs, max_ball)
        self._add_weighted_distribution(dist, freq_dist, max_ball, weights['prior'])

        total = sum(dist.values())
        if total <= 0:
            uniform = 1.0 / max_ball
            return {ball: uniform for ball in range(1, max_ball + 1)}
        return {ball: dist.get(ball, 0.0) / total for ball in range(1, max_ball + 1)}

    @staticmethod
    def _normalize_flat_distribution(distribution: Dict, max_ball: int) -> Dict[int, float]:
        normalized = defaultdict(float)
        if isinstance(distribution, dict):
            for ball, prob in distribution.items():
                try:
                    ball_int = int(ball)
                    if 1 <= ball_int <= max_ball:
                        normalized[ball_int] += float(prob)
                except (TypeError, ValueError):
                    continue

        total = sum(normalized.values())
        if total <= 0:
            uniform = 1.0 / max_ball
            return {ball: uniform for ball in range(1, max_ball + 1)}
        return {ball: normalized.get(ball, 0.0) / total for ball in range(1, max_ball + 1)}

    @staticmethod
    def _add_weighted_distribution(target: Dict[int, float], source: Dict[int, float],
                                   max_ball: int, weight: float) -> None:
        total = sum(source.values())
        if total <= 0:
            return
        for ball in range(1, max_ball + 1):
            target[ball] += weight * (source.get(ball, 0.0) / total)

    def _select_balls_with_strategy(self, dist: Dict[int, float], num_balls: int,
                                     max_ball: int, strategy_index: int, seed: int,
                                     cooccur: Dict, apply_zone_constraint: bool) -> List[int]:
        """基于融合概率分布选择号码，使用多样性策略

        策略0: 概率最优 - 按概率排序选top N
        策略1: 加权采样 - 按概率做加权随机采样
        策略2: 多样化 - top 3N中随机采样，满足区间约束
        策略3: 全局混合 - 50%概率分布 + 50%均匀分布
        """
        random.seed(seed)
        np.random.seed(seed % (2**32))

        balls_list = sorted(dist.keys())
        probs_list = np.array([dist.get(b, 0) for b in balls_list])

        # 确保概率和为1
        prob_sum = probs_list.sum()
        if prob_sum > 0:
            probs_list = probs_list / prob_sum
        else:
            probs_list = np.ones(len(balls_list)) / len(balls_list)

        strategy = strategy_index % 4
        selected = []

        if strategy == 0:
            # 概率最优: 按概率排序选top N
            sorted_indices = np.argsort(probs_list)[::-1]
            selected = [balls_list[idx] for idx in sorted_indices[:num_balls]]

        elif strategy == 1:
            # 加权采样
            if len(balls_list) >= num_balls:
                chosen_indices = np.random.choice(
                    len(balls_list), size=num_balls, replace=False, p=probs_list
                )
                selected = [balls_list[idx] for idx in chosen_indices]

        elif strategy == 2:
            # 多样化: 从top 3N中随机采样N个
            sorted_indices = np.argsort(probs_list)[::-1]
            pool_size = min(num_balls * 3, len(balls_list))
            candidate_pool = [balls_list[idx] for idx in sorted_indices[:pool_size]]
            if len(candidate_pool) >= num_balls:
                selected = random.sample(candidate_pool, num_balls)
            else:
                selected = candidate_pool

        else:
            # 全局混合: 50%概率分布 + 50%均匀分布
            uniform = np.ones(len(balls_list)) / len(balls_list)
            mixed_probs = 0.5 * probs_list + 0.5 * uniform
            mixed_probs = mixed_probs / mixed_probs.sum()
            if len(balls_list) >= num_balls:
                chosen_indices = np.random.choice(
                    len(balls_list), size=num_balls, replace=False, p=mixed_probs
                )
                selected = [balls_list[idx] for idx in chosen_indices]

        # 共现关系加成: 如果已选球之间共现分数低，替换低概率球
        if len(selected) >= 2 and cooccur:
            selected = self._apply_cooccurrence_boost(selected, dist, cooccur, num_balls, max_ball)

        # 区间分布约束（仅前区）
        if apply_zone_constraint and max_ball == 35:
            selected = self._apply_zone_constraint(selected, dist, num_balls)

        # 兜底: 确保号码数量足够
        if len(selected) < num_balls:
            remaining = [b for b in range(1, max_ball + 1) if b not in selected]
            remaining_probs = [dist.get(b, 0) for b in remaining]
            total_rp = sum(remaining_probs) or 1.0
            remaining_probs = [p / total_rp for p in remaining_probs]
            needed = num_balls - len(selected)
            if remaining:
                try:
                    extra = list(np.random.choice(
                        remaining, size=min(needed, len(remaining)),
                        replace=False, p=remaining_probs
                    ))
                    selected.extend(extra)
                except Exception:
                    selected.extend(random.sample(remaining, min(needed, len(remaining))))

        return selected[:num_balls]

    def _apply_cooccurrence_boost(self, selected: List[int], dist: Dict[int, float],
                                    cooccur: Dict, num_balls: int, max_ball: int) -> List[int]:
        """共现关系加成: 检查已选球之间的共现强度，尝试提升"""
        def cooccur_score(balls):
            score = 0.0
            for a, b in combinations(balls, 2):
                score += cooccur.get(a, {}).get(b, 0)
            return score

        best = list(selected)
        best_score = cooccur_score(selected)

        # 尝试替换每个球，看是否能提升共现分数同时保持概率
        for idx in range(len(selected)):
            original_ball = selected[idx]
            candidates = sorted(
                [b for b in range(1, max_ball + 1)
                 if b not in selected and dist.get(b, 0) >= dist.get(original_ball, 0) * 0.7],
                key=lambda b: dist.get(b, 0), reverse=True
            )
            for candidate in candidates[:5]:
                trial = list(selected)
                trial[idx] = candidate
                trial_score = cooccur_score(trial)
                if trial_score > best_score:
                    best_score = trial_score
                    best = list(trial)

        return best

    @staticmethod
    def _apply_zone_constraint(selected: List[int], dist: Dict[int, float],
                                num_balls: int) -> List[int]:
        """区间分布约束（前区）

        将1-35分为7个区间，确保至少覆盖3个不同区间
        """
        def get_zone(ball):
            return (ball - 1) // 5

        zones = [get_zone(b) for b in selected]
        unique_zones = len(set(zones))

        if unique_zones >= 3:
            return selected

        result = list(selected)

        for _ in range(num_balls):
            zones_now = [get_zone(b) for b in result]
            if len(set(zones_now)) >= 3:
                break

            # 找到出现次数最多的区间
            zone_counts = Counter(zones_now)
            most_common_zone = zone_counts.most_common(1)[0][0]

            # 在该区间中找概率最低的球
            balls_in_zone = [(b, dist.get(b, 0)) for b in result if get_zone(b) == most_common_zone]
            balls_in_zone.sort(key=lambda x: x[1])
            ball_to_replace = balls_in_zone[0][0]

            # 从未覆盖的区间中选概率最高的球
            covered_zones = set(zones_now)
            uncovered_zones = set(range(7)) - covered_zones

            best_replacement = None
            best_prob = -1
            for zone in uncovered_zones:
                zone_start = zone * 5 + 1
                zone_end = min(zone * 5 + 5, 35)
                for b in range(zone_start, zone_end + 1):
                    if b not in result and dist.get(b, 0) > best_prob:
                        best_prob = dist.get(b, 0)
                        best_replacement = b

            if best_replacement is not None:
                idx = result.index(ball_to_replace)
                result[idx] = best_replacement

        return result

    def _predict_legacy(self, count: int, periods: int, order: int,
                         order_result: Dict) -> List[Tuple[List[int], List[int]]]:
        """保持原有预测逻辑（供 order=1 和 order=3 使用）"""
        front_transitions = order_result.get('front_transition_probs', {})
        back_transitions = order_result.get('back_transition_probs', {})

        predictions = []

        condition_front = []
        condition_back = []
        for i in range(min(order, len(self.df))):
            front, back = data_manager.parse_balls(self.df.iloc[i])
            condition_front.extend(front)
            condition_back.extend(back)

        if not condition_front:
            condition_front = list(range(1, 6))
        if not condition_back:
            condition_back = [1, 2]

        condition_front_str = str(tuple(sorted(condition_front)))
        condition_back_str = str(tuple(sorted(condition_back)))

        for i in range(count):
            strategy_seed = int(time.time() * 1000000) + i * 1000

            front_balls = self._predict_balls_with_condition_diverse(
                front_transitions, condition_front_str, 5, 35, i, periods, strategy_seed
            )
            back_balls = self._predict_balls_with_condition_diverse(
                back_transitions, condition_back_str, 2, 12, i, periods, strategy_seed + 500
            )

            predictions.append((sorted(front_balls), sorted(back_balls)))

        return predictions

    def _predict_balls_with_condition_diverse(self, transitions, condition_str, num_balls, max_ball, strategy_index, periods, seed=None):
        """基于条件状态预测号码 - 多样性策略版本"""

        # 设置随机种子确保每注不同
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % 2**32)
        else:
            unique_seed = (strategy_index * 999983 + hash(condition_str) % 100000) % (2**31)
            random.seed(unique_seed)
            np.random.seed(unique_seed % (2**32))

        balls = []

        # 策略1: 最高概率策略
        if strategy_index % 4 == 0:
            if condition_str in transitions:
                trans_probs = transitions[condition_str]
                sorted_probs = sorted(trans_probs.items(), key=lambda x: x[1], reverse=True)
                high_prob_balls = [int(ball) for ball, _ in sorted_probs[:num_balls*2]]
                if len(high_prob_balls) >= num_balls:
                    balls = random.sample(high_prob_balls, num_balls)
                else:
                    balls = high_prob_balls

        # 策略2: 中等概率策略
        elif strategy_index % 4 == 1:
            if condition_str in transitions:
                trans_probs = transitions[condition_str]
                sorted_probs = sorted(trans_probs.items(), key=lambda x: x[1], reverse=True)
                mid_start = len(sorted_probs) // 4
                mid_end = len(sorted_probs) * 3 // 4
                mid_prob_balls = [int(ball) for ball, _ in sorted_probs[mid_start:mid_end]]
                if len(mid_prob_balls) >= num_balls:
                    balls = random.sample(mid_prob_balls, num_balls)
                else:
                    balls = mid_prob_balls

        # 策略3: 概率加权随机选择
        elif strategy_index % 4 == 2:
            if condition_str in transitions:
                trans_probs = transitions[condition_str]
                if trans_probs:
                    ball_list = [int(ball) for ball in trans_probs.keys()]
                    prob_list = list(trans_probs.values())
                    total_prob = sum(prob_list)
                    if total_prob > 0:
                        normalized_probs = [p/total_prob for p in prob_list]
                        if len(ball_list) >= num_balls:
                            balls = list(np.random.choice(ball_list, size=num_balls, replace=False, p=normalized_probs))
                        else:
                            balls = ball_list

        # 策略4: 全局概率分布策略
        else:
            all_probs = {}
            for cond, trans_probs in transitions.items():
                for ball, prob in trans_probs.items():
                    ball_int = int(ball)
                    all_probs[ball_int] = all_probs.get(ball_int, 0) + prob
            if all_probs:
                ball_list = list(all_probs.keys())
                prob_list = list(all_probs.values())
                total_prob = sum(prob_list)
                if total_prob > 0:
                    normalized_probs = [p/total_prob for p in prob_list]
                    if len(ball_list) >= num_balls:
                        balls = list(np.random.choice(ball_list, size=num_balls, replace=False, p=normalized_probs))
                    else:
                        balls = ball_list

        # 号码不足时，使用频率分析补充
        if len(balls) < num_balls:
            try:
                from analyzer_modules import basic_analyzer
                freq_analysis = basic_analyzer.frequency_analysis(periods)
                if max_ball == 35:
                    freq_dict = freq_analysis.get('front_frequency', {})
                else:
                    freq_dict = freq_analysis.get('back_frequency', {})
                sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
                for ball, _ in sorted_freq:
                    if len(balls) >= num_balls:
                        break
                    ball_int = int(ball)
                    if ball_int not in balls:
                        balls.append(ball_int)
            except Exception:
                pass

        # 仍然不足，随机补充
        if len(balls) < num_balls:
            remaining = [i for i in range(1, max_ball + 1) if i not in balls]
            if remaining:
                needed = num_balls - len(balls)
                balls.extend(random.sample(remaining, min(needed, len(remaining))))

        return balls[:num_balls]

    def adaptive_order_markov_predict(self, count=1, periods=500) -> List[Dict]:
        """自适应阶数马尔可夫链预测

        结合1-3阶马尔可夫链的预测结果，根据各阶的统计特性自适应选择最佳结果

        Args:
            count: 预测注数
            periods: 分析期数

        Returns:
            List[Dict]: 预测结果列表，包含详细信息
        """
        # 获取1-3阶马尔可夫分析结果
        markov_result = self.analyzer.multi_order_markov_analysis(periods, max_order=3)

        if not markov_result or 'orders' not in markov_result:
            logger_manager.error("马尔可夫分析结果不可用")
            return []

        # 计算各阶的权重
        order_weights = self._calculate_order_weights(markov_result)

        # 获取各阶的预测结果
        order_predictions = {}
        for order in range(1, 4):
            if order in markov_result['orders']:
                preds = self.multi_order_markov_predict(count, periods, order)
                order_predictions[order] = preds


        # 融合各阶预测结果
        predictions = []
        for i in range(count):
            # 收集各阶对应的第i注预测
            front_candidates = []
            back_candidates = []

            for order, preds in order_predictions.items():
                if i < len(preds):
                    front, back = preds[i]

                    # 根据权重添加多次（权重越高，添加次数越多）
                    weight = order_weights.get(order, 0.1)
                    repeat = max(1, int(weight * 10))

                    for _ in range(repeat):
                        front_candidates.extend(front)
                        back_candidates.extend(back)

            # 统计各号码出现频率
            front_counter = Counter(front_candidates)
            back_counter = Counter(back_candidates)

            # 选择出现频率最高的号码
            front_balls = [ball for ball, _ in front_counter.most_common(5)]
            back_balls = [ball for ball, _ in back_counter.most_common(2)]


            # 如果号码不足，使用简单的频率分析回退
            if len(front_balls) < 5:
                try:
                    from analyzer_modules import basic_analyzer
                    freq_analysis = basic_analyzer.frequency_analysis(periods)
                    front_freq = freq_analysis.get('front_frequency', {})
                    sorted_freq = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)
                    for ball_str, _ in sorted_freq:
                        if len(front_balls) >= 5:
                            break
                        ball_int = int(ball_str) if isinstance(ball_str, str) else ball_str
                        if ball_int not in front_balls and 1 <= ball_int <= 35:
                            front_balls.append(ball_int)
                except Exception:
                    pass

            if len(back_balls) < 2:
                try:
                    from analyzer_modules import basic_analyzer
                    freq_analysis = basic_analyzer.frequency_analysis(periods)
                    back_freq = freq_analysis.get('back_frequency', {})
                    sorted_freq = sorted(back_freq.items(), key=lambda x: x[1], reverse=True)
                    for ball_str, _ in sorted_freq:
                        if len(back_balls) >= 2:
                            break
                        ball_int = int(ball_str) if isinstance(ball_str, str) else ball_str
                        if ball_int not in back_balls and 1 <= ball_int <= 12:
                            back_balls.append(ball_int)
                except Exception:
                    pass

            # 构建预测结果
            prediction = {
                'index': i + 1,
                'front_balls': sorted(front_balls),
                'back_balls': sorted(back_balls),
                'method': 'adaptive_markov',
                'confidence': 0.85,
                'order_weights': order_weights,
                'used_orders': list(order_predictions.keys())
            }

            predictions.append(prediction)

        return predictions

    def _calculate_order_weights(self, markov_result):
        """计算各阶马尔可夫链的权重"""
        weights = {}

        for order, result in markov_result['orders'].items():
            # 确保order是整数类型
            order_int = int(order)

            # 获取前区和后区的统计信息
            front_stats = result.get('front_stats', {})
            back_stats = result.get('back_stats', {})

            # 计算权重因子
            # 1. 转移概率的最大值（越高越好）
            max_prob_factor = (front_stats.get('max_probability', 0) + back_stats.get('max_probability', 0)) / 2

            # 2. 状态数量（越多越好，表示覆盖更多情况）
            states_factor = (front_stats.get('unique_states', 0) + back_stats.get('unique_states', 0)) / 2
            states_factor = min(1.0, states_factor / 1000)  # 归一化

            # 3. 阶数因子（高阶更精确但样本更少，需要平衡）
            order_factor = 1.0 / order_int

            # 综合权重
            weight = 0.5 * max_prob_factor + 0.3 * states_factor + 0.2 * order_factor
            weights[order_int] = weight

        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}

        return weights


# 全局实例
_markov_analyzer = None
_markov_predictor = None

def get_markov_analyzer() -> EnhancedMarkovAnalyzer:
    """获取马尔可夫分析器实例"""
    global _markov_analyzer
    if _markov_analyzer is None:
        _markov_analyzer = EnhancedMarkovAnalyzer()
    return _markov_analyzer

def get_markov_predictor() -> EnhancedMarkovPredictor:
    """获取马尔可夫预测器实例"""
    global _markov_predictor
    if _markov_predictor is None:
        _markov_predictor = EnhancedMarkovPredictor()
    return _markov_predictor


if __name__ == "__main__":
    # 测试增强版马尔可夫链
    print("🔄 测试增强版马尔可夫链...")

    # 测试多阶马尔可夫分析
    analyzer = get_markov_analyzer()
    for order in range(1, 4):
        print(f"\n📊 {order}阶马尔可夫链分析...")
        result = analyzer.multi_order_markov_analysis(periods=300, max_order=order)

        if result and 'orders' in result and order in result['orders']:
            order_result = result['orders'][order]
            front_stats = order_result.get('front_stats', {})
            back_stats = order_result.get('back_stats', {})

            print(f"  前区状态数: {front_stats.get('unique_states', 0)}")
            print(f"  前区最大概率: {front_stats.get('max_probability', 0):.4f}")
            print(f"  后区状态数: {back_stats.get('unique_states', 0)}")
            print(f"  后区最大概率: {back_stats.get('max_probability', 0):.4f}")

    # 测试多阶马尔可夫预测
    predictor = get_markov_predictor()
    for order in range(1, 4):
        print(f"\n🎯 {order}阶马尔可夫链预测...")
        predictions = predictor.multi_order_markov_predict(count=2, periods=300, order=order)

        for i, (front, back) in enumerate(predictions):
            front_str = ' '.join([str(b).zfill(2) for b in front])
            back_str = ' '.join([str(b).zfill(2) for b in back])
            print(f"  第 {i+1} 注: {front_str} + {back_str}")

    # 测试自适应阶数马尔可夫预测
    print("\n🌟 自适应阶数马尔可夫链预测...")
    adaptive_predictions = predictor.adaptive_order_markov_predict(count=3, periods=300)

    for i, pred in enumerate(adaptive_predictions):
        front_str = ' '.join([str(b).zfill(2) for b in pred['front_balls']])
        back_str = ' '.join([str(b).zfill(2) for b in pred['back_balls']])
        print(f"  第 {i+1} 注: {front_str} + {back_str}")
        print(f"  阶数权重: {pred['order_weights']}")

    print("\n✅ 测试完成")
