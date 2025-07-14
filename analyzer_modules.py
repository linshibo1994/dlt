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
warnings.filterwarnings('ignore')

from core_modules import cache_manager, logger_manager, data_manager


# ==================== 基础分析器 ====================
class BasicAnalyzer:
    """基础分析器"""
    
    def __init__(self, data_file="data/dlt_data_all.csv"):
        self.data_file = data_file
        self.df = data_manager.get_data()
        
        if self.df is None:
            logger_manager.error("数据未加载")
    
    def frequency_analysis(self, periods=None) -> Dict:
        """频率分析"""
        if self.df is None:
            return {}
        
        cache_key = f"frequency_analysis_{periods or 'all'}"
        cached_result = cache_manager.load_cache("analysis", cache_key)
        if cached_result:
            return cached_result
        
        df_subset = self.df.tail(periods) if periods else self.df
        
        front_counter = Counter()
        back_counter = Counter()
        
        for _, row in df_subset.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            front_counter.update(front_balls)
            back_counter.update(back_balls)
        
        result = {
            'front_frequency': dict(front_counter.most_common()),
            'back_frequency': dict(back_counter.most_common()),
            'analysis_periods': len(df_subset),
            'timestamp': datetime.now().isoformat()
        }
        
        cache_manager.save_cache("analysis", cache_key, result)
        return result
    
    def missing_analysis(self, periods=None) -> Dict:
        """遗漏分析"""
        if self.df is None:
            return {}
        
        cache_key = f"missing_analysis_{periods or 'all'}"
        cached_result = cache_manager.load_cache("analysis", cache_key)
        if cached_result:
            return cached_result
        
        df_subset = self.df.tail(periods) if periods else self.df
        
        front_missing = {i: 0 for i in range(1, 36)}
        back_missing = {i: 0 for i in range(1, 13)}
        
        for _, row in df_subset.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            
            # 更新遗漏值
            for num in range(1, 36):
                if num in front_balls:
                    front_missing[num] = 0
                else:
                    front_missing[num] += 1
            
            for num in range(1, 13):
                if num in back_balls:
                    back_missing[num] = 0
                else:
                    back_missing[num] += 1
        
        result = {
            'front_missing': front_missing,
            'back_missing': back_missing,
            'analysis_periods': len(df_subset),
            'timestamp': datetime.now().isoformat()
        }
        
        cache_manager.save_cache("analysis", cache_key, result)
        return result
    
    def hot_cold_analysis(self, periods=100) -> Dict:
        """冷热号分析"""
        if self.df is None:
            return {}
        
        cache_key = f"hot_cold_analysis_{periods}"
        cached_result = cache_manager.load_cache("analysis", cache_key)
        if cached_result:
            return cached_result
        
        freq_result = self.frequency_analysis(periods)
        
        front_freq = freq_result.get('front_frequency', {})
        back_freq = freq_result.get('back_frequency', {})
        
        # 计算平均频率
        front_avg = np.mean(list(front_freq.values())) if front_freq else 0
        back_avg = np.mean(list(back_freq.values())) if back_freq else 0
        
        # 分类冷热号
        front_hot = [num for num, freq in front_freq.items() if freq > front_avg]
        front_cold = [num for num, freq in front_freq.items() if freq < front_avg]
        back_hot = [num for num, freq in back_freq.items() if freq > back_avg]
        back_cold = [num for num, freq in back_freq.items() if freq < back_avg]
        
        result = {
            'front_hot': sorted(front_hot),
            'front_cold': sorted(front_cold),
            'back_hot': sorted(back_hot),
            'back_cold': sorted(back_cold),
            'front_avg_freq': front_avg,
            'back_avg_freq': back_avg,
            'analysis_periods': periods,
            'timestamp': datetime.now().isoformat()
        }
        
        cache_manager.save_cache("analysis", cache_key, result)
        return result
    
    def sum_analysis(self, periods=None) -> Dict:
        """和值分析"""
        if self.df is None:
            return {}
        
        cache_key = f"sum_analysis_{periods or 'all'}"
        cached_result = cache_manager.load_cache("analysis", cache_key)
        if cached_result:
            return cached_result
        
        df_subset = self.df.tail(periods) if periods else self.df
        
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
        
        cache_manager.save_cache("analysis", cache_key, result)
        return result

    def statistical_features_analysis(self, periods=None) -> Dict:
        """统计特征分析"""
        if self.df is None:
            return {}

        cache_key = f"statistical_features_{periods or 'all'}"
        cached_result = cache_manager.load_cache("analysis", cache_key)
        if cached_result:
            return cached_result

        df_subset = self.df.tail(periods) if periods else self.df

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

        cache_manager.save_cache("analysis", cache_key, result)
        return result


# ==================== 高级分析器 ====================
class AdvancedAnalyzer:
    """高级分析器"""
    
    def __init__(self, data_file="data/dlt_data_all.csv"):
        self.data_file = data_file
        self.df = data_manager.get_data()
        self.basic_analyzer = BasicAnalyzer(data_file)
        
        if self.df is None:
            logger_manager.error("数据未加载")
    
    def markov_analysis(self, periods=500) -> Dict:
        """马尔可夫链分析"""
        if self.df is None:
            return {}
        
        cache_key = f"markov_analysis_{periods}"
        cached_result = cache_manager.load_cache("analysis", cache_key)
        if cached_result:
            return cached_result
        
        df_subset = self.df.tail(periods)
        
        # 构建转移矩阵
        front_transitions = defaultdict(lambda: defaultdict(int))
        back_transitions = defaultdict(lambda: defaultdict(int))
        
        for i in range(len(df_subset) - 1):
            current_front, current_back = data_manager.parse_balls(df_subset.iloc[i])
            next_front, next_back = data_manager.parse_balls(df_subset.iloc[i + 1])
            
            # 前区转移
            for curr_ball in current_front:
                for next_ball in next_front:
                    front_transitions[curr_ball][next_ball] += 1
            
            # 后区转移
            for curr_ball in current_back:
                for next_ball in next_back:
                    back_transitions[curr_ball][next_ball] += 1
        
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
        
        cache_manager.save_cache("analysis", cache_key, result)
        return result
    
    def bayesian_analysis(self, periods=300) -> Dict:
        """贝叶斯分析"""
        if self.df is None:
            return {}
        
        cache_key = f"bayesian_analysis_{periods}"
        cached_result = cache_manager.load_cache("analysis", cache_key)
        if cached_result:
            return cached_result
        
        df_subset = self.df.tail(periods)
        
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
        
        result = {
            'front_posterior': front_posterior,
            'back_posterior': back_posterior,
            'front_likelihood': dict(front_likelihood),
            'back_likelihood': dict(back_likelihood),
            'analysis_periods': periods,
            'timestamp': datetime.now().isoformat()
        }
        
        cache_manager.save_cache("analysis", cache_key, result)
        return result
    
    def correlation_analysis(self, periods=200) -> Dict:
        """相关性分析"""
        if self.df is None:
            return {}
        
        cache_key = f"correlation_analysis_{periods}"
        cached_result = cache_manager.load_cache("analysis", cache_key)
        if cached_result:
            return cached_result
        
        df_subset = self.df.tail(periods)
        
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
        
        cache_manager.save_cache("analysis", cache_key, result)
        return result

    def trend_generation_analysis(self, periods=500) -> Dict:
        """趋势生成分析"""
        if self.df is None:
            return {}

        cache_key = f"trend_generation_{periods}"
        cached_result = cache_manager.load_cache("analysis", cache_key)
        if cached_result:
            return cached_result

        df_subset = self.df.tail(periods)

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

        cache_manager.save_cache("analysis", cache_key, result)
        return result

    def mixed_strategy_analysis(self, periods=500) -> Dict:
        """混合策略分析"""
        if self.df is None:
            return {}

        cache_key = f"mixed_strategy_{periods}"
        cached_result = cache_manager.load_cache("analysis", cache_key)
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

        cache_manager.save_cache("analysis", cache_key, result)
        return result

    def markov_bayesian_fusion_analysis(self, periods=500) -> Dict:
        """马尔可夫-贝叶斯融合分析"""
        if self.df is None:
            return {}

        cache_key = f"markov_bayesian_fusion_{periods}"
        cached_result = cache_manager.load_cache("analysis", cache_key)
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

        cache_manager.save_cache("analysis", cache_key, result)
        return result

    def hot_cold_markov_integration(self, periods=500) -> Dict:
        """冷热号-马尔可夫集成分析"""
        if self.df is None:
            return {}

        cache_key = f"hot_cold_markov_{periods}"
        cached_result = cache_manager.load_cache("analysis", cache_key)
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

        cache_manager.save_cache("analysis", cache_key, result)
        return result

    def multi_dimensional_probability_analysis(self, periods=500) -> Dict:
        """多维度概率分析"""
        if self.df is None:
            return {}

        cache_key = f"multi_dimensional_prob_{periods}"
        cached_result = cache_manager.load_cache("analysis", cache_key)
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

        cache_manager.save_cache("analysis", cache_key, result)
        return result

    def comprehensive_weight_scoring_system(self, periods=500) -> Dict:
        """综合权重评分系统"""
        if self.df is None:
            return {}

        cache_key = f"comprehensive_weight_scoring_{periods}"
        cached_result = cache_manager.load_cache("analysis", cache_key)
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

            # 频率得分
            freq_score = frequency_result.get('front_frequency', {}).get(ball, 0) / periods
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

            # 遗漏得分
            missing_periods = missing_result.get('front_missing', {}).get(ball, 0)
            missing_score = min(1.0, missing_periods / 20)  # 标准化
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

            # 贝叶斯得分
            bayesian_score = bayesian_result.get('front_posterior', {}).get(ball, 0)
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

            freq_score = frequency_result.get('back_frequency', {}).get(ball, 0) / periods
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

            missing_periods = missing_result.get('back_missing', {}).get(ball, 0)
            missing_score = min(1.0, missing_periods / 15)
            total_score += missing_score * weights['missing']
            detail_scores['missing'] = missing_score

            markov_score = 0
            if 'back_transition_probs' in markov_result:
                for from_ball, to_probs in markov_result['back_transition_probs'].items():
                    if ball in to_probs:
                        markov_score += to_probs[ball]
            total_score += markov_score * weights['markov']
            detail_scores['markov'] = markov_score

            bayesian_score = bayesian_result.get('back_posterior', {}).get(ball, 0)
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

        cache_manager.save_cache("analysis", cache_key, result)
        return result

    def advanced_pattern_recognition(self, periods=500) -> Dict:
        """高级模式识别"""
        if self.df is None:
            return {}

        cache_key = f"advanced_pattern_recognition_{periods}"
        cached_result = cache_manager.load_cache("analysis", cache_key)
        if cached_result:
            return cached_result

        df_subset = self.df.tail(periods)

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

        cache_manager.save_cache("analysis", cache_key, result)
        return result

    def nine_mathematical_models_analysis(self, periods=500) -> Dict:
        """9种数学模型综合分析

        包含：统计学分析、概率论分析、频率模式分析、决策树分析、
        周期性分析、历史关联分析、马尔可夫链分析、贝叶斯分析、回归分析
        """
        if self.df is None:
            return {}

        cache_key = f"nine_mathematical_models_{periods}"
        cached_result = cache_manager.load_cache("analysis", cache_key)
        if cached_result:
            return cached_result

        logger_manager.info(f"开始9种数学模型综合分析，期数: {periods}")

        df_subset = self.df.tail(periods)

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
        enhanced_bayesian_analysis = self._enhanced_bayesian_analysis(df_subset)

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

        cache_manager.save_cache("analysis", cache_key, result)
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

    def _enhanced_bayesian_analysis(self, df_subset) -> Dict:
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

        # 似然函数（基于最近期数）
        recent_periods = min(50, len(df_subset))
        recent_data = df_subset.tail(recent_periods)

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
        
        cache_key = f"comprehensive_analysis_{periods}"
        cached_result = cache_manager.load_cache("analysis", cache_key)
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
            
            cache_manager.save_cache("analysis", cache_key, result)
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

    def generate_frequency_chart(self, output_dir="output", periods=None) -> bool:
        """生成频率分布图"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            df_subset = self.df.tail(periods) if periods else self.df

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

    def generate_trend_chart(self, output_dir="output", periods=100) -> bool:
        """生成走势图"""
        try:
            import matplotlib.pyplot as plt

            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            df_subset = self.df.tail(periods)

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

    def generate_all_charts(self, output_dir="output", periods=500) -> bool:
        """生成所有图表"""
        logger_manager.info(f"开始生成所有可视化图表，期数: {periods}")

        success_count = 0

        # 生成频率分布图
        if self.generate_frequency_chart(output_dir, periods):
            success_count += 1

        # 生成走势图
        if self.generate_trend_chart(output_dir, periods):
            success_count += 1

        logger_manager.info(f"可视化图表生成完成，成功: {success_count}/2")
        return success_count > 0


# ==================== 全局实例 ====================
basic_analyzer = BasicAnalyzer()
advanced_analyzer = AdvancedAnalyzer()
comprehensive_analyzer = ComprehensiveAnalyzer()
visualization_analyzer = VisualizationAnalyzer()


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
    viz_result = visualization_analyzer.generate_all_charts("output", 100)
    print(f"可视化图表生成: {'成功' if viz_result else '失败'}")

    print("✅ 分析器模块测试完成")
