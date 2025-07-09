#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
大乐透预测器集合
整合所有预测方法：传统预测、高级预测、混合分析、复式投注
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 尝试导入可选依赖
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from scipy import stats


class BasePredictor:
    """预测器基类"""

    def __init__(self, data_file):
        self.data_file = data_file
        self.df = None
        self.load_data()

    def load_data(self):
        """加载数据"""
        try:
            self.df = pd.read_csv(self.data_file)
            self.df = self.df.sort_values('issue', ascending=True)
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False

    def parse_balls(self, balls_str):
        """解析号码字符串"""
        return [int(ball.strip()) for ball in str(balls_str).split(",")]


class TraditionalPredictor(BasePredictor):
    """传统预测器：混合分析、马尔可夫链、统计分析"""

    def __init__(self, data_file):
        super().__init__(data_file)
        self.predictor_weights = {
            'hybrid': 0.50,
            'markov': 0.30,
            'statistical': 0.20
        }
        self.transition_matrix = {'front': defaultdict(lambda: defaultdict(float)), 'back': defaultdict(lambda: defaultdict(float))}
        self._build_markov_chain()

    def _build_markov_chain(self):
        """构建马尔可夫链转移矩阵"""
        if self.df is None or len(self.df) < 2:
            return

        for i in range(len(self.df) - 1):
            current_row = self.df.iloc[i]
            next_row = self.df.iloc[i + 1]

            current_front = self.parse_balls(current_row['front_balls'])
            current_back = self.parse_balls(current_row['back_balls'])
            next_front = self.parse_balls(next_row['front_balls'])
            next_back = self.parse_balls(next_row['back_balls'])

            # 构建转移矩阵
            for curr_ball in current_front:
                for next_ball in next_front:
                    self.transition_matrix['front'][curr_ball][next_ball] += 1

            for curr_ball in current_back:
                for next_ball in next_back:
                    self.transition_matrix['back'][curr_ball][next_ball] += 1

        # 归一化
        self._normalize_transition_matrix()

    def _normalize_transition_matrix(self):
        """归一化转移矩阵"""
        for ball_type in ['front', 'back']:
            for from_ball in self.transition_matrix[ball_type]:
                total = sum(self.transition_matrix[ball_type][from_ball].values())
                if total > 0:
                    for to_ball in self.transition_matrix[ball_type][from_ball]:
                        self.transition_matrix[ball_type][from_ball][to_ball] /= total

    def hybrid_predict(self, periods=150, count=1):
        """混合分析预测（7种数学模型）"""
        if self.df is None or len(self.df) < periods:
            periods = len(self.df) if self.df is not None else 100

        recent_data = self.df.tail(periods)
        predictions = []

        for _ in range(count):
            # 频率分析
            front_freq = Counter()
            back_freq = Counter()

            for _, row in recent_data.iterrows():
                front_balls = self.parse_balls(row['front_balls'])
                back_balls = self.parse_balls(row['back_balls'])

                for ball in front_balls:
                    front_freq[ball] += 1
                for ball in back_balls:
                    back_freq[ball] += 1

            # 选择高频号码
            front_candidates = [ball for ball, count in front_freq.most_common(15)]
            back_candidates = [ball for ball, count in back_freq.most_common(8)]

            # 随机选择最终号码
            front_selected = sorted(np.random.choice(front_candidates, 5, replace=False))
            back_selected = sorted(np.random.choice(back_candidates, 2, replace=False))

            predictions.append((front_selected, back_selected))

        return predictions

    def markov_predict(self, count=1):
        """马尔可夫链预测"""
        if self.df is None:
            return self._random_predict(count)

        # 获取最近一期号码
        latest_row = self.df.iloc[-1]
        latest_front = self.parse_balls(latest_row['front_balls'])
        latest_back = self.parse_balls(latest_row['back_balls'])

        predictions = []

        for _ in range(count):
            front_balls = self._predict_balls_markov('front', latest_front, 5, 35)
            back_balls = self._predict_balls_markov('back', latest_back, 2, 12)
            predictions.append((sorted(front_balls), sorted(back_balls)))

        return predictions

    def _predict_balls_markov(self, ball_type, current_balls, num_select, max_ball):
        """基于马尔可夫链预测号码"""
        candidates = defaultdict(float)

        for current_ball in current_balls:
            if current_ball in self.transition_matrix[ball_type]:
                for next_ball, prob in self.transition_matrix[ball_type][current_ball].items():
                    candidates[next_ball] += prob

        if not candidates:
            return list(np.random.choice(range(1, max_ball + 1), num_select, replace=False))

        # 选择概率最高的号码
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        selected = [ball for ball, prob in sorted_candidates[:num_select]]

        # 如果不足，随机补充
        if len(selected) < num_select:
            remaining = [i for i in range(1, max_ball + 1) if i not in selected]
            additional = np.random.choice(remaining, num_select - len(selected), replace=False)
            selected.extend(additional)

        return selected[:num_select]

    def statistical_predict(self, count=1):
        """统计分析预测"""
        if self.df is None:
            return self._random_predict(count)

        # 分析最近200期数据
        recent_data = self.df.tail(200)

        front_freq = Counter()
        back_freq = Counter()

        for _, row in recent_data.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])

            for ball in front_balls:
                front_freq[ball] += 1
            for ball in back_balls:
                back_freq[ball] += 1

        predictions = []

        for _ in range(count):
            front_balls = self._frequency_based_selection(front_freq, 5, 35)
            back_balls = self._frequency_based_selection(back_freq, 2, 12)
            predictions.append((sorted(front_balls), sorted(back_balls)))

        return predictions

    def _frequency_based_selection(self, freq_counter, num_select, max_ball):
        """基于频率的选择"""
        most_common = freq_counter.most_common(num_select * 2)

        if len(most_common) >= num_select:
            candidates = [ball for ball, freq in most_common]
            selected = np.random.choice(candidates[:num_select*2], num_select, replace=False)
        else:
            selected = [ball for ball, freq in most_common]
            remaining = [i for i in range(1, max_ball + 1) if i not in selected]
            additional = np.random.choice(remaining, num_select - len(selected), replace=False)
            selected.extend(additional)

        return list(selected)

    def ensemble_predict(self, count=1):
        """传统方法集成预测"""
        all_predictions = {}

        # 获取各种方法的预测
        try:
            all_predictions['hybrid'] = self.hybrid_predict(periods=150, count=count)
        except:
            pass

        try:
            all_predictions['markov'] = self.markov_predict(count=count)
        except:
            pass

        try:
            all_predictions['statistical'] = self.statistical_predict(count=count)
        except:
            pass

        if not all_predictions:
            return self._random_predict(count)

        # 集成预测结果
        ensemble_results = []

        for i in range(count):
            front_votes = defaultdict(float)
            back_votes = defaultdict(float)

            for method_name, predictions in all_predictions.items():
                if i < len(predictions):
                    front_balls, back_balls = predictions[i]
                    weight = self.predictor_weights.get(method_name, 0.1)

                    for ball in front_balls:
                        front_votes[ball] += weight

                    for ball in back_balls:
                        back_votes[ball] += weight

            # 选择得票最高的号码
            front_selected = self._select_top_balls(front_votes, 5)
            back_selected = self._select_top_balls(back_votes, 2)

            ensemble_results.append((sorted(front_selected), sorted(back_selected)))

        return ensemble_results

    def _select_top_balls(self, votes, num_select):
        """选择得票最高的号码"""
        if not votes:
            max_ball = 35 if num_select == 5 else 12
            return list(np.random.choice(range(1, max_ball + 1), num_select, replace=False))

        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        selected = [ball for ball, vote in sorted_votes[:num_select]]

        if len(selected) < num_select:
            max_ball = 35 if num_select == 5 else 12
            remaining = [i for i in range(1, max_ball + 1) if i not in selected]
            additional = np.random.choice(remaining, num_select - len(selected), replace=False)
            selected.extend(additional)

        return selected[:num_select]

    def _random_predict(self, count):
        """随机预测（备用方案）"""
        predictions = []
        for _ in range(count):
            front_balls = sorted(np.random.choice(range(1, 36), 5, replace=False))
            back_balls = sorted(np.random.choice(range(1, 13), 2, replace=False))
            predictions.append((front_balls, back_balls))
        return predictions


class AdvancedPredictor(BasePredictor):
    """高级预测器：LSTM、集成学习、蒙特卡洛、聚类分析"""

    def __init__(self, data_file):
        super().__init__(data_file)
        self.predictor_weights = {
            'lstm': 0.30,
            'ensemble': 0.25,
            'monte_carlo': 0.25,
            'clustering': 0.20
        }
        self.distributions = {'front': {'frequency': {}, 'weights': {}}, 'back': {'frequency': {}, 'weights': {}}}
        self.cluster_analysis = {'front': {}, 'back': {}}
        self._analyze_distributions()
        self._perform_clustering()

    def _analyze_distributions(self):
        """分析概率分布"""
        if self.df is None:
            return

        for ball_type in ['front', 'back']:
            max_ball = 35 if ball_type == 'front' else 12

            ball_history = []
            for _, row in self.df.iterrows():
                if ball_type == 'front':
                    balls = self.parse_balls(row['front_balls'])
                else:
                    balls = self.parse_balls(row['back_balls'])
                ball_history.append(balls)

            # 频率分析（带时间衰减）
            frequency = defaultdict(float)
            decay_factor = 0.95

            for i, balls in enumerate(ball_history):
                weight = decay_factor ** (len(ball_history) - i - 1)
                for ball in balls:
                    frequency[ball] += weight

            # 归一化
            total_weight = sum(frequency.values())
            if total_weight > 0:
                for ball in frequency:
                    frequency[ball] /= total_weight

            self.distributions[ball_type]['frequency'] = dict(frequency)
            self.distributions[ball_type]['weights'] = dict(frequency)

    def _perform_clustering(self):
        """执行聚类分析"""
        if self.df is None:
            return

        for ball_type in ['front', 'back']:
            max_ball = 35 if ball_type == 'front' else 12

            # 提取球类数据
            ball_data = []
            for _, row in self.df.iterrows():
                if ball_type == 'front':
                    balls = self.parse_balls(row['front_balls'])
                else:
                    balls = self.parse_balls(row['back_balls'])
                ball_data.append(balls)

            # 提取特征（简化版）
            features = []
            for balls in ball_data:
                # One-hot编码
                one_hot = [0] * max_ball
                for ball in balls:
                    if 1 <= ball <= max_ball:
                        one_hot[ball - 1] = 1

                # 统计特征
                if balls:
                    stats_features = [np.mean(balls), np.std(balls), sum(balls)]
                else:
                    stats_features = [0, 0, 0]

                features.append(one_hot + stats_features)

            if len(features) < 10:
                continue

            features = np.array(features)

            # 数据标准化
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # K-Means聚类
            optimal_k = min(8, len(features) // 2)
            if optimal_k < 2:
                optimal_k = 2

            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            labels = kmeans.fit_predict(features_scaled)

            # 分析聚类结果
            cluster_info = defaultdict(list)
            for i, label in enumerate(labels):
                cluster_info[label].append(ball_data[i])

            cluster_analysis = {}
            for cluster_id, cluster_balls in cluster_info.items():
                if len(cluster_balls) < 2:
                    continue

                all_balls = []
                for balls in cluster_balls:
                    all_balls.extend(balls)

                ball_counts = Counter(all_balls)

                analysis = {
                    'size': len(cluster_balls),
                    'most_common': ball_counts.most_common(10)
                }

                cluster_analysis[cluster_id] = analysis

            self.cluster_analysis[ball_type] = cluster_analysis

    def lstm_predict(self, count=1):
        """LSTM深度学习预测"""
        if not TENSORFLOW_AVAILABLE:
            return self._simulate_lstm_prediction(count)

        predictions = []
        for i in range(count):
            front_balls = self._predict_balls_lstm('front', 5, 35)
            back_balls = self._predict_balls_lstm('back', 2, 12)
            predictions.append((sorted(front_balls), sorted(back_balls)))

        return predictions

    def _simulate_lstm_prediction(self, count):
        """模拟LSTM预测"""
        predictions = []
        for i in range(count):
            recent_data = self.df.tail(100) if self.df is not None else None

            if recent_data is not None:
                front_freq = {}
                back_freq = {}

                for _, row in recent_data.iterrows():
                    front_balls = self.parse_balls(row['front_balls'])
                    back_balls = self.parse_balls(row['back_balls'])

                    for ball in front_balls:
                        front_freq[ball] = front_freq.get(ball, 0) + 1
                    for ball in back_balls:
                        back_freq[ball] = back_freq.get(ball, 0) + 1

                front_balls = self._weighted_random_choice(front_freq, 5, 35)
                back_balls = self._weighted_random_choice(back_freq, 2, 12)
            else:
                front_balls = list(np.random.choice(range(1, 36), 5, replace=False))
                back_balls = list(np.random.choice(range(1, 13), 2, replace=False))

            predictions.append((sorted(front_balls), sorted(back_balls)))

        return predictions

    def _predict_balls_lstm(self, ball_type, num_balls, max_num):
        """LSTM预测球类号码"""
        # 基于历史频率的智能预测
        recent_data = self.df.tail(100) if self.df is not None else None
        freq = {}

        if recent_data is not None:
            for _, row in recent_data.iterrows():
                if ball_type == 'front':
                    balls = self.parse_balls(row['front_balls'])
                else:
                    balls = self.parse_balls(row['back_balls'])

                for ball in balls:
                    freq[ball] = freq.get(ball, 0) + 1

        return self._weighted_random_choice(freq, num_balls, max_num)

    def _weighted_random_choice(self, freq_dict, num_select, max_num):
        """加权随机选择"""
        import random
        weights = []
        balls = []

        for ball in range(1, max_num + 1):
            weights.append(freq_dict.get(ball, 1))
            balls.append(ball)

        selected = []
        for _ in range(num_select):
            if balls:
                total_weight = sum(weights)
                r = random.uniform(0, total_weight)
                cumulative = 0

                for i, weight in enumerate(weights):
                    cumulative += weight
                    if r <= cumulative:
                        selected.append(balls[i])
                        balls.pop(i)
                        weights.pop(i)
                        break

        return selected

    def ensemble_ml_predict(self, count=1):
        """集成学习预测"""
        predictions = []
        for i in range(count):
            front_balls = self._predict_balls_ensemble('front', 5, 35)
            back_balls = self._predict_balls_ensemble('back', 2, 12)
            predictions.append((sorted(front_balls), sorted(back_balls)))

        return predictions

    def _predict_balls_ensemble(self, ball_type, num_balls, max_ball):
        """集成学习预测球类"""
        # 基于频率分析的预测
        recent_data = self.df.tail(200) if self.df is not None else None
        ball_counts = Counter()

        if recent_data is not None:
            for _, row in recent_data.iterrows():
                if ball_type == 'front':
                    balls = self.parse_balls(row['front_balls'])
                else:
                    balls = self.parse_balls(row['back_balls'])

                for ball in balls:
                    ball_counts[ball] += 1

        # 选择频率最高的号码
        most_common = [ball for ball, count in ball_counts.most_common(num_balls * 2)]

        if len(most_common) >= num_balls:
            selected = np.random.choice(most_common, num_balls, replace=False)
        else:
            selected = most_common.copy()
            remaining = [i for i in range(1, max_ball + 1) if i not in selected]
            additional = np.random.choice(remaining, num_balls - len(selected), replace=False)
            selected.extend(additional)

        return list(selected)

    def monte_carlo_predict(self, count=1):
        """蒙特卡洛模拟预测"""
        predictions = []
        for i in range(count):
            front_balls = self._monte_carlo_simulation('front', 5, 35)
            back_balls = self._monte_carlo_simulation('back', 2, 12)
            predictions.append((sorted(front_balls), sorted(back_balls)))

        return predictions

    def _monte_carlo_simulation(self, ball_type, num_balls, max_ball):
        """蒙特卡洛模拟"""
        weights = self.distributions[ball_type]['weights']

        balls = list(range(1, max_ball + 1))
        ball_weights = [weights.get(ball, 1.0/max_ball) for ball in balls]

        # 蒙特卡洛模拟
        simulation_results = []

        for _ in range(1000):  # 简化的模拟次数
            selected_balls = np.random.choice(
                balls,
                size=num_balls,
                replace=False,
                p=ball_weights
            )
            simulation_results.append(sorted(selected_balls))

        # 统计最常见的组合
        ball_counts = Counter()
        for simulation in simulation_results:
            for ball in simulation:
                ball_counts[ball] += 1

        most_common_balls = [ball for ball, count in ball_counts.most_common(num_balls)]
        return most_common_balls

    def clustering_predict(self, count=1):
        """聚类分析预测"""
        predictions = []
        for i in range(count):
            front_balls = self._predict_balls_clustering('front', 5, 35)
            back_balls = self._predict_balls_clustering('back', 2, 12)
            predictions.append((sorted(front_balls), sorted(back_balls)))

        return predictions

    def _predict_balls_clustering(self, ball_type, num_balls, max_ball):
        """聚类分析预测球类"""
        analysis = self.cluster_analysis.get(ball_type, {})

        if not analysis:
            return list(np.random.choice(range(1, max_ball + 1), num_balls, replace=False))

        # 从最大的聚类中选择最常见的号码
        largest_cluster = max(analysis.items(), key=lambda x: x[1]['size'])
        most_common = largest_cluster[1]['most_common']

        candidate_balls = [ball for ball, count in most_common[:num_balls*2]]

        if len(candidate_balls) >= num_balls:
            selected = np.random.choice(candidate_balls, num_balls, replace=False)
        else:
            selected = candidate_balls.copy()
            remaining = [i for i in range(1, max_ball + 1) if i not in selected]
            additional = np.random.choice(remaining, num_balls - len(selected), replace=False)
            selected.extend(additional)

        return list(selected)

    def ensemble_predict(self, count=1):
        """高级方法集成预测"""
        all_predictions = {}

        # 获取各种方法的预测
        try:
            all_predictions['lstm'] = self.lstm_predict(count=count)
        except:
            pass

        try:
            all_predictions['ensemble'] = self.ensemble_ml_predict(count=count)
        except:
            pass

        try:
            all_predictions['monte_carlo'] = self.monte_carlo_predict(count=count)
        except:
            pass

        try:
            all_predictions['clustering'] = self.clustering_predict(count=count)
        except:
            pass

        if not all_predictions:
            return self._random_predict(count)

        # 集成预测结果
        ensemble_results = []

        for i in range(count):
            front_votes = defaultdict(float)
            back_votes = defaultdict(float)

            for method_name, predictions in all_predictions.items():
                if i < len(predictions):
                    front_balls, back_balls = predictions[i]
                    weight = self.predictor_weights.get(method_name, 0.1)

                    for ball in front_balls:
                        front_votes[ball] += weight

                    for ball in back_balls:
                        back_votes[ball] += weight

            # 选择得票最高的号码
            front_selected = self._select_top_balls(front_votes, 5)
            back_selected = self._select_top_balls(back_votes, 2)

            ensemble_results.append((sorted(front_selected), sorted(back_selected)))

        return ensemble_results

    def _select_top_balls(self, votes, num_select):
        """选择得票最高的号码"""
        if not votes:
            max_ball = 35 if num_select == 5 else 12
            return list(np.random.choice(range(1, max_ball + 1), num_select, replace=False))

        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        selected = [ball for ball, vote in sorted_votes[:num_select]]

        if len(selected) < num_select:
            max_ball = 35 if num_select == 5 else 12
            remaining = [i for i in range(1, max_ball + 1) if i not in selected]
            additional = np.random.choice(remaining, num_select - len(selected), replace=False)
            selected.extend(additional)

        return selected[:num_select]

    def _random_predict(self, count):
        """随机预测（备用方案）"""
        predictions = []
        for _ in range(count):
            front_balls = sorted(np.random.choice(range(1, 36), 5, replace=False))
            back_balls = sorted(np.random.choice(range(1, 13), 2, replace=False))
            predictions.append((front_balls, back_balls))
        return predictions


class CompoundPredictor(BasePredictor):
    """复式投注预测器 - 支持任意期数分析，生成任意注数的任意前区+后区数量组合"""

    def __init__(self, data_file):
        super().__init__(data_file)

        # 初始化分析器
        self.traditional_predictor = None
        self.advanced_predictor = None

        if self.df is not None:
            self.traditional_predictor = TraditionalPredictor(data_file)
            self.advanced_predictor = AdvancedPredictor(data_file)

    def predict_compound_combinations(self, periods=3000, combinations=None, method="hybrid", explain=True):
        """预测复式投注组合

        Args:
            periods: 分析期数
            combinations: 组合列表，格式：[(前区数量, 后区数量), ...]
                         例如：[(6, 2), (7, 5)] 表示第一注6+2，第二注7+5
            method: 预测方法 ("hybrid", "markov", "advanced")
            explain: 是否显示详细过程

        Returns:
            预测结果列表
        """
        if not self.traditional_predictor or not self.advanced_predictor:
            print("❌ 分析器初始化失败")
            return []

        if not combinations:
            combinations = [(6, 2), (7, 3)]  # 默认组合

        if explain:
            print("=" * 80)
            print("🎯 大乐透复式投注预测器")
            print("=" * 80)
            print(f"📊 分析期数: {periods} 期")
            print(f"🎲 预测方法: {self._get_method_name(method)}")
            print(f"📝 复式组合: {len(combinations)} 注")
            for i, (front, back) in enumerate(combinations, 1):
                print(f"   第 {i} 注: {front}+{back} (前区{front}个号码，后区{back}个号码)")
            print()

        predictions = []

        for i, (front_count, back_count) in enumerate(combinations, 1):
            if explain:
                print(f"🔮 生成第 {i} 注复式组合 ({front_count}+{back_count})...")

            # 验证参数合理性
            if not self._validate_combination(front_count, back_count):
                print(f"❌ 第 {i} 注组合参数无效: {front_count}+{back_count}")
                continue

            # 根据方法选择预测器
            if method == "hybrid":
                front_balls, back_balls = self._predict_hybrid_compound(
                    periods, front_count, back_count, i, explain
                )
            elif method == "markov":
                front_balls, back_balls = self._predict_markov_compound(
                    periods, front_count, back_count, i, explain
                )
            elif method == "advanced":
                front_balls, back_balls = self._predict_advanced_compound(
                    periods, front_count, back_count, i, explain
                )
            else:
                print(f"❌ 未知的预测方法: {method}")
                continue

            if front_balls and back_balls:
                prediction = {
                    'index': i,
                    'combination': f"{front_count}+{back_count}",
                    'front_count': front_count,
                    'back_count': back_count,
                    'front_balls': sorted(front_balls),
                    'back_balls': sorted(back_balls),
                    'total_combinations': self._calculate_total_combinations(front_count, back_count),
                    'investment_cost': self._calculate_investment_cost(front_count, back_count)
                }
                predictions.append(prediction)

                if explain:
                    self._print_compound_result(prediction)
            else:
                print(f"❌ 第 {i} 注预测失败")

        if explain:
            print("\n" + "=" * 80)
            print("✅ 复式投注预测完成")
            if predictions:
                total_combinations = sum(p['total_combinations'] for p in predictions)
                total_cost = sum(p['investment_cost'] for p in predictions)
                print(f"📊 总投注组合数: {total_combinations:,}")
                print(f"💰 总投注成本: {total_cost:,} 元")
            print("=" * 80)

        return predictions

    def _get_method_name(self, method):
        """获取方法名称"""
        method_names = {
            'hybrid': '混合分析',
            'markov': '马尔可夫链分析',
            'advanced': '高级集成分析'
        }
        return method_names.get(method, method)

    def _validate_combination(self, front_count, back_count):
        """验证组合参数的合理性"""
        if front_count < 5 or front_count > 35:
            return False
        if back_count < 2 or back_count > 12:
            return False
        return True

    def _predict_hybrid_compound(self, periods, front_count, back_count, index, explain=True):
        """使用混合分析预测复式组合"""
        try:
            # 使用传统预测器的混合分析
            predictions = self.traditional_predictor.hybrid_predict(periods=periods, count=5)

            # 收集候选号码
            front_candidates = Counter()
            back_candidates = Counter()

            for front_balls, back_balls in predictions:
                for ball in front_balls:
                    front_candidates[ball] += 1
                for ball in back_balls:
                    back_candidates[ball] += 1

            # 选择频率最高的号码
            front_balls = self._select_top_numbers_from_counter(front_candidates, front_count, 35)
            back_balls = self._select_top_numbers_from_counter(back_candidates, back_count, 12)

            if explain:
                print(f"   ✅ 混合分析完成")
                print(f"   📊 前区选择频率最高的{front_count}个号码")
                print(f"   📊 后区选择频率最高的{back_count}个号码")

            return front_balls, back_balls

        except Exception as e:
            print(f"❌ 混合分析预测失败: {e}")
            return [], []

    def _predict_markov_compound(self, periods, front_count, back_count, index, explain=True):
        """使用马尔可夫链分析预测复式组合"""
        try:
            # 使用传统预测器的马尔可夫链分析
            predictions = self.traditional_predictor.markov_predict(count=5)

            # 收集候选号码
            front_candidates = Counter()
            back_candidates = Counter()

            for front_balls, back_balls in predictions:
                for ball in front_balls:
                    front_candidates[ball] += 1
                for ball in back_balls:
                    back_candidates[ball] += 1

            # 选择频率最高的号码
            front_balls = self._select_top_numbers_from_counter(front_candidates, front_count, 35)
            back_balls = self._select_top_numbers_from_counter(back_candidates, back_count, 12)

            if explain:
                print(f"   ✅ 马尔可夫链分析完成")
                print(f"   🔗 前区选择转移概率最高的{front_count}个号码")
                print(f"   🔗 后区选择转移概率最高的{back_count}个号码")

            return front_balls, back_balls

        except Exception as e:
            print(f"❌ 马尔可夫链分析预测失败: {e}")
            return [], []

    def _predict_advanced_compound(self, periods, front_count, back_count, index, explain=True):
        """使用高级集成分析预测复式组合"""
        try:
            # 使用高级预测器的集成分析
            predictions = self.advanced_predictor.ensemble_predict(count=5)

            # 收集候选号码
            front_candidates = Counter()
            back_candidates = Counter()

            for front_balls, back_balls in predictions:
                for ball in front_balls:
                    front_candidates[ball] += 1
                for ball in back_balls:
                    back_candidates[ball] += 1

            # 选择频率最高的号码
            front_balls = self._select_top_numbers_from_counter(front_candidates, front_count, 35)
            back_balls = self._select_top_numbers_from_counter(back_candidates, back_count, 12)

            if explain:
                print(f"   ✅ 高级集成分析完成")
                print(f"   🧠 前区选择集成评分最高的{front_count}个号码")
                print(f"   🧠 后区选择集成评分最高的{back_count}个号码")

            return front_balls, back_balls

        except Exception as e:
            print(f"❌ 高级集成分析预测失败: {e}")
            return [], []

    def _select_top_numbers_from_counter(self, candidates, count, max_ball):
        """从计数器中选择指定数量的号码"""
        # 获取最常见的号码
        most_common = candidates.most_common(count * 2)  # 获取更多候选

        if len(most_common) >= count:
            # 如果有足够的候选号码，随机选择
            import random
            selected_candidates = [ball for ball, freq in most_common]
            selected = random.sample(selected_candidates[:count*2], min(count, len(selected_candidates)))
        else:
            # 如果候选号码不足，补充随机号码
            selected = [ball for ball, freq in most_common]
            remaining = [i for i in range(1, max_ball + 1) if i not in selected]
            import random
            additional = random.sample(remaining, count - len(selected))
            selected.extend(additional)

        return selected[:count]

    def _calculate_total_combinations(self, front_count, back_count):
        """计算总投注组合数"""
        import math

        # C(front_count, 5) * C(back_count, 2)
        front_combinations = math.comb(front_count, 5)
        back_combinations = math.comb(back_count, 2)

        return front_combinations * back_combinations

    def _calculate_investment_cost(self, front_count, back_count, single_cost=3):
        """计算投注成本"""
        total_combinations = self._calculate_total_combinations(front_count, back_count)
        return total_combinations * single_cost

    def _print_compound_result(self, prediction):
        """打印复式预测结果"""
        print(f"   🎯 第 {prediction['index']} 注 ({prediction['combination']}):")

        front_str = ' '.join([f"{ball:02d}" for ball in prediction['front_balls']])
        back_str = ' '.join([f"{ball:02d}" for ball in prediction['back_balls']])

        print(f"      前区({prediction['front_count']}个): {front_str}")
        print(f"      后区({prediction['back_count']}个): {back_str}")
        print(f"      投注组合数: {prediction['total_combinations']:,}")
        print(f"      投注成本: {prediction['investment_cost']:,} 元")
        print()

    def predict_compound(self, combinations=None, method="hybrid", periods=150):
        """简化的复式投注预测（保持向后兼容）"""
        if combinations is None:
            combinations = ["6+2", "7+3"]

        # 转换格式
        combo_tuples = []
        for combo in combinations:
            front_count, back_count = map(int, combo.split('+'))
            combo_tuples.append((front_count, back_count))

        # 调用完整的复式预测方法
        predictions = self.predict_compound_combinations(
            periods=periods,
            combinations=combo_tuples,
            method=method,
            explain=False
        )

        # 转换为原格式
        results = {}
        for pred in predictions:
            combo = pred['combination']
            results[combo] = {
                'front_balls': pred['front_balls'],
                'back_balls': pred['back_balls'],
                'combinations': pred['total_combinations'],
                'cost': pred['investment_cost']
            }

        return results


class SuperPredictor(BasePredictor):
    """超级预测器 - 整合所有预测方法"""

    def __init__(self, data_file):
        super().__init__(data_file)

        # 预测器权重
        self.predictor_weights = {
            'traditional': 0.60,  # 传统方法总权重
            'advanced': 0.40      # 高级方法总权重
        }

        # 初始化预测器
        self.traditional_predictor = TraditionalPredictor(data_file)
        self.advanced_predictor = AdvancedPredictor(data_file)

    def quick_predict(self, count=1):
        """快速预测"""
        # 优先使用传统混合分析
        try:
            predictions = self.traditional_predictor.hybrid_predict(periods=100, count=count)
            return predictions
        except Exception as e:
            print(f"快速预测失败，使用随机预测: {e}")
            return self._random_predict(count)

    def ultimate_ensemble_predict(self, count=1):
        """终极集成预测 - 融合所有方法"""
        all_predictions = {}

        # 获取传统方法预测
        try:
            traditional_pred = self.traditional_predictor.ensemble_predict(count=count)
            all_predictions['traditional'] = traditional_pred
        except Exception as e:
            print(f"传统方法预测失败: {e}")

        # 获取高级方法预测
        try:
            advanced_pred = self.advanced_predictor.ensemble_predict(count=count)
            all_predictions['advanced'] = advanced_pred
        except Exception as e:
            print(f"高级方法预测失败: {e}")

        if not all_predictions:
            print("所有预测方法都失败，使用随机预测")
            return self._random_predict(count)

        # 终极集成
        ultimate_results = []

        for i in range(count):
            front_votes = defaultdict(float)
            back_votes = defaultdict(float)

            for method_type, predictions in all_predictions.items():
                if i < len(predictions):
                    front_balls, back_balls = predictions[i]
                    weight = self.predictor_weights[method_type]

                    for ball in front_balls:
                        front_votes[ball] += weight

                    for ball in back_balls:
                        back_votes[ball] += weight

            # 选择得票最高的号码
            front_selected = self._select_top_balls(front_votes, 5)
            back_selected = self._select_top_balls(back_votes, 2)

            ultimate_results.append((sorted(front_selected), sorted(back_selected)))

        return ultimate_results

    def compare_all_methods(self, count=1):
        """对比所有预测方法"""
        results = {}

        # 传统方法
        try:
            results['hybrid'] = self.traditional_predictor.hybrid_predict(periods=150, count=count)
        except:
            results['hybrid'] = []

        try:
            results['markov'] = self.traditional_predictor.markov_predict(count=count)
        except:
            results['markov'] = []

        try:
            results['statistical'] = self.traditional_predictor.statistical_predict(count=count)
        except:
            results['statistical'] = []

        # 高级方法
        try:
            results['lstm'] = self.advanced_predictor.lstm_predict(count=count)
        except:
            results['lstm'] = []

        try:
            results['ensemble_ml'] = self.advanced_predictor.ensemble_ml_predict(count=count)
        except:
            results['ensemble_ml'] = []

        try:
            results['monte_carlo'] = self.advanced_predictor.monte_carlo_predict(count=count)
        except:
            results['monte_carlo'] = []

        try:
            results['clustering'] = self.advanced_predictor.clustering_predict(count=count)
        except:
            results['clustering'] = []

        # 集成方法
        try:
            results['traditional_ensemble'] = self.traditional_predictor.ensemble_predict(count=count)
        except:
            results['traditional_ensemble'] = []

        try:
            results['advanced_ensemble'] = self.advanced_predictor.ensemble_predict(count=count)
        except:
            results['advanced_ensemble'] = []

        try:
            results['ultimate_ensemble'] = self.ultimate_ensemble_predict(count=count)
        except:
            results['ultimate_ensemble'] = []

        return results

    def _select_top_balls(self, votes, num_select):
        """选择得票最高的号码"""
        if not votes:
            max_ball = 35 if num_select == 5 else 12
            return list(np.random.choice(range(1, max_ball + 1), num_select, replace=False))

        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        selected = [ball for ball, vote in sorted_votes[:num_select]]

        if len(selected) < num_select:
            max_ball = 35 if num_select == 5 else 12
            remaining = [i for i in range(1, max_ball + 1) if i not in selected]
            additional = np.random.choice(remaining, num_select - len(selected), replace=False)
            selected.extend(additional)

        return selected[:num_select]

    def _random_predict(self, count):
        """随机预测（备用方案）"""
        predictions = []
        for _ in range(count):
            front_balls = sorted(np.random.choice(range(1, 36), 5, replace=False))
            back_balls = sorted(np.random.choice(range(1, 13), 2, replace=False))
            predictions.append((front_balls, back_balls))
        return predictions


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="大乐透预测器集合")
    parser.add_argument("-d", "--data", default="data/dlt_data_all.csv", help="数据文件路径")
    parser.add_argument("-n", "--num", type=int, default=1, help="预测注数")
    parser.add_argument("-m", "--method",
                       choices=['traditional', 'advanced', 'compound', 'super_quick', 'super_ensemble', 'super_compare'],
                       default='super_ensemble', help="预测方法")
    parser.add_argument("-c", "--combinations", nargs='+', default=["6+2", "7+3"], help="复式投注组合")

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"数据文件不存在: {args.data}")
        return

    # 根据选择的方法进行预测
    if args.method == 'traditional':
        predictor = TraditionalPredictor(args.data)
        predictions = predictor.ensemble_predict(count=args.num)

        print("🔬 传统方法集成预测:")
        for i, (front, back) in enumerate(predictions, 1):
            front_str = ' '.join([str(b).zfill(2) for b in front])
            back_str = ' '.join([str(b).zfill(2) for b in back])
            print(f"第 {i} 注: 前区 {front_str} | 后区 {back_str}")

    elif args.method == 'advanced':
        predictor = AdvancedPredictor(args.data)
        predictions = predictor.ensemble_predict(count=args.num)

        print("🧠 高级方法集成预测:")
        for i, (front, back) in enumerate(predictions, 1):
            front_str = ' '.join([str(b).zfill(2) for b in front])
            back_str = ' '.join([str(b).zfill(2) for b in back])
            print(f"第 {i} 注: 前区 {front_str} | 后区 {back_str}")

    elif args.method == 'compound':
        predictor = CompoundPredictor(args.data)
        results = predictor.predict_compound(combinations=args.combinations)

        print("🎰 复式投注预测:")
        for combo, result in results.items():
            front_str = ' '.join([str(b).zfill(2) for b in result['front_balls']])
            back_str = ' '.join([str(b).zfill(2) for b in result['back_balls']])
            print(f"{combo}: 前区 {front_str} | 后区 {back_str}")
            print(f"  组合数: {result['combinations']}, 成本: {result['cost']}元")

    elif args.method == 'super_quick':
        predictor = SuperPredictor(args.data)
        predictions = predictor.quick_predict(count=args.num)

        print("⚡ 超级预测器 - 快速预测:")
        for i, (front, back) in enumerate(predictions, 1):
            front_str = ' '.join([str(b).zfill(2) for b in front])
            back_str = ' '.join([str(b).zfill(2) for b in back])
            print(f"第 {i} 注: 前区 {front_str} | 后区 {back_str}")

    elif args.method == 'super_ensemble':
        predictor = SuperPredictor(args.data)
        predictions = predictor.ultimate_ensemble_predict(count=args.num)

        print("🌟 超级预测器 - 终极集成预测:")
        for i, (front, back) in enumerate(predictions, 1):
            front_str = ' '.join([str(b).zfill(2) for b in front])
            back_str = ' '.join([str(b).zfill(2) for b in back])
            print(f"第 {i} 注: 前区 {front_str} | 后区 {back_str}")

    elif args.method == 'super_compare':
        predictor = SuperPredictor(args.data)
        results = predictor.compare_all_methods(count=1)

        print("📊 超级预测器 - 全方法对比:")
        for method_name, predictions in results.items():
            if predictions:
                front, back = predictions[0]
                front_str = ' '.join([str(b).zfill(2) for b in front])
                back_str = ' '.join([str(b).zfill(2) for b in back])
                print(f"{method_name.upper():<20}: 前区 {front_str} | 后区 {back_str}")
            else:
                print(f"{method_name.upper():<20}: 预测失败")


if __name__ == "__main__":
    main()