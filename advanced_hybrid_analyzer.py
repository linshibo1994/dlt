#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高级混合分析预测系统
基于统计学、概率论、马尔可夫链、贝叶斯分析、冷热号分布规律等多种数学模型
"""

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime
import scipy.stats as stats
from scipy.fft import fft
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class AdvancedHybridAnalyzer:
    """高级混合分析预测器"""
    
    def __init__(self, data_file, output_dir="output/hybrid"):
        """初始化分析器
        
        Args:
            data_file: 数据文件路径
            output_dir: 输出目录
        """
        self.data_file = data_file
        self.output_dir = output_dir
        self.df = None
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 模型权重配置（基于技术文档）
        self.model_weights = {
            'statistical': 0.15,    # 统计学分析
            'probability': 0.20,    # 概率论分析
            'markov': 0.25,         # 马尔可夫链分析（最高权重）
            'bayesian': 0.15,       # 贝叶斯分析
            'hot_cold': 0.15,       # 冷热号分析
            'cycle': 0.10,          # 周期性分析
            'correlation': 0.00     # 相关性分析（验证用）
        }
        
        # 稳定性阈值配置
        self.stability_thresholds = {
            'front_position_transition': 5,
            'front_global_transition': 10,
            'back_transition': 3,
            'correlation_threshold': 0.3,
            'significance_level': 0.05
        }
        
        # 加载数据
        self.load_data()
    
    def load_data(self):
        """加载数据"""
        try:
            self.df = pd.read_csv(self.data_file)
            # 按期号排序（从早到晚）
            self.df = self.df.sort_values('issue', ascending=True)
            print(f"成功加载数据，共 {len(self.df)} 条记录")
            print(f"数据范围: {self.df.iloc[0]['issue']} - {self.df.iloc[-1]['issue']}")
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def parse_balls(self, balls_str):
        """解析号码字符串"""
        return [int(ball.strip()) for ball in str(balls_str).split(",")]
    
    def predict_with_hybrid_analysis(self, periods=100, count=1, explain=True):
        """使用混合分析进行预测
        
        Args:
            periods: 分析期数
            count: 预测注数
            explain: 是否显示详细过程
        
        Returns:
            预测结果列表
        """
        if explain:
            print("=" * 80)
            print("🔬 高级混合分析预测系统")
            print("=" * 80)
            print(f"📊 分析期数: {periods} 期")
            print(f"🎯 预测注数: {count} 注")
            print(f"📈 使用模型: 统计学、概率论、马尔可夫链、贝叶斯、冷热号、周期性、相关性")
            print()
        
        # 获取分析数据
        analysis_data = self.df.tail(periods).copy()
        
        if len(analysis_data) < 10:
            print("❌ 数据不足，至少需要10期数据")
            return []
        
        # 执行多模型分析
        if explain:
            print("🔍 开始多模型并行分析...")
        
        hybrid_analysis = self._run_hybrid_analysis(analysis_data, explain)
        
        # 生成预测
        predictions = self._generate_predictions(
            hybrid_analysis, analysis_data, count, explain
        )
        
        # 保存分析结果
        self._save_analysis_results(hybrid_analysis, predictions, periods)
        
        if explain:
            print("\n" + "=" * 80)
            print("✅ 高级混合分析完成")
            print("=" * 80)
        
        return predictions
    
    def _run_hybrid_analysis(self, data, explain=True):
        """运行混合分析"""
        analysis_results = {}
        
        # 1. 统计学分析模块 (15%)
        if explain:
            print("📈 1. 统计学分析模块 (权重: 15%)")
        analysis_results['statistical'] = self._statistical_analysis(data, explain)
        
        # 2. 概率论分析模块 (20%)
        if explain:
            print("\n🎲 2. 概率论分析模块 (权重: 20%)")
        analysis_results['probability'] = self._probability_analysis(data, explain)
        
        # 3. 马尔可夫链分析模块 (25%)
        if explain:
            print("\n🔗 3. 马尔可夫链分析模块 (权重: 25%)")
        analysis_results['markov'] = self._markov_analysis(data, explain)
        
        # 4. 贝叶斯分析模块 (15%)
        if explain:
            print("\n🧮 4. 贝叶斯分析模块 (权重: 15%)")
        analysis_results['bayesian'] = self._bayesian_analysis(data, explain)
        
        # 5. 冷热号分析模块 (15%)
        if explain:
            print("\n🌡️ 5. 冷热号分析模块 (权重: 15%)")
        analysis_results['hot_cold'] = self._hot_cold_analysis(data, explain)
        
        # 6. 周期性分析模块 (10%)
        if explain:
            print("\n🔄 6. 周期性分析模块 (权重: 10%)")
        analysis_results['cycle'] = self._cycle_analysis(data, explain)
        
        # 7. 相关性分析模块 (验证用)
        if explain:
            print("\n🔍 7. 相关性分析模块 (验证用)")
        analysis_results['correlation'] = self._correlation_analysis(data, explain)
        
        return analysis_results
    
    def _statistical_analysis(self, data, explain=True):
        """统计学分析模块"""
        results = {}
        
        # 解析前区和后区号码
        front_balls_lists = []
        back_balls_lists = []
        
        for _, row in data.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])
            front_balls_lists.append(front_balls)
            back_balls_lists.append(back_balls)
        
        # 计算和值统计特征
        front_sums = [sum(balls) for balls in front_balls_lists]
        
        # 基本统计量
        stats_features = {
            '均值': np.mean(front_sums),
            '标准差': np.std(front_sums),
            '中位数': np.median(front_sums),
            '偏度': stats.skew(front_sums),
            '峰度': stats.kurtosis(front_sums),
            '变异系数': np.std(front_sums) / np.mean(front_sums),
            '四分位距': np.percentile(front_sums, 75) - np.percentile(front_sums, 25)
        }
        
        # 正态性检验
        if len(front_sums) > 8:
            try:
                dagostino_stat, dagostino_p = stats.normaltest(front_sums)
                stats_features['正态性检验p值'] = dagostino_p
                stats_features['是否正态分布'] = dagostino_p > 0.05
            except:
                stats_features['正态性检验p值'] = 0.5
                stats_features['是否正态分布'] = True
        
        results['和值统计'] = stats_features
        
        # 计算号码评分
        target_sum = stats_features['均值']
        front_scores = {}
        
        for ball in range(1, 36):
            # 基于目标和值的适应性评分
            contribution_score = self._calculate_sum_contribution(ball, target_sum)
            front_scores[ball] = contribution_score
        
        results['前区评分'] = front_scores
        
        # 后区统计分析
        back_counts = Counter([ball for balls in back_balls_lists for ball in balls])
        back_scores = {}
        for ball in range(1, 13):
            back_scores[ball] = back_counts.get(ball, 0) / len(data)
        
        results['后区评分'] = back_scores
        
        if explain:
            print(f"   📊 和值均值: {stats_features['均值']:.2f}")
            print(f"   📊 和值标准差: {stats_features['标准差']:.2f}")
            print(f"   📊 分布偏度: {stats_features['偏度']:.3f}")
            print(f"   📊 是否正态分布: {stats_features['是否正态分布']}")
        
        return results
    
    def _calculate_sum_contribution(self, ball, target_sum):
        """计算号码对目标和值的贡献度"""
        # 假设其他4个号码的平均值
        other_avg = (target_sum - ball) / 4
        
        # 评估这个号码在目标和值中的合理性
        if 1 <= other_avg <= 35:
            # 号码越接近理想分布，得分越高
            ideal_contribution = target_sum / 5
            deviation = abs(ball - ideal_contribution)
            score = max(0, 1 - deviation / 35)
        else:
            score = 0.1  # 不合理的组合给低分
        
        return score
    
    def _probability_analysis(self, data, explain=True):
        """概率论分析模块"""
        results = {}
        
        # 计算号码出现概率
        front_counts = Counter()
        back_counts = Counter()
        
        for _, row in data.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])
            
            for ball in front_balls:
                front_counts[ball] += 1
            for ball in back_balls:
                back_counts[ball] += 1
        
        total_front_draws = len(data) * 5
        total_back_draws = len(data) * 2
        
        # 计算概率分布
        front_probs = {ball: count / total_front_draws for ball, count in front_counts.items()}
        back_probs = {ball: count / total_back_draws for ball, count in back_counts.items()}
        
        # 补充未出现的号码
        for ball in range(1, 36):
            if ball not in front_probs:
                front_probs[ball] = 0.001  # 给极小概率
        
        for ball in range(1, 13):
            if ball not in back_probs:
                back_probs[ball] = 0.001
        
        # 卡方检验（检验均匀分布假设）
        expected_front = total_front_draws / 35
        observed_front = [front_counts.get(i, 0) for i in range(1, 36)]
        
        try:
            chi2_stat, chi2_p = stats.chisquare(observed_front)
            is_uniform = chi2_p > 0.05
        except:
            chi2_stat, chi2_p = 0, 0.5
            is_uniform = True
        
        # 信息熵计算
        front_entropy = -sum(p * np.log2(p) for p in front_probs.values() if p > 0)
        
        results['前区概率'] = front_probs
        results['后区概率'] = back_probs
        results['卡方检验'] = {'统计量': chi2_stat, 'p值': chi2_p, '是否均匀': is_uniform}
        results['信息熵'] = front_entropy
        
        if explain:
            print(f"   🎲 前区信息熵: {front_entropy:.3f}")
            print(f"   🎲 卡方检验p值: {chi2_p:.3f}")
            print(f"   🎲 分布是否均匀: {is_uniform}")
        
        return results

    def _markov_analysis(self, data, explain=True):
        """马尔可夫链分析模块"""
        results = {}

        # 构建状态转移矩阵
        front_transitions = defaultdict(lambda: defaultdict(int))
        back_transitions = defaultdict(lambda: defaultdict(int))

        sorted_data = data.sort_values('issue', ascending=True).reset_index(drop=True)

        # 分析前区转移
        for i in range(len(sorted_data) - 1):
            current_front = self.parse_balls(sorted_data.iloc[i]['front_balls'])
            next_front = self.parse_balls(sorted_data.iloc[i + 1]['front_balls'])

            for current_ball in current_front:
                for next_ball in next_front:
                    front_transitions[current_ball][next_ball] += 1

        # 分析后区转移
        for i in range(len(sorted_data) - 1):
            current_back = self.parse_balls(sorted_data.iloc[i]['back_balls'])
            next_back = self.parse_balls(sorted_data.iloc[i + 1]['back_balls'])

            for current_ball in current_back:
                for next_ball in next_back:
                    back_transitions[current_ball][next_ball] += 1

        # 计算转移概率（带稳定性权重）
        front_probs = self._calculate_stable_transition_probs(
            front_transitions, self.stability_thresholds['front_global_transition'], 35
        )
        back_probs = self._calculate_stable_transition_probs(
            back_transitions, self.stability_thresholds['back_transition'], 12
        )

        results['前区转移概率'] = front_probs
        results['后区转移概率'] = back_probs

        # 计算稳定性统计
        stable_states = sum(1 for probs in front_probs.values()
                          if any(info.get('稳定性权重', 0) >= 0.5 for info in probs.values()))

        results['稳定性统计'] = {
            '总状态数': len(front_probs),
            '稳定状态数': stable_states,
            '稳定性比例': stable_states / len(front_probs) if front_probs else 0
        }

        if explain:
            print(f"   🔗 前区状态数: {len(front_probs)}")
            print(f"   🔗 稳定状态数: {stable_states}")
            print(f"   🔗 稳定性比例: {stable_states / len(front_probs) * 100:.1f}%" if front_probs else "0%")

        return results

    def _calculate_stable_transition_probs(self, transitions, threshold, max_ball):
        """计算带稳定性权重的转移概率"""
        stable_probs = {}

        for current, nexts in transitions.items():
            total_transitions = sum(nexts.values())
            stability_weight = min(1.0, total_transitions / threshold)

            stable_probs[current] = {}
            for next_ball, count in nexts.items():
                base_prob = count / total_transitions if total_transitions > 0 else 0
                uniform_prob = 1 / max_ball

                # 稳定性调整概率
                stable_prob = (base_prob * stability_weight +
                             (1 - stability_weight) * uniform_prob)

                stable_probs[current][next_ball] = {
                    '概率': stable_prob,
                    '原始概率': base_prob,
                    '出现次数': count,
                    '总转移次数': total_transitions,
                    '稳定性权重': stability_weight
                }

        return stable_probs

    def _bayesian_analysis(self, data, explain=True):
        """贝叶斯分析模块"""
        results = {}

        # 先验概率设定
        FRONT_PRIOR = 1/35
        BACK_PRIOR = 1/12

        # Beta分布参数
        ALPHA_PRIOR = 1
        BETA_PRIOR_FRONT = 35
        BETA_PRIOR_BACK = 12

        # 计算观测数据
        front_counts = {i: 1 for i in range(1, 36)}  # 加1平滑
        back_counts = {i: 1 for i in range(1, 13)}

        total_front_draws = len(data) * 5
        total_back_draws = len(data) * 2

        for _, row in data.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])

            for ball in front_balls:
                front_counts[ball] += 1
            for ball in back_balls:
                back_counts[ball] += 1

        # 贝叶斯后验分析
        front_posterior = {}
        for ball in range(1, 36):
            successes = front_counts[ball] - 1  # 减去平滑项
            trials = total_front_draws

            # Beta后验参数
            alpha_post = ALPHA_PRIOR + successes
            beta_post = BETA_PRIOR_FRONT + trials - successes

            # 后验统计
            posterior_mean = alpha_post / (alpha_post + beta_post)

            # 贝叶斯因子
            likelihood = successes / trials if trials > 0 else 0
            bayes_factor = likelihood / FRONT_PRIOR if FRONT_PRIOR > 0 else 1

            front_posterior[ball] = {
                '后验均值': posterior_mean,
                '贝叶斯因子': bayes_factor,
                '观测次数': successes,
                '总试验次数': trials
            }

        # 后区贝叶斯分析
        back_posterior = {}
        for ball in range(1, 13):
            successes = back_counts[ball] - 1
            trials = total_back_draws

            alpha_post = ALPHA_PRIOR + successes
            beta_post = BETA_PRIOR_BACK + trials - successes

            posterior_mean = alpha_post / (alpha_post + beta_post)
            likelihood = successes / trials if trials > 0 else 0
            bayes_factor = likelihood / BACK_PRIOR if BACK_PRIOR > 0 else 1

            back_posterior[ball] = {
                '后验均值': posterior_mean,
                '贝叶斯因子': bayes_factor,
                '观测次数': successes,
                '总试验次数': trials
            }

        results['前区后验分析'] = front_posterior
        results['后区后验分析'] = back_posterior

        if explain:
            avg_bayes_factor = np.mean([info['贝叶斯因子'] for info in front_posterior.values()])
            print(f"   🧮 平均贝叶斯因子: {avg_bayes_factor:.3f}")
            print(f"   🧮 前区观测期数: {len(data)}")

        return results

    def _hot_cold_analysis(self, data, explain=True):
        """冷热号分析模块"""
        results = {}

        # 多时间窗口分析
        windows = [10, 20, 30]  # 最近10期、20期、30期

        front_heat_analysis = {}
        back_heat_analysis = {}

        for window in windows:
            if len(data) < window:
                continue

            recent_data = data.tail(window)

            # 计算热度指数
            front_counts = Counter()
            back_counts = Counter()

            for _, row in recent_data.iterrows():
                front_balls = self.parse_balls(row['front_balls'])
                back_balls = self.parse_balls(row['back_balls'])

                for ball in front_balls:
                    front_counts[ball] += 1
                for ball in back_balls:
                    back_counts[ball] += 1

            # 计算热度指数 = 实际频率 / 期望频率
            expected_front = window * 5 / 35
            expected_back = window * 2 / 12

            front_heat = {}
            for ball in range(1, 36):
                actual_freq = front_counts.get(ball, 0)
                heat_index = actual_freq / expected_front if expected_front > 0 else 0
                front_heat[ball] = heat_index

            back_heat = {}
            for ball in range(1, 13):
                actual_freq = back_counts.get(ball, 0)
                heat_index = actual_freq / expected_back if expected_back > 0 else 0
                back_heat[ball] = heat_index

            front_heat_analysis[f'{window}期'] = front_heat
            back_heat_analysis[f'{window}期'] = back_heat

        # 综合热度评分（多周期加权平均）
        front_comprehensive_heat = {}
        back_comprehensive_heat = {}

        for ball in range(1, 36):
            heat_scores = []
            for window_key, heat_data in front_heat_analysis.items():
                if ball in heat_data:
                    # 中心化处理：减去1.0，使得平均热度为0
                    centered_heat = (heat_data[ball] - 1.0) * 0.5
                    heat_scores.append(centered_heat)

            front_comprehensive_heat[ball] = np.mean(heat_scores) if heat_scores else 0

        for ball in range(1, 13):
            heat_scores = []
            for window_key, heat_data in back_heat_analysis.items():
                if ball in heat_data:
                    centered_heat = (heat_data[ball] - 1.0) * 0.5
                    heat_scores.append(centered_heat)

            back_comprehensive_heat[ball] = np.mean(heat_scores) if heat_scores else 0

        results['前区热度分析'] = front_heat_analysis
        results['后区热度分析'] = back_heat_analysis
        results['前区综合热度'] = front_comprehensive_heat
        results['后区综合热度'] = back_comprehensive_heat

        # 分类冷热号
        front_hot = [ball for ball, heat in front_comprehensive_heat.items() if heat > 0.3]
        front_cold = [ball for ball, heat in front_comprehensive_heat.items() if heat < -0.3]

        if explain:
            print(f"   🌡️ 前区热号: {len(front_hot)} 个")
            print(f"   🌡️ 前区冷号: {len(front_cold)} 个")
            if front_hot:
                print(f"   🌡️ 热号示例: {sorted(front_hot)[:5]}")

        return results

    def _cycle_analysis(self, data, explain=True):
        """周期性分析模块"""
        results = {}

        # 构建时间序列数据
        front_series = []
        back_series = []

        for _, row in data.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])

            # 使用和值作为时间序列特征
            front_series.append(sum(front_balls))
            back_series.append(sum(back_balls))

        # FFT频域分析
        if len(front_series) >= 16:  # 至少需要16个数据点
            front_fft = np.abs(fft(front_series))
            back_fft = np.abs(fft(back_series))

            # 找到主要频率成分
            front_freqs = np.fft.fftfreq(len(front_series))
            back_freqs = np.fft.fftfreq(len(back_series))

            # 排除直流分量，找到最强的周期
            front_main_freq_idx = np.argmax(front_fft[1:len(front_fft)//2]) + 1
            back_main_freq_idx = np.argmax(back_fft[1:len(back_fft)//2]) + 1

            front_period = 1 / abs(front_freqs[front_main_freq_idx]) if front_freqs[front_main_freq_idx] != 0 else float('inf')
            back_period = 1 / abs(back_freqs[back_main_freq_idx]) if back_freqs[back_main_freq_idx] != 0 else float('inf')

            results['前区主周期'] = front_period
            results['后区主周期'] = back_period
            results['前区频谱强度'] = front_fft[front_main_freq_idx]
            results['后区频谱强度'] = back_fft[back_main_freq_idx]
        else:
            results['前区主周期'] = float('inf')
            results['后区主周期'] = float('inf')
            results['前区频谱强度'] = 0
            results['后区频谱强度'] = 0

        # 自相关分析
        front_autocorr = self._calculate_autocorrelation(front_series, max_lag=min(20, len(front_series)//2))
        back_autocorr = self._calculate_autocorrelation(back_series, max_lag=min(10, len(back_series)//2))

        results['前区自相关'] = front_autocorr
        results['后区自相关'] = back_autocorr

        # 趋势分析
        front_trend = np.polyfit(range(len(front_series)), front_series, 1)[0]
        back_trend = np.polyfit(range(len(back_series)), back_series, 1)[0]

        results['前区趋势'] = front_trend
        results['后区趋势'] = back_trend

        if explain:
            print(f"   🔄 前区主周期: {front_period:.1f} 期" if front_period != float('inf') else "   🔄 前区主周期: 无明显周期")
            print(f"   🔄 前区趋势: {'上升' if front_trend > 0 else '下降' if front_trend < 0 else '平稳'}")

        return results

    def _calculate_autocorrelation(self, series, max_lag):
        """计算自相关函数"""
        autocorr = {}
        series = np.array(series)
        n = len(series)

        # 标准化
        mean = np.mean(series)
        std = np.std(series)

        if std == 0:
            return {lag: 0 for lag in range(max_lag + 1)}

        normalized_series = (series - mean) / std

        for lag in range(max_lag + 1):
            if lag == 0:
                autocorr[lag] = 1.0
            elif lag < n:
                correlation = np.corrcoef(normalized_series[:-lag], normalized_series[lag:])[0, 1]
                autocorr[lag] = correlation if not np.isnan(correlation) else 0
            else:
                autocorr[lag] = 0

        return autocorr

    def _correlation_analysis(self, data, explain=True):
        """相关性分析模块"""
        results = {}

        # 构建特征矩阵
        features = []

        for _, row in data.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])

            # 特征工程
            feature_vector = [
                sum(front_balls),                    # 前区和值
                max(front_balls) - min(front_balls), # 前区跨度
                len(set(front_balls)),               # 前区唯一数（应该总是5）
                sum(back_balls),                     # 后区和值
                max(back_balls) - min(back_balls),   # 后区跨度
                # 奇偶比例
                sum(1 for ball in front_balls if ball % 2 == 1) / 5,  # 前区奇数比例
                sum(1 for ball in back_balls if ball % 2 == 1) / 2,   # 后区奇数比例
                # 大小比例
                sum(1 for ball in front_balls if ball > 17.5) / 5,    # 前区大数比例
                sum(1 for ball in back_balls if ball > 6) / 2,        # 后区大数比例
            ]
            features.append(feature_vector)

        features = np.array(features)

        if len(features) > 5:
            # 计算相关性矩阵
            correlation_matrix = np.corrcoef(features.T)

            # PCA分析
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)

            pca = PCA()
            pca_result = pca.fit_transform(scaled_features)

            # 主成分贡献率
            explained_variance_ratio = pca.explained_variance_ratio_

            results['相关性矩阵'] = correlation_matrix.tolist()
            results['主成分贡献率'] = explained_variance_ratio.tolist()
            results['累计贡献率'] = np.cumsum(explained_variance_ratio).tolist()

            # 特征重要性
            feature_names = ['前区和值', '前区跨度', '前区唯一数', '后区和值', '后区跨度',
                           '前区奇数比例', '后区奇数比例', '前区大数比例', '后区大数比例']

            # 第一主成分的特征权重
            first_pc_weights = pca.components_[0]
            feature_importance = {name: abs(weight) for name, weight in zip(feature_names, first_pc_weights)}

            results['特征重要性'] = feature_importance

            if explain:
                print(f"   🔍 第一主成分贡献率: {explained_variance_ratio[0]:.3f}")
                most_important = max(feature_importance.items(), key=lambda x: x[1])
                print(f"   🔍 最重要特征: {most_important[0]} ({most_important[1]:.3f})")
        else:
            results['相关性矩阵'] = []
            results['主成分贡献率'] = []
            results['特征重要性'] = {}

        return results

    def _generate_predictions(self, hybrid_analysis, data, count, explain=True):
        """生成预测结果"""
        if explain:
            print("\n🎯 开始生成预测...")

        # 获取最近一期的号码作为马尔可夫链的起始状态
        latest_row = data.iloc[-1]
        latest_front = self.parse_balls(latest_row['front_balls'])
        latest_back = self.parse_balls(latest_row['back_balls'])

        predictions = []
        used_combinations = set()

        for prediction_num in range(count):
            if explain:
                print(f"\n🔮 生成第 {prediction_num + 1} 注预测...")

            # 计算综合评分
            front_scores, back_scores = self._calculate_comprehensive_scores(
                hybrid_analysis, latest_front, latest_back, prediction_num, explain
            )

            # 选择号码
            front_balls, back_balls = self._select_numbers_with_diversity(
                front_scores, back_scores, prediction_num, used_combinations
            )

            # 记录已使用的组合
            combination = (tuple(sorted(front_balls)), tuple(sorted(back_balls)))
            used_combinations.add(combination)

            predictions.append((front_balls, back_balls))

            if explain:
                front_str = ' '.join([str(b).zfill(2) for b in sorted(front_balls)])
                back_str = ' '.join([str(b).zfill(2) for b in sorted(back_balls)])
                print(f"   第 {prediction_num + 1} 注: 前区 {front_str} | 后区 {back_str}")

        return predictions

    def _calculate_comprehensive_scores(self, hybrid_analysis, latest_front, latest_back, prediction_num, explain=True):
        """计算综合评分"""
        front_scores = {i: 0.0 for i in range(1, 36)}
        back_scores = {i: 0.0 for i in range(1, 13)}

        if explain:
            print("   📊 多模型评分计算:")

        # 1. 统计学模型评分 (15%)
        self._apply_statistical_scoring(front_scores, back_scores, hybrid_analysis['statistical'], 0.15, explain)

        # 2. 概率论模型评分 (20%)
        self._apply_probability_scoring(front_scores, back_scores, hybrid_analysis['probability'], 0.20, explain)

        # 3. 马尔可夫链模型评分 (25%)
        self._apply_markov_scoring(front_scores, back_scores, hybrid_analysis['markov'],
                                 latest_front, latest_back, 0.25, explain)

        # 4. 贝叶斯模型评分 (15%)
        self._apply_bayesian_scoring(front_scores, back_scores, hybrid_analysis['bayesian'], 0.15, explain)

        # 5. 冷热号模型评分 (15%)
        self._apply_hot_cold_scoring(front_scores, back_scores, hybrid_analysis['hot_cold'], 0.15, explain)

        # 6. 周期性模型评分 (10%)
        self._apply_cycle_scoring(front_scores, back_scores, hybrid_analysis['cycle'], 0.10, explain)

        return front_scores, back_scores

    def _apply_statistical_scoring(self, front_scores, back_scores, analysis, weight, explain=True):
        """应用统计学评分"""
        if '前区评分' in analysis:
            for ball, score in analysis['前区评分'].items():
                front_scores[ball] += score * weight

        if '后区评分' in analysis:
            for ball, score in analysis['后区评分'].items():
                back_scores[ball] += score * weight

        if explain:
            print(f"     ✓ 统计学评分 (权重: {weight:.0%})")

    def _apply_probability_scoring(self, front_scores, back_scores, analysis, weight, explain=True):
        """应用概率论评分"""
        if '前区概率' in analysis:
            for ball, prob in analysis['前区概率'].items():
                front_scores[ball] += prob * weight * 20  # 放大概率值

        if '后区概率' in analysis:
            for ball, prob in analysis['后区概率'].items():
                back_scores[ball] += prob * weight * 20

        if explain:
            print(f"     ✓ 概率论评分 (权重: {weight:.0%})")

    def _apply_markov_scoring(self, front_scores, back_scores, analysis, latest_front, latest_back, weight, explain=True):
        """应用马尔可夫链评分"""
        # 前区马尔可夫评分
        if '前区转移概率' in analysis:
            front_transitions = analysis['前区转移概率']
            for current_ball in latest_front:
                if current_ball in front_transitions:
                    for next_ball, info in front_transitions[current_ball].items():
                        prob = info.get('概率', 0)
                        front_scores[next_ball] += prob * weight

        # 后区马尔可夫评分
        if '后区转移概率' in analysis:
            back_transitions = analysis['后区转移概率']
            for current_ball in latest_back:
                if current_ball in back_transitions:
                    for next_ball, info in back_transitions[current_ball].items():
                        prob = info.get('概率', 0)
                        back_scores[next_ball] += prob * weight

        if explain:
            print(f"     ✓ 马尔可夫链评分 (权重: {weight:.0%})")

    def _apply_bayesian_scoring(self, front_scores, back_scores, analysis, weight, explain=True):
        """应用贝叶斯评分"""
        if '前区后验分析' in analysis:
            for ball, info in analysis['前区后验分析'].items():
                posterior_mean = info.get('后验均值', 0)
                front_scores[ball] += posterior_mean * weight * 10  # 放大后验概率

        if '后区后验分析' in analysis:
            for ball, info in analysis['后区后验分析'].items():
                posterior_mean = info.get('后验均值', 0)
                back_scores[ball] += posterior_mean * weight * 10

        if explain:
            print(f"     ✓ 贝叶斯评分 (权重: {weight:.0%})")

    def _apply_hot_cold_scoring(self, front_scores, back_scores, analysis, weight, explain=True):
        """应用冷热号评分"""
        if '前区综合热度' in analysis:
            for ball, heat in analysis['前区综合热度'].items():
                front_scores[ball] += heat * weight

        if '后区综合热度' in analysis:
            for ball, heat in analysis['后区综合热度'].items():
                back_scores[ball] += heat * weight

        if explain:
            print(f"     ✓ 冷热号评分 (权重: {weight:.0%})")

    def _apply_cycle_scoring(self, front_scores, back_scores, analysis, weight, explain=True):
        """应用周期性评分"""
        # 基于趋势的简单评分
        front_trend = analysis.get('前区趋势', 0)
        back_trend = analysis.get('后区趋势', 0)

        # 如果有上升趋势，给较大号码更高分数
        if front_trend > 0:
            for ball in range(18, 36):
                front_scores[ball] += weight * 0.1
        elif front_trend < 0:
            for ball in range(1, 18):
                front_scores[ball] += weight * 0.1

        if back_trend > 0:
            for ball in range(7, 13):
                back_scores[ball] += weight * 0.1
        elif back_trend < 0:
            for ball in range(1, 7):
                back_scores[ball] += weight * 0.1

        if explain:
            print(f"     ✓ 周期性评分 (权重: {weight:.0%})")

    def _select_numbers_with_diversity(self, front_scores, back_scores, prediction_num, used_combinations):
        """带多样性的号码选择"""
        # 差异化选择策略
        choice_offset = prediction_num * 0.1

        # 前区号码选择
        sorted_front = sorted(front_scores.items(), key=lambda x: x[1], reverse=True)

        front_balls = []
        for i, (ball, score) in enumerate(sorted_front):
            # 引入随机性，避免总是选择最高分
            if len(front_balls) < 5:
                # 第1注选择最高分，后续注数引入偏移
                if prediction_num == 0 or np.random.random() > choice_offset:
                    front_balls.append(ball)
                elif i < len(sorted_front) - 1:
                    # 跳过当前选择，选择下一个
                    continue

        # 如果选择不足5个，补充剩余的高分号码
        if len(front_balls) < 5:
            remaining_balls = [ball for ball, _ in sorted_front if ball not in front_balls]
            front_balls.extend(remaining_balls[:5-len(front_balls)])

        front_balls = sorted(front_balls[:5])

        # 后区号码选择
        sorted_back = sorted(back_scores.items(), key=lambda x: x[1], reverse=True)

        back_balls = []
        for i, (ball, score) in enumerate(sorted_back):
            if len(back_balls) < 2:
                if prediction_num == 0 or np.random.random() > choice_offset:
                    back_balls.append(ball)
                elif i < len(sorted_back) - 1:
                    continue

        if len(back_balls) < 2:
            remaining_balls = [ball for ball, _ in sorted_back if ball not in back_balls]
            back_balls.extend(remaining_balls[:2-len(back_balls)])

        back_balls = sorted(back_balls[:2])

        # 检查是否与已有组合重复
        combination = (tuple(front_balls), tuple(back_balls))
        if combination in used_combinations:
            # 如果重复，随机调整一个号码
            import random
            if random.random() < 0.5 and len(front_balls) > 0:
                # 调整前区
                replace_idx = random.randint(0, len(front_balls) - 1)
                available_balls = [ball for ball in range(1, 36) if ball not in front_balls]
                if available_balls:
                    front_balls[replace_idx] = random.choice(available_balls)
                    front_balls = sorted(front_balls)
            else:
                # 调整后区
                if len(back_balls) > 0:
                    replace_idx = random.randint(0, len(back_balls) - 1)
                    available_balls = [ball for ball in range(1, 13) if ball not in back_balls]
                    if available_balls:
                        back_balls[replace_idx] = random.choice(available_balls)
                        back_balls = sorted(back_balls)

        return front_balls, back_balls

    def _save_analysis_results(self, hybrid_analysis, predictions, periods):
        """保存分析结果"""
        try:
            # 保存详细分析结果
            analysis_file = os.path.join(self.output_dir, f"hybrid_analysis_{periods}periods.json")

            # 转换numpy类型为Python原生类型
            serializable_analysis = self._make_serializable(hybrid_analysis)

            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_analysis, f, ensure_ascii=False, indent=2, default=str)

            # 保存预测结果
            predictions_file = os.path.join(self.output_dir, f"predictions_{periods}periods.json")

            predictions_data = {
                'timestamp': datetime.now().isoformat(),
                'periods': periods,
                'model_weights': self.model_weights,
                'predictions': [
                    {
                        'index': i + 1,
                        'front_balls': front_balls,
                        'back_balls': back_balls,
                        'formatted': f"前区 {' '.join([str(b).zfill(2) for b in sorted(front_balls)])} | 后区 {' '.join([str(b).zfill(2) for b in sorted(back_balls)])}"
                    }
                    for i, (front_balls, back_balls) in enumerate(predictions)
                ]
            }

            with open(predictions_file, 'w', encoding='utf-8') as f:
                json.dump(predictions_data, f, ensure_ascii=False, indent=2)

            print(f"\n💾 分析结果已保存:")
            print(f"   📄 详细分析: {analysis_file}")
            print(f"   🎯 预测结果: {predictions_file}")

        except Exception as e:
            print(f"保存分析结果失败: {e}")

    def _make_serializable(self, obj):
        """将对象转换为可序列化的格式"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="高级混合分析预测系统")
    parser.add_argument("-d", "--data", default="data/dlt_data_all.csv", help="数据文件路径")
    parser.add_argument("-p", "--periods", type=int, default=100, help="分析期数")
    parser.add_argument("-c", "--count", type=int, default=1, help="预测注数")
    parser.add_argument("--explain", action="store_true", help="显示详细分析过程")

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"❌ 数据文件不存在: {args.data}")
        print("请先运行数据爬虫获取数据")
        return

    # 创建分析器
    analyzer = AdvancedHybridAnalyzer(args.data)

    # 执行混合分析预测
    predictions = analyzer.predict_with_hybrid_analysis(
        periods=args.periods,
        count=args.count,
        explain=args.explain
    )

    if predictions:
        print(f"\n🎉 高级混合分析预测完成！")
        print(f"📊 基于 {args.periods} 期数据的 {len(predictions)} 注预测:")

        for i, (front_balls, back_balls) in enumerate(predictions, 1):
            front_str = ' '.join([str(b).zfill(2) for b in sorted(front_balls)])
            back_str = ' '.join([str(b).zfill(2) for b in sorted(back_balls)])
            print(f"第 {i} 注: 前区 {front_str} | 后区 {back_str}")
    else:
        print("❌ 预测失败")


if __name__ == "__main__":
    main()
