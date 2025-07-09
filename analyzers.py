#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
大乐透分析器集合
整合所有分析功能：基础分析、高级分析、数据分析、混合分析
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from scipy import stats
from scipy.fft import fft
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class BaseAnalyzer:
    """分析器基类"""
    
    def __init__(self, data_file, output_dir="./output"):
        self.data_file = data_file
        self.output_dir = output_dir
        self.df = None
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 加载数据
        self.load_data()
    
    def load_data(self):
        """加载数据"""
        try:
            self.df = pd.read_csv(self.data_file)
            self.df = self.df.sort_values('issue', ascending=True)
            print(f"分析器加载数据: {len(self.df)} 条记录")
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def parse_balls(self, balls_str):
        """解析号码字符串"""
        return [int(ball.strip()) for ball in str(balls_str).split(",")]


class BasicAnalyzer(BaseAnalyzer):
    """基础分析器：频率、遗漏、热冷号分析"""
    
    def __init__(self, data_file, output_dir="./output/basic"):
        super().__init__(data_file, output_dir)
        
        # 解析前区和后区号码
        self._parse_ball_numbers()
    
    def _parse_ball_numbers(self):
        """解析前区和后区号码"""
        if self.df is None:
            return
        
        self.front_balls_lists = []
        self.back_balls_lists = []
        
        for _, row in self.df.iterrows():
            front_balls = self.parse_balls(row["front_balls"])
            back_balls = self.parse_balls(row["back_balls"])
            self.front_balls_lists.append(front_balls)
            self.back_balls_lists.append(back_balls)
    
    def analyze_frequency(self, save_result=True):
        """分析号码出现频率"""
        print("分析号码出现频率...")
        
        # 统计前区号码频率
        front_balls_flat = [ball for sublist in self.front_balls_lists for ball in sublist]
        front_counter = Counter(front_balls_flat)
        
        # 确保所有可能的前区号码都在字典中
        for i in range(1, 36):
            if i not in front_counter:
                front_counter[i] = 0
        
        # 统计后区号码频率
        back_balls_flat = [ball for sublist in self.back_balls_lists for ball in sublist]
        back_counter = Counter(back_balls_flat)
        
        # 确保所有可能的后区号码都在字典中
        for i in range(1, 13):
            if i not in back_counter:
                back_counter[i] = 0
        
        # 转换为字典并排序
        front_freq = dict(sorted(front_counter.items()))
        back_freq = dict(sorted(back_counter.items()))
        
        if save_result:
            # 保存结果
            result = {
                "front_frequency": front_freq,
                "back_frequency": back_freq,
                "analysis_date": pd.Timestamp.now().isoformat(),
                "total_periods": len(self.df)
            }
            
            with open(f"{self.output_dir}/frequency_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # 生成频率图表
            self._plot_frequency(front_freq, back_freq)
        
        return front_freq, back_freq
    
    def _plot_frequency(self, front_freq, back_freq):
        """绘制频率分布图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 前区频率图
        front_balls = list(front_freq.keys())
        front_counts = list(front_freq.values())
        
        ax1.bar(front_balls, front_counts, color='skyblue', alpha=0.7)
        ax1.set_title('前区号码频率分布', fontsize=16, fontweight='bold')
        ax1.set_xlabel('号码', fontsize=12)
        ax1.set_ylabel('出现次数', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 后区频率图
        back_balls = list(back_freq.keys())
        back_counts = list(back_freq.values())
        
        ax2.bar(back_balls, back_counts, color='lightcoral', alpha=0.7)
        ax2.set_title('后区号码频率分布', fontsize=16, fontweight='bold')
        ax2.set_xlabel('号码', fontsize=12)
        ax2.set_ylabel('出现次数', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/frequency_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_missing_values(self, save_result=True):
        """分析号码遗漏值"""
        print("分析号码遗漏值...")
        
        # 计算前区遗漏值
        front_missing = {}
        for ball in range(1, 36):
            missing_count = 0
            for front_list in self.front_balls_lists:
                if ball not in front_list:
                    missing_count += 1
                else:
                    break
            front_missing[ball] = missing_count
        
        # 计算后区遗漏值
        back_missing = {}
        for ball in range(1, 13):
            missing_count = 0
            for back_list in self.back_balls_lists:
                if ball not in back_list:
                    missing_count += 1
                else:
                    break
            back_missing[ball] = missing_count
        
        if save_result:
            result = {
                "front_missing": front_missing,
                "back_missing": back_missing,
                "analysis_date": pd.Timestamp.now().isoformat(),
                "total_periods": len(self.df)
            }
            
            with open(f"{self.output_dir}/missing_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        return front_missing, back_missing
    
    def analyze_hot_cold_numbers(self, recent_periods=30, save_result=True):
        """分析热门和冷门号码"""
        print(f"分析最近{recent_periods}期热门和冷门号码...")
        
        # 获取最近N期数据
        recent_front_lists = self.front_balls_lists[:recent_periods]
        recent_back_lists = self.back_balls_lists[:recent_periods]
        
        # 统计前区号码频率
        front_balls_flat = [ball for sublist in recent_front_lists for ball in sublist]
        front_counter = Counter(front_balls_flat)
        
        # 确保所有可能的前区号码都在字典中
        for i in range(1, 36):
            if i not in front_counter:
                front_counter[i] = 0
        
        # 统计后区号码频率
        back_balls_flat = [ball for sublist in recent_back_lists for ball in sublist]
        back_counter = Counter(back_balls_flat)
        
        # 确保所有可能的后区号码都在字典中
        for i in range(1, 13):
            if i not in back_counter:
                back_counter[i] = 0
        
        # 获取热门号码（出现频率最高的前10个）
        front_hot = [ball for ball, count in front_counter.most_common(10)]
        back_hot = [ball for ball, count in back_counter.most_common(6)]
        
        # 获取冷门号码（出现频率最低的前10个）
        front_cold = [ball for ball, count in sorted(front_counter.items(), key=lambda x: x[1])[:10]]
        back_cold = [ball for ball, count in sorted(back_counter.items(), key=lambda x: x[1])[:6]]
        
        if save_result:
            result = {
                "recent_periods": recent_periods,
                "front_hot_numbers": front_hot,
                "back_hot_numbers": back_hot,
                "front_cold_numbers": front_cold,
                "back_cold_numbers": back_cold,
                "front_frequency": dict(front_counter),
                "back_frequency": dict(back_counter),
                "analysis_date": pd.Timestamp.now().isoformat()
            }
            
            with open(f"{self.output_dir}/hot_cold_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        return {
            'front_hot': front_hot,
            'back_hot': back_hot,
            'front_cold': front_cold,
            'back_cold': back_cold
        }
    
    def analyze_distribution(self, save_result=True):
        """分析号码分布特征"""
        print("分析号码分布特征...")
        
        # 计算和值分布
        front_sums = [sum(front_list) for front_list in self.front_balls_lists]
        back_sums = [sum(back_list) for back_list in self.back_balls_lists]
        
        # 计算奇偶分布
        front_odd_counts = [sum(1 for ball in front_list if ball % 2 == 1) for front_list in self.front_balls_lists]
        back_odd_counts = [sum(1 for ball in back_list if ball % 2 == 1) for back_list in self.back_balls_lists]
        
        # 计算大小分布（前区：1-17为小，18-35为大；后区：1-6为小，7-12为大）
        front_small_counts = [sum(1 for ball in front_list if ball <= 17) for front_list in self.front_balls_lists]
        back_small_counts = [sum(1 for ball in back_list if ball <= 6) for back_list in self.back_balls_lists]
        
        distribution_stats = {
            "front_sum": {
                "mean": float(np.mean(front_sums)),
                "std": float(np.std(front_sums)),
                "min": int(np.min(front_sums)),
                "max": int(np.max(front_sums))
            },
            "back_sum": {
                "mean": float(np.mean(back_sums)),
                "std": float(np.std(back_sums)),
                "min": int(np.min(back_sums)),
                "max": int(np.max(back_sums))
            },
            "front_odd_ratio": float(np.mean(front_odd_counts) / 5),
            "back_odd_ratio": float(np.mean(back_odd_counts) / 2),
            "front_small_ratio": float(np.mean(front_small_counts) / 5),
            "back_small_ratio": float(np.mean(back_small_counts) / 2)
        }
        
        if save_result:
            result = {
                "distribution_statistics": distribution_stats,
                "analysis_date": pd.Timestamp.now().isoformat(),
                "total_periods": len(self.df)
            }
            
            with open(f"{self.output_dir}/distribution_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        return distribution_stats
    
    def run_basic_analysis(self):
        """运行所有基础分析"""
        print("开始基础分析...")
        
        results = {}
        
        # 分析号码出现频率
        front_freq, back_freq = self.analyze_frequency()
        results["frequency"] = {"front": front_freq, "back": back_freq}
        
        # 分析号码遗漏值
        front_missing, back_missing = self.analyze_missing_values()
        results["missing"] = {"front": front_missing, "back": back_missing}
        
        # 分析热门和冷门号码
        hot_cold = self.analyze_hot_cold_numbers()
        results["hot_cold"] = hot_cold
        
        # 分析号码分布
        distribution_stats = self.analyze_distribution()
        results["distribution"] = distribution_stats
        
        print("基础分析完成")
        return results


class AdvancedAnalyzer(BaseAnalyzer):
    """高级分析器：马尔可夫链、贝叶斯、概率分布、频率模式分析"""

    def __init__(self, data_file, output_dir="./output/advanced"):
        super().__init__(data_file, output_dir)

        # 解析前区和后区号码
        self._parse_ball_numbers()

    def _parse_ball_numbers(self):
        """解析前区和后区号码"""
        if self.df is None:
            return

        self.front_balls_lists = []
        self.back_balls_lists = []

        for _, row in self.df.iterrows():
            front_balls = self.parse_balls(row["front_balls"])
            back_balls = self.parse_balls(row["back_balls"])
            self.front_balls_lists.append(front_balls)
            self.back_balls_lists.append(back_balls)

    def analyze_statistical_features(self, save_result=True):
        """分析统计学特征"""
        print("分析统计学特征...")

        # 计算和值统计
        front_sums = [sum(front_list) for front_list in self.front_balls_lists]
        back_sums = [sum(back_list) for back_list in self.back_balls_lists]

        # 计算极差统计
        front_ranges = [max(front_list) - min(front_list) for front_list in self.front_balls_lists]
        back_ranges = [max(back_list) - min(back_list) for back_list in self.back_balls_lists]

        # 计算AC值（算术复杂性）
        front_ac_values = []
        for front_list in self.front_balls_lists:
            sorted_balls = sorted(front_list)
            ac_value = 0
            for i in range(len(sorted_balls) - 1):
                ac_value += abs(sorted_balls[i+1] - sorted_balls[i])
            front_ac_values.append(ac_value)

        # 统计特征
        features = {
            "front_sum_stats": {
                "mean": float(np.mean(front_sums)),
                "std": float(np.std(front_sums)),
                "min": int(np.min(front_sums)),
                "max": int(np.max(front_sums)),
                "median": float(np.median(front_sums))
            },
            "back_sum_stats": {
                "mean": float(np.mean(back_sums)),
                "std": float(np.std(back_sums)),
                "min": int(np.min(back_sums)),
                "max": int(np.max(back_sums)),
                "median": float(np.median(back_sums))
            },
            "front_range_stats": {
                "mean": float(np.mean(front_ranges)),
                "std": float(np.std(front_ranges)),
                "min": int(np.min(front_ranges)),
                "max": int(np.max(front_ranges))
            },
            "back_range_stats": {
                "mean": float(np.mean(back_ranges)),
                "std": float(np.std(back_ranges)),
                "min": int(np.min(back_ranges)),
                "max": int(np.max(back_ranges))
            },
            "front_ac_stats": {
                "mean": float(np.mean(front_ac_values)),
                "std": float(np.std(front_ac_values)),
                "min": int(np.min(front_ac_values)),
                "max": int(np.max(front_ac_values))
            }
        }

        if save_result:
            with open(f"{self.output_dir}/statistical_features.json", 'w', encoding='utf-8') as f:
                json.dump(features, f, ensure_ascii=False, indent=2)

        return features

    def analyze_markov_chain(self, save_result=True):
        """分析马尔可夫链"""
        print("分析马尔可夫链...")

        # 构建转移矩阵
        front_transitions = defaultdict(lambda: defaultdict(int))
        back_transitions = defaultdict(lambda: defaultdict(int))

        # 分析前区转移
        for i in range(len(self.front_balls_lists) - 1):
            current_front = set(self.front_balls_lists[i])
            next_front = set(self.front_balls_lists[i + 1])

            for current_ball in current_front:
                for next_ball in next_front:
                    front_transitions[current_ball][next_ball] += 1

        # 分析后区转移
        for i in range(len(self.back_balls_lists) - 1):
            current_back = set(self.back_balls_lists[i])
            next_back = set(self.back_balls_lists[i + 1])

            for current_ball in current_back:
                for next_ball in next_back:
                    back_transitions[current_ball][next_ball] += 1

        # 计算转移概率
        front_transition_probs = {}
        for from_ball, to_dict in front_transitions.items():
            total = sum(to_dict.values())
            if total > 0:
                front_transition_probs[from_ball] = {
                    to_ball: count / total for to_ball, count in to_dict.items()
                }

        back_transition_probs = {}
        for from_ball, to_dict in back_transitions.items():
            total = sum(to_dict.values())
            if total > 0:
                back_transition_probs[from_ball] = {
                    to_ball: count / total for to_ball, count in to_dict.items()
                }

        markov_analysis = {
            "front_transition_matrix": dict(front_transitions),
            "back_transition_matrix": dict(back_transitions),
            "front_transition_probs": front_transition_probs,
            "back_transition_probs": back_transition_probs
        }

        if save_result:
            with open(f"{self.output_dir}/markov_chain.json", 'w', encoding='utf-8') as f:
                json.dump(markov_analysis, f, ensure_ascii=False, indent=2)

        return markov_analysis

    def analyze_probability_distribution(self, save_result=True):
        """分析概率分布"""
        print("分析概率分布...")

        # 计算各号码的理论概率和实际概率
        front_theoretical_prob = 5 / 35  # 前区理论概率
        back_theoretical_prob = 2 / 12   # 后区理论概率

        # 统计实际出现次数
        front_counts = Counter()
        back_counts = Counter()

        for front_list in self.front_balls_lists:
            for ball in front_list:
                front_counts[ball] += 1

        for back_list in self.back_balls_lists:
            for ball in back_list:
                back_counts[ball] += 1

        total_periods = len(self.front_balls_lists)

        # 计算实际概率
        front_actual_probs = {}
        for ball in range(1, 36):
            front_actual_probs[ball] = front_counts.get(ball, 0) / total_periods

        back_actual_probs = {}
        for ball in range(1, 13):
            back_actual_probs[ball] = back_counts.get(ball, 0) / total_periods

        # 卡方检验
        front_expected = [front_theoretical_prob * total_periods] * 35
        front_observed = [front_counts.get(i, 0) for i in range(1, 36)]
        front_chi2, front_p_value = stats.chisquare(front_observed, front_expected)

        back_expected = [back_theoretical_prob * total_periods] * 12
        back_observed = [back_counts.get(i, 0) for i in range(1, 13)]
        back_chi2, back_p_value = stats.chisquare(back_observed, back_expected)

        prob_analysis = {
            "front_theoretical_prob": float(front_theoretical_prob),
            "back_theoretical_prob": float(back_theoretical_prob),
            "front_actual_probs": {k: float(v) for k, v in front_actual_probs.items()},
            "back_actual_probs": {k: float(v) for k, v in back_actual_probs.items()},
            "front_chi2_test": {
                "chi2": float(front_chi2),
                "p_value": float(front_p_value),
                "is_random": bool(front_p_value > 0.05)
            },
            "back_chi2_test": {
                "chi2": float(back_chi2),
                "p_value": float(back_p_value),
                "is_random": bool(back_p_value > 0.05)
            }
        }

        if save_result:
            with open(f"{self.output_dir}/probability_distribution.json", 'w', encoding='utf-8') as f:
                json.dump(prob_analysis, f, ensure_ascii=False, indent=2)

        return prob_analysis

    def analyze_bayesian(self, save_result=True):
        """贝叶斯分析"""
        print("进行贝叶斯分析...")

        # 计算先验概率（基于历史频率）
        front_prior = {}
        back_prior = {}

        for ball in range(1, 36):
            count = sum(1 for front_list in self.front_balls_lists if ball in front_list)
            front_prior[ball] = count / len(self.front_balls_lists)

        for ball in range(1, 13):
            count = sum(1 for back_list in self.back_balls_lists if ball in back_list)
            back_prior[ball] = count / len(self.back_balls_lists)

        # 计算最近期数的似然概率
        recent_periods = min(50, len(self.front_balls_lists))
        recent_front_lists = self.front_balls_lists[:recent_periods]
        recent_back_lists = self.back_balls_lists[:recent_periods]

        front_likelihood = {}
        back_likelihood = {}

        for ball in range(1, 36):
            count = sum(1 for front_list in recent_front_lists if ball in front_list)
            front_likelihood[ball] = count / recent_periods

        for ball in range(1, 13):
            count = sum(1 for back_list in recent_back_lists if ball in back_list)
            back_likelihood[ball] = count / recent_periods

        # 计算后验概率（简化的贝叶斯更新）
        front_posterior = {}
        back_posterior = {}

        for ball in range(1, 36):
            # 简化的贝叶斯更新：posterior ∝ likelihood × prior
            front_posterior[ball] = front_likelihood[ball] * front_prior[ball]

        for ball in range(1, 13):
            back_posterior[ball] = back_likelihood[ball] * back_prior[ball]

        # 归一化后验概率
        front_total = sum(front_posterior.values())
        if front_total > 0:
            front_posterior = {ball: prob / front_total for ball, prob in front_posterior.items()}

        back_total = sum(back_posterior.values())
        if back_total > 0:
            back_posterior = {ball: prob / back_total for ball, prob in back_posterior.items()}

        bayesian_analysis = {
            "front_prior": front_prior,
            "back_prior": back_prior,
            "front_likelihood": front_likelihood,
            "back_likelihood": back_likelihood,
            "front_posterior": front_posterior,
            "back_posterior": back_posterior,
            "recent_periods": recent_periods
        }

        if save_result:
            with open(f"{self.output_dir}/bayesian_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(bayesian_analysis, f, ensure_ascii=False, indent=2)

        return bayesian_analysis

    def run_advanced_analysis(self):
        """运行所有高级分析"""
        print("开始高级分析...")

        results = {}

        # 分析统计学特征
        stats_results = self.analyze_statistical_features()
        results["statistical_features"] = stats_results

        # 分析概率分布
        prob_results = self.analyze_probability_distribution()
        results["probability_distribution"] = prob_results

        # 分析马尔可夫链
        markov_results = self.analyze_markov_chain()
        results["markov_chain"] = markov_results

        # 贝叶斯分析
        bayesian_results = self.analyze_bayesian()
        results["bayesian"] = bayesian_results

        print("高级分析完成")
        return results


class ComprehensiveAnalyzer(BaseAnalyzer):
    """综合分析器：整合所有分析功能"""

    def __init__(self, data_file, output_dir="./output"):
        super().__init__(data_file, output_dir)

        # 初始化子分析器
        self.basic_analyzer = BasicAnalyzer(data_file, f"{output_dir}/basic")
        self.advanced_analyzer = AdvancedAnalyzer(data_file, f"{output_dir}/advanced")

    def run_all_analysis(self, periods=0, save_results=True):
        """运行所有分析"""
        print("🔍 开始综合分析...")

        # 如果指定了期数，截取数据
        if periods > 0 and self.df is not None:
            original_df = self.df.copy()
            self.df = self.df.tail(periods)
            self.basic_analyzer.df = self.df
            self.advanced_analyzer.df = self.df

            # 重新解析号码
            self.basic_analyzer._parse_ball_numbers()
            self.advanced_analyzer._parse_ball_numbers()

        all_results = {}

        # 基础分析
        print("📊 执行基础分析...")
        basic_results = self.basic_analyzer.run_basic_analysis()
        all_results["basic"] = basic_results

        # 高级分析
        print("🧠 执行高级分析...")
        advanced_results = self.advanced_analyzer.run_advanced_analysis()
        all_results["advanced"] = advanced_results

        # 综合分析报告
        print("📋 生成综合分析报告...")
        summary = self._generate_summary(basic_results, advanced_results)
        all_results["summary"] = summary

        if save_results:
            # 保存综合结果
            with open(f"{self.output_dir}/comprehensive_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

            # 生成文本报告
            self._generate_text_report(all_results)

        # 恢复原始数据
        if periods > 0 and 'original_df' in locals():
            self.df = original_df
            self.basic_analyzer.df = original_df
            self.advanced_analyzer.df = original_df

        print("✅ 综合分析完成")
        return all_results

    def _generate_summary(self, basic_results, advanced_results):
        """生成分析摘要"""
        summary = {
            "analysis_date": datetime.now().isoformat(),
            "data_overview": {
                "total_periods": len(self.df) if self.df is not None else 0,
                "date_range": f"{self.df['date'].min()} - {self.df['date'].max()}" if self.df is not None else "N/A",
                "issue_range": f"{self.df['issue'].min()} - {self.df['issue'].max()}" if self.df is not None else "N/A"
            }
        }

        # 基础分析摘要
        if "frequency" in basic_results:
            front_freq = basic_results["frequency"]["front"]
            back_freq = basic_results["frequency"]["back"]

            # 最热和最冷号码
            front_hot = max(front_freq.items(), key=lambda x: x[1])
            front_cold = min(front_freq.items(), key=lambda x: x[1])
            back_hot = max(back_freq.items(), key=lambda x: x[1])
            back_cold = min(back_freq.items(), key=lambda x: x[1])

            summary["frequency_summary"] = {
                "front_hottest": {"number": front_hot[0], "count": front_hot[1]},
                "front_coldest": {"number": front_cold[0], "count": front_cold[1]},
                "back_hottest": {"number": back_hot[0], "count": back_hot[1]},
                "back_coldest": {"number": back_cold[0], "count": back_cold[1]}
            }

        # 高级分析摘要
        if "probability_distribution" in advanced_results:
            prob_dist = advanced_results["probability_distribution"]
            summary["randomness_test"] = {
                "front_is_random": prob_dist["front_chi2_test"]["is_random"],
                "back_is_random": prob_dist["back_chi2_test"]["is_random"],
                "front_p_value": prob_dist["front_chi2_test"]["p_value"],
                "back_p_value": prob_dist["back_chi2_test"]["p_value"]
            }

        return summary

    def _generate_text_report(self, all_results):
        """生成文本报告"""
        report_path = f"{self.output_dir}/analysis_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("大乐透数据综合分析报告\n")
            f.write("=" * 50 + "\n\n")

            # 数据概览
            summary = all_results.get("summary", {})
            data_overview = summary.get("data_overview", {})

            f.write("数据概览:\n")
            f.write(f"  总期数: {data_overview.get('total_periods', 'N/A')}\n")
            f.write(f"  日期范围: {data_overview.get('date_range', 'N/A')}\n")
            f.write(f"  期号范围: {data_overview.get('issue_range', 'N/A')}\n")
            f.write(f"  分析时间: {summary.get('analysis_date', 'N/A')}\n\n")

            # 频率分析摘要
            freq_summary = summary.get("frequency_summary", {})
            if freq_summary:
                f.write("频率分析摘要:\n")
                f.write(f"  前区最热号码: {freq_summary.get('front_hottest', {}).get('number', 'N/A')}号 ")
                f.write(f"({freq_summary.get('front_hottest', {}).get('count', 'N/A')}次)\n")
                f.write(f"  前区最冷号码: {freq_summary.get('front_coldest', {}).get('number', 'N/A')}号 ")
                f.write(f"({freq_summary.get('front_coldest', {}).get('count', 'N/A')}次)\n")
                f.write(f"  后区最热号码: {freq_summary.get('back_hottest', {}).get('number', 'N/A')}号 ")
                f.write(f"({freq_summary.get('back_hottest', {}).get('count', 'N/A')}次)\n")
                f.write(f"  后区最冷号码: {freq_summary.get('back_coldest', {}).get('number', 'N/A')}号 ")
                f.write(f"({freq_summary.get('back_coldest', {}).get('count', 'N/A')}次)\n\n")

            # 随机性检验
            randomness = summary.get("randomness_test", {})
            if randomness:
                f.write("随机性检验:\n")
                f.write(f"  前区随机性: {'通过' if randomness.get('front_is_random', False) else '不通过'}\n")
                f.write(f"  后区随机性: {'通过' if randomness.get('back_is_random', False) else '不通过'}\n")
                f.write(f"  前区P值: {randomness.get('front_p_value', 'N/A'):.4f}\n")
                f.write(f"  后区P值: {randomness.get('back_p_value', 'N/A'):.4f}\n\n")

            f.write("详细分析结果请查看对应的JSON文件。\n")

        print(f"文本报告已保存到: {report_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="大乐透分析器集合")
    parser.add_argument("-d", "--data", default="data/dlt_data_all.csv", help="数据文件路径")
    parser.add_argument("-o", "--output", default="output", help="输出目录")
    parser.add_argument("-m", "--method",
                       choices=['basic', 'advanced', 'comprehensive'],
                       default='comprehensive', help="分析方法")
    parser.add_argument("-p", "--periods", type=int, default=0, help="分析期数，0表示全部")

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"数据文件不存在: {args.data}")
        return

    # 根据选择的方法进行分析
    if args.method == 'basic':
        analyzer = BasicAnalyzer(args.data, f"{args.output}/basic")
        results = analyzer.run_basic_analysis()
        print(f"基础分析完成，结果保存在: {args.output}/basic")

    elif args.method == 'advanced':
        analyzer = AdvancedAnalyzer(args.data, f"{args.output}/advanced")
        results = analyzer.run_advanced_analysis()
        print(f"高级分析完成，结果保存在: {args.output}/advanced")

    elif args.method == 'comprehensive':
        analyzer = ComprehensiveAnalyzer(args.data, args.output)
        results = analyzer.run_all_analysis(periods=args.periods)
        print(f"综合分析完成，结果保存在: {args.output}")


if __name__ == "__main__":
    main()
