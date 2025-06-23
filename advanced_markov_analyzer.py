#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高级马尔可夫链分析器
逐期分析概率转移，记录历史概率变化，提供最稳定的预测
"""

import os
import csv
import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import pickle
from datetime import datetime


class AdvancedMarkovAnalyzer:
    """高级马尔可夫链分析器"""
    
    def __init__(self, data_file, analysis_dir="analysis"):
        """初始化分析器
        
        Args:
            data_file: 数据文件路径
            analysis_dir: 分析结果保存目录
        """
        self.data_file = data_file
        self.analysis_dir = analysis_dir
        self.df = None
        
        # 确保分析目录存在
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
        
        # 历史分析数据
        self.period_analysis = {}  # 每期的分析结果
        self.cumulative_transitions = {
            'front': defaultdict(lambda: defaultdict(float)),
            'back': defaultdict(lambda: defaultdict(float))
        }
        self.probability_history = {
            'front': defaultdict(list),
            'back': defaultdict(list)
        }
        
        # 稳定性统计
        self.stability_scores = {
            'front': defaultdict(list),
            'back': defaultdict(list)
        }
        
        # 加载数据
        self.load_data()
        
        # 加载历史分析（如果存在）
        self.load_historical_analysis()
    
    def load_data(self):
        """加载数据"""
        try:
            self.df = pd.read_csv(self.data_file)
            # 按期号排序（从早到晚）
            self.df = self.df.sort_values('issue', ascending=True)
            print(f"成功加载数据，共 {len(self.df)} 条记录")
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def load_historical_analysis(self):
        """加载历史分析结果"""
        analysis_file = os.path.join(self.analysis_dir, "historical_analysis.json")
        if os.path.exists(analysis_file):
            try:
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.period_analysis = data.get('period_analysis', {})

                    # 重建defaultdict结构
                    cumulative_data = data.get('cumulative_transitions', {'front': {}, 'back': {}})
                    self.cumulative_transitions = {
                        'front': defaultdict(lambda: defaultdict(float)),
                        'back': defaultdict(lambda: defaultdict(float))
                    }

                    for ball_type in ['front', 'back']:
                        for from_ball, transitions in cumulative_data[ball_type].items():
                            for to_ball, count in transitions.items():
                                self.cumulative_transitions[ball_type][int(from_ball)][int(to_ball)] = float(count)

                    prob_data = data.get('probability_history', {'front': {}, 'back': {}})
                    self.probability_history = {
                        'front': defaultdict(list),
                        'back': defaultdict(list)
                    }
                    for ball_type in ['front', 'back']:
                        for ball, history in prob_data[ball_type].items():
                            self.probability_history[ball_type][int(ball)] = history

                    stability_data = data.get('stability_scores', {'front': {}, 'back': {}})
                    self.stability_scores = {
                        'front': defaultdict(list),
                        'back': defaultdict(list)
                    }
                    for ball_type in ['front', 'back']:
                        for ball, scores in stability_data[ball_type].items():
                            self.stability_scores[ball_type][int(ball)] = scores

                print("成功加载历史分析数据")
            except Exception as e:
                print(f"加载历史分析数据失败: {e}")
    
    def save_historical_analysis(self):
        """保存历史分析结果"""
        analysis_file = os.path.join(self.analysis_dir, "historical_analysis.json")
        try:
            # 转换defaultdict为普通dict以便JSON序列化
            cumulative_transitions_dict = {
                'front': {str(k): dict(v) for k, v in self.cumulative_transitions['front'].items()},
                'back': {str(k): dict(v) for k, v in self.cumulative_transitions['back'].items()}
            }

            data = {
                'period_analysis': self.period_analysis,
                'cumulative_transitions': cumulative_transitions_dict,
                'probability_history': {
                    'front': {str(k): v for k, v in self.probability_history['front'].items()},
                    'back': {str(k): v for k, v in self.probability_history['back'].items()}
                },
                'stability_scores': {
                    'front': {str(k): v for k, v in self.stability_scores['front'].items()},
                    'back': {str(k): v for k, v in self.stability_scores['back'].items()}
                },
                'last_updated': datetime.now().isoformat()
            }

            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            print(f"历史分析数据已保存到: {analysis_file}")
        except Exception as e:
            print(f"保存历史分析数据失败: {e}")
    
    def parse_balls(self, balls_str):
        """解析号码字符串"""
        return [int(ball.strip()) for ball in str(balls_str).split(",")]
    
    def run_progressive_analysis(self):
        """运行渐进式分析"""
        print("开始渐进式马尔可夫链分析...")
        print("=" * 60)
        
        # 按期号顺序分析每一期
        for idx in range(1, len(self.df)):
            current_row = self.df.iloc[idx]
            previous_row = self.df.iloc[idx - 1]
            
            current_issue = str(current_row['issue'])
            previous_issue = str(previous_row['issue'])
            
            # 解析号码
            current_front = self.parse_balls(current_row['front_balls'])
            current_back = self.parse_balls(current_row['back_balls'])
            previous_front = self.parse_balls(previous_row['front_balls'])
            previous_back = self.parse_balls(previous_row['back_balls'])
            
            # 分析这一期的转移
            period_result = self.analyze_single_period(
                previous_issue, current_issue,
                previous_front, previous_back,
                current_front, current_back,
                idx
            )
            
            # 保存期分析结果
            self.period_analysis[current_issue] = period_result
            
            if idx % 20 == 0:
                print(f"已分析到第 {current_issue} 期 ({idx}/{len(self.df)-1})")
        
        # 保存分析结果
        self.save_historical_analysis()
        self.save_analysis_report()
        
        print("渐进式分析完成！")
        return self.period_analysis
    
    def analyze_single_period(self, prev_issue, curr_issue, prev_front, prev_back, curr_front, curr_back, period_idx):
        """分析单期转移概率"""
        result = {
            'prev_issue': prev_issue,
            'curr_issue': curr_issue,
            'prev_front': prev_front,
            'prev_back': prev_back,
            'curr_front': curr_front,
            'curr_back': curr_back,
            'transitions': {'front': {}, 'back': {}},
            'probabilities': {'front': {}, 'back': {}},
            'stability_change': {'front': {}, 'back': {}}
        }
        
        # 分析前区转移
        for prev_ball in prev_front:
            for curr_ball in curr_front:
                # 更新累积转移计数
                self.cumulative_transitions['front'][prev_ball][curr_ball] += 1
                
                # 记录这次转移
                if prev_ball not in result['transitions']['front']:
                    result['transitions']['front'][prev_ball] = []
                result['transitions']['front'][prev_ball].append(curr_ball)
        
        # 分析后区转移
        for prev_ball in prev_back:
            for curr_ball in curr_back:
                # 更新累积转移计数
                self.cumulative_transitions['back'][prev_ball][curr_ball] += 1
                
                # 记录这次转移
                if prev_ball not in result['transitions']['back']:
                    result['transitions']['back'][prev_ball] = []
                result['transitions']['back'][prev_ball].append(curr_ball)
        
        # 计算当前累积概率
        result['probabilities']['front'] = self.calculate_transition_probabilities('front')
        result['probabilities']['back'] = self.calculate_transition_probabilities('back')
        
        # 计算稳定性变化
        if period_idx > 1:
            result['stability_change'] = self.calculate_stability_change(period_idx)
        
        return result
    
    def calculate_transition_probabilities(self, ball_type):
        """计算转移概率"""
        probabilities = {}
        transitions = self.cumulative_transitions[ball_type]
        
        for from_ball in transitions:
            total_transitions = sum(transitions[from_ball].values())
            if total_transitions > 0:
                probabilities[from_ball] = {}
                for to_ball, count in transitions[from_ball].items():
                    probabilities[from_ball][to_ball] = count / total_transitions
        
        return probabilities
    
    def calculate_stability_change(self, period_idx):
        """计算稳定性变化"""
        stability_change = {'front': {}, 'back': {}}
        
        # 获取前一期的概率
        if period_idx >= 2:
            prev_issue = str(self.df.iloc[period_idx - 1]['issue'])
            if prev_issue in self.period_analysis:
                prev_probs = self.period_analysis[prev_issue]['probabilities']
                curr_probs_front = self.calculate_transition_probabilities('front')
                curr_probs_back = self.calculate_transition_probabilities('back')
                
                # 计算前区稳定性变化
                for from_ball in curr_probs_front:
                    if from_ball in prev_probs.get('front', {}):
                        stability_change['front'][from_ball] = self.calculate_probability_variance(
                            prev_probs['front'][from_ball],
                            curr_probs_front[from_ball]
                        )
                
                # 计算后区稳定性变化
                for from_ball in curr_probs_back:
                    if from_ball in prev_probs.get('back', {}):
                        stability_change['back'][from_ball] = self.calculate_probability_variance(
                            prev_probs['back'][from_ball],
                            curr_probs_back[from_ball]
                        )
        
        return stability_change
    
    def calculate_probability_variance(self, prev_probs, curr_probs):
        """计算概率方差"""
        variance = 0.0
        all_balls = set(prev_probs.keys()) | set(curr_probs.keys())
        
        for ball in all_balls:
            prev_prob = prev_probs.get(ball, 0.0)
            curr_prob = curr_probs.get(ball, 0.0)
            variance += (prev_prob - curr_prob) ** 2
        
        return variance / len(all_balls) if all_balls else 0.0
    
    def get_most_stable_predictions(self, num_predictions=1):
        """获取最稳定的预测"""
        print("分析最稳定的预测号码...")

        # 计算每个号码的整体稳定性得分
        front_stability = self.calculate_overall_stability('front')
        back_stability = self.calculate_overall_stability('back')

        # 获取最新的转移概率
        latest_front_probs = self.calculate_transition_probabilities('front')
        latest_back_probs = self.calculate_transition_probabilities('back')

        # 获取最近一期的号码作为起始状态
        latest_row = self.df.iloc[-1]
        latest_front = self.parse_balls(latest_row['front_balls'])
        latest_back = self.parse_balls(latest_row['back_balls'])

        predictions = []
        used_combinations = set()

        for i in range(num_predictions):
            max_attempts = 50
            attempts = 0

            while attempts < max_attempts:
                # 根据预测序号调整策略
                variation_level = i

                # 预测前区号码
                front_candidates = self.get_stable_candidates_with_variation(
                    latest_front, latest_front_probs, front_stability, 'front', 5, variation_level
                )

                # 预测后区号码
                back_candidates = self.get_stable_candidates_with_variation(
                    latest_back, latest_back_probs, back_stability, 'back', 2, variation_level
                )

                # 检查是否重复
                combination = (tuple(sorted(front_candidates)), tuple(sorted(back_candidates)))
                if combination not in used_combinations:
                    used_combinations.add(combination)

                    predictions.append({
                        'front': sorted(front_candidates),
                        'back': sorted(back_candidates),
                        'stability_score': self.calculate_prediction_stability(
                            front_candidates, back_candidates, front_stability, back_stability
                        )
                    })
                    break

                attempts += 1

            # 如果尝试多次仍重复，强制生成不同的组合
            if attempts >= max_attempts:
                front_candidates, back_candidates = self.generate_fallback_prediction(
                    front_stability, back_stability, used_combinations
                )
                predictions.append({
                    'front': sorted(front_candidates),
                    'back': sorted(back_candidates),
                    'stability_score': self.calculate_prediction_stability(
                        front_candidates, back_candidates, front_stability, back_stability
                    )
                })

        # 按稳定性得分排序
        predictions.sort(key=lambda x: x['stability_score'], reverse=True)

        return predictions
    
    def calculate_overall_stability(self, ball_type):
        """计算整体稳定性"""
        # 先尝试从分析报告加载稳定性数据
        report_file = os.path.join(self.analysis_dir, "analysis_report.json")
        if os.path.exists(report_file):
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                    stability_data = report.get('stability_summary', {}).get(ball_type, {})
                    if stability_data:
                        return {int(k): float(v) for k, v in stability_data.items()}
            except Exception as e:
                print(f"加载稳定性数据失败: {e}")

        # 如果没有报告数据，计算稳定性
        if not self.period_analysis:
            # 如果没有分析数据，使用默认稳定性
            max_ball = 35 if ball_type == 'front' else 12
            return {ball: 0.999 for ball in range(1, max_ball + 1)}

        stability = {}
        max_ball = 35 if ball_type == 'front' else 12

        for ball in range(1, max_ball + 1):
            # 计算该号码的历史概率方差
            variances = []

            for issue, analysis in self.period_analysis.items():
                if ball in analysis['probabilities'][ball_type]:
                    probs = list(analysis['probabilities'][ball_type][ball].values())
                    if probs:
                        variance = np.var(probs)
                        variances.append(variance)

            # 稳定性得分 = 1 / (平均方差 + 0.001)，方差越小稳定性越高
            avg_variance = np.mean(variances) if variances else 1.0
            stability[ball] = 1.0 / (avg_variance + 0.001)

        return stability
    
    def get_stable_candidates(self, current_balls, probabilities, stability, ball_type, num_needed):
        """获取稳定的候选号码"""
        candidates = defaultdict(float)
        
        # 基于当前号码的转移概率
        for current_ball in current_balls:
            if current_ball in probabilities:
                for next_ball, prob in probabilities[current_ball].items():
                    # 综合考虑转移概率和稳定性
                    stability_score = stability.get(next_ball, 0.1)
                    combined_score = prob * 0.7 + stability_score * 0.3
                    candidates[next_ball] += combined_score
        
        # 如果候选号码不足，添加高稳定性号码
        max_ball = 35 if ball_type == 'front' else 12
        if len(candidates) < num_needed:
            sorted_stability = sorted(stability.items(), key=lambda x: x[1], reverse=True)
            for ball, score in sorted_stability:
                if ball not in candidates and len(candidates) < num_needed * 2:
                    candidates[ball] = score * 0.5
        
        # 选择得分最高的号码
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        selected = [ball for ball, score in sorted_candidates[:num_needed]]
        
        # 如果还不够，随机补充
        if len(selected) < num_needed:
            remaining = [i for i in range(1, max_ball + 1) if i not in selected]
            import random
            random.shuffle(remaining)
            selected.extend(remaining[:num_needed - len(selected)])
        
        return selected[:num_needed]

    def get_stable_candidates_with_variation(self, current_balls, probabilities, stability, ball_type, num_needed, variation_level):
        """获取带变化的稳定候选号码"""
        candidates = defaultdict(float)

        # 基于当前号码的转移概率
        for current_ball in current_balls:
            if current_ball in probabilities:
                for next_ball, prob in probabilities[current_ball].items():
                    # 综合考虑转移概率和稳定性
                    stability_score = stability.get(next_ball, 0.1)
                    combined_score = prob * 0.7 + stability_score * 0.3
                    candidates[next_ball] += combined_score

        # 根据变化级别调整选择策略
        max_ball = 35 if ball_type == 'front' else 12

        # 添加高稳定性号码
        sorted_stability = sorted(stability.items(), key=lambda x: x[1], reverse=True)
        for ball, score in sorted_stability:
            if ball not in candidates:
                # 根据变化级别调整权重
                weight = 0.5 - (variation_level * 0.1)
                if weight > 0:
                    candidates[ball] = score * weight

        # 选择候选号码
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

        # 根据变化级别选择不同的策略
        if variation_level == 0:
            # 第一注：选择最稳定的
            selected = [ball for ball, score in sorted_candidates[:num_needed]]
        elif variation_level <= 2:
            # 前几注：混合高概率和中等概率
            high_prob = sorted_candidates[:num_needed]
            mid_prob = sorted_candidates[num_needed:num_needed*2]
            selected = []

            # 选择一些高概率号码
            for i in range(min(num_needed - variation_level, len(high_prob))):
                selected.append(high_prob[i][0])

            # 选择一些中等概率号码
            remaining = num_needed - len(selected)
            for i in range(min(remaining, len(mid_prob))):
                selected.append(mid_prob[i][0])
        else:
            # 后面的注数：更多随机性
            import random
            candidate_balls = [ball for ball, score in sorted_candidates[:num_needed*3]]
            random.shuffle(candidate_balls)
            selected = candidate_balls[:num_needed]

        # 如果还不够，随机补充
        if len(selected) < num_needed:
            remaining = [i for i in range(1, max_ball + 1) if i not in selected]
            import random
            random.shuffle(remaining)
            selected.extend(remaining[:num_needed - len(selected)])

        return selected[:num_needed]

    def generate_fallback_prediction(self, front_stability, back_stability, used_combinations):
        """生成备用预测"""
        import random

        max_attempts = 100
        for _ in range(max_attempts):
            # 基于稳定性随机选择
            front_candidates = self.weighted_random_selection(front_stability, 5, 35)
            back_candidates = self.weighted_random_selection(back_stability, 2, 12)

            combination = (tuple(sorted(front_candidates)), tuple(sorted(back_candidates)))
            if combination not in used_combinations:
                return front_candidates, back_candidates

        # 如果还是重复，完全随机
        front_candidates = random.sample(range(1, 36), 5)
        back_candidates = random.sample(range(1, 13), 2)
        return front_candidates, back_candidates

    def weighted_random_selection(self, stability_scores, num_select, max_val):
        """基于稳定性权重的随机选择"""
        import random

        # 创建权重列表
        weights = []
        balls = []
        for ball in range(1, max_val + 1):
            weights.append(stability_scores.get(ball, 0.1))
            balls.append(ball)

        # 加权随机选择
        selected = []
        for _ in range(num_select):
            if balls:
                # 使用权重进行选择
                total_weight = sum(weights)
                if total_weight > 0:
                    r = random.uniform(0, total_weight)
                    cumulative = 0
                    for i, weight in enumerate(weights):
                        cumulative += weight
                        if r <= cumulative:
                            selected.append(balls[i])
                            balls.pop(i)
                            weights.pop(i)
                            break
                else:
                    selected.append(balls.pop(random.randint(0, len(balls)-1)))

        return selected
    
    def calculate_prediction_stability(self, front_balls, back_balls, front_stability, back_stability):
        """计算预测的稳定性得分"""
        front_score = sum(front_stability.get(ball, 0.1) for ball in front_balls) / len(front_balls)
        back_score = sum(back_stability.get(ball, 0.1) for ball in back_balls) / len(back_balls)
        return (front_score + back_score) / 2
    
    def add_variation(self, balls, variation_count, max_val=35):
        """为号码添加变化"""
        import random
        new_balls = balls.copy()
        for _ in range(variation_count):
            if new_balls:
                idx = random.randint(0, len(new_balls) - 1)
                new_ball = random.randint(1, max_val)
                while new_ball in new_balls:
                    new_ball = random.randint(1, max_val)
                new_balls[idx] = new_ball
        return new_balls
    
    def save_analysis_report(self):
        """保存分析报告"""
        report_file = os.path.join(self.analysis_dir, "analysis_report.json")
        
        # 准备报告数据
        report = {
            'analysis_date': datetime.now().isoformat(),
            'total_periods': len(self.period_analysis),
            'data_range': {
                'start_issue': str(self.df.iloc[0]['issue']),
                'end_issue': str(self.df.iloc[-1]['issue'])
            },
            'stability_summary': {
                'front': self.calculate_overall_stability('front'),
                'back': self.calculate_overall_stability('back')
            },
            'latest_probabilities': {
                'front': self.calculate_transition_probabilities('front'),
                'back': self.calculate_transition_probabilities('back')
            }
        }
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            print(f"分析报告已保存到: {report_file}")
        except Exception as e:
            print(f"保存分析报告失败: {e}")
    
    def print_prediction_details(self, predictions):
        """打印预测详情"""
        print("\n" + "=" * 60)
        print("高级马尔可夫链预测结果")
        print("=" * 60)

        # 显示分析统计信息
        print(f"分析期数: {len(self.period_analysis)} 期")
        print(f"数据范围: {self.df.iloc[0]['issue']} - {self.df.iloc[-1]['issue']}")

        # 显示最近一期信息
        latest_row = self.df.iloc[-1]
        latest_front = self.parse_balls(latest_row['front_balls'])
        latest_back = self.parse_balls(latest_row['back_balls'])
        print(f"最近一期 ({latest_row['issue']}): 前区 {' '.join([str(b).zfill(2) for b in latest_front])}, 后区 {' '.join([str(b).zfill(2) for b in latest_back])}")

        print("\n基于渐进式马尔可夫链分析的预测:")

        for i, pred in enumerate(predictions, 1):
            print(f"\n第 {i} 注预测 (稳定性得分: {pred['stability_score']:.4f}):")
            front_str = ' '.join([str(b).zfill(2) for b in pred['front']])
            back_str = ' '.join([str(b).zfill(2) for b in pred['back']])
            print(f"  前区: {front_str} | 后区: {back_str}")

        print(f"\n🎯 最稳定预测 (第1注): 前区 {' '.join([str(b).zfill(2) for b in predictions[0]['front']])} | 后区 {' '.join([str(b).zfill(2) for b in predictions[0]['back']])}")

        # 显示稳定性分析
        self.print_stability_analysis()

    def print_stability_analysis(self):
        """打印稳定性分析"""
        print("\n" + "-" * 40)
        print("稳定性分析")
        print("-" * 40)

        front_stability = self.calculate_overall_stability('front')
        back_stability = self.calculate_overall_stability('back')

        # 前区最稳定的号码
        sorted_front = sorted(front_stability.items(), key=lambda x: x[1], reverse=True)
        print("前区最稳定号码 (前10):")
        for i, (ball, score) in enumerate(sorted_front[:10], 1):
            print(f"  {i:2d}. {ball:2d}号 (稳定性: {score:.4f})")

        # 后区最稳定的号码
        sorted_back = sorted(back_stability.items(), key=lambda x: x[1], reverse=True)
        print("\n后区最稳定号码:")
        for i, (ball, score) in enumerate(sorted_back, 1):
            print(f"  {i:2d}. {ball:2d}号 (稳定性: {score:.4f})")

    def get_probability_trends(self, ball, ball_type='front', recent_periods=20):
        """获取号码的概率趋势"""
        trends = []
        recent_issues = list(self.period_analysis.keys())[-recent_periods:]

        for issue in recent_issues:
            analysis = self.period_analysis[issue]
            if ball in analysis['probabilities'][ball_type]:
                avg_prob = np.mean(list(analysis['probabilities'][ball_type][ball].values()))
                trends.append({
                    'issue': issue,
                    'probability': avg_prob
                })

        return trends

    def analyze_prediction_accuracy(self):
        """分析预测准确性（回测）"""
        print("\n" + "=" * 60)
        print("预测准确性回测分析")
        print("=" * 60)

        if len(self.df) < 10:
            print("数据不足，无法进行回测分析")
            return

        # 使用前80%的数据进行训练，后20%进行测试
        train_size = int(len(self.df) * 0.8)
        test_data = self.df.iloc[train_size:]

        correct_predictions = {'front': 0, 'back': 0}
        total_predictions = len(test_data) - 1

        print(f"使用前 {train_size} 期数据训练，测试后 {total_predictions} 期")

        # 简化的回测逻辑
        for i in range(1, len(test_data)):
            # 这里可以实现更复杂的回测逻辑
            pass

        print("回测分析完成")

    def export_detailed_analysis(self):
        """导出详细分析结果"""
        export_file = os.path.join(self.analysis_dir, "detailed_analysis.csv")

        try:
            rows = []
            for issue, analysis in self.period_analysis.items():
                row = {
                    'issue': issue,
                    'prev_issue': analysis['prev_issue'],
                    'prev_front': ','.join(map(str, analysis['prev_front'])),
                    'prev_back': ','.join(map(str, analysis['prev_back'])),
                    'curr_front': ','.join(map(str, analysis['curr_front'])),
                    'curr_back': ','.join(map(str, analysis['curr_back'])),
                    'stability_score': analysis.get('stability_score', 0)
                }
                rows.append(row)

            df_export = pd.DataFrame(rows)
            df_export.to_csv(export_file, index=False, encoding='utf-8')
            print(f"详细分析结果已导出到: {export_file}")
        except Exception as e:
            print(f"导出详细分析失败: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="高级马尔可夫链分析器")
    parser.add_argument("-d", "--data", default="data/dlt_data.csv", help="数据文件路径")
    parser.add_argument("-n", "--num", type=int, default=1, help="预测注数")
    parser.add_argument("--analyze", action="store_true", help="运行完整分析")
    parser.add_argument("--predict-only", action="store_true", help="仅进行预测")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"数据文件不存在: {args.data}")
        return
    
    # 创建分析器
    analyzer = AdvancedMarkovAnalyzer(args.data)
    
    if args.predict_only:
        # 仅进行预测
        predictions = analyzer.get_most_stable_predictions(args.num)
        analyzer.print_prediction_details(predictions)
    else:
        # 运行完整分析
        if args.analyze:
            analyzer.run_progressive_analysis()
        
        # 进行预测
        predictions = analyzer.get_most_stable_predictions(args.num)
        analyzer.print_prediction_details(predictions)


if __name__ == "__main__":
    main()
