#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自适应学习模块集成
整合多臂老虎机、强化学习、自适应权重调整等智能学习功能
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, Counter, deque
import copy
import math

from core_modules import cache_manager, logger_manager, data_manager, task_manager
from predictor_modules import traditional_predictor, advanced_predictor, super_predictor


# ==================== 多臂老虎机算法 ====================
class MultiArmedBandit:
    """多臂老虎机算法"""
    
    def __init__(self, n_arms: int, algorithm: str = "ucb1"):
        """初始化多臂老虎机
        
        Args:
            n_arms: 臂的数量（算法数量）
            algorithm: 算法类型 (epsilon_greedy, ucb1, thompson_sampling)
        """
        self.n_arms = n_arms
        self.algorithm = algorithm
        
        # 统计信息
        self.counts = np.zeros(n_arms)  # 每个臂被选择的次数
        self.values = np.zeros(n_arms)  # 每个臂的平均奖励
        self.total_rewards = np.zeros(n_arms)  # 每个臂的总奖励
        
        # 算法参数
        self.epsilon = 0.1  # epsilon-greedy参数
        self.c = 2.0  # UCB1参数
        
        # Thompson Sampling参数
        self.alpha = np.ones(n_arms)  # Beta分布的alpha参数
        self.beta = np.ones(n_arms)   # Beta分布的beta参数
    
    def select_arm(self) -> int:
        """选择一个臂"""
        if self.algorithm == "epsilon_greedy":
            return self._epsilon_greedy()
        elif self.algorithm == "ucb1":
            return self._ucb1()
        elif self.algorithm == "thompson_sampling":
            return self._thompson_sampling()
        else:
            return np.random.randint(self.n_arms)
    
    def update(self, arm: int, reward: float):
        """更新臂的统计信息"""
        self.counts[arm] += 1
        self.total_rewards[arm] += reward
        self.values[arm] = self.total_rewards[arm] / self.counts[arm]
        
        # 更新Thompson Sampling参数
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
    
    def _epsilon_greedy(self) -> int:
        """Epsilon-Greedy算法"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.values)
    
    def _ucb1(self) -> int:
        """UCB1算法"""
        if 0 in self.counts:
            return np.where(self.counts == 0)[0][0]
        
        total_counts = np.sum(self.counts)
        ucb_values = self.values + self.c * np.sqrt(np.log(total_counts) / self.counts)
        return np.argmax(ucb_values)
    
    def _thompson_sampling(self) -> int:
        """Thompson Sampling算法"""
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)


# ==================== 准确率跟踪器 ====================
class AccuracyTracker:
    """准确率跟踪器"""
    
    def __init__(self):
        self.predictions = []
        self.results = []
        self.accuracy_stats = defaultdict(lambda: defaultdict(int))
    
    def record_prediction(self, predictor_name: str, prediction: Dict, issue: str):
        """记录预测"""
        record = {
            'predictor': predictor_name,
            'issue': issue,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        }
        self.predictions.append(record)
    
    def verify_prediction(self, issue: str, actual_front: List[int], actual_back: List[int]):
        """验证预测结果"""
        for pred_record in self.predictions:
            if pred_record['issue'] == issue:
                prediction = pred_record['prediction']
                predictor = pred_record['predictor']
                
                # 计算中奖情况
                prize_level, front_hits, back_hits = self._calculate_prize_level(
                    prediction['front_balls'], prediction['back_balls'],
                    actual_front, actual_back
                )
                
                result = {
                    'predictor': predictor,
                    'issue': issue,
                    'predicted_front': prediction['front_balls'],
                    'predicted_back': prediction['back_balls'],
                    'actual_front': actual_front,
                    'actual_back': actual_back,
                    'front_hits': front_hits,
                    'back_hits': back_hits,
                    'prize_level': prize_level,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.results.append(result)
                self.accuracy_stats[predictor][prize_level] += 1
    
    def _calculate_prize_level(self, predicted_front: List[int], predicted_back: List[int],
                              actual_front: List[int], actual_back: List[int]) -> Tuple[str, int, int]:
        """计算中奖等级"""
        front_hits = len(set(predicted_front) & set(actual_front))
        back_hits = len(set(predicted_back) & set(actual_back))
        
        # 判断中奖等级
        if front_hits == 5 and back_hits == 2:
            return "一等奖", front_hits, back_hits
        elif front_hits == 5 and back_hits == 1:
            return "二等奖", front_hits, back_hits
        elif front_hits == 5 and back_hits == 0:
            return "三等奖", front_hits, back_hits
        elif front_hits == 4 and back_hits == 2:
            return "四等奖", front_hits, back_hits
        elif front_hits == 4 and back_hits == 1:
            return "五等奖", front_hits, back_hits
        elif front_hits == 3 and back_hits == 2:
            return "六等奖", front_hits, back_hits
        elif front_hits == 4 and back_hits == 0:
            return "七等奖", front_hits, back_hits
        elif front_hits == 3 and back_hits == 1:
            return "八等奖", front_hits, back_hits
        elif front_hits == 2 and back_hits == 2:
            return "八等奖", front_hits, back_hits
        elif front_hits == 3 and back_hits == 0:
            return "九等奖", front_hits, back_hits
        elif front_hits == 1 and back_hits == 2:
            return "九等奖", front_hits, back_hits
        elif front_hits == 2 and back_hits == 1:
            return "九等奖", front_hits, back_hits
        elif front_hits == 0 and back_hits == 2:
            return "九等奖", front_hits, back_hits
        else:
            return "未中奖", front_hits, back_hits
    
    def get_accuracy_report(self, predictor_name: str = None) -> Dict:
        """获取准确率报告"""
        if predictor_name:
            stats = {predictor_name: self.accuracy_stats[predictor_name]}
        else:
            stats = dict(self.accuracy_stats)
        
        report = {}
        
        for predictor, prize_stats in stats.items():
            total_predictions = sum(prize_stats.values())
            win_count = sum(count for prize, count in prize_stats.items() if prize != "未中奖")
            
            report[predictor] = {
                'total_predictions': total_predictions,
                'win_count': win_count,
                'win_rate': win_count / total_predictions if total_predictions > 0 else 0,
                'prize_distribution': dict(prize_stats)
            }
        
        return report


# ==================== 增强版自适应学习预测器 ====================
class EnhancedAdaptiveLearningPredictor:
    """增强版自适应学习预测器"""
    
    def __init__(self, data_file="data/dlt_data_all.csv"):
        """初始化增强版自适应学习预测器"""
        self.data_file = data_file
        self.df = data_manager.get_data()
        
        # 预测器配置（延迟初始化）
        self.predictors = {}
        self._predictors_initialized = False
        
        self.predictor_names = [
            'traditional_frequency',
            'traditional_hot_cold',
            'traditional_missing',
            'advanced_markov',
            'advanced_bayesian',
            'advanced_ensemble',
            'super_predictor'
        ]
        
        # 多臂老虎机
        self.bandit = MultiArmedBandit(len(self.predictor_names), algorithm="ucb1")
        
        # 准确率跟踪
        self.accuracy_tracker = AccuracyTracker()
        
        # 学习历史
        self.learning_history = []
        self.performance_window = deque(maxlen=50)
        
        # 动态权重
        self.dynamic_weights = {}
        self.weight_history = []
        
        # 强化学习参数
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.2
        self.exploration_decay = 0.995
        
        # 性能统计
        self.predictor_performance = {}
        self.recent_performance = {}
        
        # 初始化性能统计
        for name in self.predictor_names:
            self.predictor_performance[name] = {
                'total_predictions': 0,
                'total_score': 0.0,
                'win_count': 0,
                'recent_scores': deque(maxlen=20),
                'confidence': 0.5
            }
            self.recent_performance[name] = deque(maxlen=10)
        
        if self.df is None:
            logger_manager.error("数据未加载")

    def _initialize_predictors(self):
        """初始化预测器"""
        if self._predictors_initialized:
            return

        from predictor_modules import get_traditional_predictor, get_advanced_predictor, get_super_predictor

        traditional_pred = get_traditional_predictor()
        advanced_pred = get_advanced_predictor()
        super_pred = get_super_predictor()

        self.predictors = {
            'traditional_frequency': traditional_pred,
            'traditional_hot_cold': traditional_pred,
            'traditional_missing': traditional_pred,
            'advanced_markov': advanced_pred,
            'advanced_bayesian': advanced_pred,
            'advanced_ensemble': advanced_pred,
            'super_predictor': super_pred
        }

        self._predictors_initialized = True
    
    def _predict_with_predictor(self, predictor_name: str, periods: int = 500) -> Tuple[List[int], List[int], float]:
        """使用指定预测器进行预测

        Args:
            predictor_name: 预测器名称
            periods: 分析期数
        """
        # 确保预测器已初始化
        if not self._predictors_initialized:
            self._initialize_predictors()

        try:
            if predictor_name == 'traditional_frequency':
                result = self.predictors['traditional_frequency'].frequency_predict(1, periods)
                front_balls = [int(x) for x in result[0][0]]
                back_balls = [int(x) for x in result[0][1]]
                return front_balls, back_balls, 0.5

            elif predictor_name == 'traditional_hot_cold':
                result = self.predictors['traditional_hot_cold'].hot_cold_predict(1, periods)
                front_balls = [int(x) for x in result[0][0]]
                back_balls = [int(x) for x in result[0][1]]
                return front_balls, back_balls, 0.5

            elif predictor_name == 'traditional_missing':
                result = self.predictors['traditional_missing'].missing_predict(1, periods)
                front_balls = [int(x) for x in result[0][0]]
                back_balls = [int(x) for x in result[0][1]]
                return front_balls, back_balls, 0.5

            elif predictor_name == 'advanced_markov':
                result = self.predictors['advanced_markov'].markov_predict(1, periods)
                front_balls = [int(x) for x in result[0][0]]
                back_balls = [int(x) for x in result[0][1]]
                return front_balls, back_balls, 0.6

            elif predictor_name == 'advanced_bayesian':
                result = self.predictors['advanced_bayesian'].bayesian_predict(1, periods)
                front_balls = [int(x) for x in result[0][0]]
                back_balls = [int(x) for x in result[0][1]]
                return front_balls, back_balls, 0.6

            elif predictor_name == 'advanced_ensemble':
                result = self.predictors['advanced_ensemble'].ensemble_predict(1)
                front_balls = [int(x) for x in result[0][0]]
                back_balls = [int(x) for x in result[0][1]]
                return front_balls, back_balls, 0.7

            elif predictor_name == 'super_predictor':
                result = self.predictors['super_predictor'].predict_super(1)
                if result:
                    front_balls = [int(x) for x in result[0]['front_balls']]
                    back_balls = [int(x) for x in result[0]['back_balls']]
                    return front_balls, back_balls, result[0].get('confidence', 0.8)

        except Exception as e:
            logger_manager.error(f"{predictor_name} 预测失败", e)

        # 默认随机预测
        front_balls = sorted([int(x) for x in np.random.choice(range(1, 36), 5, replace=False)])
        back_balls = sorted([int(x) for x in np.random.choice(range(1, 13), 2, replace=False)])
        return front_balls, back_balls, 0.1
    
    def _calculate_prize_score(self, prize_level: str) -> float:
        """计算中奖得分"""
        prize_scores = {
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
        }
        return prize_scores.get(prize_level, 0.0)
    
    def _update_predictor_performance(self, predictor_name: str, score: float, prize_level: str):
        """更新预测器性能"""
        perf = self.predictor_performance[predictor_name]
        
        perf['total_predictions'] += 1
        perf['total_score'] += score
        perf['recent_scores'].append(score)
        
        if prize_level != "未中奖":
            perf['win_count'] += 1
        
        # 更新置信度
        if len(perf['recent_scores']) > 0:
            recent_avg = np.mean(perf['recent_scores'])
            perf['confidence'] = min(0.9, max(0.1, recent_avg / 10.0))
        
        # 更新最近性能
        self.recent_performance[predictor_name].append(score)
    
    def enhanced_adaptive_learning(self, start_period: int = 100, test_periods: int = 1000) -> Dict:
        """增强版自适应学习"""
        logger_manager.info(f"启动增强版自适应学习，起始期数: {start_period}, 测试期数: {test_periods}")

        if self.df is None or len(self.df) == 0:
            logger_manager.error("数据未加载或为空")
            return {}

        # 计算起始索引位置（从最新数据开始倒数start_period期）
        total_periods = len(self.df)
        if start_period >= total_periods:
            logger_manager.error(f"起始期数 {start_period} 超过可用数据期数 {total_periods}")
            return {}

        # 从最新数据开始倒数start_period期作为起始位置
        start_idx = total_periods - start_period

        # 检查是否有足够的数据进行测试
        available_periods = len(self.df) - start_idx
        if available_periods < test_periods:
            logger_manager.warning(f"可用期数 {available_periods} 少于请求的测试期数 {test_periods}，将使用可用期数")
            test_periods = available_periods
        
        detailed_results = []
        total_score = 0.0
        win_count = 0
        
        # 创建进度条
        progress_bar = task_manager.create_progress_bar(test_periods, "增强学习进度")
        
        try:
            # 预热阶段：让每个预测器都被选择几次
            warmup_rounds = min(20, len(self.predictor_names) * 3)
            
            for i in range(test_periods):
                if task_manager.is_interrupted():
                    break

                # 使用DataFrame索引而不是期号
                current_idx = start_idx + i

                if current_idx >= len(self.df):
                    break

                # 获取当前期的真实开奖号码
                current_row = self.df.iloc[current_idx]
                actual_front, actual_back = data_manager.parse_balls(current_row)
                
                # 选择预测器
                if i < warmup_rounds:
                    # 预热阶段：轮流选择
                    selected_arm = i % len(self.predictor_names)
                else:
                    # 正式阶段：使用多臂老虎机选择
                    selected_arm = self.bandit.select_arm()
                
                selected_predictor = self.predictor_names[selected_arm]
                
                # 进行预测
                predicted_front, predicted_back, confidence = self._predict_with_predictor(selected_predictor)
                
                # 计算中奖情况
                prize_level, front_hits, back_hits = self.accuracy_tracker._calculate_prize_level(
                    predicted_front, predicted_back, actual_front, actual_back
                )
                
                # 计算得分
                score = self._calculate_prize_score(prize_level)
                total_score += score
                
                if prize_level != "未中奖":
                    win_count += 1
                
                # 更新多臂老虎机
                normalized_reward = min(1.0, score / 10.0)  # 标准化奖励
                self.bandit.update(selected_arm, normalized_reward)
                
                # 更新预测器性能
                self._update_predictor_performance(selected_predictor, score, prize_level)
                
                # 记录详细结果
                result = {
                    'period': i + 1,
                    'issue': current_row['issue'],
                    'date': current_row['date'],
                    'selected_predictor': selected_predictor,
                    'predicted_front': predicted_front,
                    'predicted_back': predicted_back,
                    'actual_front': actual_front,
                    'actual_back': actual_back,
                    'front_hits': front_hits,
                    'back_hits': back_hits,
                    'prize_level': prize_level,
                    'score': score,
                    'confidence': confidence,
                    'cumulative_score': total_score,
                    'win_rate': win_count / (i + 1),
                    'bandit_values': self.bandit.values.copy(),
                    'bandit_counts': self.bandit.counts.copy()
                }
                
                detailed_results.append(result)
                
                # 更新探索率
                self.exploration_rate *= self.exploration_decay
                self.exploration_rate = max(0.05, self.exploration_rate)
                
                # 更新进度条
                progress_bar.update(1, f"期数: {current_row['issue']}, 预测器: {selected_predictor}, 中奖: {prize_level}")
            
            progress_bar.finish("学习完成")
            
        except KeyboardInterrupt:
            progress_bar.interrupt("用户中断")
            logger_manager.warning("学习被用户中断")
        except Exception as e:
            progress_bar.interrupt(f"错误: {e}")
            logger_manager.error("学习过程出错", e)
        
        # 汇总结果
        final_results = {
            'total_periods': len(detailed_results),
            'total_wins': win_count,
            'win_rate': win_count / len(detailed_results) if detailed_results else 0,
            'total_score': total_score,
            'average_score': total_score / len(detailed_results) if detailed_results else 0,
            'predictor_performance': copy.deepcopy(self.predictor_performance),
            'bandit_final_values': self.bandit.values.copy(),
            'bandit_final_counts': self.bandit.counts.copy(),
            'detailed_results': detailed_results,
            'timestamp': datetime.now().isoformat()
        }
        
        self.learning_history.append(final_results)
        logger_manager.info(f"增强学习完成，中奖率: {final_results['win_rate']:.3f}")
        
        return final_results
    
    def generate_enhanced_prediction(self, count: int = 1, periods: int = 500) -> List[Dict]:
        """生成增强预测

        Args:
            count: 生成注数
            periods: 分析期数
        """
        logger_manager.info(f"生成增强预测，注数: {count}, 分析期数: {periods}")
        
        if len(self.predictor_names) == 0:
            logger_manager.error("预测器未初始化")
            return []
        
        predictions = []
        
        # 计算当前最优预测器
        best_arm = np.argmax(self.bandit.values)
        best_predictor = self.predictor_names[best_arm]
        
        logger_manager.info(f"最优预测器: {best_predictor} (奖励: {self.bandit.values[best_arm]:.3f})")
        
        for i in range(count):
            # 使用多臂老虎机选择预测器
            selected_arm = self.bandit.select_arm()
            selected_predictor = self.predictor_names[selected_arm]
            
            try:
                # 使用选中的预测器进行预测
                front_balls, back_balls, confidence = self._predict_with_predictor(selected_predictor, periods)
                
                prediction = {
                    'index': i + 1,
                    'predictor_used': selected_predictor,
                    'front_balls': front_balls,
                    'back_balls': back_balls,
                    'confidence': confidence,
                    'expected_reward': self.bandit.values[selected_arm]
                }
                
                predictions.append(prediction)
                
            except Exception as e:
                logger_manager.error(f"第 {i+1} 注预测失败", e)
        
        return predictions

    def smart_predict_compound(self, front_count: int = 8, back_count: int = 4, periods: int = 500) -> Dict:
        """智能复式预测（基于学习结果的最优预测器）

        Args:
            front_count: 前区号码数量
            back_count: 后区号码数量
            periods: 分析期数
        """
        logger_manager.info(f"智能复式预测: {front_count}+{back_count}, 分析期数: {periods}")

        if len(self.predictor_names) == 0:
            logger_manager.error("预测器未初始化")
            return {}

        try:
            # 获取多个最优预测器的预测结果
            top_predictors = np.argsort(self.bandit.values)[-3:]  # 选择前3个最优预测器

            front_candidates = set()
            back_candidates = set()

            # 使用多个最优预测器生成候选号码
            for arm in top_predictors:
                predictor_name = self.predictor_names[arm]
                try:
                    # 生成多注预测来增加候选号码的多样性
                    for _ in range(3):
                        front_balls, back_balls, _ = self._predict_with_predictor(predictor_name, periods)
                        front_candidates.update(front_balls)
                        back_candidates.update(back_balls)
                except Exception as e:
                    logger_manager.error(f"预测器 {predictor_name} 失败", e)
                    continue

            # 补充候选号码到所需数量（使用频率分析而不是随机数）
            if len(front_candidates) < front_count:
                from analyzer_modules import basic_analyzer
                freq_analysis = basic_analyzer.frequency_analysis()
                front_freq = freq_analysis.get('front_frequency', {})
                sorted_freq = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)

                for ball, _ in sorted_freq:
                    if len(front_candidates) >= front_count:
                        break
                    ball_int = int(ball) if isinstance(ball, str) else ball
                    front_candidates.add(ball_int)

            if len(back_candidates) < back_count:
                from analyzer_modules import basic_analyzer
                freq_analysis = basic_analyzer.frequency_analysis()
                back_freq = freq_analysis.get('back_frequency', {})
                sorted_freq = sorted(back_freq.items(), key=lambda x: x[1], reverse=True)

                for ball, _ in sorted_freq:
                    if len(back_candidates) >= back_count:
                        break
                    ball_int = int(ball) if isinstance(ball, str) else ball
                    back_candidates.add(ball_int)

            # 选择最终号码（基于预测器权重排序）
            front_balls = sorted(list(front_candidates))[:front_count]
            back_balls = sorted(list(back_candidates))[:back_count]

            # 计算组合数和投注金额
            from math import comb
            total_combinations = comb(front_count, 5) * comb(back_count, 2)
            total_cost = total_combinations * 3  # 每注3元

            # 计算综合置信度
            top_confidence = np.mean([self.bandit.values[arm] for arm in top_predictors])

            result = {
                'front_balls': front_balls,
                'back_balls': back_balls,
                'front_count': front_count,
                'back_count': back_count,
                'total_combinations': total_combinations,
                'total_cost': total_cost,
                'method': 'smart_compound',
                'confidence': min(top_confidence, 0.9),  # 限制最大置信度
                'top_predictors': [self.predictor_names[arm] for arm in top_predictors],
                'predictor_weights': [self.bandit.values[arm] for arm in top_predictors]
            }

            return result

        except Exception as e:
            logger_manager.error("智能复式预测失败", e)
            return {}

    def smart_predict_duplex(self, front_dan_count: int = 2, back_dan_count: int = 1,
                            front_tuo_count: int = 6, back_tuo_count: int = 4, periods: int = 500) -> Dict:
        """智能胆拖预测（基于学习结果的最优预测器）

        Args:
            front_dan_count: 前区胆码数量
            back_dan_count: 后区胆码数量
            front_tuo_count: 前区拖码数量
            back_tuo_count: 后区拖码数量
            periods: 分析期数
        """
        logger_manager.info(f"智能胆拖预测: 前区{front_dan_count}胆{front_tuo_count}拖, 后区{back_dan_count}胆{back_tuo_count}拖, 分析期数: {periods}")

        if len(self.predictor_names) == 0:
            logger_manager.error("预测器未初始化")
            return {}

        try:
            # 选择最优预测器作为胆码生成器
            best_arm = np.argmax(self.bandit.values)
            best_predictor = self.predictor_names[best_arm]

            # 生成胆码（使用最优预测器的高置信度预测）
            front_dan = []
            back_dan = []

            # 多次预测取最频繁的号码作为胆码
            front_freq = {}
            back_freq = {}

            for _ in range(10):  # 进行10次预测
                front_balls, back_balls, _ = self._predict_with_predictor(best_predictor, periods)
                for ball in front_balls:
                    front_freq[ball] = front_freq.get(ball, 0) + 1
                for ball in back_balls:
                    back_freq[ball] = back_freq.get(ball, 0) + 1

            # 选择频率最高的号码作为胆码
            front_dan = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)[:front_dan_count]
            front_dan = [ball for ball, freq in front_dan]

            back_dan = sorted(back_freq.items(), key=lambda x: x[1], reverse=True)[:back_dan_count]
            back_dan = [ball for ball, freq in back_dan]

            # 生成拖码（使用其他预测器）
            front_tuo_candidates = set()
            back_tuo_candidates = set()

            # 使用多个预测器生成拖码候选
            for arm in range(len(self.predictor_names)):
                if arm == best_arm:
                    continue  # 跳过已用于胆码的预测器

                predictor_name = self.predictor_names[arm]
                try:
                    front_balls, back_balls, _ = self._predict_with_predictor(predictor_name)
                    # 排除胆码
                    front_tuo_candidates.update([b for b in front_balls if b not in front_dan])
                    back_tuo_candidates.update([b for b in back_balls if b not in back_dan])
                except Exception as e:
                    logger_manager.error(f"预测器 {predictor_name} 失败", e)
                    continue

            # 补充拖码到所需数量（使用频率分析而不是随机数）
            if len(front_tuo_candidates) < front_tuo_count:
                from analyzer_modules import basic_analyzer
                freq_analysis = basic_analyzer.frequency_analysis()
                front_freq = freq_analysis.get('front_frequency', {})
                sorted_freq = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)

                for ball, _ in sorted_freq:
                    if len(front_tuo_candidates) >= front_tuo_count:
                        break
                    ball_int = int(ball) if isinstance(ball, str) else ball
                    if ball_int not in front_dan:
                        front_tuo_candidates.add(ball_int)

            while len(back_tuo_candidates) < back_tuo_count:
                candidate = int(np.random.randint(1, 13))
                if candidate not in back_dan:
                    back_tuo_candidates.add(candidate)

            front_tuo = sorted(list(front_tuo_candidates))[:front_tuo_count]
            back_tuo = sorted(list(back_tuo_candidates))[:back_tuo_count]

            # 计算组合数和投注金额
            from math import comb
            front_combinations = comb(front_tuo_count, 5 - front_dan_count)
            back_combinations = comb(back_tuo_count, 2 - back_dan_count)
            total_combinations = front_combinations * back_combinations
            total_cost = total_combinations * 3

            result = {
                'front_dan': sorted(front_dan),
                'front_tuo': front_tuo,
                'back_dan': sorted(back_dan),
                'back_tuo': back_tuo,
                'total_combinations': total_combinations,
                'total_cost': total_cost,
                'method': 'smart_duplex',
                'confidence': self.bandit.values[best_arm],
                'best_predictor': best_predictor
            }

            return result

        except Exception as e:
            logger_manager.error("智能胆拖预测失败", e)
            return {}

    def save_enhanced_results(self, filename: str = None) -> str:
        """保存增强学习结果"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_learning_{timestamp}.json"
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'learning_history': self.learning_history,
            'predictor_performance': self.predictor_performance,
            'bandit_values': self.bandit.values.tolist(),
            'bandit_counts': self.bandit.counts.tolist(),
            'predictor_names': self.predictor_names,
            'accuracy_stats': dict(self.accuracy_tracker.accuracy_stats)
        }
        
        try:
            cache_manager.save_cache("analysis", f"enhanced_learning_{datetime.now().strftime('%Y%m%d')}", results)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            logger_manager.info(f"增强学习结果已保存: {filename}")
            return filename
        except Exception as e:
            logger_manager.error("保存增强学习结果失败", e)
            return ""
    
    def load_enhanced_results(self, filename: str) -> bool:
        """加载增强学习结果"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            self.learning_history = results.get('learning_history', [])
            self.predictor_performance = results.get('predictor_performance', {})
            
            # 恢复多臂老虎机状态
            if 'bandit_values' in results and 'bandit_counts' in results:
                self.bandit.values = np.array(results['bandit_values'])
                self.bandit.counts = np.array(results['bandit_counts'])
                self.bandit.total_rewards = self.bandit.values * self.bandit.counts
            
            logger_manager.info(f"增强学习结果已加载: {filename}")
            return True
        except Exception as e:
            logger_manager.error("加载增强学习结果失败", e)
            return False

    def parameter_optimization(self, test_periods=100, optimization_rounds=10) -> Dict:
        """参数优化功能"""
        logger_manager.info(f"开始参数优化: 测试期数{test_periods}, 优化轮数{optimization_rounds}")

        best_params = {}
        best_score = 0.0
        optimization_history = []

        # 参数搜索空间
        param_space = {
            'epsilon': [0.1, 0.2, 0.3],
            'ucb_c': [1.0, 1.5, 2.0],
            'thompson_alpha': [1.0, 2.0, 3.0],
            'thompson_beta': [1.0, 2.0, 3.0]
        }

        for round_idx in range(optimization_rounds):
            logger_manager.info(f"优化轮次 {round_idx + 1}/{optimization_rounds}")

            # 随机选择参数组合
            current_params = {}
            for param, values in param_space.items():
                current_params[param] = np.random.choice(values)

            # 应用参数
            self.bandit.epsilon = current_params['epsilon']
            self.bandit.c = current_params['ucb_c']

            # 进行测试
            results = self.enhanced_adaptive_learning(
                start_period=100,
                test_periods=test_periods
            )

            if results:
                score = results['win_rate'] * 0.7 + results['average_score'] * 0.3

                optimization_history.append({
                    'round': round_idx + 1,
                    'params': current_params.copy(),
                    'win_rate': results['win_rate'],
                    'average_score': results['average_score'],
                    'score': score
                })

                if score > best_score:
                    best_score = score
                    best_params = current_params.copy()
                    logger_manager.info(f"发现更好参数: 得分 {score:.3f}")

        # 应用最佳参数
        if best_params:
            self.bandit.epsilon = best_params['epsilon']
            self.bandit.c = best_params['ucb_c']
            logger_manager.info(f"应用最佳参数: {best_params}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_history': optimization_history
        }


# ==================== 全局实例 ====================
enhanced_adaptive_predictor = EnhancedAdaptiveLearningPredictor()


if __name__ == "__main__":
    # 测试自适应学习模块
    print("🔧 测试自适应学习模块...")

    # 测试多臂老虎机
    print("🎰 测试多臂老虎机...")
    bandit = MultiArmedBandit(3, "ucb1")
    for i in range(10):
        arm = bandit.select_arm()
        reward = np.random.random()
        bandit.update(arm, reward)
    print(f"多臂老虎机测试完成，最优臂: {np.argmax(bandit.values)}")

    # 测试增强学习
    print("🚀 测试增强学习...")
    results = enhanced_adaptive_predictor.enhanced_adaptive_learning(100, 20)
    print(f"增强学习测试完成，中奖率: {results.get('win_rate', 0):.3f}")

    # 测试增强预测
    print("🎯 测试增强预测...")
    predictions = enhanced_adaptive_predictor.generate_enhanced_prediction(1)
    if predictions:
        pred = predictions[0]
        print(f"增强预测: 前区 {pred['front_balls']}, 后区 {pred['back_balls']}")

    print("✅ 自适应学习模块测试完成")
