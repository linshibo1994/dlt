#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强马尔可夫链模块
提供多阶马尔可夫链和自适应马尔可夫链预测
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from collections import defaultdict, Counter
from datetime import datetime

# 尝试导入核心模块
try:
    from core_modules import logger_manager, data_manager, cache_manager
except ImportError:
    # 如果在不同目录运行，添加父目录到路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core_modules import logger_manager, data_manager, cache_manager


class EnhancedMarkovAnalyzer:
    """增强马尔可夫链分析器"""
    
    def __init__(self):
        self.df = data_manager.get_data()
        if self.df is None:
            logger_manager.error("数据未加载")
    
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
        
        cache_key = f"multi_order_markov_analysis_{periods}_{max_order}"
        cached_result = cache_manager.load_cache("analysis", cache_key)
        if cached_result:
            return cached_result
        
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
        
        cache_manager.save_cache("analysis", cache_key, result)
        return result
    
    def _analyze_nth_order_markov(self, df_subset, order=1) -> Dict:
        """分析n阶马尔可夫链
        
        Args:
            df_subset: 数据子集
            order: 马尔可夫链阶数
        
        Returns:
            Dict: n阶马尔可夫链分析结果
        """
        # 前区和后区的转移矩阵
        front_transitions = defaultdict(lambda: defaultdict(int))
        back_transitions = defaultdict(lambda: defaultdict(int))
        
        # 对于n阶马尔可夫链，我们需要n个连续的状态作为条件
        for i in range(len(df_subset) - order):
            # 构建条件状态（前n期的号码）
            condition_front = []
            condition_back = []
            
            for j in range(order):
                front, back = data_manager.parse_balls(df_subset.iloc[i + j])
                condition_front.extend(front)
                condition_back.extend(back)
            
            # 获取下一期的号码（要预测的状态）
            next_front, next_back = data_manager.parse_balls(df_subset.iloc[i + order])
            
            # 将条件状态转换为元组（作为字典键）
            condition_front_tuple = tuple(sorted(condition_front))
            condition_back_tuple = tuple(sorted(condition_back))
            
            # 更新转移计数
            for next_ball in next_front:
                front_transitions[condition_front_tuple][next_ball] += 1
            
            for next_ball in next_back:
                back_transitions[condition_back_tuple][next_ball] += 1
        
        # 转换为概率
        front_probs = {}
        for condition, to_dict in front_transitions.items():
            total = sum(to_dict.values())
            if total > 0:
                # 使用字符串作为键，因为字典键需要可哈希
                front_probs[str(condition)] = {to_ball: count/total for to_ball, count in to_dict.items()}
        
        back_probs = {}
        for condition, to_dict in back_transitions.items():
            total = sum(to_dict.values())
            if total > 0:
                back_probs[str(condition)] = {to_ball: count/total for to_ball, count in to_dict.items()}
        
        # 计算状态转移矩阵的统计信息
        front_stats = self._calculate_transition_stats(front_transitions)
        back_stats = self._calculate_transition_stats(back_transitions)
        
        return {
            'front_transition_probs': front_probs,
            'back_transition_probs': back_probs,
            'front_stats': front_stats,
            'back_stats': back_stats,
            'order': order
        }
    
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
    
    def multi_order_markov_predict(self, count=1, periods=500, order=1) -> List[Tuple[List[int], List[int]]]:
        """多阶马尔可夫链预测
        
        Args:
            count: 预测注数
            periods: 分析期数
            order: 马尔可夫链阶数 (1-3)
        
        Returns:
            List[Tuple[List[int], List[int]]]: 预测结果列表，每个元素为(前区号码, 后区号码)
        """
        # 检查参数有效性
        order = min(max(1, order), 3)  # 限制在1-3之间
        
        # 获取马尔可夫分析结果
        markov_result = self.analyzer.multi_order_markov_analysis(periods, max_order=order)
        
        if not markov_result or 'orders' not in markov_result or str(order) not in markov_result['orders']:
            logger_manager.error(f"{order}阶马尔可夫分析结果不可用")
            return []

        order_result = markov_result['orders'][str(order)]
        front_transitions = order_result.get('front_transition_probs', {})
        back_transitions = order_result.get('back_transition_probs', {})
        
        predictions = []
        
        # 获取最近n期的号码作为条件状态
        condition_front = []
        condition_back = []
        
        for i in range(min(order, len(self.df))):
            front, back = data_manager.parse_balls(self.df.iloc[i])
            condition_front.extend(front)
            condition_back.extend(back)
        
        # 如果数据不足，使用默认值
        if not condition_front:
            condition_front = list(range(1, 6))
        if not condition_back:
            condition_back = [1, 2]
        
        # 将条件状态转换为字符串（作为字典键）
        condition_front_str = str(tuple(sorted(condition_front)))
        condition_back_str = str(tuple(sorted(condition_back)))
        
        for i in range(count):
            # 为每注使用不同的马尔可夫策略，添加时间戳确保随机性
            import time
            strategy_seed = int(time.time() * 1000000) + i * 1000

            # 预测前区号码
            front_balls = self._predict_balls_with_condition_diverse(
                front_transitions, condition_front_str, 5, 35, i, strategy_seed
            )

            # 预测后区号码
            back_balls = self._predict_balls_with_condition_diverse(
                back_transitions, condition_back_str, 2, 12, i, strategy_seed + 500
            )

            predictions.append((sorted(front_balls), sorted(back_balls)))

        return predictions

    def _predict_balls_with_condition_diverse(self, transitions, condition_str, num_balls, max_ball, strategy_index, seed=None):
        """基于条件状态预测号码 - 多样性策略版本"""
        import random
        import numpy as np

        # 设置随机种子确保每注不同
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % 2**32)

        balls = []

        # 策略1: 最高概率策略 (第1注)
        if strategy_index % 4 == 0 and strategy_index < 4:
            # 如果条件状态存在于转移矩阵中
            if condition_str in transitions:
                trans_probs = transitions[condition_str]

                # 按概率排序，选择前几个高概率号码
                sorted_probs = sorted(trans_probs.items(), key=lambda x: x[1], reverse=True)
                high_prob_balls = [int(ball) for ball, _ in sorted_probs[:num_balls*2]]

                if len(high_prob_balls) >= num_balls:
                    balls = random.sample(high_prob_balls, num_balls)
                else:
                    balls = high_prob_balls

        # 策略2: 中等概率策略 (第2注)
        elif strategy_index % 4 == 1 and strategy_index < 4:
            if condition_str in transitions:
                trans_probs = transitions[condition_str]
                sorted_probs = sorted(trans_probs.items(), key=lambda x: x[1], reverse=True)

                # 选择中等概率的号码
                mid_start = len(sorted_probs) // 4
                mid_end = len(sorted_probs) * 3 // 4
                mid_prob_balls = [int(ball) for ball, _ in sorted_probs[mid_start:mid_end]]

                if len(mid_prob_balls) >= num_balls:
                    balls = random.sample(mid_prob_balls, num_balls)
                else:
                    balls = mid_prob_balls

        # 策略3: 概率加权随机选择 (第3注)
        elif strategy_index % 4 == 2 and strategy_index < 4:
            if condition_str in transitions:
                trans_probs = transitions[condition_str]

                if trans_probs:
                    ball_list = [int(ball) for ball in trans_probs.keys()]
                    prob_list = list(trans_probs.values())

                    # 归一化概率
                    total_prob = sum(prob_list)
                    if total_prob > 0:
                        normalized_probs = [p/total_prob for p in prob_list]

                        # 概率加权随机选择
                        if len(ball_list) >= num_balls:
                            balls = list(np.random.choice(ball_list, size=num_balls, replace=False, p=normalized_probs))
                        else:
                            balls = ball_list

        # 策略4: 全局概率分布策略 (第4注及以后)
        else:
            # 为第4注以后添加更多随机性
            import time
            random.seed(int(time.time() * 1000) + strategy_index)
            # 从所有转移概率中选择
            all_probs = {}

            for cond, trans_probs in transitions.items():
                for ball, prob in trans_probs.items():
                    ball_int = int(ball)
                    all_probs[ball_int] = all_probs.get(ball_int, 0) + prob

            if all_probs:
                # 概率加权随机选择
                ball_list = list(all_probs.keys())
                prob_list = list(all_probs.values())

                total_prob = sum(prob_list)
                if total_prob > 0:
                    normalized_probs = [p/total_prob for p in prob_list]

                    if len(ball_list) >= num_balls:
                        balls = list(np.random.choice(ball_list, size=num_balls, replace=False, p=normalized_probs))
                    else:
                        balls = ball_list

        # 如果号码不足，使用频率分析补充
        if len(balls) < num_balls:
            from analyzer_modules import basic_analyzer
            freq_analysis = basic_analyzer.frequency_analysis()

            if max_ball == 35:  # 前区
                freq_dict = freq_analysis.get('front_frequency', {})
            else:  # 后区
                freq_dict = freq_analysis.get('back_frequency', {})

            sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
            for ball, _ in sorted_freq:
                if len(balls) >= num_balls:
                    break

                ball_int = int(ball)
                if ball_int not in balls:
                    balls.append(ball_int)

        # 如果仍然不足，随机补充
        if len(balls) < num_balls:
            remaining = [i for i in range(1, max_ball + 1) if i not in balls]
            if remaining:
                needed = num_balls - len(balls)
                balls.extend(random.sample(remaining, min(needed, len(remaining))))

        return balls[:num_balls]

    def _predict_balls_with_condition(self, transitions, condition_str, num_balls, max_ball):
        """基于条件状态预测号码"""
        balls = []
        
        # 如果条件状态存在于转移矩阵中
        if condition_str in transitions:
            trans_probs = transitions[condition_str]
            
            # 按概率排序
            sorted_probs = sorted(trans_probs.items(), key=lambda x: x[1], reverse=True)
            
            # 选择概率最高的号码
            for ball, _ in sorted_probs:
                if len(balls) >= num_balls:
                    break
                
                ball_int = int(ball)
                if ball_int not in balls:
                    balls.append(ball_int)
        
        # 如果号码不足，从所有转移概率中选择
        if len(balls) < num_balls:
            all_probs = {}
            
            for cond, trans_probs in transitions.items():
                for ball, prob in trans_probs.items():
                    ball_int = int(ball)
                    if ball_int not in balls:
                        all_probs[ball_int] = all_probs.get(ball_int, 0) + prob
            
            # 按概率排序
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            
            # 选择概率最高的号码
            for ball, _ in sorted_probs:
                if len(balls) >= num_balls:
                    break
                
                if ball not in balls:
                    balls.append(ball)
        
        # 如果仍然不足，使用频率分析补充
        if len(balls) < num_balls:
            from analyzer_modules import basic_analyzer
            freq_analysis = basic_analyzer.frequency_analysis()

            if max_ball == 35:  # 前区
                freq_dict = freq_analysis.get('front_frequency', {})
            else:  # 后区
                freq_dict = freq_analysis.get('back_frequency', {})

            # 按频率排序
            sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)

            for ball_str, _ in sorted_freq:
                if len(balls) >= num_balls:
                    break
                ball_int = int(ball_str) if isinstance(ball_str, str) else ball_str
                if ball_int not in balls and 1 <= ball_int <= max_ball:
                    balls.append(ball_int)

        # 如果还是不足，使用默认序列
        while len(balls) < num_balls:
            for i in range(1, max_ball + 1):
                if len(balls) >= num_balls:
                    break
                if i not in balls:
                    balls.append(i)
        
        return balls
    
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
            if str(order) in markov_result['orders']:
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
            
            # 如果号码不足，使用频率分析补充
            if len(front_balls) < 5:
                from analyzer_modules import basic_analyzer
                freq_analysis = basic_analyzer.frequency_analysis()
                front_freq = freq_analysis.get('front_frequency', {})
                sorted_freq = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)

                for ball_str, _ in sorted_freq:
                    if len(front_balls) >= 5:
                        break
                    ball_int = int(ball_str) if isinstance(ball_str, str) else ball_str
                    if ball_int not in front_balls and 1 <= ball_int <= 35:
                        front_balls.append(ball_int)

            if len(back_balls) < 2:
                from analyzer_modules import basic_analyzer
                freq_analysis = basic_analyzer.frequency_analysis()
                back_freq = freq_analysis.get('back_frequency', {})
                sorted_freq = sorted(back_freq.items(), key=lambda x: x[1], reverse=True)

                for ball_str, _ in sorted_freq:
                    if len(back_balls) >= 2:
                        break
                    ball_int = int(ball_str) if isinstance(ball_str, str) else ball_str
                    if ball_int not in back_balls and 1 <= ball_int <= 12:
                        back_balls.append(ball_int)
            
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