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
    from smart_cache_system import smart_cache_manager
except ImportError:
    # 如果在不同目录运行，添加父目录到路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core_modules import logger_manager, data_manager, cache_manager
    from smart_cache_system import smart_cache_manager


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
        
        method_name = "multi_order_markov_analysis"
        cached_result = smart_cache_manager.load_cache("analysis", method_name, periods, max_order=max_order)
        if cached_result:
            return cached_result
        
        df_subset = self.df.tail(periods)
        
        result = {
            'orders': {},
            'analysis_periods': periods,
            'max_order': max_order
        }
        
        # 对每个阶数进行分析
        for order in range(1, max_order + 1):
            order_result = self._analyze_nth_order_markov(df_subset, order)
            result['orders'][order] = order_result
        
        smart_cache_manager.save_cache("analysis", method_name, result, periods, max_order=max_order)
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
        
        if not markov_result or 'orders' not in markov_result or order not in markov_result['orders']:
            logger_manager.error(f"{order}阶马尔可夫分析结果不可用")
            return []

        order_result = markov_result['orders'][order]
        front_transitions = order_result.get('front_transition_probs', {})
        back_transitions = order_result.get('back_transition_probs', {})
        
        predictions = []
        
        # 获取最近n期的号码作为条件状态
        condition_front = []
        condition_back = []

        for i in range(min(order, len(self.df))):
            # 从最新的数据开始取，而不是从最早的数据开始
            front, back = data_manager.parse_balls(self.df.iloc[-(i+1)])
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

            # 为每注使用不同的选择策略，确保多样性
            front_balls = self._select_balls_with_diversity(front_counter, 5, i, periods)
            back_balls = self._select_balls_with_diversity(back_counter, 2, i, periods)


            
            # 如果号码不足，使用多样化的回退策略
            if len(front_balls) < 5:
                front_balls = self._fallback_ball_selection(front_balls, 5, 35, i, 'front')

            if len(back_balls) < 2:
                back_balls = self._fallback_ball_selection(back_balls, 2, 12, i, 'back')
            
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

    def _select_balls_with_diversity(self, counter, target_count, prediction_index, periods=500):
        """根据预测索引使用不同策略选择号码，确保多样性"""
        import random
        import numpy as np
        import time

        # 为每注设置不同的随机种子，同时考虑期数的影响
        seed_base = prediction_index * 1000 + int(time.time() * 1000) % 10000 + periods
        random.seed(seed_base)
        np.random.seed(seed_base % 2**32)

        if not counter:
            return []

        # 获取所有候选号码和频率
        all_balls = list(counter.keys())
        all_counts = list(counter.values())

        if len(all_balls) <= target_count:
            return all_balls

        selected_balls = []

        # 策略1：第1注 - 期数敏感的频率策略
        if prediction_index == 0:
            # 根据分析期数调整选择策略
            if periods <= 300:
                # 短期分析：纯频率最高
                selected_balls = [ball for ball, _ in counter.most_common(target_count)]
            elif periods <= 800:
                # 中期分析：80%高频 + 20%随机
                high_freq_count = max(1, int(target_count * 0.8))
                high_freq_balls = [ball for ball, _ in counter.most_common(high_freq_count)]
                selected_balls.extend(high_freq_balls)

                remaining_balls = [ball for ball in all_balls if ball not in high_freq_balls]
                if remaining_balls and len(selected_balls) < target_count:
                    random_count = target_count - len(selected_balls)
                    random_balls = random.sample(remaining_balls, min(random_count, len(remaining_balls)))
                    selected_balls.extend(random_balls)
            else:
                # 长期分析：70%高频 + 30%加权随机
                high_freq_count = max(1, int(target_count * 0.7))
                high_freq_balls = [ball for ball, _ in counter.most_common(high_freq_count)]
                selected_balls.extend(high_freq_balls)

                # 剩余位置用加权随机填充
                remaining_balls = [ball for ball in all_balls if ball not in high_freq_balls]
                remaining_counts = [counter[ball] for ball in remaining_balls]

                if remaining_balls and len(selected_balls) < target_count:
                    random_count = target_count - len(selected_balls)
                    if sum(remaining_counts) > 0:
                        probabilities = [count / sum(remaining_counts) for count in remaining_counts]
                        weighted_balls = list(np.random.choice(
                            remaining_balls, size=min(random_count, len(remaining_balls)),
                            replace=False, p=probabilities
                        ))
                        selected_balls.extend(weighted_balls)
                    else:
                        random_balls = random.sample(remaining_balls, min(random_count, len(remaining_balls)))
                        selected_balls.extend(random_balls)

        # 策略2：第2注 - 频率加权随机选择
        elif prediction_index == 1:
            # 计算概率权重
            total_count = sum(all_counts)
            probabilities = [count / total_count for count in all_counts]

            # 概率加权随机选择
            selected_balls = list(np.random.choice(
                all_balls, size=min(target_count, len(all_balls)),
                replace=False, p=probabilities
            ))

        # 策略3：第3注 - 混合策略（50%高频 + 50%随机）
        elif prediction_index == 2:
            high_freq_count = max(1, target_count // 2)
            random_count = target_count - high_freq_count

            # 选择高频号码
            high_freq_balls = [ball for ball, _ in counter.most_common(high_freq_count)]
            selected_balls.extend(high_freq_balls)

            # 从剩余号码中随机选择
            remaining_balls = [ball for ball in all_balls if ball not in high_freq_balls]
            if remaining_balls and random_count > 0:
                random_balls = random.sample(remaining_balls, min(random_count, len(remaining_balls)))
                selected_balls.extend(random_balls)

        # 策略4：第4注及以后 - 平衡策略
        else:
            # 将号码按频率分为三档
            sorted_balls = [ball for ball, _ in counter.most_common()]
            total_balls = len(sorted_balls)

            tier1_end = max(1, total_balls // 3)
            tier2_end = max(2, total_balls * 2 // 3)

            tier1_balls = sorted_balls[:tier1_end]  # 高频
            tier2_balls = sorted_balls[tier1_end:tier2_end]  # 中频
            tier3_balls = sorted_balls[tier2_end:]  # 低频

            # 按比例从各档选择：60%高频，30%中频，10%低频
            tier1_count = max(1, int(target_count * 0.6))
            tier2_count = max(0, int(target_count * 0.3))
            tier3_count = target_count - tier1_count - tier2_count

            # 从各档随机选择
            if tier1_balls:
                selected_balls.extend(random.sample(tier1_balls, min(tier1_count, len(tier1_balls))))
            if tier2_balls and tier2_count > 0:
                selected_balls.extend(random.sample(tier2_balls, min(tier2_count, len(tier2_balls))))
            if tier3_balls and tier3_count > 0:
                selected_balls.extend(random.sample(tier3_balls, min(tier3_count, len(tier3_balls))))

        # 确保返回正确数量的号码
        if len(selected_balls) < target_count:
            # 从剩余号码中补充
            remaining = [ball for ball in all_balls if ball not in selected_balls]
            if remaining:
                need_count = target_count - len(selected_balls)
                additional = random.sample(remaining, min(need_count, len(remaining)))
                selected_balls.extend(additional)

        return selected_balls[:target_count]

    def _fallback_ball_selection(self, existing_balls, target_count, max_ball, prediction_index, ball_type):
        """多样化的回退号码选择策略"""
        import random
        import time

        # 为每注设置不同的随机种子
        random.seed(prediction_index * 1000 + int(time.time() * 1000) % 10000)

        # 获取频率分析结果
        from analyzer_modules import basic_analyzer
        freq_analysis = basic_analyzer.frequency_analysis()

        if ball_type == 'front':
            freq_data = freq_analysis.get('front_frequency', {})
        else:
            freq_data = freq_analysis.get('back_frequency', {})

        # 转换为整数并排序
        freq_balls = []
        for ball_str, freq in freq_data.items():
            try:
                ball_int = int(ball_str) if isinstance(ball_str, str) else ball_str
                if 1 <= ball_int <= max_ball:
                    freq_balls.append((ball_int, freq))
            except (ValueError, TypeError):
                continue

        freq_balls.sort(key=lambda x: x[1], reverse=True)

        # 为每注使用不同的选择策略
        need_count = target_count - len(existing_balls)
        selected_balls = list(existing_balls)

        if prediction_index == 0:
            # 第1注：高频优先
            for ball, _ in freq_balls:
                if len(selected_balls) >= target_count:
                    break
                if ball not in selected_balls:
                    selected_balls.append(ball)

        elif prediction_index == 1:
            # 第2注：随机选择
            available_balls = [ball for ball, _ in freq_balls if ball not in selected_balls]
            if available_balls and need_count > 0:
                random_balls = random.sample(available_balls, min(need_count, len(available_balls)))
                selected_balls.extend(random_balls)

        else:
            # 第3注及以后：混合策略
            high_freq_balls = [ball for ball, _ in freq_balls[:len(freq_balls)//2] if ball not in selected_balls]
            low_freq_balls = [ball for ball, _ in freq_balls[len(freq_balls)//2:] if ball not in selected_balls]

            # 50%高频，50%低频
            high_count = need_count // 2
            low_count = need_count - high_count

            if high_freq_balls and high_count > 0:
                selected_balls.extend(random.sample(high_freq_balls, min(high_count, len(high_freq_balls))))

            if low_freq_balls and low_count > 0:
                selected_balls.extend(random.sample(low_freq_balls, min(low_count, len(low_freq_balls))))

        # 如果还不够，随机补充
        if len(selected_balls) < target_count:
            remaining_balls = [i for i in range(1, max_ball + 1) if i not in selected_balls]
            if remaining_balls:
                need_more = target_count - len(selected_balls)
                selected_balls.extend(random.sample(remaining_balls, min(need_more, len(remaining_balls))))

        return selected_balls[:target_count]

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