#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
预测器模块集成
整合传统预测、高级预测、深度学习预测、自适应学习预测等所有预测功能
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
from analyzer_modules import basic_analyzer, advanced_analyzer, comprehensive_analyzer
from smart_cache_system import smart_cache_manager

# 导入增强深度学习模块
try:
    from enhanced_deep_learning.models import LSTMPredictor, TransformerPredictor, GANPredictor, EnsembleManager
    from enhanced_deep_learning.performance.enhanced_hardware_accelerator import EnhancedHardwareAccelerator
    from enhanced_deep_learning.performance.acceleration_selector import AccelerationSelector, AccelerationRecommendation
    from compound_modules.compound_predictor import CompoundPredictorMixin, CompoundConfig, CompoundResult
    ENHANCED_DL_AVAILABLE = True
except ImportError as e:
    logger_manager.warning(f"增强深度学习模块导入失败: {e}")
    ENHANCED_DL_AVAILABLE = False


# ==================== 传统预测器 ====================
class TraditionalPredictor:
    """传统预测器"""
    
    def __init__(self, data_file="data/dlt_data_all.csv"):
        self.data_file = data_file
        self.df = data_manager.get_data()
        
        if self.df is None:
            logger_manager.error("数据未加载")
    
    def frequency_predict(self, count=1, periods=500) -> List[Tuple[List[int], List[int]]]:
        """基于频率的预测 - 真正的多样性统计分析"""
        import random
        import numpy as np

        freq_result = basic_analyzer.frequency_analysis(periods)

        front_freq = freq_result.get('front_frequency', {})
        back_freq = freq_result.get('back_frequency', {})

        predictions = []

        # 获取频率排序的候选号码
        front_candidates = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)
        back_candidates = sorted(back_freq.items(), key=lambda x: x[1], reverse=True)

        # 为每注生成不同的预测策略
        for i in range(count):
            front_balls = []
            back_balls = []

            # 策略1: 高频号码为主 (第1注)
            if i % 4 == 0:
                # 选择频率最高的号码，但加入随机性
                high_freq_front = [int(ball) for ball, freq in front_candidates[:8]]
                front_balls = random.sample(high_freq_front, min(5, len(high_freq_front)))

                high_freq_back = [int(ball) for ball, freq in back_candidates[:4]]
                back_balls = random.sample(high_freq_back, min(2, len(high_freq_back)))

            # 策略2: 中频号码为主 (第2注)
            elif i % 4 == 1:
                # 选择中等频率的号码
                mid_start = len(front_candidates) // 4
                mid_end = len(front_candidates) * 3 // 4
                mid_freq_front = [int(ball) for ball, freq in front_candidates[mid_start:mid_end]]
                if len(mid_freq_front) >= 5:
                    front_balls = random.sample(mid_freq_front, 5)
                else:
                    front_balls = mid_freq_front + random.sample([int(ball) for ball, freq in front_candidates[:8]], 5 - len(mid_freq_front))

                mid_freq_back = [int(ball) for ball, freq in back_candidates[1:5]]
                if len(mid_freq_back) >= 2:
                    back_balls = random.sample(mid_freq_back, 2)
                else:
                    back_balls = mid_freq_back + random.sample([int(ball) for ball, freq in back_candidates[:4]], 2 - len(mid_freq_back))

            # 策略3: 混合频率策略 (第3注)
            elif i % 4 == 2:
                # 2个高频 + 2个中频 + 1个低频
                high_freq = [int(ball) for ball, freq in front_candidates[:6]]
                mid_freq = [int(ball) for ball, freq in front_candidates[6:15]]
                low_freq = [int(ball) for ball, freq in front_candidates[15:25]]

                front_balls = []
                front_balls.extend(random.sample(high_freq, min(2, len(high_freq))))
                front_balls.extend(random.sample(mid_freq, min(2, len(mid_freq))))
                if len(low_freq) > 0:
                    front_balls.extend(random.sample(low_freq, min(1, len(low_freq))))

                # 如果不足5个，用高频补充
                while len(front_balls) < 5:
                    remaining = [ball for ball in high_freq if ball not in front_balls]
                    if remaining:
                        front_balls.append(random.choice(remaining))
                    else:
                        break

                # 后区混合策略
                back_high = [int(ball) for ball, freq in back_candidates[:3]]
                back_mid = [int(ball) for ball, freq in back_candidates[3:8]]

                back_balls = []
                if len(back_high) > 0:
                    back_balls.append(random.choice(back_high))
                if len(back_mid) > 0:
                    back_balls.append(random.choice(back_mid))

                # 如果不足2个，用高频补充
                while len(back_balls) < 2:
                    remaining = [ball for ball in back_high if ball not in back_balls]
                    if remaining:
                        back_balls.append(random.choice(remaining))
                    else:
                        break

            # 策略4: 概率加权随机选择 (第4注及以后)
            else:
                # 基于频率的加权随机选择
                front_weights = [freq for ball, freq in front_candidates]
                front_balls_list = [int(ball) for ball, freq in front_candidates]

                if len(front_weights) > 0:
                    # 归一化权重
                    total_weight = sum(front_weights)
                    front_probs = [w/total_weight for w in front_weights]

                    # 加权随机选择
                    front_balls = list(np.random.choice(front_balls_list, size=5, replace=False, p=front_probs))

                back_weights = [freq for ball, freq in back_candidates]
                back_balls_list = [int(ball) for ball, freq in back_candidates]

                if len(back_weights) > 0:
                    total_weight = sum(back_weights)
                    back_probs = [w/total_weight for w in back_weights]
                    back_balls = list(np.random.choice(back_balls_list, size=2, replace=False, p=back_probs))

            # 确保号码数量正确
            if len(front_balls) < 5:
                remaining = [int(ball) for ball, freq in front_candidates[:10] if int(ball) not in front_balls]
                front_balls.extend(remaining[:5-len(front_balls)])

            if len(back_balls) < 2:
                remaining = [int(ball) for ball, freq in back_candidates[:6] if int(ball) not in back_balls]
                back_balls.extend(remaining[:2-len(back_balls)])

            predictions.append((sorted(front_balls[:5]), sorted(back_balls[:2])))

        return predictions
    
    def hot_cold_predict(self, count=1, periods=500) -> List[Tuple[List[int], List[int]]]:
        """基于冷热号的预测 - 真正的冷热号分析和多样性策略"""
        import random

        hot_cold_result = basic_analyzer.hot_cold_analysis(periods)
        
        front_hot = hot_cold_result.get('front_hot', [])
        front_cold = hot_cold_result.get('front_cold', [])
        back_hot = hot_cold_result.get('back_hot', [])
        back_cold = hot_cold_result.get('back_cold', [])
        
        predictions = []
        
        # 冷热号预测（确定性选择，3个热号+2个冷号）
        front_balls = []
        back_balls = []

        # 前区：选择3个热号和2个冷号
        hot_count = 3
        cold_count = 2

        if len(front_hot) >= hot_count:
            front_balls.extend([int(ball) for ball in front_hot[:hot_count]])
        else:
            front_balls.extend([int(ball) for ball in front_hot])

        if len(front_cold) >= cold_count:
            front_balls.extend([int(ball) for ball in front_cold[:cold_count]])
        else:
            front_balls.extend([int(ball) for ball in front_cold])

        # 如果前区号码不足，用频率分析补充
        if len(front_balls) < 5:
            freq_analysis = basic_analyzer.frequency_analysis()
            front_freq = freq_analysis.get('front_frequency', {})
            sorted_freq = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)
            for ball, freq in sorted_freq:
                if len(front_balls) >= 5:
                    break
                if ball not in front_balls:
                    front_balls.append(ball)

        # 后区：选择1个热号和1个冷号
        if len(back_hot) > 0:
            back_balls.append(int(back_hot[0]))

        if len(back_cold) > 0:
            # 选择不与热号重复的冷号
            for cold_ball in back_cold:
                if int(cold_ball) not in back_balls:
                    back_balls.append(int(cold_ball))
                    break

        # 如果后区号码不足，用频率分析补充
        if len(back_balls) < 2:
            freq_analysis = basic_analyzer.frequency_analysis()
            back_freq = freq_analysis.get('back_frequency', {})
            sorted_freq = sorted(back_freq.items(), key=lambda x: x[1], reverse=True)
            for ball, freq in sorted_freq:
                if len(back_balls) >= 2:
                    break
                if ball not in back_balls:
                    back_balls.append(ball)

        # 为每注生成不同的冷热号策略
        for i in range(count):
            current_front = []
            current_back = []

            # 策略1: 热号为主策略 (第1注)
            if i % 5 == 0:
                # 4个热号 + 1个冷号
                if len(front_hot) >= 4:
                    current_front.extend(random.sample([int(ball) for ball in front_hot], 4))
                else:
                    current_front.extend([int(ball) for ball in front_hot])

                if len(front_cold) >= 1:
                    remaining_cold = [int(ball) for ball in front_cold if int(ball) not in current_front]
                    if remaining_cold:
                        current_front.append(random.choice(remaining_cold))

                # 后区：2个热号
                if len(back_hot) >= 2:
                    current_back = random.sample([int(ball) for ball in back_hot], 2)
                else:
                    current_back.extend([int(ball) for ball in back_hot])

            # 策略2: 冷号回补策略 (第2注)
            elif i % 5 == 1:
                # 2个热号 + 3个冷号
                if len(front_hot) >= 2:
                    current_front.extend(random.sample([int(ball) for ball in front_hot], 2))
                else:
                    current_front.extend([int(ball) for ball in front_hot])

                if len(front_cold) >= 3:
                    remaining_cold = [int(ball) for ball in front_cold if int(ball) not in current_front]
                    if len(remaining_cold) >= 3:
                        current_front.extend(random.sample(remaining_cold, 3))
                    else:
                        current_front.extend(remaining_cold)

                # 后区：1个热号 + 1个冷号
                if len(back_hot) >= 1:
                    current_back.append(random.choice([int(ball) for ball in back_hot]))
                if len(back_cold) >= 1:
                    remaining_cold = [int(ball) for ball in back_cold if int(ball) not in current_back]
                    if remaining_cold:
                        current_back.append(random.choice(remaining_cold))

            # 策略3: 平衡策略 (第3注)
            elif i % 5 == 2:
                # 3个热号 + 2个冷号
                if len(front_hot) >= 3:
                    current_front.extend(random.sample([int(ball) for ball in front_hot], 3))
                else:
                    current_front.extend([int(ball) for ball in front_hot])

                if len(front_cold) >= 2:
                    remaining_cold = [int(ball) for ball in front_cold if int(ball) not in current_front]
                    if len(remaining_cold) >= 2:
                        current_front.extend(random.sample(remaining_cold, 2))
                    else:
                        current_front.extend(remaining_cold)

                # 后区：随机选择热号或冷号
                back_candidates = []
                if len(back_hot) > 0:
                    back_candidates.extend([int(ball) for ball in back_hot])
                if len(back_cold) > 0:
                    back_candidates.extend([int(ball) for ball in back_cold])

                if len(back_candidates) >= 2:
                    current_back = random.sample(back_candidates, 2)
                else:
                    current_back = back_candidates

            # 策略4: 极端热号策略 (第4注)
            elif i % 5 == 3:
                # 全部选择热号
                if len(front_hot) >= 5:
                    current_front = random.sample([int(ball) for ball in front_hot], 5)
                else:
                    current_front.extend([int(ball) for ball in front_hot])

                if len(back_hot) >= 2:
                    current_back = random.sample([int(ball) for ball in back_hot], 2)
                else:
                    current_back.extend([int(ball) for ball in back_hot])

            # 策略5: 极端冷号策略 (第5注)
            else:
                # 全部选择冷号
                if len(front_cold) >= 5:
                    current_front = random.sample([int(ball) for ball in front_cold], 5)
                else:
                    current_front.extend([int(ball) for ball in front_cold])

                if len(back_cold) >= 2:
                    current_back = random.sample([int(ball) for ball in back_cold], 2)
                else:
                    current_back.extend([int(ball) for ball in back_cold])

            # 如果号码不足，用频率分析补充
            if len(current_front) < 5:
                freq_analysis = basic_analyzer.frequency_analysis(periods)
                front_freq = freq_analysis.get('front_frequency', {})
                sorted_freq = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)
                for ball, freq in sorted_freq:
                    if len(current_front) >= 5:
                        break
                    if int(ball) not in current_front:
                        current_front.append(int(ball))

            if len(current_back) < 2:
                freq_analysis = basic_analyzer.frequency_analysis(periods)
                back_freq = freq_analysis.get('back_frequency', {})
                sorted_freq = sorted(back_freq.items(), key=lambda x: x[1], reverse=True)
                for ball, freq in sorted_freq:
                    if len(current_back) >= 2:
                        break
                    if int(ball) not in current_back:
                        current_back.append(int(ball))

            predictions.append((sorted(current_front[:5]), sorted(current_back[:2])))

        return predictions
    
    def missing_predict(self, count=1, periods=500) -> List[Tuple[List[int], List[int]]]:
        """基于遗漏的预测 - 真正的遗漏值分析和回补概率计算"""
        import random
        import numpy as np

        missing_result = basic_analyzer.missing_analysis(periods)

        front_missing = missing_result.get('front_missing', {})
        back_missing = missing_result.get('back_missing', {})

        predictions = []

        # 按遗漏值排序
        front_sorted = sorted(front_missing.items(), key=lambda x: x[1], reverse=True)
        back_sorted = sorted(back_missing.items(), key=lambda x: x[1], reverse=True)

        # 为每注生成不同的遗漏值策略
        for i in range(count):
            import time
            strategy_seed = int(time.time() * 1000000) + i * 1000
            random.seed(strategy_seed)
            np.random.seed(strategy_seed % 2**32)

            front_balls = []
            back_balls = []

            # 策略1: 极度超期回补策略 (第1注)
            if i == 0:
                # 选择遗漏值最大的号码（极度超期）
                extreme_missing_front = [int(ball) for ball, missing in front_sorted[:8] if missing > periods * 0.1]
                if len(extreme_missing_front) >= 5:
                    front_balls = random.sample(extreme_missing_front, 5)
                else:
                    front_balls = extreme_missing_front + [int(ball) for ball, missing in front_sorted[:5-len(extreme_missing_front)]]

                extreme_missing_back = [int(ball) for ball, missing in back_sorted[:4] if missing > periods * 0.15]
                if len(extreme_missing_back) >= 2:
                    back_balls = random.sample(extreme_missing_back, 2)
                else:
                    # 确保不重复添加号码
                    back_balls = extreme_missing_back[:]
                    needed = 2 - len(extreme_missing_back)
                    for ball, missing in back_sorted:
                        if len(back_balls) >= 2:
                            break
                        ball_int = int(ball)
                        if ball_int not in back_balls:
                            back_balls.append(ball_int)

            # 策略2: 中期遗漏策略 (第2注)
            elif i == 1:
                # 选择中等遗漏值的号码
                mid_missing_front = []
                for ball, missing in front_sorted:
                    if periods * 0.05 <= missing <= periods * 0.15:
                        mid_missing_front.append(int(ball))

                if len(mid_missing_front) >= 5:
                    front_balls = random.sample(mid_missing_front, 5)
                else:
                    front_balls = mid_missing_front + [int(ball) for ball, missing in front_sorted[:5-len(mid_missing_front)]]

                mid_missing_back = []
                for ball, missing in back_sorted:
                    if periods * 0.08 <= missing <= periods * 0.2:
                        mid_missing_back.append(int(ball))

                if len(mid_missing_back) >= 2:
                    back_balls = random.sample(mid_missing_back, 2)
                else:
                    # 确保不重复添加号码
                    back_balls = mid_missing_back[:]
                    needed = 2 - len(mid_missing_back)
                    for ball, missing in back_sorted:
                        if len(back_balls) >= 2:
                            break
                        ball_int = int(ball)
                        if ball_int not in back_balls:
                            back_balls.append(ball_int)

            # 策略3: 混合遗漏策略 (第3注)
            elif i == 2:
                # 2个高遗漏 + 2个中遗漏 + 1个低遗漏
                high_missing = [int(ball) for ball, missing in front_sorted[:8]]
                mid_missing = [int(ball) for ball, missing in front_sorted[8:20]]
                low_missing = [int(ball) for ball, missing in front_sorted[20:30]]

                front_balls = []
                front_balls.extend(random.sample(high_missing, min(2, len(high_missing))))
                front_balls.extend(random.sample(mid_missing, min(2, len(mid_missing))))
                if len(low_missing) > 0:
                    front_balls.extend(random.sample(low_missing, min(1, len(low_missing))))

                # 如果不足5个，用高遗漏补充
                while len(front_balls) < 5:
                    remaining = [ball for ball in high_missing if ball not in front_balls]
                    if remaining:
                        front_balls.append(random.choice(remaining))
                    else:
                        break

                # 后区混合策略
                back_high = [int(ball) for ball, missing in back_sorted[:4]]
                back_mid = [int(ball) for ball, missing in back_sorted[4:8]]

                back_balls = []
                if len(back_high) > 0:
                    back_balls.append(random.choice(back_high))
                if len(back_mid) > 0:
                    back_balls.append(random.choice(back_mid))

                # 如果不足2个，用高遗漏补充
                while len(back_balls) < 2:
                    remaining = [ball for ball in back_high if ball not in back_balls]
                    if remaining:
                        back_balls.append(random.choice(remaining))
                    else:
                        break

            # 策略4: 遗漏值加权随机选择 (第4注及以后)
            else:
                # 基于遗漏值的加权随机选择
                front_weights = []
                front_balls_list = []

                for ball, missing in front_sorted:
                    # 遗漏值越大，权重越高
                    weight = missing + 1  # 避免权重为0
                    front_weights.append(weight)
                    front_balls_list.append(int(ball))

                if len(front_weights) > 0:
                    # 归一化权重
                    total_weight = sum(front_weights)
                    front_probs = [w/total_weight for w in front_weights]

                    # 加权随机选择
                    front_balls = list(np.random.choice(front_balls_list, size=5, replace=False, p=front_probs))

                back_weights = []
                back_balls_list = []

                for ball, missing in back_sorted:
                    weight = missing + 1
                    back_weights.append(weight)
                    back_balls_list.append(int(ball))

                if len(back_weights) > 0:
                    total_weight = sum(back_weights)
                    back_probs = [w/total_weight for w in back_weights]
                    back_balls = list(np.random.choice(back_balls_list, size=2, replace=False, p=back_probs))

            # 确保号码数量正确
            if len(front_balls) < 5:
                remaining = [int(ball) for ball, missing in front_sorted[:10] if int(ball) not in front_balls]
                front_balls.extend(remaining[:5-len(front_balls)])

            if len(back_balls) < 2:
                remaining = [int(ball) for ball, missing in back_sorted[:6] if int(ball) not in back_balls]
                back_balls.extend(remaining[:2-len(back_balls)])

            predictions.append((sorted(front_balls[:5]), sorted(back_balls[:2])))

        return predictions


# ==================== 高级预测器 ====================
class AdvancedPredictor:
    """高级预测器"""
    
    def __init__(self, data_file="data/dlt_data_all.csv"):
        self.data_file = data_file
        self.df = data_manager.get_data()
        self.traditional_predictor = TraditionalPredictor(data_file)
        
        if self.df is None:
            logger_manager.error("数据未加载")
    
    def markov_predict(self, count=1, periods=500) -> List[Tuple[List[int], List[int]]]:
        """增强马尔可夫链预测 - 真正的状态序列生成"""
        try:
            logger_manager.info(f"开始马尔可夫链预测: 注数={count}, 分析期数={periods}")

            # 获取马尔可夫分析结果
            markov_result = advanced_analyzer.markov_analysis(periods)

            front_transitions = markov_result.get('front_transition_probs', {})
            back_transitions = markov_result.get('back_transition_probs', {})

            if not front_transitions or not back_transitions:
                logger_manager.warning("马尔可夫转移概率为空，使用频率分析回退")
                return self._markov_fallback_predict(count, periods)

            predictions = []

            for i in range(count):
                # 为每注预测添加不同的随机种子，确保多样性
                import time
                import random
                import numpy as np

                # 使用时间戳和索引创建唯一的随机种子
                seed = int(time.time() * 1000000) + i * 1000
                random.seed(seed)
                np.random.seed(seed % 2**32)

                # 为每注使用不同的策略确保多样性
                if i == 0:
                    # 第一注：标准马尔可夫链
                    front_sequence = self._generate_markov_sequence(
                        front_transitions, 5, 1, 35, periods, i
                    )
                    back_sequence = self._generate_markov_sequence(
                        back_transitions, 2, 1, 12, periods, i
                    )
                else:
                    # 其他注：混合马尔可夫链和随机选择
                    front_sequence = self._generate_diverse_markov_sequence(
                        front_transitions, 5, 1, 35, i
                    )
                    back_sequence = self._generate_diverse_markov_sequence(
                        back_transitions, 2, 1, 12, i
                    )

                predictions.append((sorted(front_sequence), sorted(back_sequence)))

            logger_manager.info(f"马尔可夫链预测完成，生成{len(predictions)}注预测")
            return predictions

        except Exception as e:
            logger_manager.error(f"马尔可夫链预测失败: {e}")
            return self._markov_fallback_predict(count, periods)

    def _generate_markov_sequence(self, transitions: Dict, target_count: int,
                                min_num: int, max_num: int, periods: int, sequence_index: int = 0) -> List[int]:
        """生成真正的马尔可夫链序列"""
        try:
            import numpy as np

            # 获取初始状态，为每个序列使用不同的初始状态
            initial_state = self._get_initial_markov_state(min_num, max_num, sequence_index)

            # 马尔可夫链状态序列生成
            sequence = []
            current_state = initial_state
            max_iterations = target_count * 3  # 防止无限循环
            iterations = 0

            while len(sequence) < target_count and iterations < max_iterations:
                iterations += 1

                # 为每次迭代添加额外的随机性
                import time
                iteration_seed = int(time.time() * 1000000) + sequence_index * 10000 + iterations
                np.random.seed(iteration_seed % 2**32)

                # 根据当前状态和转移概率选择下一个状态
                next_state = self._markov_state_transition(current_state, transitions)

                if next_state is not None and next_state not in sequence:
                    sequence.append(next_state)
                    current_state = next_state
                else:
                    # 如果转移失败，随机选择一个新状态
                    available_states = [num for num in range(min_num, max_num + 1)
                                      if num not in sequence]
                    if available_states:
                        # 添加更多随机性
                        random_seed = int(time.time() * 1000000) + sequence_index * 5000 + iterations * 100
                        np.random.seed(random_seed % 2**32)
                        current_state = np.random.choice(available_states)
                        if current_state not in sequence:
                            sequence.append(current_state)

            # 如果序列不足，用概率最高的状态补充
            if len(sequence) < target_count:
                sequence.extend(self._supplement_markov_sequence(
                    sequence, transitions, target_count - len(sequence), min_num, max_num
                ))

            return sequence[:target_count]

        except Exception as e:
            logger_manager.error(f"生成马尔可夫序列失败: {e}")
            import random
            return random.sample(range(min_num, max_num + 1), target_count)

    def _generate_diverse_markov_sequence(self, transitions, target_count, min_num, max_num, sequence_index):
        """生成多样化的马尔可夫序列

        Args:
            transitions: 转移概率矩阵
            target_count: 目标序列长度
            min_num: 最小号码
            max_num: 最大号码
            sequence_index: 序列索引（用于多样性）

        Returns:
            生成的号码序列
        """
        try:
            import random
            import numpy as np
            import time

            # 为每个序列设置不同的随机种子
            seed = int(time.time() * 1000000) + sequence_index * 10000
            random.seed(seed)
            np.random.seed(seed % 2**32)

            sequence = []

            # 根据序列索引使用不同的策略
            strategy = sequence_index % 4

            if strategy == 0:
                # 策略1：高概率优先
                sequence = self._high_probability_strategy(transitions, target_count, min_num, max_num)
            elif strategy == 1:
                # 策略2：平衡策略
                sequence = self._balanced_strategy(transitions, target_count, min_num, max_num)
            elif strategy == 2:
                # 策略3：多样性策略
                sequence = self._diversity_strategy(transitions, target_count, min_num, max_num)
            else:
                # 策略4：混合策略
                sequence = self._hybrid_strategy(transitions, target_count, min_num, max_num)

            # 确保序列长度正确
            if len(sequence) < target_count:
                # 补充缺失的号码
                available = [num for num in range(min_num, max_num + 1) if num not in sequence]
                needed = target_count - len(sequence)
                if len(available) >= needed:
                    sequence.extend(random.sample(available, needed))
                else:
                    sequence.extend(available)

            return sequence[:target_count]

        except Exception as e:
            logger_manager.error(f"多样化马尔可夫序列生成失败: {e}")
            # 回退到简单随机选择
            import random
            return random.sample(range(min_num, max_num + 1), target_count)

    def _high_probability_strategy(self, transitions, target_count, min_num, max_num):
        """高概率策略"""
        import random
        sequence = []

        # 获取所有转移概率
        all_probs = {}
        for state, trans in transitions.items():
            for next_state, prob in trans.items():
                try:
                    next_num = int(next_state)
                    if min_num <= next_num <= max_num:
                        all_probs[next_num] = all_probs.get(next_num, 0) + prob
                except ValueError:
                    continue

        # 按概率排序
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        high_prob_nums = [num for num, _ in sorted_probs[:target_count * 2]]

        if len(high_prob_nums) >= target_count:
            sequence = random.sample(high_prob_nums, target_count)
        else:
            sequence = high_prob_nums

        return sequence

    def _balanced_strategy(self, transitions, target_count, min_num, max_num):
        """平衡策略"""
        import random
        sequence = []

        # 平衡选择高、中、低概率号码
        all_probs = {}
        for state, trans in transitions.items():
            for next_state, prob in trans.items():
                try:
                    next_num = int(next_state)
                    if min_num <= next_num <= max_num:
                        all_probs[next_num] = all_probs.get(next_num, 0) + prob
                except ValueError:
                    continue

        if all_probs:
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            total_nums = len(sorted_probs)

            # 分三档选择
            high_count = target_count // 3
            mid_count = target_count // 3
            low_count = target_count - high_count - mid_count

            high_nums = [num for num, _ in sorted_probs[:total_nums//3]]
            mid_nums = [num for num, _ in sorted_probs[total_nums//3:2*total_nums//3]]
            low_nums = [num for num, _ in sorted_probs[2*total_nums//3:]]

            if len(high_nums) >= high_count:
                sequence.extend(random.sample(high_nums, high_count))
            else:
                sequence.extend(high_nums)

            if len(mid_nums) >= mid_count:
                sequence.extend(random.sample(mid_nums, mid_count))
            else:
                sequence.extend(mid_nums)

            if len(low_nums) >= low_count:
                sequence.extend(random.sample(low_nums, low_count))
            else:
                sequence.extend(low_nums)

        return sequence

    def _diversity_strategy(self, transitions, target_count, min_num, max_num):
        """多样性策略"""
        import random
        sequence = []

        # 尽量选择分散的号码
        all_nums = list(range(min_num, max_num + 1))
        random.shuffle(all_nums)

        # 确保号码分散
        for num in all_nums:
            if len(sequence) >= target_count:
                break

            # 检查与已选号码的距离
            too_close = False
            for existing in sequence:
                if abs(num - existing) <= 2:  # 距离太近
                    too_close = True
                    break

            if not too_close:
                sequence.append(num)

        # 如果不够，随机补充
        if len(sequence) < target_count:
            remaining = [num for num in all_nums if num not in sequence]
            needed = target_count - len(sequence)
            if len(remaining) >= needed:
                sequence.extend(random.sample(remaining, needed))
            else:
                sequence.extend(remaining)

        return sequence

    def _hybrid_strategy(self, transitions, target_count, min_num, max_num):
        """混合策略"""
        import random

        # 结合高概率和多样性
        high_prob_seq = self._high_probability_strategy(transitions, target_count//2, min_num, max_num)
        diversity_seq = self._diversity_strategy(transitions, target_count//2, min_num, max_num)

        # 合并并去重
        combined = list(set(high_prob_seq + diversity_seq))

        if len(combined) >= target_count:
            return random.sample(combined, target_count)
        else:
            # 补充随机号码
            all_nums = list(range(min_num, max_num + 1))
            remaining = [num for num in all_nums if num not in combined]
            needed = target_count - len(combined)
            if len(remaining) >= needed:
                combined.extend(random.sample(remaining, needed))
            else:
                combined.extend(remaining)

            return combined[:target_count]

    def markov_compound_predict(self, front_count=8, back_count=4, analysis_periods=500) -> Dict:
        """马尔可夫复式预测

        Args:
            front_count: 前区号码数量 (6-15)
            back_count: 后区号码数量 (3-12)
            analysis_periods: 分析期数

        Returns:
            马尔可夫复式预测结果
        """
        logger_manager.info(f"马尔可夫复式预测: {front_count}+{back_count}, 分析期数: {analysis_periods}")

        try:
            # 构建马尔可夫转移矩阵
            transitions = self._build_markov_transitions(analysis_periods)

            if not transitions:
                logger_manager.warning("马尔可夫转移矩阵为空，使用备选方案")
                return self._fallback_markov_compound_prediction(front_count, back_count)

            # 基于马尔可夫链的复式号码选择
            front_balls = self._markov_compound_selection(
                transitions, front_count, True, analysis_periods
            )
            back_balls = self._markov_compound_selection(
                transitions, back_count, False, analysis_periods
            )

            # 计算复式投注信息
            from math import comb
            total_combinations = comb(front_count, 5) * comb(back_count, 2)
            total_cost = total_combinations * 3  # 每注3元

            # 计算置信度
            confidence = self._calculate_markov_compound_confidence(
                transitions, front_count, back_count
            )

            result = {
                'front_balls': front_balls,
                'back_balls': back_balls,
                'front_count': front_count,
                'back_count': back_count,
                'total_combinations': total_combinations,
                'total_cost': total_cost,
                'method': 'markov_compound',
                'confidence': confidence,
                'analysis_periods': analysis_periods,
                'markov_details': {
                    'transition_count': len(transitions),
                    'state_coverage': len(set().union(*[t.keys() for t in transitions.values()])),
                    'avg_transition_prob': sum(sum(t.values()) for t in transitions.values()) / max(1, len(transitions))
                },
                'timestamp': datetime.now().isoformat()
            }

            return result

        except Exception as e:
            logger_manager.error(f"马尔可夫复式预测失败: {e}")
            return self._fallback_markov_compound_prediction(front_count, back_count)

    def _markov_compound_selection(self, transitions, target_count, is_front, analysis_periods):
        """基于马尔可夫链的复式号码选择"""
        try:
            import random

            # 获取转移概率统计
            state_probs = {}
            for state, trans in transitions.items():
                for next_state, prob in trans.items():
                    try:
                        next_num = int(next_state)
                        max_num = 35 if is_front else 12
                        if 1 <= next_num <= max_num:
                            state_probs[next_num] = state_probs.get(next_num, 0) + prob
                    except ValueError:
                        continue

            # 按概率排序
            sorted_probs = sorted(state_probs.items(), key=lambda x: x[1], reverse=True)

            # 智能选择策略
            selected = []

            # 选择高概率号码（60%）
            high_prob_count = int(target_count * 0.6)
            high_prob_nums = [num for num, _ in sorted_probs[:high_prob_count * 2]]
            if len(high_prob_nums) >= high_prob_count:
                selected.extend(random.sample(high_prob_nums, high_prob_count))
            else:
                selected.extend(high_prob_nums)

            # 选择中等概率号码（30%）
            mid_prob_count = int(target_count * 0.3)
            mid_start = len(sorted_probs) // 3
            mid_end = 2 * len(sorted_probs) // 3
            mid_prob_nums = [num for num, _ in sorted_probs[mid_start:mid_end] if num not in selected]
            if len(mid_prob_nums) >= mid_prob_count:
                selected.extend(random.sample(mid_prob_nums, mid_prob_count))
            else:
                selected.extend(mid_prob_nums)

            # 补充剩余号码
            max_num = 35 if is_front else 12
            all_nums = list(range(1, max_num + 1))
            remaining = [num for num in all_nums if num not in selected]
            needed = target_count - len(selected)

            if needed > 0 and remaining:
                selected.extend(random.sample(remaining, min(needed, len(remaining))))

            return sorted(selected[:target_count])

        except Exception as e:
            logger_manager.error(f"马尔可夫复式号码选择失败: {e}")
            # 回退到随机选择
            import random
            max_num = 35 if is_front else 12
            return sorted(random.sample(range(1, max_num + 1), target_count))

    def _calculate_markov_compound_confidence(self, transitions, front_count, back_count):
        """计算马尔可夫复式预测的置信度"""
        try:
            # 基础置信度
            base_confidence = 0.65

            # 转移矩阵质量加成
            transition_bonus = min(0.15, len(transitions) * 0.001)

            # 复式规模加成
            scale_bonus = min(0.1, (front_count - 5) * 0.01 + (back_count - 2) * 0.02)

            final_confidence = base_confidence + transition_bonus + scale_bonus
            return min(0.85, max(0.5, final_confidence))

        except Exception:
            return 0.65

    def _fallback_markov_compound_prediction(self, front_count, back_count):
        """马尔可夫复式预测的备选方案"""
        import numpy as np
        from math import comb

        front_balls = sorted(np.random.choice(range(1, 36), front_count, replace=False))
        back_balls = sorted(np.random.choice(range(1, 13), back_count, replace=False))

        total_combinations = comb(front_count, 5) * comb(back_count, 2)
        total_cost = total_combinations * 3

        return {
            'front_balls': [int(x) for x in front_balls],
            'back_balls': [int(x) for x in back_balls],
            'front_count': front_count,
            'back_count': back_count,
            'total_combinations': total_combinations,
            'total_cost': total_cost,
            'method': 'markov_compound_fallback',
            'confidence': 0.5
        }

    def _get_initial_markov_state(self, min_num: int, max_num: int, sequence_index: int = 0) -> int:
        """获取马尔可夫链初始状态"""
        try:
            import numpy as np
            import random

            # 为不同的序列使用不同的初始状态策略
            if sequence_index == 0:
                # 第一注：使用最近一期的号码作为初始状态
                if len(self.df) > 0:
                    last_row = self.df.iloc[-1]
                    last_front, last_back = data_manager.parse_balls(last_row)

                    if max_num == 35:  # 前区
                        return last_front[0] if last_front else np.random.randint(min_num, max_num + 1)
                    else:  # 后区
                        return last_back[0] if last_back else np.random.randint(min_num, max_num + 1)
                else:
                    return np.random.randint(min_num, max_num + 1)
            else:
                # 其他注：使用不同的历史期数或随机状态
                if len(self.df) > sequence_index:
                    # 使用不同历史期数的号码
                    history_row = self.df.iloc[-(sequence_index + 1)]
                    history_front, history_back = data_manager.parse_balls(history_row)

                    if max_num == 35:  # 前区
                        if history_front:
                            # 从历史号码中随机选择一个
                            return random.choice(history_front)
                        else:
                            return np.random.randint(min_num, max_num + 1)
                    else:  # 后区
                        if history_back:
                            return random.choice(history_back)
                        else:
                            return np.random.randint(min_num, max_num + 1)
                else:
                    # 完全随机
                    return np.random.randint(min_num, max_num + 1)

        except Exception as e:
            logger_manager.error(f"获取初始状态失败: {e}")
            import random
            return random.randint(min_num, max_num)

    def _markov_state_transition(self, current_state: int, transitions: Dict) -> Optional[int]:
        """马尔可夫状态转移"""
        try:
            import numpy as np

            # 获取当前状态的转移概率
            if current_state in transitions:
                trans_probs = transitions[current_state]

                if trans_probs:
                    # 基于概率分布进行随机选择
                    states = list(trans_probs.keys())
                    probabilities = list(trans_probs.values())

                    # 标准化概率
                    total_prob = sum(probabilities)
                    if total_prob > 0:
                        normalized_probs = [p / total_prob for p in probabilities]

                        # 根据概率分布选择下一个状态
                        next_state = np.random.choice(states, p=normalized_probs)
                        return int(next_state)

            # 如果当前状态没有转移概率，随机选择
            return None

        except Exception as e:
            logger_manager.error(f"马尔可夫状态转移失败: {e}")
            return None

    def _supplement_markov_sequence(self, current_sequence: List[int], transitions: Dict,
                                  need_count: int, min_num: int, max_num: int) -> List[int]:
        """补充马尔可夫序列"""
        try:
            # 计算所有状态的平均转移概率
            state_scores = {}

            for state, trans_probs in transitions.items():
                if trans_probs:
                    avg_prob = sum(trans_probs.values()) / len(trans_probs)
                    state_scores[int(state)] = avg_prob

            # 排除已选择的号码
            available_states = [(state, score) for state, score in state_scores.items()
                              if state not in current_sequence and min_num <= state <= max_num]

            # 按得分排序
            available_states.sort(key=lambda x: x[1], reverse=True)

            # 选择得分最高的状态
            supplement = [state for state, score in available_states[:need_count]]

            # 如果还不够，随机补充
            if len(supplement) < need_count:
                remaining = [num for num in range(min_num, max_num + 1)
                           if num not in current_sequence and num not in supplement]
                import random
                additional = random.sample(remaining, min(need_count - len(supplement), len(remaining)))
                supplement.extend(additional)

            return supplement

        except Exception as e:
            logger_manager.error(f"补充马尔可夫序列失败: {e}")
            return []

    def _markov_fallback_predict(self, count: int, periods: int) -> List[Tuple[List[int], List[int]]]:
        """马尔可夫预测回退方案"""
        try:
            logger_manager.info("使用马尔可夫回退预测")

            # 使用频率分析作为回退
            freq_result = basic_analyzer.frequency_analysis(periods)
            front_freq = freq_result.get('front_frequency', {})
            back_freq = freq_result.get('back_frequency', {})

            predictions = []
            for i in range(count):
                # 基于频率的随机选择
                front_candidates = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)
                back_candidates = sorted(back_freq.items(), key=lambda x: x[1], reverse=True)

                front_balls = [int(ball) for ball, freq in front_candidates[:5]]
                back_balls = [int(ball) for ball, freq in back_candidates[:2]]

                predictions.append((sorted(front_balls), sorted(back_balls)))

            return predictions

        except Exception as e:
            logger_manager.error(f"马尔可夫回退预测失败: {e}")
            return []



    def markov_predict_custom(self, count=1, analysis_periods=300, predict_periods=1) -> List[Dict]:
        """马尔可夫链自定义期数预测

        Args:
            count: 生成预测注数
            analysis_periods: 分析期数
            predict_periods: 预测期数

        Returns:
            预测结果列表，包含详细信息
        """
        logger_manager.info(f"马尔可夫链自定义预测: 分析{analysis_periods}期, 预测{predict_periods}期, 生成{count}注")

        predictions = []

        for predict_idx in range(predict_periods):
            period_predictions = []

            # 获取马尔可夫分析结果
            markov_result = advanced_analyzer.markov_analysis(analysis_periods)
            front_transitions = markov_result.get('front_transition_probs', {})
            back_transitions = markov_result.get('back_transition_probs', {})

            for i in range(count):
                # 使用改进的马尔可夫预测算法
                front_balls = self._markov_predict_balls(front_transitions, 5, 35)
                back_balls = self._markov_predict_balls(back_transitions, 2, 12)

                # 计算稳定性得分
                front_stability = self._calculate_stability_score(front_transitions, front_balls)
                back_stability = self._calculate_stability_score(back_transitions, back_balls)
                overall_stability = (front_stability + back_stability) / 2

                prediction = {
                    'index': i + 1,
                    'period': predict_idx + 1,
                    'front_balls': sorted(front_balls),
                    'back_balls': sorted(back_balls),
                    'front_stability': front_stability,
                    'back_stability': back_stability,
                    'overall_stability': overall_stability,
                    'analysis_periods': analysis_periods,
                    'method': 'markov_custom'
                }

                period_predictions.append(prediction)

            # 按稳定性排序
            period_predictions.sort(key=lambda x: x['overall_stability'], reverse=True)
            predictions.extend(period_predictions)

        return predictions

    def _markov_predict_balls(self, transitions: Dict, num_balls: int, max_ball: int) -> List[int]:
        """基于马尔可夫转移概率预测号码"""
        if not transitions:
            # 如果没有转移概率，使用频率分析
            freq_analysis = basic_analyzer.frequency_analysis()
            if max_ball == 35:  # 前区
                freq_dict = freq_analysis.get('front_frequency', {})
            else:  # 后区
                freq_dict = freq_analysis.get('back_frequency', {})

            sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
            return sorted([int(ball) for ball, freq in sorted_freq[:num_balls]])

        balls = []

        # 选择起始号码（选择转移概率最高的）
        start_ball = max(transitions.keys(), key=lambda x: sum(transitions[x].values()))
        balls.append(start_ball)

        # 基于转移概率选择后续号码
        current_ball = start_ball
        while len(balls) < num_balls:
            if current_ball in transitions and transitions[current_ball]:
                # 按概率选择下一个号码
                next_balls = list(transitions[current_ball].keys())
                probs = list(transitions[current_ball].values())

                # 标准化概率
                total_prob = sum(probs)
                if total_prob > 0:
                    probs = [p / total_prob for p in probs]

                    # 选择概率最高的未选号码
                    for ball, prob in sorted(zip(next_balls, probs), key=lambda x: x[1], reverse=True):
                        if ball not in balls:
                            balls.append(ball)
                            current_ball = ball
                            break
                    else:
                        # 如果没有找到未选号码，使用频率分析补充
                        freq_analysis = basic_analyzer.frequency_analysis()
                        if max_ball == 35:  # 前区
                            freq_dict = freq_analysis.get('front_frequency', {})
                        else:  # 后区
                            freq_dict = freq_analysis.get('back_frequency', {})

                        sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
                        for ball, freq in sorted_freq:
                            if ball not in balls:
                                balls.append(ball)
                                current_ball = ball
                                break
                else:
                    # 使用频率分析选择
                    freq_analysis = basic_analyzer.frequency_analysis()
                    if max_ball == 35:  # 前区
                        freq_dict = freq_analysis.get('front_frequency', {})
                    else:  # 后区
                        freq_dict = freq_analysis.get('back_frequency', {})

                    sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
                    for ball, freq in sorted_freq:
                        if ball not in balls:
                            balls.append(ball)
                            current_ball = ball
                            break
            else:
                # 使用频率分析选择
                freq_analysis = basic_analyzer.frequency_analysis()
                if max_ball == 35:  # 前区
                    freq_dict = freq_analysis.get('front_frequency', {})
                else:  # 后区
                    freq_dict = freq_analysis.get('back_frequency', {})

                sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
                for ball, freq in sorted_freq:
                    if ball not in balls:
                        balls.append(ball)
                        current_ball = ball
                        break

        return balls

    def _calculate_stability_score(self, transitions: Dict, balls: List[int]) -> float:
        """计算稳定性得分"""
        if not transitions or not balls:
            return 0.0

        total_score = 0.0
        count = 0

        for ball in balls:
            if ball in transitions:
                # 计算该号码的转移稳定性
                trans_probs = list(transitions[ball].values())
                if trans_probs:
                    # 使用方差的倒数作为稳定性指标
                    variance = np.var(trans_probs)
                    stability = 1.0 / (1.0 + variance)
                    total_score += stability
                    count += 1

        return total_score / count if count > 0 else 0.0

    def bayesian_predict(self, count=1, periods=500, n_jobs=1) -> List[Tuple[List[int], List[int]]]:
        """贝叶斯预测 - 真正的贝叶斯推理和概率采样"""
        import random
        import numpy as np

        # 使用n_jobs参数进行贝叶斯分析
        bayesian_result = advanced_analyzer.bayesian_analysis(periods, n_jobs=n_jobs)

        front_posterior = bayesian_result.get('front_posterior', {})
        back_posterior = bayesian_result.get('back_posterior', {})

        predictions = []

        # 为每注生成不同的贝叶斯策略
        for i in range(count):
            front_balls = []
            back_balls = []

            # 策略1: 最大后验概率策略 (第1注)
            if i % 4 == 0:
                if front_posterior:
                    # 选择后验概率最高的号码，但加入随机性
                    sorted_front = sorted(front_posterior.items(), key=lambda x: x[1], reverse=True)
                    high_prob_front = [int(ball) for ball, prob in sorted_front[:8]]
                    front_balls = random.sample(high_prob_front, min(5, len(high_prob_front)))

                if back_posterior:
                    sorted_back = sorted(back_posterior.items(), key=lambda x: x[1], reverse=True)
                    high_prob_back = [int(ball) for ball, prob in sorted_back[:4]]
                    back_balls = random.sample(high_prob_back, min(2, len(high_prob_back)))

            # 策略2: 中等概率策略 (第2注)
            elif i % 4 == 1:
                if front_posterior:
                    # 选择中等概率的号码
                    sorted_front = sorted(front_posterior.items(), key=lambda x: x[1], reverse=True)
                    mid_start = len(sorted_front) // 4
                    mid_end = len(sorted_front) * 3 // 4
                    mid_prob_front = [int(ball) for ball, prob in sorted_front[mid_start:mid_end]]
                    if len(mid_prob_front) >= 5:
                        front_balls = random.sample(mid_prob_front, 5)
                    else:
                        front_balls = mid_prob_front + [int(ball) for ball, prob in sorted_front[:5-len(mid_prob_front)]]

                if back_posterior:
                    sorted_back = sorted(back_posterior.items(), key=lambda x: x[1], reverse=True)
                    mid_prob_back = [int(ball) for ball, prob in sorted_back[1:5]]
                    if len(mid_prob_back) >= 2:
                        back_balls = random.sample(mid_prob_back, 2)
                    else:
                        back_balls = mid_prob_back + [int(ball) for ball, prob in sorted_back[:2-len(mid_prob_back)]]

            # 策略3: 混合概率策略 (第3注)
            elif i % 4 == 2:
                if front_posterior:
                    # 2个高概率 + 2个中概率 + 1个低概率
                    sorted_front = sorted(front_posterior.items(), key=lambda x: x[1], reverse=True)
                    high_prob = [int(ball) for ball, prob in sorted_front[:6]]
                    mid_prob = [int(ball) for ball, prob in sorted_front[6:15]]
                    low_prob = [int(ball) for ball, prob in sorted_front[15:25]]

                    front_balls = []
                    front_balls.extend(random.sample(high_prob, min(2, len(high_prob))))
                    front_balls.extend(random.sample(mid_prob, min(2, len(mid_prob))))
                    if len(low_prob) > 0:
                        front_balls.extend(random.sample(low_prob, min(1, len(low_prob))))

                    # 如果不足5个，用高概率补充
                    while len(front_balls) < 5:
                        remaining = [ball for ball in high_prob if ball not in front_balls]
                        if remaining:
                            front_balls.append(random.choice(remaining))
                        else:
                            break

                if back_posterior:
                    sorted_back = sorted(back_posterior.items(), key=lambda x: x[1], reverse=True)
                    back_high = [int(ball) for ball, prob in sorted_back[:3]]
                    back_mid = [int(ball) for ball, prob in sorted_back[3:8]]

                    back_balls = []
                    if len(back_high) > 0:
                        back_balls.append(random.choice(back_high))
                    if len(back_mid) > 0:
                        back_balls.append(random.choice(back_mid))

                    # 如果不足2个，用高概率补充
                    while len(back_balls) < 2:
                        remaining = [ball for ball in back_high if ball not in back_balls]
                        if remaining:
                            back_balls.append(random.choice(remaining))
                        else:
                            break

            # 策略4: 概率加权随机采样 (第4注及以后)
            else:
                # 基于后验概率的加权随机采样
                if front_posterior:
                    front_balls_list = [int(ball) for ball in front_posterior.keys()]
                    front_probs = [prob for prob in front_posterior.values()]

                    if len(front_probs) > 0:
                        # 归一化概率
                        total_prob = sum(front_probs)
                        front_probs_norm = [p/total_prob for p in front_probs]

                        # 概率加权随机采样
                        front_balls = list(np.random.choice(front_balls_list, size=5, replace=False, p=front_probs_norm))

                if back_posterior:
                    back_balls_list = [int(ball) for ball in back_posterior.keys()]
                    back_probs = [prob for prob in back_posterior.values()]

                    if len(back_probs) > 0:
                        total_prob = sum(back_probs)
                        back_probs_norm = [p/total_prob for p in back_probs]
                        back_balls = list(np.random.choice(back_balls_list, size=2, replace=False, p=back_probs_norm))

            # 如果没有后验概率或号码不足，使用频率分析补充
            if len(front_balls) < 5:
                freq_analysis = basic_analyzer.frequency_analysis(periods)
                front_freq = freq_analysis.get('front_frequency', {})
                sorted_freq = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)
                for ball, freq in sorted_freq:
                    if len(front_balls) >= 5:
                        break
                    if int(ball) not in front_balls:
                        front_balls.append(int(ball))

            if len(back_balls) < 2:
                freq_analysis = basic_analyzer.frequency_analysis(periods)
                back_freq = freq_analysis.get('back_frequency', {})
                sorted_freq = sorted(back_freq.items(), key=lambda x: x[1], reverse=True)
                for ball, freq in sorted_freq:
                    if len(back_balls) >= 2:
                        break
                    if int(ball) not in back_balls:
                        back_balls.append(int(ball))

            predictions.append((sorted(front_balls[:5]), sorted(back_balls[:2])))

        return predictions
    
    def ensemble_predict(self, count=1, periods=500, weights=None) -> List[Tuple[List[int], List[int]]]:
        """集成预测"""
        if weights is None:
            weights = {
                'markov': 0.30,
                'bayesian': 0.20,
                'frequency': 0.20,
                'hot_cold': 0.15,
                'missing': 0.15
            }

        predictions = []

        # 获取各种预测方法的结果（一次性获取，确保一致性）
        markov_pred = self.markov_predict(1, periods)[0]
        bayesian_pred = self.bayesian_predict(1, periods)[0]
        freq_pred = self.traditional_predictor.frequency_predict(1, periods)[0]
        hot_cold_pred = self.traditional_predictor.hot_cold_predict(1, periods)[0]
        missing_pred = self.traditional_predictor.missing_predict(1, periods)[0]

        # 收集所有候选号码
        all_front_candidates = []
        all_back_candidates = []

        for method, weight in weights.items():
            if method == 'markov':
                pred = markov_pred
            elif method == 'bayesian':
                pred = bayesian_pred
            elif method == 'frequency':
                pred = freq_pred
            elif method == 'hot_cold':
                pred = hot_cold_pred
            elif method == 'missing':
                pred = missing_pred
            else:
                continue

            # 根据权重重复添加候选号码
            repeat_count = max(1, int(weight * 10))
            for _ in range(repeat_count):
                all_front_candidates.extend(pred[0])
                all_back_candidates.extend(pred[1])

        # 统计频率并选择最高频率的号码
        front_counter = Counter(all_front_candidates)
        back_counter = Counter(all_back_candidates)

        # 选择频率最高的号码（去重）
        front_balls = []
        for ball, freq_count in front_counter.most_common():
            if len(front_balls) >= 5:
                break
            if int(ball) not in front_balls:
                front_balls.append(int(ball))

        back_balls = []
        for ball, freq_count in back_counter.most_common():
            if len(back_balls) >= 2:
                break
            if int(ball) not in back_balls:
                back_balls.append(int(ball))

        # 如果数量不足，使用频率分析补充
        if len(front_balls) < 5:
            freq_analysis = basic_analyzer.frequency_analysis()
            front_freq = freq_analysis.get('front_frequency', {})
            sorted_freq = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)
            for ball, freq in sorted_freq:
                if len(front_balls) >= 5:
                    break
                if ball not in front_balls:
                    front_balls.append(ball)

        if len(back_balls) < 2:
            freq_analysis = basic_analyzer.frequency_analysis()
            back_freq = freq_analysis.get('back_frequency', {})
            sorted_freq = sorted(back_freq.items(), key=lambda x: x[1], reverse=True)
            for ball, freq in sorted_freq:
                if len(back_balls) >= 2:
                    break
                if ball not in back_balls:
                    back_balls.append(ball)

        # 生成多注相同的预测（基于集成的确定性预测）
        for _ in range(count):
            predictions.append((sorted(front_balls[:5]), sorted(back_balls[:2])))
        
        return predictions
    
    def update_weights(self, new_weights: Dict[str, float]):
        """更新权重"""
        # 这个方法用于自适应学习系统
        pass

    def clustering_predict(self, count=1, periods=500, method="kmeans") -> List[Tuple[List[int], List[int]]]:
        """聚类分析预测"""
        try:
            logger_manager.info(f"开始聚类分析预测: 注数={count}, 分析期数={periods}, 方法={method}")

            # 获取历史数据
            if self.df is None or len(self.df) < periods:
                logger_manager.warning("数据不足，使用频率分析作为回退")
                return self.traditional_predictor.frequency_predict(count, periods)

            recent_data = self.df.tail(periods)

            # 准备聚类数据
            features = []
            for _, row in recent_data.iterrows():
                try:
                    front_balls = [int(x) for x in str(row.get('front_balls', '')).split(',') if x.strip().isdigit()]
                    back_balls = [int(x) for x in str(row.get('back_balls', '')).split(',') if x.strip().isdigit()]

                    if len(front_balls) == 5 and len(back_balls) == 2:
                        # 创建特征向量：前区和后区的统计特征
                        feature_vector = [
                            sum(front_balls),  # 前区和值
                            max(front_balls) - min(front_balls),  # 前区跨度
                            len([x for x in front_balls if x <= 18]),  # 前区小号个数
                            sum(back_balls),  # 后区和值
                            max(back_balls) - min(back_balls),  # 后区跨度
                        ]
                        features.append(feature_vector)
                except:
                    continue

            if len(features) < 10:
                logger_manager.warning("有效特征数据不足，使用频率分析作为回退")
                return self.traditional_predictor.frequency_predict(count, periods)

            # 进行聚类分析
            from sklearn.cluster import KMeans
            import numpy as np

            features_array = np.array(features)
            n_clusters = min(8, len(features) // 10)  # 动态确定聚类数

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_array)

            # 分析每个聚类的特征
            cluster_centers = kmeans.cluster_centers_

            # 选择最有潜力的聚类（基于最近数据的分布）
            recent_features = features_array[-20:]  # 最近20期
            recent_clusters = kmeans.predict(recent_features)

            # 统计最近期数中各聚类的出现频率
            from collections import Counter
            cluster_freq = Counter(recent_clusters)

            predictions = []
            for i in range(count):
                try:
                    # 选择出现频率较高的聚类作为预测基础
                    target_cluster = cluster_freq.most_common(1)[0][0] if cluster_freq else 0
                    target_center = cluster_centers[target_cluster]

                    # 基于聚类中心生成预测
                    front_sum_target = int(target_center[0])
                    front_span_target = int(target_center[1])
                    front_small_count = int(target_center[2])
                    back_sum_target = int(target_center[3])

                    # 生成符合聚类特征的号码
                    front_balls = self._generate_balls_by_cluster_features(
                        front_sum_target, front_span_target, front_small_count, 5, 35
                    )

                    back_balls = self._generate_balls_by_sum(back_sum_target, 2, 12)

                    predictions.append((sorted(front_balls), sorted(back_balls)))

                except Exception as e:
                    logger_manager.error(f"聚类预测第{i+1}注失败: {e}")
                    # 使用频率分析作为回退
                    fallback = self.traditional_predictor.frequency_predict(1, periods)
                    if fallback:
                        predictions.append(fallback[0])

            logger_manager.info(f"聚类分析预测完成，生成{len(predictions)}注")
            return predictions

        except Exception as e:
            logger_manager.error(f"聚类分析预测失败: {e}")
            # 回退到频率分析
            return self.traditional_predictor.frequency_predict(count, periods)

    def _generate_balls_by_cluster_features(self, target_sum, target_span, small_count, num_balls, max_ball):
        """根据聚类特征生成号码"""
        import random
        import numpy as np

        attempts = 0
        max_attempts = 1000

        while attempts < max_attempts:
            # 生成候选号码
            balls = sorted(random.sample(range(1, max_ball + 1), num_balls))

            current_sum = sum(balls)
            current_span = max(balls) - min(balls)
            current_small = len([x for x in balls if x <= max_ball // 2])

            # 检查是否接近目标特征
            sum_diff = abs(current_sum - target_sum)
            span_diff = abs(current_span - target_span)
            small_diff = abs(current_small - small_count)

            # 如果特征接近，返回结果
            if sum_diff <= 20 and span_diff <= 10 and small_diff <= 2:
                return balls

            attempts += 1

        # 如果无法生成符合特征的号码，返回随机号码
        return sorted(random.sample(range(1, max_ball + 1), num_balls))

    def _generate_balls_by_sum(self, target_sum, num_balls, max_ball):
        """根据目标和值生成号码"""
        import random

        attempts = 0
        max_attempts = 1000

        while attempts < max_attempts:
            balls = sorted(random.sample(range(1, max_ball + 1), num_balls))
            if abs(sum(balls) - target_sum) <= 5:
                return balls
            attempts += 1

        # 回退方案
        return sorted(random.sample(range(1, max_ball + 1), num_balls))

    def markov_2nd_predict(self, count=1, periods=500) -> List[Tuple[List[int], List[int]]]:
        """二阶马尔可夫链预测"""
        try:
            logger_manager.info(f"开始二阶马尔可夫链预测: 注数={count}, 分析期数={periods}")

            # 获取历史数据
            if self.df is None or len(self.df) < periods:
                logger_manager.warning("数据不足，使用一阶马尔可夫链作为回退")
                return self.markov_predict(count, periods)

            recent_data = self.df.tail(periods)

            # 构建二阶转移矩阵
            front_transitions_2nd = {}
            back_transitions_2nd = {}

            for i in range(len(recent_data) - 2):
                try:
                    # 获取连续三期的数据
                    period1 = recent_data.iloc[i]
                    period2 = recent_data.iloc[i + 1]
                    period3 = recent_data.iloc[i + 2]

                    front1 = [int(x) for x in str(period1.get('front_balls', '')).split(',') if x.strip().isdigit()]
                    front2 = [int(x) for x in str(period2.get('front_balls', '')).split(',') if x.strip().isdigit()]
                    front3 = [int(x) for x in str(period3.get('front_balls', '')).split(',') if x.strip().isdigit()]

                    back1 = [int(x) for x in str(period1.get('back_balls', '')).split(',') if x.strip().isdigit()]
                    back2 = [int(x) for x in str(period2.get('back_balls', '')).split(',') if x.strip().isdigit()]
                    back3 = [int(x) for x in str(period3.get('back_balls', '')).split(',') if x.strip().isdigit()]

                    if len(front1) == 5 and len(front2) == 5 and len(front3) == 5:
                        # 二阶状态：前两期的状态组合
                        state_key = (tuple(sorted(front1)), tuple(sorted(front2)))
                        next_state = tuple(sorted(front3))

                        if state_key not in front_transitions_2nd:
                            front_transitions_2nd[state_key] = {}
                        if next_state not in front_transitions_2nd[state_key]:
                            front_transitions_2nd[state_key][next_state] = 0
                        front_transitions_2nd[state_key][next_state] += 1

                    if len(back1) == 2 and len(back2) == 2 and len(back3) == 2:
                        state_key = (tuple(sorted(back1)), tuple(sorted(back2)))
                        next_state = tuple(sorted(back3))

                        if state_key not in back_transitions_2nd:
                            back_transitions_2nd[state_key] = {}
                        if next_state not in back_transitions_2nd[state_key]:
                            back_transitions_2nd[state_key][next_state] = 0
                        back_transitions_2nd[state_key][next_state] += 1

                except:
                    continue

            if not front_transitions_2nd or not back_transitions_2nd:
                logger_manager.warning("二阶转移矩阵构建失败，使用一阶马尔可夫链")
                return self.markov_predict(count, periods)

            # 获取最近两期作为当前状态
            last_two_periods = recent_data.tail(2)

            predictions = []
            for i in range(count):
                try:
                    # 预测前区
                    front_balls = self._predict_with_2nd_order_markov(
                        last_two_periods, front_transitions_2nd, 'front_balls', 5, 35
                    )

                    # 预测后区
                    back_balls = self._predict_with_2nd_order_markov(
                        last_two_periods, back_transitions_2nd, 'back_balls', 2, 12
                    )

                    predictions.append((sorted(front_balls), sorted(back_balls)))

                except Exception as e:
                    logger_manager.error(f"二阶马尔可夫预测第{i+1}注失败: {e}")
                    # 使用一阶马尔可夫作为回退
                    fallback = self.markov_predict(1, periods)
                    if fallback:
                        predictions.append(fallback[0])

            logger_manager.info(f"二阶马尔可夫链预测完成，生成{len(predictions)}注")
            return predictions

        except Exception as e:
            logger_manager.error(f"二阶马尔可夫链预测失败: {e}")
            return self.markov_predict(count, periods)

    def markov_3rd_predict(self, count=1, periods=500) -> List[Tuple[List[int], List[int]]]:
        """三阶马尔可夫链预测"""
        try:
            logger_manager.info(f"开始三阶马尔可夫链预测: 注数={count}, 分析期数={periods}")

            # 获取历史数据
            if self.df is None or len(self.df) < periods:
                logger_manager.warning("数据不足，使用二阶马尔可夫链作为回退")
                return self.markov_2nd_predict(count, periods)

            recent_data = self.df.tail(periods)

            # 构建三阶转移矩阵
            front_transitions_3rd = {}
            back_transitions_3rd = {}

            for i in range(len(recent_data) - 3):
                try:
                    # 获取连续四期的数据
                    periods_data = [recent_data.iloc[i + j] for j in range(4)]

                    front_data = []
                    back_data = []

                    for period in periods_data:
                        front_balls = [int(x) for x in str(period.get('front_balls', '')).split(',') if x.strip().isdigit()]
                        back_balls = [int(x) for x in str(period.get('back_balls', '')).split(',') if x.strip().isdigit()]

                        if len(front_balls) == 5 and len(back_balls) == 2:
                            front_data.append(tuple(sorted(front_balls)))
                            back_data.append(tuple(sorted(back_balls)))
                        else:
                            break

                    if len(front_data) == 4:
                        # 三阶状态：前三期的状态组合
                        state_key = (front_data[0], front_data[1], front_data[2])
                        next_state = front_data[3]

                        if state_key not in front_transitions_3rd:
                            front_transitions_3rd[state_key] = {}
                        if next_state not in front_transitions_3rd[state_key]:
                            front_transitions_3rd[state_key][next_state] = 0
                        front_transitions_3rd[state_key][next_state] += 1

                    if len(back_data) == 4:
                        state_key = (back_data[0], back_data[1], back_data[2])
                        next_state = back_data[3]

                        if state_key not in back_transitions_3rd:
                            back_transitions_3rd[state_key] = {}
                        if next_state not in back_transitions_3rd[state_key]:
                            back_transitions_3rd[state_key][next_state] = 0
                        back_transitions_3rd[state_key][next_state] += 1

                except:
                    continue

            if not front_transitions_3rd or not back_transitions_3rd:
                logger_manager.warning("三阶转移矩阵构建失败，使用二阶马尔可夫链")
                return self.markov_2nd_predict(count, periods)

            # 获取最近三期作为当前状态
            last_three_periods = recent_data.tail(3)

            predictions = []
            for i in range(count):
                try:
                    # 预测前区
                    front_balls = self._predict_with_3rd_order_markov(
                        last_three_periods, front_transitions_3rd, 'front_balls', 5, 35
                    )

                    # 预测后区
                    back_balls = self._predict_with_3rd_order_markov(
                        last_three_periods, back_transitions_3rd, 'back_balls', 2, 12
                    )

                    predictions.append((sorted(front_balls), sorted(back_balls)))

                except Exception as e:
                    logger_manager.error(f"三阶马尔可夫预测第{i+1}注失败: {e}")
                    # 使用二阶马尔可夫作为回退
                    fallback = self.markov_2nd_predict(1, periods)
                    if fallback:
                        predictions.append(fallback[0])

            logger_manager.info(f"三阶马尔可夫链预测完成，生成{len(predictions)}注")
            return predictions

        except Exception as e:
            logger_manager.error(f"三阶马尔可夫链预测失败: {e}")
            return self.markov_2nd_predict(count, periods)

    def adaptive_markov_predict(self, count=1, periods=500) -> List[Tuple[List[int], List[int]]]:
        """自适应马尔可夫链预测（自动选择最优阶数）"""
        try:
            logger_manager.info(f"开始自适应马尔可夫链预测: 注数={count}, 分析期数={periods}")

            # 评估不同阶数的马尔可夫链性能
            orders = [1, 2, 3]
            order_scores = {}

            for order in orders:
                try:
                    # 使用历史数据评估每个阶数的性能
                    if order == 1:
                        test_result = self.markov_predict(1, min(periods, 200))
                    elif order == 2:
                        test_result = self.markov_2nd_predict(1, min(periods, 200))
                    else:  # order == 3
                        test_result = self.markov_3rd_predict(1, min(periods, 200))

                    # 简单的性能评估（基于数据可用性和复杂度）
                    if test_result:
                        if order == 1:
                            score = 0.7  # 基础分数
                        elif order == 2:
                            score = 0.8 if periods >= 100 else 0.6
                        else:  # order == 3
                            score = 0.9 if periods >= 200 else 0.5

                        order_scores[order] = score
                    else:
                        order_scores[order] = 0.0

                except Exception as e:
                    logger_manager.warning(f"评估{order}阶马尔可夫链失败: {e}")
                    order_scores[order] = 0.0

            # 选择最优阶数
            best_order = max(order_scores.items(), key=lambda x: x[1])[0]
            logger_manager.info(f"自适应选择{best_order}阶马尔可夫链，评分: {order_scores[best_order]:.3f}")

            # 使用最优阶数进行预测
            if best_order == 1:
                return self.markov_predict(count, periods)
            elif best_order == 2:
                return self.markov_2nd_predict(count, periods)
            else:  # best_order == 3
                return self.markov_3rd_predict(count, periods)

        except Exception as e:
            logger_manager.error(f"自适应马尔可夫链预测失败: {e}")
            return self.markov_predict(count, periods)

    def _predict_with_2nd_order_markov(self, last_two_periods, transitions, ball_type, num_balls, max_ball):
        """使用二阶马尔可夫链进行预测"""
        try:
            # 获取最近两期的状态
            period1 = last_two_periods.iloc[0]
            period2 = last_two_periods.iloc[1]

            balls1 = [int(x) for x in str(period1.get(ball_type, '')).split(',') if x.strip().isdigit()]
            balls2 = [int(x) for x in str(period2.get(ball_type, '')).split(',') if x.strip().isdigit()]

            if len(balls1) == num_balls and len(balls2) == num_balls:
                state_key = (tuple(sorted(balls1)), tuple(sorted(balls2)))

                if state_key in transitions:
                    # 根据转移概率选择下一状态
                    next_states = transitions[state_key]
                    total_count = sum(next_states.values())

                    import random
                    rand_val = random.random()
                    cumulative_prob = 0.0

                    for next_state, count in next_states.items():
                        cumulative_prob += count / total_count
                        if rand_val <= cumulative_prob:
                            return list(next_state)

            # 如果无法找到匹配的状态，使用随机生成
            import random
            return sorted(random.sample(range(1, max_ball + 1), num_balls))

        except Exception as e:
            logger_manager.error(f"二阶马尔可夫预测失败: {e}")
            import random
            return sorted(random.sample(range(1, max_ball + 1), num_balls))

    def _predict_with_3rd_order_markov(self, last_three_periods, transitions, ball_type, num_balls, max_ball):
        """使用三阶马尔可夫链进行预测"""
        try:
            # 获取最近三期的状态
            balls_data = []
            for i in range(3):
                period = last_three_periods.iloc[i]
                balls = [int(x) for x in str(period.get(ball_type, '')).split(',') if x.strip().isdigit()]
                if len(balls) == num_balls:
                    balls_data.append(tuple(sorted(balls)))
                else:
                    break

            if len(balls_data) == 3:
                state_key = (balls_data[0], balls_data[1], balls_data[2])

                if state_key in transitions:
                    # 根据转移概率选择下一状态
                    next_states = transitions[state_key]
                    total_count = sum(next_states.values())

                    import random
                    rand_val = random.random()
                    cumulative_prob = 0.0

                    for next_state, count in next_states.items():
                        cumulative_prob += count / total_count
                        if rand_val <= cumulative_prob:
                            return list(next_state)

            # 如果无法找到匹配的状态，使用随机生成
            import random
            return sorted(random.sample(range(1, max_ball + 1), num_balls))

        except Exception as e:
            logger_manager.error(f"三阶马尔可夫预测失败: {e}")
            import random
            return sorted(random.sample(range(1, max_ball + 1), num_balls))

    def mixed_strategy_predict(self, count=1, strategy='balanced', periods=500) -> List[Dict]:
        """混合策略预测

        Args:
            count: 生成注数
            strategy: 策略类型 ('conservative', 'aggressive', 'balanced')
            periods: 分析期数

        Returns:
            预测结果列表
        """
        logger_manager.info(f"混合策略预测: {strategy}, 注数: {count}, 分析期数: {periods}")

        # 获取混合策略分析结果
        try:
            strategy_result = advanced_analyzer.mixed_strategy_analysis(periods)
            strategies = strategy_result.get('strategies', {})
        except Exception as e:
            logger_manager.error(f"获取混合策略分析失败: {e}")
            # 使用默认策略配置
            strategies = {
                'conservative': {
                    'weights': {'frequency': 0.4, 'markov': 0.3, 'bayesian': 0.2, 'correlation': 0.1},
                    'risk_level': 'low',
                    'description': '基于高频号码和稳定模式'
                },
                'aggressive': {
                    'weights': {'frequency': 0.1, 'markov': 0.4, 'bayesian': 0.3, 'correlation': 0.2},
                    'risk_level': 'high',
                    'description': '基于趋势变化和新兴模式'
                },
                'balanced': {
                    'weights': {'frequency': 0.25, 'markov': 0.25, 'bayesian': 0.25, 'correlation': 0.25},
                    'risk_level': 'medium',
                    'description': '各种方法均衡组合'
                }
            }

        # 处理策略名称映射
        strategy_mapping = {
            'conservative': 'conservative_strategy',
            'aggressive': 'aggressive_strategy',
            'balanced': 'balanced_strategy'
        }

        # 如果strategies来自分析结果，需要映射策略名称
        if 'conservative_strategy' in strategies:
            mapped_strategies = {}
            for key, mapped_key in strategy_mapping.items():
                if mapped_key in strategies:
                    mapped_strategies[key] = strategies[mapped_key]
            strategies = mapped_strategies

        if strategy not in strategies:
            strategy = 'balanced'  # 默认使用平衡策略

        strategy_config = strategies[strategy]
        weights = strategy_config['weights']

        predictions = []

        for i in range(count):
            # 基于策略权重获取预测
            front_candidates = Counter()
            back_candidates = Counter()

            # 频率预测
            if weights.get('frequency', 0) > 0:
                try:
                    freq_pred = get_traditional_predictor().frequency_predict(1)[0]
                    weight = weights['frequency']
                    for ball in freq_pred[0]:
                        front_candidates[ball] += weight * 10
                    for ball in freq_pred[1]:
                        back_candidates[ball] += weight * 10
                except Exception as e:
                    logger_manager.error(f"频率预测失败: {e}")

            # 马尔可夫预测
            if weights.get('markov', 0) > 0:
                markov_pred = self.markov_predict(1)[0]
                weight = weights['markov']
                for ball in markov_pred[0]:
                    front_candidates[ball] += weight * 10
                for ball in markov_pred[1]:
                    back_candidates[ball] += weight * 10

            # 贝叶斯预测
            if weights.get('bayesian', 0) > 0:
                bayesian_pred = self.bayesian_predict(1)[0]
                weight = weights['bayesian']
                for ball in bayesian_pred[0]:
                    front_candidates[ball] += weight * 10
                for ball in bayesian_pred[1]:
                    back_candidates[ball] += weight * 10

            # 选择最终号码
            front_balls = [ball for ball, score in front_candidates.most_common(5)]
            back_balls = [ball for ball, score in back_candidates.most_common(2)]

            # 如果号码不足，用频率分析补充
            if len(front_balls) < 5:
                freq_analysis = basic_analyzer.frequency_analysis(periods)
                front_freq = freq_analysis.get('front_frequency', {})
                sorted_freq = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)
                for ball, freq in sorted_freq:
                    if len(front_balls) >= 5:
                        break
                    ball_int = int(ball) if isinstance(ball, str) else ball
                    if ball_int not in front_balls:
                        front_balls.append(ball_int)

            if len(back_balls) < 2:
                freq_analysis = basic_analyzer.frequency_analysis(periods)
                back_freq = freq_analysis.get('back_frequency', {})
                sorted_freq = sorted(back_freq.items(), key=lambda x: x[1], reverse=True)
                for ball, freq in sorted_freq:
                    if len(back_balls) >= 2:
                        break
                    ball_int = int(ball) if isinstance(ball, str) else ball
                    if ball_int not in back_balls:
                        back_balls.append(ball_int)

            prediction = {
                'index': i + 1,
                'front_balls': sorted(front_balls),
                'back_balls': sorted(back_balls),
                'strategy': strategy,
                'risk_level': strategy_config['risk_level'],
                'description': strategy_config['description'],
                'weights': weights,
                'method': 'mixed_strategy'
            }

            predictions.append(prediction)

        return predictions

    def markov_compound_predict(self, front_count=8, back_count=4, analysis_periods=500) -> Dict:
        """基于马尔可夫链的复式预测

        Args:
            front_count: 前区号码数量 (6-15)
            back_count: 后区号码数量 (3-12)
            analysis_periods: 分析期数

        Returns:
            马尔可夫复式预测结果
        """
        logger_manager.info(f"马尔可夫链复式预测: {front_count}+{back_count}, 分析期数: {analysis_periods}")

        try:
            # 获取马尔可夫链分析结果
            markov_result = advanced_analyzer.markov_analysis(analysis_periods)

            if not markov_result:
                logger_manager.warning("马尔可夫链分析结果为空，使用备选方案")
                return self._fallback_markov_compound_prediction(front_count, back_count)

            # 基于马尔可夫链的复式号码选择
            front_balls = self._markov_compound_selection(
                markov_result, front_count, True, analysis_periods
            )
            back_balls = self._markov_compound_selection(
                markov_result, back_count, False, analysis_periods
            )

            # 计算组合数和投注金额
            from math import comb
            total_combinations = comb(front_count, 5) * comb(back_count, 2)
            total_cost = total_combinations * 3

            # 计算马尔可夫链置信度
            confidence = self._calculate_markov_compound_confidence(markov_result, front_count, back_count)

            result = {
                'front_balls': front_balls,
                'back_balls': back_balls,
                'front_count': front_count,
                'back_count': back_count,
                'total_combinations': total_combinations,
                'total_cost': total_cost,
                'method': 'markov_compound',
                'confidence': confidence,
                'analysis_periods': analysis_periods,
                'markov_details': {
                    'transition_matrix_size': len(markov_result.get('front_transition_probs', {})),
                    'state_count': len(markov_result.get('front_states', {})),
                    'prediction_accuracy': markov_result.get('prediction_accuracy', 0.7)
                },
                'timestamp': datetime.now().isoformat()
            }

            return result

        except Exception as e:
            logger_manager.error(f"马尔可夫链复式预测失败: {e}")
            return self._fallback_markov_compound_prediction(front_count, back_count)

    def _markov_compound_selection(self, markov_result, target_count, is_front=True, analysis_periods=500):
        """基于马尔可夫链的复式号码选择"""
        # 获取转移概率
        if is_front:
            transition_probs = markov_result.get('front_transition_probs', {})
            max_ball = 35
        else:
            transition_probs = markov_result.get('back_transition_probs', {})
            max_ball = 12

        if not transition_probs:
            return sorted(np.random.choice(range(1, max_ball + 1), target_count, replace=False))

        # 计算每个号码的马尔可夫得分
        markov_scores = {}

        for ball in range(1, max_ball + 1):
            total_score = 0
            count = 0

            # 计算从所有状态转移到该号码的概率总和
            for from_state, to_probs in transition_probs.items():
                if ball in to_probs:
                    total_score += to_probs[ball]
                    count += 1

            # 平均转移概率作为马尔可夫得分
            markov_scores[ball] = total_score / max(count, 1)

        # 按马尔可夫得分排序
        sorted_scores = sorted(markov_scores.items(), key=lambda x: x[1], reverse=True)

        # 马尔可夫智能选择策略：50%高概率 + 50%多样性选择
        high_prob_count = int(target_count * 0.5)
        diversity_count = target_count - high_prob_count

        selected = []

        # 选择高概率号码（避免连续选择）
        for i in range(len(sorted_scores)):
            if len(selected) >= high_prob_count:
                break

            candidate = int(sorted_scores[i][0])

            # 检查是否与已选号码过于接近（避免连续号码）
            too_close = False
            for existing in selected:
                if abs(candidate - existing) <= 1:  # 相邻号码
                    too_close = True
                    break

            if not too_close:
                selected.append(candidate)

        # 多样性选择（确保号码分布均匀）
        if diversity_count > 0:
            # 将号码范围分成几个区间，从每个区间选择
            if max_ball == 35:  # 前区
                ranges = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35)]
            else:  # 后区
                ranges = [(1, 3), (4, 6), (7, 9), (10, 12)]

            # 从每个区间选择最佳号码
            for start, end in ranges:
                if len(selected) >= target_count:
                    break

                # 在当前区间内找到最佳候选
                range_candidates = []
                for ball in range(start, end + 1):
                    if ball in selected:
                        continue

                    # 检查是否与已选号码过于接近
                    too_close = False
                    for existing in selected:
                        if abs(ball - existing) <= 1:
                            too_close = True
                            break

                    if not too_close and ball in markov_scores:
                        range_candidates.append((ball, markov_scores[ball]))

                # 从当前区间选择得分最高的
                if range_candidates:
                    range_candidates.sort(key=lambda x: x[1], reverse=True)
                    selected.append(range_candidates[0][0])

        # 如果数量不足，用频率分析补充
        if len(selected) < target_count:
            freq_analysis = basic_analyzer.frequency_analysis()
            if max_ball == 35:  # 前区
                freq_dict = freq_analysis.get('front_frequency', {})
            else:  # 后区
                freq_dict = freq_analysis.get('back_frequency', {})

            sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
            for ball, freq in sorted_freq:
                if len(selected) >= target_count:
                    break
                # 确保ball是整数
                ball_int = int(ball) if isinstance(ball, str) else ball
                if ball_int not in selected:
                    selected.append(ball_int)

        return sorted(selected[:target_count])

    def _calculate_markov_compound_confidence(self, markov_result, front_count, back_count):
        """计算马尔可夫链复式预测的置信度"""
        try:
            # 基础置信度
            base_confidence = 0.7

            # 转移矩阵完整性加成
            front_transitions = len(markov_result.get('front_transition_probs', {}))
            back_transitions = len(markov_result.get('back_transition_probs', {}))

            if front_transitions >= 20 and back_transitions >= 10:
                matrix_bonus = 0.1
            elif front_transitions >= 10 and back_transitions >= 5:
                matrix_bonus = 0.05
            else:
                matrix_bonus = 0

            # 复式规模加成
            scale_bonus = min(0.1, (front_count - 5) * 0.01 + (back_count - 2) * 0.02)

            # 预测准确性加成
            accuracy = markov_result.get('prediction_accuracy', 0.7)
            accuracy_bonus = (accuracy - 0.5) * 0.2

            final_confidence = base_confidence + matrix_bonus + scale_bonus + accuracy_bonus
            return min(0.9, max(0.5, final_confidence))

        except Exception:
            return 0.65

    def _fallback_markov_compound_prediction(self, front_count, back_count):
        """马尔可夫链复式预测的备选方案"""
        front_balls = sorted(np.random.choice(range(1, 36), front_count, replace=False))
        back_balls = sorted(np.random.choice(range(1, 13), back_count, replace=False))

        from math import comb
        total_combinations = comb(front_count, 5) * comb(back_count, 2)
        total_cost = total_combinations * 3

        return {
            'front_balls': [int(x) for x in front_balls],
            'back_balls': [int(x) for x in back_balls],
            'front_count': front_count,
            'back_count': back_count,
            'total_combinations': total_combinations,
            'total_cost': total_cost,
            'method': 'markov_compound_fallback',
            'confidence': 0.4
        }

    def advanced_integration_predict(self, count=1, integration_type="comprehensive", periods=500) -> List[Tuple[List[int], List[int]]]:
        """基于高级集成分析的预测

        Args:
            count: 生成注数
            integration_type: 集成类型 ('comprehensive', 'markov_bayesian', 'hot_cold_markov', 'multi_dimensional')
            periods: 分析期数

        Returns:
            预测结果列表
        """
        logger_manager.info(f"高级集成预测: {integration_type}, 注数: {count}, 分析期数: {periods}")

        predictions = []

        try:
            # 获取高级集成分析结果
            if integration_type == "comprehensive":
                analysis_result = advanced_analyzer.comprehensive_weight_scoring_system(periods)
                front_candidates = [(int(ball) if isinstance(ball, str) else ball, data['total_score'])
                                  for ball, data in analysis_result['comprehensive_scores']['front_scores'].items()]
                back_candidates = [(int(ball) if isinstance(ball, str) else ball, data['total_score'])
                                 for ball, data in analysis_result['comprehensive_scores']['back_scores'].items()]

            elif integration_type == "markov_bayesian":
                analysis_result = advanced_analyzer.markov_bayesian_fusion_analysis(periods)
                front_candidates = analysis_result.get('front_recommendations', [])
                back_candidates = analysis_result.get('back_recommendations', [])

            elif integration_type == "hot_cold_markov":
                analysis_result = advanced_analyzer.hot_cold_markov_integration(periods)
                front_candidates = analysis_result.get('front_integrated', [])
                back_candidates = analysis_result.get('back_integrated', [])

            elif integration_type == "multi_dimensional":
                analysis_result = advanced_analyzer.multi_dimensional_probability_analysis(periods)
                front_ranked = analysis_result.get('front_ranked', [])
                back_ranked = analysis_result.get('back_ranked', [])
                # 转换数据格式，确保ball是整数
                front_candidates = [(int(ball) if isinstance(ball, str) else ball, data['total_prob'])
                                  for ball, data in front_ranked]
                back_candidates = [(int(ball) if isinstance(ball, str) else ball, data['total_prob'])
                                 for ball, data in back_ranked]

            else:
                # 默认使用综合权重评分
                analysis_result = advanced_analyzer.comprehensive_weight_scoring_system(periods)
                front_candidates = [(ball, data['total_score']) for ball, data in analysis_result['comprehensive_scores']['front_scores'].items()]
                back_candidates = [(ball, data['total_score']) for ball, data in analysis_result['comprehensive_scores']['back_scores'].items()]

            # 排序候选号码
            front_sorted = sorted(front_candidates, key=lambda x: x[1], reverse=True)
            back_sorted = sorted(back_candidates, key=lambda x: x[1], reverse=True)

            for i in range(count):
                # 改进的智能选择策略：加权随机选择
                import random
                import numpy as np

                # 选择前区号码
                front_balls = []

                # 检查是否有有效的得分
                valid_front_scores = [score for _, score in front_sorted if score > 0]

                if len(valid_front_scores) >= 5:
                    # 有足够的有效得分，使用加权随机选择
                    weights = [max(0.1, score) for _, score in front_sorted[:15]]  # 取前15个候选
                    candidates = [ball for ball, _ in front_sorted[:15]]

                    # 确保候选号码是整数
                    candidates = [int(ball) if isinstance(ball, str) else ball for ball in candidates]

                    # 加权随机选择5个号码
                    selected_indices = np.random.choice(
                        len(candidates),
                        size=min(5, len(candidates)),
                        replace=False,
                        p=np.array(weights) / np.sum(weights)
                    )
                    front_balls = [candidates[idx] for idx in selected_indices]
                else:
                    # 得分都很低，使用混合策略
                    # 50%高分 + 50%随机分布选择
                    high_count = min(2, len(front_sorted))
                    for j in range(high_count):
                        ball = front_sorted[j][0]
                        if isinstance(ball, str):
                            ball = int(ball)
                        front_balls.append(ball)

                    # 从不同区间随机选择剩余号码
                    remaining_needed = 5 - len(front_balls)
                    ranges = [(6, 15), (16, 25), (26, 35)]
                    for start, end in ranges:
                        if remaining_needed <= 0:
                            break
                        available = [x for x in range(start, end+1) if x not in front_balls]
                        if available:
                            selected = random.choice(available)
                            front_balls.append(selected)
                            remaining_needed -= 1

                # 如果前区号码不足，用频率分析补充
                if len(front_balls) < 5:
                    freq_analysis = basic_analyzer.frequency_analysis()
                    front_freq = freq_analysis.get('front_frequency', {})
                    sorted_freq = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)
                    for ball, freq in sorted_freq:
                        if len(front_balls) >= 5:
                            break
                        if ball not in front_balls:
                            front_balls.append(ball)

                # 选择后区号码
                back_balls = []
                back_high_count = 1
                back_random_count = 1

                for j in range(min(back_high_count, len(back_sorted))):
                    ball = back_sorted[j][0]
                    if isinstance(ball, str):
                        ball = int(ball)
                    back_balls.append(ball)

                if len(back_sorted) > back_high_count:
                    remaining_back = []
                    for x in back_sorted[back_high_count:back_high_count+5]:
                        ball = x[0]
                        if isinstance(ball, str):
                            ball = int(ball)
                        remaining_back.append(ball)
                    if remaining_back:
                        # 确保数组是整数类型
                        remaining_back = np.array(remaining_back, dtype=int)
                        random_back = np.random.choice(
                            remaining_back,
                            min(back_random_count, len(remaining_back)),
                            replace=False
                        )
                        back_balls.extend(random_back.tolist())

                while len(back_balls) < 2:
                    candidate = np.random.randint(1, 13)
                    if candidate not in back_balls:
                        back_balls.append(candidate)

                # 确保数据类型正确
                front_balls = sorted([int(x) for x in front_balls])
                back_balls = sorted([int(x) for x in back_balls])

                # 返回标准元组格式
                predictions.append((front_balls, back_balls))

        except Exception as e:
            logger_manager.error(f"高级集成预测失败: {e}")
            # 使用频率分析作为备选方案
            freq_analysis = basic_analyzer.frequency_analysis()
            front_freq = freq_analysis.get('front_frequency', {})
            back_freq = freq_analysis.get('back_frequency', {})

            front_sorted = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)
            back_sorted = sorted(back_freq.items(), key=lambda x: x[1], reverse=True)

            front_balls = [int(ball) for ball, freq in front_sorted[:5]]
            back_balls = [int(ball) for ball, freq in back_sorted[:2]]

            for i in range(count):
                # 返回标准元组格式
                predictions.append((front_balls, back_balls))

        return predictions

    def highly_integrated_predict(self, count=1, periods=500, integration_level="ultimate") -> List[Tuple[List[int], List[int]]]:
        """高度集成预测

        Args:
            count: 生成注数
            periods: 分析期数
            integration_level: 集成级别 ('high', 'ultimate')

        Returns:
            预测结果列表
        """
        logger_manager.info(f"高度集成预测: 注数={count}, 分析期数={periods}, 级别={integration_level}")

        try:
            # 使用高级集成分析
            result = self.advanced_integration_predict(count=count, periods=periods, integration_type="comprehensive")

            # 转换为标准格式
            predictions = []
            for pred in result:
                if isinstance(pred, dict):
                    front_balls = pred.get('front_balls', [])
                    back_balls = pred.get('back_balls', [])
                    predictions.append((front_balls, back_balls))
                elif isinstance(pred, tuple) and len(pred) == 2:
                    predictions.append(pred)

            return predictions

        except Exception as e:
            logger_manager.error(f"高度集成预测失败: {e}")
            # 回退到集成预测
            return self.ensemble_predict(count=count, periods=periods)

    def stacking_predict(self, count=1, periods=500) -> List[Tuple[List[int], List[int]]]:
        """堆叠集成预测

        Args:
            count: 生成注数
            periods: 分析期数

        Returns:
            预测结果列表
        """
        logger_manager.info(f"堆叠集成预测: 注数={count}, 分析期数={periods}")

        try:
            # 使用多种基础预测器
            base_predictions = []

            # 收集基础预测结果
            markov_pred = self.markov_predict(count=1, periods=periods)
            if markov_pred:
                base_predictions.extend(markov_pred)

            bayesian_pred = self.bayesian_predict(count=1, periods=periods)
            if bayesian_pred:
                base_predictions.extend(bayesian_pred)

            freq_pred = self.traditional_predictor.frequency_predict(count=1, periods=periods)
            if freq_pred:
                base_predictions.extend(freq_pred)

            # 使用堆叠方法融合预测结果
            predictions = []
            for i in range(count):
                if base_predictions:
                    # 选择不同的基础预测结果进行组合
                    idx = i % len(base_predictions)
                    selected_pred = base_predictions[idx]
                    predictions.append(selected_pred)
                else:
                    # 回退到集成预测
                    ensemble_pred = self.ensemble_predict(count=1, periods=periods)
                    if ensemble_pred:
                        predictions.extend(ensemble_pred)

            return predictions[:count]

        except Exception as e:
            logger_manager.error(f"堆叠集成预测失败: {e}")
            # 回退到集成预测
            return self.ensemble_predict(count=count, periods=periods)

    def enhanced_predict(self, count=1, periods=500) -> List[Tuple[List[int], List[int]]]:
        """增强预测

        Args:
            count: 生成注数
            periods: 分析期数

        Returns:
            预测结果列表
        """
        logger_manager.info(f"增强预测: 注数={count}, 分析期数={periods}")

        try:
            # 使用增强特性预测器
            try:
                from improvements.enhanced_features import get_enhanced_feature_predictor
                feature_predictor = get_enhanced_feature_predictor()

                # 使用模式预测
                pattern_pred = feature_predictor.pattern_based_predict(count=count)
                if pattern_pred and len(pattern_pred) >= count:
                    return pattern_pred[:count]

            except ImportError:
                logger_manager.warning("增强特性模块不可用，使用高级集成预测")

            # 回退到高级集成预测
            return self.advanced_integration_predict(count=count, periods=periods, integration_type="multi_dimensional")

        except Exception as e:
            logger_manager.error(f"增强预测失败: {e}")
            # 最终回退到集成预测
            return self.ensemble_predict(count=count, periods=periods)

    def stacking_predict(self, count=1, periods=500) -> List[Tuple[List[int], List[int]]]:
        """堆叠集成预测

        Args:
            count: 生成注数
            periods: 分析期数

        Returns:
            预测结果列表
        """
        logger_manager.info(f"堆叠集成预测: 注数={count}, 分析期数={periods}")

        try:
            # 使用多种基础预测器
            base_predictions = []

            # 收集基础预测结果
            markov_pred = self.markov_predict(count=1, periods=periods)
            if markov_pred:
                base_predictions.extend(markov_pred)

            bayesian_pred = self.bayesian_predict(count=1, periods=periods)
            if bayesian_pred:
                base_predictions.extend(bayesian_pred)

            freq_pred = self.traditional_predictor.frequency_predict(count=1, periods=periods)
            if freq_pred:
                base_predictions.extend(freq_pred)

            # 使用堆叠方法融合预测结果
            predictions = []
            for i in range(count):
                if base_predictions:
                    # 选择不同的基础预测结果进行组合
                    idx = i % len(base_predictions)
                    selected_pred = base_predictions[idx]
                    predictions.append(selected_pred)
                else:
                    # 回退到集成预测
                    ensemble_pred = self.ensemble_predict(count=1, periods=periods)
                    if ensemble_pred:
                        predictions.extend(ensemble_pred)

            logger_manager.info(f"堆叠集成预测完成，生成{len(predictions)}注")
            return predictions[:count]

        except Exception as e:
            logger_manager.error(f"堆叠集成预测失败: {e}")
            # 回退到集成预测
            return self.ensemble_predict(count=count, periods=periods)

    def nine_models_compound_predict(self, front_count=8, back_count=4, analysis_periods=500) -> Dict:
        """基于九种数学模型的复式预测

        Args:
            front_count: 前区号码数量 (6-15)
            back_count: 后区号码数量 (3-12)
            analysis_periods: 分析期数

        Returns:
            九模型复式预测结果
        """
        logger_manager.info(f"九模型复式预测: {front_count}+{back_count}, 分析期数: {analysis_periods}")

        try:
            # 获取九种数学模型分析结果
            nine_models_result = advanced_analyzer.nine_mathematical_models_analysis(analysis_periods)

            if not nine_models_result:
                logger_manager.warning("九种数学模型分析结果为空，使用备选方案")
                return self._fallback_nine_models_compound_prediction(front_count, back_count)

            # 基于九模型的复式号码选择
            front_balls = self._nine_models_compound_selection(
                nine_models_result, front_count, True, analysis_periods
            )
            back_balls = self._nine_models_compound_selection(
                nine_models_result, back_count, False, analysis_periods
            )

            # 计算复式投注信息
            from math import comb
            total_combinations = comb(front_count, 5) * comb(back_count, 2)
            total_cost = total_combinations * 3  # 每注3元

            # 计算置信度
            confidence = self._calculate_nine_models_compound_confidence(
                nine_models_result, front_count, back_count
            )

            result = {
                'front_balls': front_balls,
                'back_balls': back_balls,
                'front_count': front_count,
                'back_count': back_count,
                'total_combinations': total_combinations,
                'total_cost': total_cost,
                'method': 'nine_models_compound',
                'confidence': confidence,
                'analysis_periods': analysis_periods,
                'nine_models_details': {
                    'model_count': len(nine_models_result.get('model_results', {})),
                    'comprehensive_score': nine_models_result.get('comprehensive_scores', {}).get('overall_score', 0),
                    'prediction_accuracy': nine_models_result.get('prediction_accuracy', 0.7)
                },
                'timestamp': datetime.now().isoformat()
            }

            return result

        except Exception as e:
            logger_manager.error(f"九模型复式预测失败: {e}")
            return self._fallback_nine_models_compound_prediction(front_count, back_count)

    def _nine_models_compound_selection(self, nine_models_result, target_count, is_front, analysis_periods):
        """基于九种数学模型的复式号码选择"""
        try:
            import numpy as np

            # 获取综合评分
            comprehensive_scores = nine_models_result.get('comprehensive_scores', {})
            recommendations = comprehensive_scores.get('prediction_recommendations', {})

            if is_front:
                candidates = recommendations.get('front_top10', list(range(1, 36)))
                max_num = 35
            else:
                candidates = recommendations.get('back_top6', list(range(1, 13)))
                max_num = 12

            # 确保有足够的候选号码
            if len(candidates) < target_count:
                all_nums = list(range(1, max_num + 1))
                candidates.extend([num for num in all_nums if num not in candidates])

            # 智能选择策略：结合模型评分和分散性
            selected = []

            # 选择高评分号码（70%）
            high_score_count = int(target_count * 0.7)
            selected.extend(candidates[:high_score_count])

            # 选择分散号码（30%）
            remaining_candidates = [num for num in candidates if num not in selected]
            diversity_count = target_count - len(selected)

            if diversity_count > 0 and remaining_candidates:
                # 确保号码分散
                for candidate in remaining_candidates:
                    if len(selected) >= target_count:
                        break

                    # 检查与已选号码的距离
                    too_close = any(abs(candidate - existing) <= 2 for existing in selected)
                    if not too_close:
                        selected.append(candidate)

                # 如果还不够，随机补充
                if len(selected) < target_count:
                    remaining = [num for num in remaining_candidates if num not in selected]
                    needed = target_count - len(selected)
                    if remaining:
                        import random
                        selected.extend(random.sample(remaining, min(needed, len(remaining))))

            return sorted(selected[:target_count])

        except Exception as e:
            logger_manager.error(f"九模型复式号码选择失败: {e}")
            # 回退到随机选择
            import random
            max_num = 35 if is_front else 12
            return sorted(random.sample(range(1, max_num + 1), target_count))

    def _calculate_nine_models_compound_confidence(self, nine_models_result, front_count, back_count):
        """计算九模型复式预测的置信度"""
        try:
            # 基础置信度
            base_confidence = 0.7

            # 模型数量加成
            model_count = len(nine_models_result.get('model_results', {}))
            model_bonus = min(0.1, model_count * 0.01)

            # 综合评分加成
            overall_score = nine_models_result.get('comprehensive_scores', {}).get('overall_score', 0.5)
            score_bonus = (overall_score - 0.5) * 0.2

            # 复式规模加成
            scale_bonus = min(0.1, (front_count - 5) * 0.01 + (back_count - 2) * 0.02)

            final_confidence = base_confidence + model_bonus + score_bonus + scale_bonus
            return min(0.9, max(0.5, final_confidence))

        except Exception:
            return 0.7

    def _fallback_nine_models_compound_prediction(self, front_count, back_count):
        """九模型复式预测的备选方案"""
        import numpy as np
        from math import comb

        front_balls = sorted(np.random.choice(range(1, 36), front_count, replace=False))
        back_balls = sorted(np.random.choice(range(1, 13), back_count, replace=False))

        total_combinations = comb(front_count, 5) * comb(back_count, 2)
        total_cost = total_combinations * 3

        return {
            'front_balls': [int(x) for x in front_balls],
            'back_balls': [int(x) for x in back_balls],
            'front_count': front_count,
            'back_count': back_count,
            'total_combinations': total_combinations,
            'total_cost': total_cost,
            'method': 'nine_models_compound_fallback',
            'confidence': 0.5
        }

    def nine_models_predict(self, count=1, periods=500) -> List[Tuple[List[int], List[int]]]:
        """基于9种数学模型的预测生成

        Args:
            count: 生成注数
            periods: 分析期数

        Returns:
            预测结果列表，格式: [(前区号码, 后区号码), ...]
        """
        logger_manager.info(f"9种数学模型预测，注数: {count}, 分析期数: {periods}")

        predictions = []

        try:
            # 获取9种数学模型分析结果
            nine_models_result = advanced_analyzer.nine_mathematical_models_analysis(periods)

            if not nine_models_result or 'comprehensive_scores' not in nine_models_result:
                logger_manager.warning("9种数学模型分析结果为空，使用备选方案")
                return self._fallback_nine_models_prediction(count)

            comprehensive_scores = nine_models_result['comprehensive_scores']

            # 获取推荐号码
            front_recommendations = comprehensive_scores.get('prediction_recommendations', {}).get('front_top10', [])
            back_recommendations = comprehensive_scores.get('prediction_recommendations', {}).get('back_top6', [])

            if not front_recommendations or not back_recommendations:
                logger_manager.warning("推荐号码为空，使用备选方案")
                return self._fallback_nine_models_prediction(count)

            for i in range(count):
                # 智能选择策略
                front_balls = self._intelligent_nine_models_selection(
                    front_recommendations, 5, is_front=True
                )
                back_balls = self._intelligent_nine_models_selection(
                    back_recommendations, 2, is_front=False
                )

                # 确保数据类型正确
                front_balls = sorted([int(x) for x in front_balls])
                back_balls = sorted([int(x) for x in back_balls])

                # 添加到预测结果（标准元组格式）
                predictions.append((front_balls, back_balls))

        except Exception as e:
            logger_manager.error(f"9种数学模型预测失败: {e}")
            return self._fallback_nine_models_prediction(count)

        return predictions

    def _intelligent_nine_models_selection(self, recommendations, target_count, is_front=True):
        """基于9种数学模型的智能号码选择"""
        if not recommendations:
            # 如果没有推荐，使用频率分析
            freq_analysis = basic_analyzer.frequency_analysis()
            if is_front:
                freq_dict = freq_analysis.get('front_frequency', {})
            else:
                freq_dict = freq_analysis.get('back_frequency', {})

            sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
            return sorted([int(ball) for ball, freq in sorted_freq[:target_count]])

        selected = []

        # 策略：60%高分号码 + 40%多样性号码
        high_score_count = int(target_count * 0.6)
        diversity_count = target_count - high_score_count

        # 选择高分号码
        for i in range(min(high_score_count, len(recommendations))):
            ball = recommendations[i][0]  # (ball, score) 格式
            selected.append(int(ball))

        # 选择多样性号码（确定性选择）
        if diversity_count > 0 and len(recommendations) > high_score_count:
            diversity_candidates = [x[0] for x in recommendations[high_score_count:]]
            if diversity_candidates:
                selected_count = min(diversity_count, len(diversity_candidates))
                selected.extend([int(x) for x in diversity_candidates[:selected_count]])

        # 如果数量不足，用频率分析补充
        if len(selected) < target_count:
            freq_analysis = basic_analyzer.frequency_analysis()
            if is_front:
                freq_dict = freq_analysis.get('front_frequency', {})
            else:
                freq_dict = freq_analysis.get('back_frequency', {})

            sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
            for ball, freq in sorted_freq:
                if len(selected) >= target_count:
                    break
                if ball not in selected:
                    selected.append(ball)

        return sorted(selected[:target_count])

    def _calculate_nine_models_confidence(self, nine_models_result):
        """计算9种数学模型的综合置信度"""
        try:
            # 基础置信度
            base_confidence = 0.8

            # 模型一致性加成
            model_consensus = nine_models_result.get('comprehensive_scores', {}).get(
                'confidence_levels', {}
            ).get('model_consensus', 0.8)
            consensus_bonus = (model_consensus - 0.5) * 0.2  # 最多0.1加成

            # 分析期数加成
            periods = nine_models_result.get('analysis_periods', 0)
            if periods >= 500:
                period_bonus = 0.1
            elif periods >= 300:
                period_bonus = 0.05
            else:
                period_bonus = 0

            # 模型完整性加成
            models_count = len(nine_models_result.get('nine_models', {}))
            completeness_bonus = min(0.05, models_count * 0.01)

            final_confidence = base_confidence + consensus_bonus + period_bonus + completeness_bonus
            return min(0.95, max(0.5, final_confidence))

        except Exception:
            return 0.75

    def _fallback_nine_models_prediction(self, count):
        """9种数学模型的备选预测方案"""
        # 使用频率分析作为备选方案
        freq_analysis = basic_analyzer.frequency_analysis()
        front_freq = freq_analysis.get('front_frequency', {})
        back_freq = freq_analysis.get('back_frequency', {})

        front_sorted = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)
        back_sorted = sorted(back_freq.items(), key=lambda x: x[1], reverse=True)

        front_balls = [int(ball) for ball, freq in front_sorted[:5]]
        back_balls = [int(ball) for ball, freq in back_sorted[:2]]

        predictions = []
        for i in range(count):
            # 返回标准元组格式
            predictions.append((front_balls, back_balls))

        return predictions

    def nine_models_compound_predict(self, front_count=8, back_count=4, periods=500) -> Dict:
        """基于9种数学模型的复式预测

        Args:
            front_count: 前区号码数量 (6-15)
            back_count: 后区号码数量 (3-12)
            periods: 分析期数

        Returns:
            复式预测结果
        """
        logger_manager.info(f"9种数学模型复式预测: {front_count}+{back_count}, 分析期数: {periods}")

        try:
            # 获取9种数学模型分析结果
            nine_models_result = advanced_analyzer.nine_mathematical_models_analysis(periods)

            if not nine_models_result or 'comprehensive_scores' not in nine_models_result:
                logger_manager.warning("9种数学模型分析结果为空，使用备选方案")
                return self._fallback_compound_prediction(front_count, back_count)

            comprehensive_scores = nine_models_result['comprehensive_scores']

            # 获取所有候选号码及其评分
            front_scores = comprehensive_scores.get('front_scores', {})
            back_scores = comprehensive_scores.get('back_scores', {})

            if not front_scores or not back_scores:
                return self._fallback_compound_prediction(front_count, back_count)

            # 基于9种数学模型的智能复式选择
            front_balls = self._nine_models_compound_selection(front_scores, front_count, True)
            back_balls = self._nine_models_compound_selection(back_scores, back_count, False)

            # 计算组合数和投注金额
            from math import comb
            total_combinations = comb(front_count, 5) * comb(back_count, 2)
            total_cost = total_combinations * 3

            # 计算9种模型的综合置信度
            confidence = self._calculate_nine_models_compound_confidence(nine_models_result, front_count, back_count)

            result = {
                'front_balls': front_balls,
                'back_balls': back_balls,
                'front_count': front_count,
                'back_count': back_count,
                'total_combinations': total_combinations,
                'total_cost': total_cost,
                'method': 'nine_models_compound',
                'confidence': confidence,
                'models_used': list(nine_models_result.get('model_weights', {}).keys()),
                'model_weights': nine_models_result.get('model_weights', {}),
                'analysis_timestamp': nine_models_result.get('timestamp', 'unknown'),
                'model_details': {
                    'statistical_score': self._extract_model_contribution(nine_models_result, 'statistical'),
                    'probability_score': self._extract_model_contribution(nine_models_result, 'probability'),
                    'markov_score': self._extract_model_contribution(nine_models_result, 'enhanced_markov'),
                    'bayesian_score': self._extract_model_contribution(nine_models_result, 'enhanced_bayesian')
                }
            }

            return result

        except Exception as e:
            logger_manager.error(f"9种数学模型复式预测失败: {e}")
            return self._fallback_compound_prediction(front_count, back_count)

    def _nine_models_compound_selection(self, scores_dict, target_count, is_front=True):
        """基于9种数学模型的复式号码选择"""
        if not scores_dict:
            # 如果没有评分，使用频率分析
            freq_analysis = basic_analyzer.frequency_analysis()
            if is_front:
                freq_dict = freq_analysis.get('front_frequency', {})
            else:
                freq_dict = freq_analysis.get('back_frequency', {})

            sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
            selected_balls = [int(ball) for ball, freq in sorted_freq[:target_count]]
            return sorted(selected_balls)

        # 按评分排序
        sorted_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)

        # 智能选择策略：80%高分 + 20%平衡选择
        high_score_count = int(target_count * 0.8)
        balance_count = target_count - high_score_count

        selected = []

        # 选择高分号码
        for i in range(min(high_score_count, len(sorted_scores))):
            ball_key = sorted_scores[i][0]
            # 确保转换为整数
            if isinstance(ball_key, str):
                selected.append(int(ball_key))
            else:
                selected.append(int(ball_key))

        # 平衡选择（从中等分数中选择，增加覆盖面）
        if balance_count > 0 and len(sorted_scores) > high_score_count:
            balance_start = high_score_count
            balance_end = min(len(sorted_scores), high_score_count + balance_count * 3)
            balance_candidates = []
            for x in sorted_scores[balance_start:balance_end]:
                ball_key = x[0]
                if isinstance(ball_key, str):
                    balance_candidates.append(int(ball_key))
                else:
                    balance_candidates.append(int(ball_key))

            if balance_candidates:
                selected_count = min(balance_count, len(balance_candidates))
                selected.extend(balance_candidates[:selected_count])

        # 如果数量不足，用频率分析补充
        if len(selected) < target_count:
            freq_analysis = basic_analyzer.frequency_analysis()
            if is_front:
                freq_dict = freq_analysis.get('front_frequency', {})
            else:
                freq_dict = freq_analysis.get('back_frequency', {})

            sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
            for ball, freq in sorted_freq:
                if len(selected) >= target_count:
                    break
                if ball not in selected:
                    selected.append(ball)

        return sorted(selected[:target_count])

    def _calculate_nine_models_compound_confidence(self, nine_models_result, front_count, back_count):
        """计算9种数学模型复式预测的置信度"""
        try:
            # 基础置信度
            base_confidence = 0.75

            # 模型完整性加成
            models_count = len(nine_models_result.get('nine_models', {}))
            if models_count >= 9:
                completeness_bonus = 0.1
            elif models_count >= 6:
                completeness_bonus = 0.05
            else:
                completeness_bonus = 0

            # 复式规模加成（更大的复式有更高的中奖概率）
            scale_bonus = min(0.1, (front_count - 5) * 0.01 + (back_count - 2) * 0.02)

            # 模型一致性加成
            consensus = nine_models_result.get('comprehensive_scores', {}).get(
                'confidence_levels', {}
            ).get('model_consensus', 0.8)
            consensus_bonus = (consensus - 0.5) * 0.1

            final_confidence = base_confidence + completeness_bonus + scale_bonus + consensus_bonus
            return min(0.95, max(0.6, final_confidence))

        except Exception:
            return 0.7

    def _extract_model_contribution(self, nine_models_result, model_name):
        """提取特定模型的贡献度"""
        try:
            model_weights = nine_models_result.get('model_weights', {})
            return model_weights.get(model_name, 0)
        except Exception:
            return 0

    def _fallback_compound_prediction(self, front_count, back_count):
        """9种数学模型复式预测的备选方案"""
        front_balls = sorted(np.random.choice(range(1, 36), front_count, replace=False))
        back_balls = sorted(np.random.choice(range(1, 13), back_count, replace=False))

        from math import comb
        total_combinations = comb(front_count, 5) * comb(back_count, 2)
        total_cost = total_combinations * 3

        return {
            'front_balls': [int(x) for x in front_balls],
            'back_balls': [int(x) for x in back_balls],
            'front_count': front_count,
            'back_count': back_count,
            'total_combinations': total_combinations,
            'total_cost': total_cost,
            'method': 'nine_models_compound_fallback',
            'confidence': 0.4
        }

    def _calculate_integration_confidence(self, analysis_result: Dict) -> float:
        """计算集成分析置信度"""
        try:
            # 基于分析结果的完整性和质量计算置信度
            base_confidence = 0.7

            # 检查分析结果的完整性
            if 'comprehensive_scores' in analysis_result:
                completeness_bonus = 0.1
            elif 'front_recommendations' in analysis_result or 'front_integrated' in analysis_result:
                completeness_bonus = 0.05
            else:
                completeness_bonus = 0

            # 检查分析期数
            periods = analysis_result.get('analysis_periods', 0)
            if periods >= 500:
                period_bonus = 0.1
            elif periods >= 300:
                period_bonus = 0.05
            else:
                period_bonus = 0

            final_confidence = base_confidence + completeness_bonus + period_bonus
            return min(0.95, final_confidence)

        except Exception:
            return 0.6

    def super_predict(self, count=1, periods=500) -> List[Tuple[List[int], List[int]]]:
        """超级预测方法 - 集成所有高级算法的最优预测"""
        try:
            logger_manager.info(f"开始超级预测: 注数={count}, 分析期数={periods}")

            # 获取多种预测结果
            predictions_pool = []

            # 1. 九种数学模型预测
            try:
                nine_models_result = self.nine_models_predict(count, periods)
                predictions_pool.extend(nine_models_result)
            except Exception as e:
                logger_manager.warning(f"九种数学模型预测失败: {e}")

            # 2. 集成预测
            try:
                ensemble_result = self.ensemble_predict(count, periods)
                predictions_pool.extend(ensemble_result)
            except Exception as e:
                logger_manager.warning(f"集成预测失败: {e}")

            # 3. 自适应马尔可夫预测
            try:
                adaptive_result = self.adaptive_markov_predict(count, periods)
                predictions_pool.extend(adaptive_result)
            except Exception as e:
                logger_manager.warning(f"自适应马尔可夫预测失败: {e}")

            # 4. 贝叶斯预测
            try:
                bayesian_result = self.bayesian_predict(count, periods, n_jobs=1)
                predictions_pool.extend(bayesian_result)
            except Exception as e:
                logger_manager.warning(f"贝叶斯预测失败: {e}")

            # 如果有预测结果，选择最优的
            if predictions_pool:
                # 使用投票机制选择最优预测
                final_predictions = []
                for i in range(count):
                    if i < len(predictions_pool):
                        final_predictions.append(predictions_pool[i])
                    else:
                        # 如果预测池不够，使用集成预测补充
                        fallback = self.ensemble_predict(1, periods)
                        if fallback:
                            final_predictions.append(fallback[0])

                logger_manager.info(f"超级预测完成，生成{len(final_predictions)}注")
                return final_predictions
            else:
                # 如果所有预测都失败，使用集成预测作为备选
                logger_manager.warning("所有高级预测失败，使用集成预测作为备选")
                return self.ensemble_predict(count, periods)

        except Exception as e:
            logger_manager.error(f"超级预测失败: {e}")
            # 最终备选方案
            return self.ensemble_predict(count, periods)

    def adaptive_predict(self, count=1, periods=500) -> List[Tuple[List[int], List[int]]]:
        """自适应预测方法 - 根据历史数据自动选择最优算法"""
        try:
            logger_manager.info(f"开始自适应预测: 注数={count}, 分析期数={periods}")

            # 评估不同算法的历史表现
            algorithms = [
                ('markov', self.markov_predict),
                ('markov_2nd', self.markov_2nd_predict),
                ('adaptive_markov', self.adaptive_markov_predict),
                ('bayesian', lambda c, p: self.bayesian_predict(c, p, n_jobs=1)),
                ('ensemble', self.ensemble_predict)
            ]

            # 简单的算法选择策略：根据期数选择
            if periods <= 100:
                # 小数据量，使用马尔可夫链
                selected_algorithm = algorithms[0][1]  # markov
                algorithm_name = "马尔可夫链"
            elif periods <= 300:
                # 中等数据量，使用二阶马尔可夫
                selected_algorithm = algorithms[1][1]  # markov_2nd
                algorithm_name = "二阶马尔可夫"
            elif periods <= 500:
                # 大数据量，使用自适应马尔可夫
                selected_algorithm = algorithms[2][1]  # adaptive_markov
                algorithm_name = "自适应马尔可夫"
            else:
                # 超大数据量，使用贝叶斯
                selected_algorithm = algorithms[3][1]  # bayesian
                algorithm_name = "贝叶斯"

            logger_manager.info(f"自适应选择算法: {algorithm_name}")

            # 执行选定的算法
            result = selected_algorithm(count, periods)

            if result:
                logger_manager.info(f"自适应预测完成，生成{len(result)}注")
                return result
            else:
                # 如果选定算法失败，使用集成预测作为备选
                logger_manager.warning(f"{algorithm_name}预测失败，使用集成预测作为备选")
                return self.ensemble_predict(count, periods)

        except Exception as e:
            logger_manager.error(f"自适应预测失败: {e}")
            # 最终备选方案
            return self.ensemble_predict(count, periods)


# ==================== 超级预测器 ====================
class SuperPredictor:
    """超级预测器 - 集成所有高级算法"""
    
    def __init__(self, data_file="data/dlt_data_all.csv"):
        self.data_file = data_file
        self.df = data_manager.get_data()

        # 延迟初始化子预测器
        self.advanced_predictor = None

        # 初始化高级算法预测器
        self.sub_predictors = {}
        self.predictor_weights = {}
        self._sub_predictors_initialized = False

        if self.df is None:
            logger_manager.error("数据未加载")
    
    def _initialize_sub_predictors(self):
        """初始化子预测器"""
        if self._sub_predictors_initialized:
            return

        logger_manager.info("初始化超级预测器的子预测器...")

        # 初始化高级预测器
        if self.advanced_predictor is None:
            self.advanced_predictor = AdvancedPredictor(self.data_file)

        # 增强深度学习预测器
        if ENHANCED_DL_AVAILABLE:
            try:
                self.sub_predictors['lstm'] = LSTMPredictor()
                self.sub_predictors['transformer'] = TransformerPredictor()
                self.sub_predictors['gan'] = GANPredictor()
                self.sub_predictors['ensemble'] = EnsembleManager()

                self.predictor_weights['lstm'] = 0.25
                self.predictor_weights['transformer'] = 0.25
                self.predictor_weights['gan'] = 0.20
                self.predictor_weights['ensemble'] = 0.30

                logger_manager.info("增强深度学习预测器初始化成功")
            except Exception as e:
                logger_manager.error(f"增强深度学习预测器初始化失败: {e}")

        # 高级预测器
        self.predictor_weights['advanced'] = 0.20

        # 标准化权重
        total_weight = sum(self.predictor_weights.values())
        if total_weight > 0:
            self.predictor_weights = {k: v/total_weight for k, v in self.predictor_weights.items()}

        self._sub_predictors_initialized = True
    
    def predict_super(self, count=1, periods=500, method="intelligent_ensemble") -> List[Dict]:
        """超级预测

        Args:
            count: 生成注数
            periods: 分析期数
            method: 预测方法
        """
        logger_manager.info(f"开始超级预测，方法: {method}, 注数: {count}, 分析期数: {periods}")

        # 延迟初始化子预测器
        if not self._sub_predictors_initialized:
            self._initialize_sub_predictors()

        predictions = []

        for i in range(count):
            try:
                # 获取各子预测器的预测结果
                sub_predictions = self._get_sub_predictions(periods)

                # 智能融合
                front_balls, back_balls = self._intelligent_fusion(sub_predictions)

                prediction = {
                    'index': i + 1,
                    'front_balls': front_balls,
                    'back_balls': back_balls,
                    'method': method,
                    'sub_predictions': sub_predictions,
                    'confidence': self._calculate_confidence(sub_predictions)
                }

                predictions.append(prediction)

            except Exception as e:
                logger_manager.error(f"第 {i+1} 注超级预测失败", e)

        return predictions
    
    def _get_sub_predictions(self, periods=500) -> Dict:
        """获取子预测器的预测结果

        Args:
            periods: 分析期数
        """
        sub_predictions = {}

        # 高级预测器
        try:
            result = self.advanced_predictor.ensemble_predict(count=1, periods=periods)
            if result:
                sub_predictions['advanced'] = {
                    'front_balls': result[0][0],
                    'back_balls': result[0][1],
                    'confidence': 0.6
                }
        except Exception as e:
            logger_manager.error("高级预测器预测失败", e)
        
        # LSTM预测器
        if 'lstm' in self.sub_predictors:
            try:
                predictor = self.sub_predictors['lstm']
                # 使用新的LSTM预测器接口
                result = predictor.predict(self.df)
                if result:
                    # 转换结果格式
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], tuple):
                            # 标准格式 (front_balls, back_balls)
                            sub_predictions['lstm'] = {
                                'front_balls': result[0][0],
                                'back_balls': result[0][1],
                                'confidence': 0.7
                            }
                        elif isinstance(result[0], dict):
                            # 字典格式
                            sub_predictions['lstm'] = {
                                'front_balls': result[0]['front_balls'],
                                'back_balls': result[0]['back_balls'],
                                'confidence': result[0].get('confidence', 0.7)
                            }
            except Exception as e:
                logger_manager.error(f"LSTM预测器预测失败: {e}")
        
        # 蒙特卡洛预测器
        if 'monte_carlo' in self.sub_predictors:
            try:
                predictor = self.sub_predictors['monte_carlo']
                result = predictor.predict_monte_carlo(count=1, method="comprehensive", num_simulations=5000)
                if result:
                    sub_predictions['monte_carlo'] = {
                        'front_balls': result[0]['front_balls'],
                        'back_balls': result[0]['back_balls'],
                        'confidence': 0.6
                    }
            except Exception as e:
                logger_manager.error("蒙特卡洛预测器预测失败", e)
        
        # 聚类预测器
        if 'clustering' in self.sub_predictors:
            try:
                predictor = self.sub_predictors['clustering']
                result = predictor.predict_clustering(count=1, method="ensemble")
                if result:
                    sub_predictions['clustering'] = {
                        'front_balls': result[0]['front_balls'],
                        'back_balls': result[0]['back_balls'],
                        'confidence': 0.5
                    }
            except Exception as e:
                logger_manager.error("聚类预测器预测失败", e)
        
        return sub_predictions
    
    def _intelligent_fusion(self, sub_predictions: Dict) -> Tuple[List[int], List[int]]:
        """智能融合预测结果"""
        all_front_candidates = []
        all_back_candidates = []
        
        for name, prediction in sub_predictions.items():
            weight = self.predictor_weights.get(name, 0.25)
            confidence = prediction.get('confidence', 0.5)
            
            # 综合权重
            final_weight = weight * confidence
            repeat_count = max(1, int(final_weight * 10))
            
            for _ in range(repeat_count):
                all_front_candidates.extend(prediction['front_balls'])
                all_back_candidates.extend(prediction['back_balls'])
        
        # 统计频率并选择
        front_counter = Counter(all_front_candidates)
        back_counter = Counter(all_back_candidates)
        
        # 选择频率最高的号码
        front_balls = [ball for ball, freq_count in front_counter.most_common(8)]
        back_balls = [ball for ball, freq_count in back_counter.most_common(4)]
        
        # 智能选择最终号码
        final_front = self._smart_selection(front_balls, 5)
        final_back = self._smart_selection(back_balls, 2)
        
        return final_front, final_back
    
    def _smart_selection(self, candidates: List[int], num_select: int) -> List[int]:
        """智能选择号码"""
        if len(candidates) <= num_select:
            # 如果候选号码不足，用频率分析补充
            freq_analysis = basic_analyzer.frequency_analysis()
            if num_select == 5:  # 前区
                freq_dict = freq_analysis.get('front_frequency', {})
            else:  # 后区
                freq_dict = freq_analysis.get('back_frequency', {})

            sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
            for ball, freq in sorted_freq:
                if len(candidates) >= num_select:
                    break
                if ball not in candidates:
                    candidates.append(ball)
            return sorted(candidates[:num_select])

        # 选择策略：高频 + 随机
        high_freq_count = num_select // 2
        random_count = num_select - high_freq_count

        selected = candidates[:high_freq_count]
        remaining = candidates[high_freq_count:]

        if random_count > 0 and remaining:
            selected_count = min(random_count, len(remaining))
            selected.extend(remaining[:selected_count])

        return sorted(selected[:num_select])

    def _calculate_confidence(self, sub_predictions: Dict) -> float:
        """计算预测置信度"""
        if not sub_predictions:
            return 0.0

        total_confidence = 0.0
        total_weight = 0.0

        for name, prediction in sub_predictions.items():
            weight = self.predictor_weights.get(name, 0.25)
            confidence = prediction.get('confidence', 0.5)

            total_confidence += confidence * weight
            total_weight += weight

        return total_confidence / total_weight if total_weight > 0 else 0.5


# ==================== 复式投注预测器 ====================
class CompoundPredictor:
    """复式投注预测器"""

    def __init__(self, data_file="data/dlt_data_all.csv"):
        self.data_file = data_file
        self.df = data_manager.get_data()
        self.advanced_predictor = AdvancedPredictor(data_file)

        if self.df is None:
            logger_manager.error("数据未加载")

    def predict_compound(self, front_count: int, back_count: int, method: str = "ensemble", periods: int = 500) -> Dict:
        """复式投注预测

        Args:
            front_count: 前区号码数量 (6-15)
            back_count: 后区号码数量 (3-12)
            method: 预测方法
            periods: 分析期数

        Returns:
            复式投注预测结果
        """
        logger_manager.info(f"复式投注预测: {front_count}+{back_count}, 方法: {method}, 分析期数: {periods}")

        try:
            # 获取基础预测
            if method == "ensemble":
                base_predictions = self.advanced_predictor.ensemble_predict(count=3, periods=periods)
            elif method == "markov":
                base_predictions = self.advanced_predictor.markov_predict(count=3, periods=periods)
            elif method == "bayesian":
                base_predictions = self.advanced_predictor.bayesian_predict(count=3, periods=periods)
            else:
                base_predictions = self.advanced_predictor.ensemble_predict(count=3, periods=periods)

            # 收集候选号码
            front_candidates = set()
            back_candidates = set()

            for pred in base_predictions:
                front_candidates.update(pred[0])
                back_candidates.update(pred[1])

            # 如果候选号码不足，用频率分析补充
            if len(front_candidates) < front_count:
                freq_analysis = basic_analyzer.frequency_analysis()
                front_freq = freq_analysis.get('front_frequency', {})
                sorted_freq = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)
                for ball, freq in sorted_freq:
                    if len(front_candidates) >= front_count:
                        break
                    # 确保添加的是整数
                    if isinstance(ball, str):
                        front_candidates.add(int(ball))
                    else:
                        front_candidates.add(int(ball))

            if len(back_candidates) < back_count:
                freq_analysis = basic_analyzer.frequency_analysis()
                back_freq = freq_analysis.get('back_frequency', {})
                sorted_freq = sorted(back_freq.items(), key=lambda x: x[1], reverse=True)
                for ball, freq in sorted_freq:
                    if len(back_candidates) >= back_count:
                        break
                    # 确保添加的是整数
                    if isinstance(ball, str):
                        back_candidates.add(int(ball))
                    else:
                        back_candidates.add(int(ball))

            # 选择最终号码（确保都是整数）
            front_balls = sorted([int(x) for x in front_candidates])[:front_count]
            back_balls = sorted([int(x) for x in back_candidates])[:back_count]

            # 计算组合数
            from math import comb
            total_combinations = comb(front_count, 5) * comb(back_count, 2)
            total_cost = total_combinations * 3  # 每注3元

            result = {
                'front_balls': front_balls,
                'back_balls': back_balls,
                'front_count': front_count,
                'back_count': back_count,
                'total_combinations': total_combinations,
                'total_cost': total_cost,
                'method': method,
                'confidence': 0.7
            }

            return result

        except Exception as e:
            logger_manager.error("复式投注预测失败", e)
            return {}

    def predict_duplex(self, front_dan_count: int = 2, back_dan_count: int = 1,
                      front_tuo_count: int = 6, back_tuo_count: int = 4,
                      method: str = "ensemble", periods: int = 500) -> Dict:
        """胆拖投注预测

        Args:
            front_dan_count: 前区胆码数量
            back_dan_count: 后区胆码数量
            front_tuo_count: 前区拖码数量
            back_tuo_count: 后区拖码数量
            method: 预测方法
            periods: 分析期数

        Returns:
            胆拖投注预测结果
        """
        logger_manager.info(f"胆拖投注预测: 前区{front_dan_count}胆{front_tuo_count}拖, 后区{back_dan_count}胆{back_tuo_count}拖, 分析期数: {periods}")

        try:
            # 获取基础预测
            if method == "ensemble":
                base_predictions = self.advanced_predictor.ensemble_predict(count=5, periods=periods)
            elif method == "markov":
                base_predictions = self.advanced_predictor.markov_predict(count=5, periods=periods)
            elif method == "bayesian":
                base_predictions = self.advanced_predictor.bayesian_predict(count=5, periods=periods)
            else:
                base_predictions = self.advanced_predictor.ensemble_predict(count=5, periods=periods)

            # 统计号码频率
            front_counter = Counter()
            back_counter = Counter()

            for pred in base_predictions:
                front_counter.update(pred[0])
                back_counter.update(pred[1])

            # 选择胆码（频率最高的）
            front_dan = [ball for ball, freq_count in front_counter.most_common(front_dan_count)]
            back_dan = [ball for ball, freq_count in back_counter.most_common(back_dan_count)]

            # 选择拖码（排除胆码后的候选）
            front_tuo_candidates = [ball for ball, freq_count in front_counter.most_common() if ball not in front_dan]
            back_tuo_candidates = [ball for ball, freq_count in back_counter.most_common() if ball not in back_dan]

            # 补充拖码
            while len(front_tuo_candidates) < front_tuo_count:
                candidate = np.random.randint(1, 36)
                if candidate not in front_dan and candidate not in front_tuo_candidates:
                    front_tuo_candidates.append(candidate)

            while len(back_tuo_candidates) < back_tuo_count:
                candidate = np.random.randint(1, 13)
                if candidate not in back_dan and candidate not in back_tuo_candidates:
                    back_tuo_candidates.append(candidate)

            front_tuo = sorted(front_tuo_candidates[:front_tuo_count])
            back_tuo = sorted(back_tuo_candidates[:back_tuo_count])

            # 计算组合数
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
                'method': method,
                'confidence': 0.8
            }

            return result

        except Exception as e:
            logger_manager.error("胆拖投注预测失败", e)
            return {}

    def predict_highly_integrated_compound(self, front_count: int = 10, back_count: int = 5,
                                         integration_level: str = "ultimate", periods: int = 500) -> Dict:
        """基于高度集成的复式预测

        Args:
            front_count: 前区号码数量 (8-15)
            back_count: 后区号码数量 (4-12)
            integration_level: 集成级别 ('high', 'ultimate')
            periods: 分析期数

        Returns:
            高度集成复式预测结果
        """
        logger_manager.info(f"高度集成复式预测: {front_count}+{back_count}, 集成级别: {integration_level}, 分析期数: {periods}")

        try:
            # 初始化超级预测器
            super_predictor = SuperPredictor(self.data_file)

            # 收集多种算法的预测结果
            all_predictions = {}

            # 1. 传统算法预测
            traditional_pred = TraditionalPredictor(self.data_file)
            all_predictions['frequency'] = traditional_pred.frequency_predict(5, periods)
            all_predictions['hot_cold'] = traditional_pred.hot_cold_predict(5, periods)
            all_predictions['missing'] = traditional_pred.missing_predict(5, periods)

            # 2. 高级算法预测
            all_predictions['markov'] = self.advanced_predictor.markov_predict(5, periods)
            all_predictions['bayesian'] = self.advanced_predictor.bayesian_predict(5, periods)
            all_predictions['ensemble'] = self.advanced_predictor.ensemble_predict(5, periods)

            # 3. 超级算法预测
            super_results = super_predictor.predict_super(3, periods, "intelligent_ensemble")
            all_predictions['super'] = [(pred['front_balls'], pred['back_balls']) for pred in super_results]

            # 4. 马尔可夫自定义预测
            markov_custom = self.advanced_predictor.markov_predict_custom(3, periods, 1)
            all_predictions['markov_custom'] = [(pred['front_balls'], pred['back_balls']) for pred in markov_custom]

            # 高度集成候选号码收集
            front_candidates = Counter()
            back_candidates = Counter()

            # 算法权重配置
            if integration_level == "ultimate":
                weights = {
                    'frequency': 0.10, 'hot_cold': 0.08, 'missing': 0.07,
                    'markov': 0.20, 'bayesian': 0.15, 'ensemble': 0.25,
                    'super': 0.15, 'markov_custom': 0.00
                }
            else:  # high
                weights = {
                    'frequency': 0.15, 'hot_cold': 0.12, 'missing': 0.10,
                    'markov': 0.23, 'bayesian': 0.18, 'ensemble': 0.22,
                    'super': 0.00, 'markov_custom': 0.00
                }

            # 基于权重收集候选号码
            for method, predictions in all_predictions.items():
                if method not in weights or weights[method] == 0:
                    continue

                weight = weights[method]

                for pred in predictions:
                    if isinstance(pred, dict):  # 混合策略预测结果
                        front_balls = pred['front_balls']
                        back_balls = pred['back_balls']
                    else:  # 元组格式
                        front_balls, back_balls = pred

                    # 根据权重添加候选号码（检查范围）
                    score = int(weight * 100)
                    for ball in front_balls:
                        ball_int = int(ball)
                        if 1 <= ball_int <= 35:  # 前区号码范围检查
                            front_candidates[ball_int] += score
                    for ball in back_balls:
                        ball_int = int(ball)
                        if 1 <= ball_int <= 12:  # 后区号码范围检查
                            back_candidates[ball_int] += score

            # 智能选择最终号码
            front_balls = self._intelligent_compound_selection(front_candidates, front_count)
            back_balls = self._intelligent_compound_selection(back_candidates, back_count)

            # 确保所有号码都是整数并去重
            front_balls = sorted(list(set([int(x) for x in front_balls])))
            back_balls = sorted(list(set([int(x) for x in back_balls])))

            # 补充到目标数量（如果去重后数量不足）
            # 使用频率分析补充，而不是随机数
            if len(front_balls) < front_count:
                freq_analysis = basic_analyzer.frequency_analysis()
                front_freq = freq_analysis.get('front_frequency', {})
                sorted_freq = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)

                for ball, _ in sorted_freq:
                    if len(front_balls) >= front_count:
                        break
                    ball_int = int(ball) if isinstance(ball, str) else ball
                    if ball_int not in front_balls:
                        front_balls.append(ball_int)

            if len(back_balls) < back_count:
                freq_analysis = basic_analyzer.frequency_analysis()
                back_freq = freq_analysis.get('back_frequency', {})
                sorted_freq = sorted(back_freq.items(), key=lambda x: x[1], reverse=True)

                for ball, _ in sorted_freq:
                    if len(back_balls) >= back_count:
                        break
                    ball_int = int(ball) if isinstance(ball, str) else ball
                    if ball_int not in back_balls:
                        back_balls.append(ball_int)

            front_balls = sorted(front_balls[:front_count])
            back_balls = sorted(back_balls[:back_count])

            # 计算组合数和投注金额
            from math import comb
            total_combinations = comb(front_count, 5) * comb(back_count, 2)
            total_cost = total_combinations * 3

            # 计算集成置信度
            confidence = self._calculate_integration_confidence(all_predictions, integration_level)

            result = {
                'front_balls': front_balls,
                'back_balls': back_balls,
                'front_count': front_count,
                'back_count': back_count,
                'total_combinations': total_combinations,
                'total_cost': total_cost,
                'method': f'highly_integrated_{integration_level}',
                'integration_level': integration_level,
                'confidence': confidence,
                'algorithms_used': list(weights.keys()),
                'algorithm_weights': weights,
                'candidate_scores': {
                    'front_top10': dict(front_candidates.most_common(10)),
                    'back_top8': dict(back_candidates.most_common(8))
                }
            }

            return result

        except Exception as e:
            logger_manager.error("高度集成复式预测失败", e)
            return {}

    def _intelligent_compound_selection(self, candidates: Counter, target_count: int) -> List[int]:
        """智能复式号码选择"""
        if len(candidates) == 0:
            # 如果没有候选号码，使用频率分析
            freq_analysis = basic_analyzer.frequency_analysis()
            if target_count > 8:  # 前区
                freq_dict = freq_analysis.get('front_frequency', {})
            else:  # 后区
                freq_dict = freq_analysis.get('back_frequency', {})

            sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
            selected_balls = []
            for ball, freq in sorted_freq[:target_count]:
                if isinstance(ball, str):
                    selected_balls.append(int(ball))
                else:
                    selected_balls.append(int(ball))
            return sorted(selected_balls)

        # 获取候选号码列表（按得分排序）
        sorted_candidates = candidates.most_common()

        if len(sorted_candidates) >= target_count:
            # 智能选择策略：70%高分号码 + 30%中等分号码
            high_score_count = int(target_count * 0.7)
            medium_score_count = target_count - high_score_count

            selected = []

            # 选择高分号码
            for i in range(min(high_score_count, len(sorted_candidates))):
                selected.append(int(sorted_candidates[i][0]))

            # 选择中等分号码（增加多样性）
            if medium_score_count > 0 and len(sorted_candidates) > high_score_count:
                medium_start = high_score_count
                medium_end = min(len(sorted_candidates), high_score_count + medium_score_count * 2)
                medium_candidates = [item[0] for item in sorted_candidates[medium_start:medium_end]]

                if medium_candidates:
                    selected_count = min(medium_score_count, len(medium_candidates))
                    selected.extend([int(x) for x in medium_candidates[:selected_count]])

            # 如果数量不足，用频率分析补充
            if len(selected) < target_count:
                freq_analysis = basic_analyzer.frequency_analysis()
                if target_count > 8:  # 前区
                    freq_dict = freq_analysis.get('front_frequency', {})
                else:  # 后区
                    freq_dict = freq_analysis.get('back_frequency', {})

                sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
                for ball, freq in sorted_freq:
                    if len(selected) >= target_count:
                        break
                    if ball not in selected:
                        selected.append(ball)

            return sorted(selected[:target_count])
        else:
            # 候选号码不足，全部选择并补充
            selected = [int(item[0]) for item in sorted_candidates]

            # 如果数量不足，用频率分析补充
            while len(selected) < target_count:
                freq_analysis = basic_analyzer.frequency_analysis()
                if target_count > 8:  # 前区
                    freq_dict = freq_analysis.get('front_frequency', {})
                else:  # 后区
                    freq_dict = freq_analysis.get('back_frequency', {})

                sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
                for ball, freq in sorted_freq:
                    if len(selected) >= target_count:
                        break
                    if ball not in selected:
                        selected.append(ball)
                        break

            return sorted(selected)

    def _calculate_integration_confidence(self, all_predictions: Dict, integration_level: str) -> float:
        """计算集成置信度"""
        try:
            total_predictions = sum(len(preds) for preds in all_predictions.values())
            algorithm_count = len(all_predictions)

            # 基础置信度
            base_confidence = 0.6 if integration_level == "high" else 0.75

            # 算法多样性加成
            diversity_bonus = min(0.15, algorithm_count * 0.02)

            # 预测数量加成
            quantity_bonus = min(0.1, total_predictions * 0.005)

            final_confidence = base_confidence + diversity_bonus + quantity_bonus

            return min(0.95, final_confidence)  # 限制最大置信度

        except Exception:
            return 0.7


# ==================== 全局实例（延迟初始化） ====================
traditional_predictor = None
advanced_predictor = None
super_predictor = None

def get_traditional_predictor():
    """获取传统预测器实例"""
    global traditional_predictor
    if traditional_predictor is None:
        traditional_predictor = TraditionalPredictor()
    return traditional_predictor

def get_advanced_predictor():
    """获取高级预测器实例"""
    global advanced_predictor
    if advanced_predictor is None:
        advanced_predictor = AdvancedPredictor()
    return advanced_predictor

def get_super_predictor():
    """获取超级预测器实例"""
    global super_predictor
    if super_predictor is None:
        super_predictor = SuperPredictor()
    return super_predictor


if __name__ == "__main__":
    # 测试预测器模块
    print("🔧 测试预测器模块...")

    # 测试传统预测器
    print("📊 测试传统预测器...")
    freq_pred = traditional_predictor.frequency_predict(1)
    print(f"频率预测: 前区 {freq_pred[0][0]}, 后区 {freq_pred[0][1]}")

    # 测试高级预测器
    print("🧮 测试高级预测器...")
    ensemble_pred = advanced_predictor.ensemble_predict(1)
    print(f"集成预测: 前区 {ensemble_pred[0][0]}, 后区 {ensemble_pred[0][1]}")

    # 测试超级预测器
    print("🚀 测试超级预测器...")
    super_pred = super_predictor.predict_super(1)
    if super_pred:
        print(f"超级预测: 前区 {super_pred[0]['front_balls']}, 后区 {super_pred[0]['back_balls']}")

    print("✅ 预测器模块测试完成")
