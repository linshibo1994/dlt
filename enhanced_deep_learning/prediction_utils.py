#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
预测工具模块
提供预测结果处理和评估功能
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter

from core_modules import logger_manager


class PredictionProcessor:
    """预测结果处理器"""
    
    def __init__(self, front_range=(1, 35), back_range=(1, 12)):
        """
        初始化预测结果处理器
        
        Args:
            front_range: 前区号码范围
            back_range: 后区号码范围
        """
        self.front_min, self.front_max = front_range
        self.back_min, self.back_max = back_range
    
    def process_raw_prediction(self, raw_prediction: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        处理原始预测结果
        
        Args:
            raw_prediction: 原始预测结果数组
            
        Returns:
            处理后的前区和后区号码
        """
        # 分离前区和后区预测
        front_pred = raw_prediction[:5]
        back_pred = raw_prediction[5:7]
        
        # 转换为彩票号码
        front_balls = [max(self.front_min, min(self.front_max, int(round(x)))) for x in front_pred]
        back_balls = [max(self.back_min, min(self.back_max, int(round(x)))) for x in back_pred]
        
        # 确保号码唯一性
        front_balls = self._ensure_unique_numbers(front_balls, self.front_min, self.front_max, 5)
        back_balls = self._ensure_unique_numbers(back_balls, self.back_min, self.back_max, 2)
        
        return sorted(front_balls), sorted(back_balls)
    
    def _ensure_unique_numbers(self, numbers: List[int], min_val: int, max_val: int, target_count: int) -> List[int]:
        """
        确保号码唯一性
        
        Args:
            numbers: 号码列表
            min_val: 最小值
            max_val: 最大值
            target_count: 目标数量
            
        Returns:
            唯一号码列表
        """
        unique_numbers = list(set(numbers))
        
        # 如果数量不足，随机补充
        while len(unique_numbers) < target_count:
            candidate = np.random.randint(min_val, max_val + 1)
            if candidate not in unique_numbers:
                unique_numbers.append(candidate)
        
        # 如果数量过多，截取
        if len(unique_numbers) > target_count:
            unique_numbers = unique_numbers[:target_count]
        
        return unique_numbers
    
    def format_prediction(self, prediction: Tuple[List[int], List[int]]) -> str:
        """
        格式化预测结果
        
        Args:
            prediction: 预测结果元组
            
        Returns:
            格式化的预测结果字符串
        """
        front, back = prediction
        front_str = ' '.join([str(b).zfill(2) for b in front])
        back_str = ' '.join([str(b).zfill(2) for b in back])
        return f"{front_str} + {back_str}"
    
    def calculate_confidence(self, predictions: List[Tuple[List[int], List[int]]]) -> float:
        """
        计算预测结果的置信度
        
        Args:
            predictions: 多组预测结果
            
        Returns:
            置信度分数 (0.0-1.0)
        """
        if not predictions:
            return 0.0
        
        # 统计号码出现频率
        front_counter = Counter()
        back_counter = Counter()
        
        for front, back in predictions:
            for num in front:
                front_counter[num] += 1
            for num in back:
                back_counter[num] += 1
        
        # 计算前区一致性
        front_total = sum(front_counter.values())
        front_max_possible = len(predictions) * 5  # 每组预测5个前区号码
        front_consistency = front_total / front_max_possible if front_max_possible > 0 else 0
        
        # 计算后区一致性
        back_total = sum(back_counter.values())
        back_max_possible = len(predictions) * 2  # 每组预测2个后区号码
        back_consistency = back_total / back_max_possible if back_max_possible > 0 else 0
        
        # 综合置信度
        confidence = (front_consistency * 0.7 + back_consistency * 0.3)
        
        return min(0.95, confidence)  # 最高置信度限制在0.95


class PredictionEvaluator:
    """预测结果评估器"""
    
    def __init__(self):
        """初始化预测结果评估器"""
        pass
    
    def evaluate_prediction(self, prediction: Tuple[List[int], List[int]], 
                           actual: Tuple[List[int], List[int]]) -> Dict[str, Any]:
        """
        评估单组预测结果
        
        Args:
            prediction: 预测结果
            actual: 实际结果
            
        Returns:
            评估结果字典
        """
        pred_front, pred_back = prediction
        actual_front, actual_back = actual
        
        # 计算前区匹配数
        front_matches = len(set(pred_front) & set(actual_front))
        
        # 计算后区匹配数
        back_matches = len(set(pred_back) & set(actual_back))
        
        # 计算奖级
        prize_level = self._calculate_prize_level(front_matches, back_matches)
        
        # 计算匹配率
        front_match_rate = front_matches / len(actual_front)
        back_match_rate = back_matches / len(actual_back)
        overall_match_rate = (front_match_rate * len(actual_front) + back_match_rate * len(actual_back)) / (len(actual_front) + len(actual_back))
        
        return {
            'front_matches': front_matches,
            'back_matches': back_matches,
            'prize_level': prize_level,
            'front_match_rate': front_match_rate,
            'back_match_rate': back_match_rate,
            'overall_match_rate': overall_match_rate,
            'has_prize': prize_level > 0
        }
    
    def evaluate_multiple_predictions(self, predictions: List[Tuple[List[int], List[int]]], 
                                     actuals: List[Tuple[List[int], List[int]]]) -> Dict[str, Any]:
        """
        评估多组预测结果
        
        Args:
            predictions: 预测结果列表
            actuals: 实际结果列表
            
        Returns:
            评估结果字典
        """
        if not predictions or not actuals:
            return {
                'avg_front_matches': 0,
                'avg_back_matches': 0,
                'prize_count': 0,
                'prize_rate': 0,
                'avg_match_rate': 0
            }
        
        # 评估每组预测
        evaluations = []
        for i, pred in enumerate(predictions):
            if i < len(actuals):
                eval_result = self.evaluate_prediction(pred, actuals[i])
                evaluations.append(eval_result)
        
        # 计算平均指标
        avg_front_matches = sum(e['front_matches'] for e in evaluations) / len(evaluations)
        avg_back_matches = sum(e['back_matches'] for e in evaluations) / len(evaluations)
        prize_count = sum(1 for e in evaluations if e['has_prize'])
        prize_rate = prize_count / len(evaluations)
        avg_match_rate = sum(e['overall_match_rate'] for e in evaluations) / len(evaluations)
        
        return {
            'avg_front_matches': avg_front_matches,
            'avg_back_matches': avg_back_matches,
            'prize_count': prize_count,
            'prize_rate': prize_rate,
            'avg_match_rate': avg_match_rate
        }
    
    def _calculate_prize_level(self, front_matches: int, back_matches: int) -> int:
        """
        计算奖级
        
        Args:
            front_matches: 前区匹配数
            back_matches: 后区匹配数
            
        Returns:
            奖级（0表示未中奖）
        """
        # 大乐透奖级规则
        if front_matches == 5 and back_matches == 2:
            return 1  # 一等奖
        elif front_matches == 5 and back_matches == 1:
            return 2  # 二等奖
        elif front_matches == 5 and back_matches == 0:
            return 3  # 三等奖
        elif front_matches == 4 and back_matches == 2:
            return 4  # 四等奖
        elif front_matches == 4 and back_matches == 1:
            return 5  # 五等奖
        elif front_matches == 3 and back_matches == 2:
            return 6  # 六等奖
        elif front_matches == 4 and back_matches == 0:
            return 7  # 七等奖
        elif front_matches == 3 and back_matches == 1:
            return 8  # 八等奖
        elif front_matches == 2 and back_matches == 2:
            return 9  # 九等奖
        elif (front_matches == 3 and back_matches == 0) or (front_matches == 2 and back_matches == 1) or (front_matches == 1 and back_matches == 2) or (front_matches == 0 and back_matches == 2):
            return 10  # 十等奖
        else:
            return 0  # 未中奖