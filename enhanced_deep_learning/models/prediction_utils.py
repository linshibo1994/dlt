#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
预测工具模块
Prediction Utilities Module

提供预测处理和评估的通用工具。
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from datetime import datetime
import core_modules as cm

logger_manager = cm.logger_manager


class PredictionProcessor:
    """预测结果处理器"""
    
    def __init__(self):
        """初始化预测处理器"""
        self.confidence_threshold = 0.5

    def process_raw_prediction(self, raw_prediction: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        处理原始预测结果，确保无重复号码

        Args:
            raw_prediction: 原始预测数组，前7个元素分别为前区5个号码和后区2个号码

        Returns:
            处理后的预测结果 (前区号码列表, 后区号码列表)
        """
        try:
            # 确保输入是numpy数组
            if not isinstance(raw_prediction, np.ndarray):
                raw_prediction = np.array(raw_prediction)

            # 提取前区和后区的原始预测
            front_raw = raw_prediction[:5] if len(raw_prediction) >= 5 else raw_prediction
            back_raw = raw_prediction[5:7] if len(raw_prediction) >= 7 else raw_prediction[-2:]

            # 转换为有效范围内的整数
            front_candidates = [max(1, min(35, int(round(float(x))))) for x in front_raw]
            back_candidates = [max(1, min(12, int(round(float(x))))) for x in back_raw]

            # 去除重复并确保数量正确 - 前区5个不重复号码
            front_balls = []
            front_set = set()
            for num in front_candidates:
                if num not in front_set and len(front_balls) < 5:
                    front_balls.append(num)
                    front_set.add(num)

            # 如果前区号码不够5个，随机补充
            if len(front_balls) < 5:
                import random
                available_front = [i for i in range(1, 36) if i not in front_set]
                random.shuffle(available_front)
                front_balls.extend(available_front[:5-len(front_balls)])

            # 去除重复并确保数量正确 - 后区2个不重复号码
            back_balls = []
            back_set = set()
            for num in back_candidates:
                if num not in back_set and len(back_balls) < 2:
                    back_balls.append(num)
                    back_set.add(num)

            # 如果后区号码不够2个，随机补充
            if len(back_balls) < 2:
                import random
                available_back = [i for i in range(1, 13) if i not in back_set]
                random.shuffle(available_back)
                back_balls.extend(available_back[:2-len(back_balls)])

            # 排序
            front_balls = sorted(front_balls)
            back_balls = sorted(back_balls)

            return front_balls, back_balls

        except Exception as e:
            logger_manager.error(f"原始预测处理失败: {e}")
            # 返回随机号码作为备选
            import random
            front_balls = sorted(random.sample(range(1, 36), 5))
            back_balls = sorted(random.sample(range(1, 13), 2))
            return front_balls, back_balls

    def format_prediction(self, prediction: Tuple[List[int], List[int]]) -> str:
        """
        格式化单个预测结果

        Args:
            prediction: 预测结果 (前区, 后区)

        Returns:
            格式化的字符串
        """
        front_balls, back_balls = prediction
        front_str = ' '.join([str(b).zfill(2) for b in front_balls])
        back_str = ' '.join([str(b).zfill(2) for b in back_balls])
        return f"{front_str} + {back_str}"

    def calculate_confidence(self, predictions: List[Tuple[List[int], List[int]]]) -> float:
        """
        计算预测置信度

        Args:
            predictions: 预测结果列表

        Returns:
            置信度分数 (0-1)
        """
        try:
            if not predictions:
                return 0.0

            # 简单的置信度计算：基于号码分布的均匀性
            all_front = []
            all_back = []

            for front, back in predictions:
                all_front.extend(front)
                all_back.extend(back)

            # 计算号码分布的标准差，标准差越小，置信度越高
            front_std = np.std(all_front) if all_front else 0
            back_std = np.std(all_back) if all_back else 0

            # 归一化到0-1范围
            front_confidence = max(0, 1 - front_std / 35)
            back_confidence = max(0, 1 - back_std / 12)

            return (front_confidence + back_confidence) / 2

        except Exception as e:
            logger_manager.error(f"置信度计算失败: {e}")
            return 0.5
        
    def process_predictions(self, predictions: List[Tuple[List[int], List[int]]], 
                          confidence_scores: List[float] = None) -> Dict[str, Any]:
        """
        处理预测结果
        
        Args:
            predictions: 预测结果列表
            confidence_scores: 置信度分数列表
            
        Returns:
            处理后的预测结果字典
        """
        try:
            if not predictions:
                return {'error': '没有预测结果'}
            
            # 计算统计信息
            total_predictions = len(predictions)
            
            # 分析前区和后区号码分布
            front_numbers = []
            back_numbers = []
            
            for front, back in predictions:
                front_numbers.extend(front)
                back_numbers.extend(back)
            
            # 计算号码频率
            front_freq = {}
            back_freq = {}
            
            for num in front_numbers:
                front_freq[num] = front_freq.get(num, 0) + 1
                
            for num in back_numbers:
                back_freq[num] = back_freq.get(num, 0) + 1
            
            # 计算平均置信度
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.8
            
            result = {
                'predictions': predictions,
                'total_count': total_predictions,
                'front_frequency': front_freq,
                'back_frequency': back_freq,
                'average_confidence': avg_confidence,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger_manager.error(f"预测结果处理失败: {e}")
            return {'error': str(e)}
    
    def format_predictions(self, predictions: List[Tuple[List[int], List[int]]]) -> List[str]:
        """
        格式化预测结果为字符串
        
        Args:
            predictions: 预测结果列表
            
        Returns:
            格式化后的字符串列表
        """
        formatted = []
        
        for i, (front, back) in enumerate(predictions, 1):
            front_str = ' '.join([f"{num:02d}" for num in sorted(front)])
            back_str = ' '.join([f"{num:02d}" for num in sorted(back)])
            formatted.append(f"第{i}注: 前区 {front_str} | 后区 {back_str}")
        
        return formatted


class PredictionEvaluator:
    """预测结果评估器"""
    
    def __init__(self):
        """初始化评估器"""
        pass
    
    def evaluate_single_prediction(self, prediction: Tuple[List[int], List[int]], 
                                 actual: Tuple[List[int], List[int]]) -> Dict[str, Any]:
        """
        评估单个预测结果
        
        Args:
            prediction: 预测结果 (前区, 后区)
            actual: 实际结果 (前区, 后区)
            
        Returns:
            评估结果字典
        """
        try:
            pred_front, pred_back = prediction
            actual_front, actual_back = actual
            
            # 计算命中数
            front_hits = len(set(pred_front) & set(actual_front))
            back_hits = len(set(pred_back) & set(actual_back))
            
            # 计算准确率
            front_accuracy = front_hits / len(actual_front) if actual_front else 0
            back_accuracy = back_hits / len(actual_back) if actual_back else 0
            
            # 总体准确率
            total_accuracy = (front_hits + back_hits) / (len(actual_front) + len(actual_back))
            
            result = {
                'front_hits': front_hits,
                'back_hits': back_hits,
                'front_accuracy': front_accuracy,
                'back_accuracy': back_accuracy,
                'total_accuracy': total_accuracy,
                'is_jackpot': front_hits == len(actual_front) and back_hits == len(actual_back)
            }
            
            return result
            
        except Exception as e:
            logger_manager.error(f"单个预测评估失败: {e}")
            return {'error': str(e)}
    
    def evaluate_multiple_predictions(self, predictions: List[Tuple[List[int], List[int]]], 
                                    actuals: List[Tuple[List[int], List[int]]]) -> Dict[str, Any]:
        """
        评估多个预测结果
        
        Args:
            predictions: 预测结果列表
            actuals: 实际结果列表
            
        Returns:
            评估结果字典
        """
        try:
            if len(predictions) != len(actuals):
                return {'error': '预测结果和实际结果数量不匹配'}
            
            evaluations = []
            total_front_hits = 0
            total_back_hits = 0
            jackpot_count = 0
            
            for pred, actual in zip(predictions, actuals):
                eval_result = self.evaluate_single_prediction(pred, actual)
                evaluations.append(eval_result)
                
                if 'error' not in eval_result:
                    total_front_hits += eval_result['front_hits']
                    total_back_hits += eval_result['back_hits']
                    if eval_result['is_jackpot']:
                        jackpot_count += 1
            
            # 计算总体统计
            total_predictions = len(predictions)
            avg_front_accuracy = np.mean([e.get('front_accuracy', 0) for e in evaluations if 'error' not in e])
            avg_back_accuracy = np.mean([e.get('back_accuracy', 0) for e in evaluations if 'error' not in e])
            avg_total_accuracy = np.mean([e.get('total_accuracy', 0) for e in evaluations if 'error' not in e])
            
            result = {
                'individual_evaluations': evaluations,
                'total_predictions': total_predictions,
                'total_front_hits': total_front_hits,
                'total_back_hits': total_back_hits,
                'jackpot_count': jackpot_count,
                'jackpot_rate': jackpot_count / total_predictions,
                'average_front_accuracy': avg_front_accuracy,
                'average_back_accuracy': avg_back_accuracy,
                'average_total_accuracy': avg_total_accuracy
            }
            
            return result
            
        except Exception as e:
            logger_manager.error(f"多个预测评估失败: {e}")
            return {'error': str(e)}


class PredictionValidator:
    """预测结果验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.front_range = (1, 35)
        self.back_range = (1, 12)
        self.front_count = 5
        self.back_count = 2
    
    def validate_prediction(self, prediction: Tuple[List[int], List[int]]) -> Dict[str, Any]:
        """
        验证单个预测结果
        
        Args:
            prediction: 预测结果 (前区, 后区)
            
        Returns:
            验证结果字典
        """
        try:
            front, back = prediction
            errors = []
            
            # 验证前区
            if len(front) != self.front_count:
                errors.append(f"前区号码数量错误: 期望{self.front_count}个，实际{len(front)}个")
            
            for num in front:
                if not (self.front_range[0] <= num <= self.front_range[1]):
                    errors.append(f"前区号码{num}超出范围{self.front_range}")
            
            if len(set(front)) != len(front):
                errors.append("前区号码有重复")
            
            # 验证后区
            if len(back) != self.back_count:
                errors.append(f"后区号码数量错误: 期望{self.back_count}个，实际{len(back)}个")
            
            for num in back:
                if not (self.back_range[0] <= num <= self.back_range[1]):
                    errors.append(f"后区号码{num}超出范围{self.back_range}")
            
            if len(set(back)) != len(back):
                errors.append("后区号码有重复")
            
            result = {
                'is_valid': len(errors) == 0,
                'errors': errors,
                'prediction': prediction
            }
            
            return result
            
        except Exception as e:
            logger_manager.error(f"预测验证失败: {e}")
            return {'is_valid': False, 'errors': [str(e)]}
