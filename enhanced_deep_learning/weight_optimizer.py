#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动权重调整系统
基于模型性能动态调整集成权重
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from collections import defaultdict
from datetime import datetime

from .performance_tracker import PerformanceTracker
from core_modules import logger_manager, cache_manager


class WeightOptimizer:
    """权重优化器"""
    
    def __init__(self, performance_tracker: Optional[PerformanceTracker] = None,
               learning_rate: float = 0.05, window_size: int = 10):
        """
        初始化权重优化器
        
        Args:
            performance_tracker: 性能跟踪器，如果为None则创建新的
            learning_rate: 学习率
            window_size: 窗口大小
        """
        self.performance_tracker = performance_tracker or PerformanceTracker()
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.weights = {}
        self.weight_history = defaultdict(list)
        self.update_history = []
        
        logger_manager.info(f"初始化权重优化器，学习率: {learning_rate}, 窗口大小: {window_size}")
    
    def initialize_weights(self, model_ids: List[str], initial_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        初始化权重
        
        Args:
            model_ids: 模型ID列表
            initial_weights: 初始权重字典，如果为None则平均分配
            
        Returns:
            初始化后的权重字典
        """
        if not model_ids:
            return {}
        
        # 如果提供了初始权重，使用它们
        if initial_weights:
            for model_id in model_ids:
                if model_id in initial_weights:
                    self.weights[model_id] = initial_weights[model_id]
                else:
                    self.weights[model_id] = 1.0 / len(model_ids)
        else:
            # 平均分配权重
            equal_weight = 1.0 / len(model_ids)
            for model_id in model_ids:
                self.weights[model_id] = equal_weight
        
        # 记录初始权重
        for model_id, weight in self.weights.items():
            self.weight_history[model_id].append(weight)
        
        logger_manager.info(f"初始化权重: {self.weights}")
        
        return self.weights
    
    def update_weights(self, metric_name: str = 'overall_score') -> Dict[str, float]:
        """
        更新权重
        
        Args:
            metric_name: 用于更新权重的指标名称
            
        Returns:
            更新后的权重字典
        """
        # 获取所有模型的性能指标
        model_metrics = self.performance_tracker.get_all_model_metrics()
        
        if not model_metrics:
            logger_manager.warning("没有模型性能指标，无法更新权重")
            return self.weights
        
        # 计算各模型的性能得分
        model_scores = {}
        for model_id, metrics in model_metrics.items():
            avg_metrics = metrics.get_average_metrics(self.window_size)
            if metric_name in avg_metrics:
                model_scores[model_id] = avg_metrics[metric_name]
        
        if not model_scores:
            logger_manager.warning(f"没有模型有 {metric_name} 指标，无法更新权重")
            return self.weights
        
        # 计算总得分
        total_score = sum(model_scores.values())
        
        if total_score == 0:
            logger_manager.warning("总得分为0，无法更新权重")
            return self.weights
        
        # 计算目标权重
        target_weights = {model_id: score / total_score for model_id, score in model_scores.items()}
        
        # 渐进式更新权重
        new_weights = {}
        for model_id in self.weights:
            if model_id in target_weights:
                # 使用学习率进行渐进式更新
                new_weight = self.weights[model_id] + self.learning_rate * (target_weights[model_id] - self.weights[model_id])
                new_weights[model_id] = max(0.01, min(0.99, new_weight))  # 限制权重范围
            else:
                # 如果模型不在目标权重中，保持原权重
                new_weights[model_id] = self.weights[model_id]
        
        # 归一化权重
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            for model_id in new_weights:
                new_weights[model_id] /= total_weight
        
        # 更新权重
        self.weights = new_weights
        
        # 记录权重历史
        for model_id, weight in self.weights.items():
            self.weight_history[model_id].append(weight)
        
        # 记录更新历史
        self.update_history.append({
            'timestamp': datetime.now().isoformat(),
            'weights': self.weights.copy(),
            'scores': model_scores
        })
        
        logger_manager.info(f"更新权重: {self.weights}")
        
        return self.weights
    
    def get_weights(self) -> Dict[str, float]:
        """
        获取当前权重
        
        Returns:
            当前权重字典
        """
        return self.weights
    
    def get_weight_history(self, model_id: str) -> List[float]:
        """
        获取权重历史
        
        Args:
            model_id: 模型ID
            
        Returns:
            权重历史列表
        """
        return self.weight_history.get(model_id, [])
    
    def get_update_history(self) -> List[Dict[str, Any]]:
        """
        获取更新历史
        
        Returns:
            更新历史列表
        """
        return self.update_history
    
    def get_weight_explanation(self) -> Dict[str, Any]:
        """
        获取权重解释
        
        Returns:
            权重解释字典
        """
        if not self.update_history:
            return {
                'explanation': "尚未进行权重更新",
                'weights': self.weights
            }
        
        # 获取最近的更新
        latest_update = self.update_history[-1]
        
        # 计算权重变化
        weight_changes = {}
        if len(self.update_history) > 1:
            previous_update = self.update_history[-2]
            for model_id in latest_update['weights']:
                if model_id in previous_update['weights']:
                    weight_changes[model_id] = latest_update['weights'][model_id] - previous_update['weights'][model_id]
        
        # 生成解释
        explanation = {
            'timestamp': latest_update['timestamp'],
            'weights': latest_update['weights'],
            'scores': latest_update['scores'],
            'changes': weight_changes,
            'explanation': self._generate_explanation(latest_update, weight_changes)
        }
        
        return explanation
    
    def _generate_explanation(self, update: Dict[str, Any], changes: Dict[str, float]) -> str:
        """
        生成权重更新解释
        
        Args:
            update: 更新信息
            changes: 权重变化
            
        Returns:
            解释文本
        """
        explanation = "权重更新基于模型性能:\n"
        
        # 添加各模型的得分和权重
        for model_id, score in update['scores'].items():
            weight = update['weights'].get(model_id, 0.0)
            change = changes.get(model_id, 0.0)
            
            explanation += f"- {model_id}: 得分 {score:.4f}, 权重 {weight:.4f}"
            
            if change > 0:
                explanation += f" (增加 {change:.4f})"
            elif change < 0:
                explanation += f" (减少 {abs(change):.4f})"
            else:
                explanation += " (无变化)"
            
            explanation += "\n"
        
        return explanation
    
    def save_weights(self, file_path: Optional[str] = None) -> bool:
        """
        保存权重
        
        Args:
            file_path: 文件路径，如果为None则使用默认路径
            
        Returns:
            是否保存成功
        """
        try:
            # 准备保存数据
            data = {
                'weights': self.weights,
                'weight_history': dict(self.weight_history),
                'update_history': self.update_history,
                'learning_rate': self.learning_rate,
                'window_size': self.window_size,
                'timestamp': datetime.now().isoformat()
            }
            
            if file_path:
                # 保存到指定文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            else:
                # 保存到缓存
                cache_manager.save_cache("data", "weight_optimizer", data)
            
            logger_manager.info(f"权重已保存")
            return True
        except Exception as e:
            logger_manager.error(f"保存权重失败: {e}")
            return False
    
    def load_weights(self, file_path: Optional[str] = None) -> bool:
        """
        加载权重
        
        Args:
            file_path: 文件路径，如果为None则使用默认路径
            
        Returns:
            是否加载成功
        """
        try:
            if file_path and os.path.exists(file_path):
                # 从指定文件加载
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                # 从缓存加载
                data = cache_manager.load_cache("data", "weight_optimizer")
            
            if data:
                self.weights = data.get('weights', {})
                self.weight_history = defaultdict(list, data.get('weight_history', {}))
                self.update_history = data.get('update_history', [])
                self.learning_rate = data.get('learning_rate', self.learning_rate)
                self.window_size = data.get('window_size', self.window_size)
                
                logger_manager.info(f"权重已加载: {self.weights}")
                return True
            else:
                logger_manager.warning("没有找到权重数据")
                return False
        except Exception as e:
            logger_manager.error(f"加载权重失败: {e}")
            return False


class AdaptiveWeightOptimizer(WeightOptimizer):
    """自适应权重优化器"""
    
    def __init__(self, performance_tracker: Optional[PerformanceTracker] = None,
               initial_learning_rate: float = 0.05, window_size: int = 10,
               min_learning_rate: float = 0.01, max_learning_rate: float = 0.2):
        """
        初始化自适应权重优化器
        
        Args:
            performance_tracker: 性能跟踪器，如果为None则创建新的
            initial_learning_rate: 初始学习率
            window_size: 窗口大小
            min_learning_rate: 最小学习率
            max_learning_rate: 最大学习率
        """
        super().__init__(performance_tracker, initial_learning_rate, window_size)
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.learning_rate_history = []
        
        logger_manager.info(f"初始化自适应权重优化器，学习率范围: [{min_learning_rate}, {max_learning_rate}]")
    
    def update_weights(self, metric_name: str = 'overall_score') -> Dict[str, float]:
        """
        更新权重
        
        Args:
            metric_name: 用于更新权重的指标名称
            
        Returns:
            更新后的权重字典
        """
        # 调整学习率
        self._adjust_learning_rate()
        
        # 调用父类方法更新权重
        return super().update_weights(metric_name)
    
    def _adjust_learning_rate(self) -> None:
        """调整学习率"""
        # 如果没有足够的更新历史，不调整学习率
        if len(self.update_history) < 2:
            return
        
        # 获取最近两次更新的权重
        latest_weights = self.update_history[-1]['weights']
        previous_weights = self.update_history[-2]['weights']
        
        # 计算权重变化的平均绝对值
        weight_changes = []
        for model_id in latest_weights:
            if model_id in previous_weights:
                change = abs(latest_weights[model_id] - previous_weights[model_id])
                weight_changes.append(change)
        
        if not weight_changes:
            return
        
        avg_change = sum(weight_changes) / len(weight_changes)
        
        # 根据权重变化调整学习率
        if avg_change < 0.01:
            # 权重变化很小，增加学习率
            self.learning_rate = min(self.learning_rate * 1.2, self.max_learning_rate)
        elif avg_change > 0.1:
            # 权重变化很大，减小学习率
            self.learning_rate = max(self.learning_rate * 0.8, self.min_learning_rate)
        
        # 记录学习率历史
        self.learning_rate_history.append(self.learning_rate)
        
        logger_manager.info(f"调整学习率: {self.learning_rate}")
    
    def get_learning_rate_history(self) -> List[float]:
        """
        获取学习率历史
        
        Returns:
            学习率历史列表
        """
        return self.learning_rate_history


if __name__ == "__main__":
    # 测试权重优化器
    print("⚖️ 测试权重优化器...")
    
    # 创建性能跟踪器
    tracker = PerformanceTracker()
    
    # 跟踪模型性能
    tracker.track_performance("model1", {
        'overall_score': 0.6
    })
    
    tracker.track_performance("model2", {
        'overall_score': 0.8
    })
    
    # 创建权重优化器
    optimizer = WeightOptimizer(tracker)
    
    # 初始化权重
    optimizer.initialize_weights(["model1", "model2"])
    
    # 更新权重
    weights = optimizer.update_weights()
    
    print(f"更新后的权重: {weights}")
    
    # 获取权重解释
    explanation = optimizer.get_weight_explanation()
    print(f"权重解释: {explanation['explanation']}")
    
    # 测试自适应权重优化器
    print("\n⚖️ 测试自适应权重优化器...")
    
    # 创建自适应权重优化器
    adaptive_optimizer = AdaptiveWeightOptimizer(tracker)
    
    # 初始化权重
    adaptive_optimizer.initialize_weights(["model1", "model2"])
    
    # 更新权重
    weights = adaptive_optimizer.update_weights()
    
    print(f"更新后的权重: {weights}")
    
    print("权重优化器测试完成")