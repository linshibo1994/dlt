#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
性能跟踪系统
跟踪和管理模型性能指标
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional, Union
from collections import defaultdict, deque
from datetime import datetime
import matplotlib.pyplot as plt

from core_modules import logger_manager, cache_manager


class ModelPerformanceMetrics:
    """模型性能指标数据结构"""
    
    def __init__(self, model_id: str):
        """
        初始化性能指标
        
        Args:
            model_id: 模型ID
        """
        self.model_id = model_id
        self.hit_rates = []  # 命中率历史
        self.accuracies = []  # 准确率历史
        self.confidence_scores = []  # 置信度历史
        self.resource_usage = []  # 资源使用历史
        self.prediction_times = []  # 预测时间历史
        self.timestamps = []  # 时间戳历史
        self.prize_rates = []  # 中奖率历史
        self.front_matches = []  # 前区匹配数历史
        self.back_matches = []  # 后区匹配数历史
        self.overall_scores = []  # 综合得分历史
    
    def add_metrics(self, metrics: Dict[str, float], timestamp: Optional[str] = None) -> None:
        """
        添加性能指标
        
        Args:
            metrics: 性能指标字典
            timestamp: 时间戳，如果为None则使用当前时间
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # 添加各项指标
        if 'hit_rate' in metrics:
            self.hit_rates.append(metrics['hit_rate'])
        
        if 'accuracy' in metrics:
            self.accuracies.append(metrics['accuracy'])
        
        if 'confidence' in metrics:
            self.confidence_scores.append(metrics['confidence'])
        
        if 'resource_usage' in metrics:
            self.resource_usage.append(metrics['resource_usage'])
        
        if 'prediction_time' in metrics:
            self.prediction_times.append(metrics['prediction_time'])
        
        if 'prize_rate' in metrics:
            self.prize_rates.append(metrics['prize_rate'])
        
        if 'front_matches' in metrics:
            self.front_matches.append(metrics['front_matches'])
        
        if 'back_matches' in metrics:
            self.back_matches.append(metrics['back_matches'])
        
        # 计算综合得分
        overall_score = self._calculate_overall_score(metrics)
        self.overall_scores.append(overall_score)
        
        # 添加时间戳
        self.timestamps.append(timestamp)
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """
        计算综合得分
        
        Args:
            metrics: 性能指标字典
            
        Returns:
            综合得分
        """
        # 权重配置
        weights = {
            'hit_rate': 0.2,
            'accuracy': 0.3,
            'confidence': 0.1,
            'prize_rate': 0.3,
            'front_matches': 0.05,
            'back_matches': 0.05
        }
        
        # 计算加权得分
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
                total_weight += weight
        
        # 归一化得分
        if total_weight > 0:
            score /= total_weight
        
        return score
    
    def get_average_metrics(self, window: Optional[int] = None) -> Dict[str, float]:
        """
        获取平均性能指标
        
        Args:
            window: 窗口大小，如果为None则使用全部历史
            
        Returns:
            平均性能指标字典
        """
        result = {}
        
        # 计算各项指标的平均值
        if self.hit_rates:
            result['hit_rate'] = self._get_average(self.hit_rates, window)
        
        if self.accuracies:
            result['accuracy'] = self._get_average(self.accuracies, window)
        
        if self.confidence_scores:
            result['confidence'] = self._get_average(self.confidence_scores, window)
        
        if self.resource_usage:
            result['resource_usage'] = self._get_average(self.resource_usage, window)
        
        if self.prediction_times:
            result['prediction_time'] = self._get_average(self.prediction_times, window)
        
        if self.prize_rates:
            result['prize_rate'] = self._get_average(self.prize_rates, window)
        
        if self.front_matches:
            result['front_matches'] = self._get_average(self.front_matches, window)
        
        if self.back_matches:
            result['back_matches'] = self._get_average(self.back_matches, window)
        
        if self.overall_scores:
            result['overall_score'] = self._get_average(self.overall_scores, window)
        
        return result
    
    def _get_average(self, values: List[float], window: Optional[int] = None) -> float:
        """
        计算平均值
        
        Args:
            values: 值列表
            window: 窗口大小，如果为None则使用全部历史
            
        Returns:
            平均值
        """
        if not values:
            return 0.0
        
        if window is not None and window > 0:
            values = values[-window:]
        
        return sum(values) / len(values)
    
    def get_trend(self, metric_name: str, window: Optional[int] = None) -> List[float]:
        """
        获取指标趋势
        
        Args:
            metric_name: 指标名称
            window: 窗口大小，如果为None则使用全部历史
            
        Returns:
            指标趋势列表
        """
        # 获取指标值列表
        if metric_name == 'hit_rate':
            values = self.hit_rates
        elif metric_name == 'accuracy':
            values = self.accuracies
        elif metric_name == 'confidence':
            values = self.confidence_scores
        elif metric_name == 'resource_usage':
            values = self.resource_usage
        elif metric_name == 'prediction_time':
            values = self.prediction_times
        elif metric_name == 'prize_rate':
            values = self.prize_rates
        elif metric_name == 'front_matches':
            values = self.front_matches
        elif metric_name == 'back_matches':
            values = self.back_matches
        elif metric_name == 'overall_score':
            values = self.overall_scores
        else:
            return []
        
        # 应用窗口
        if window is not None and window > 0 and len(values) > window:
            values = values[-window:]
        
        return values
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            字典表示
        """
        return {
            'model_id': self.model_id,
            'hit_rates': self.hit_rates,
            'accuracies': self.accuracies,
            'confidence_scores': self.confidence_scores,
            'resource_usage': self.resource_usage,
            'prediction_times': self.prediction_times,
            'timestamps': self.timestamps,
            'prize_rates': self.prize_rates,
            'front_matches': self.front_matches,
            'back_matches': self.back_matches,
            'overall_scores': self.overall_scores
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelPerformanceMetrics':
        """
        从字典创建实例
        
        Args:
            data: 字典数据
            
        Returns:
            ModelPerformanceMetrics实例
        """
        metrics = cls(data['model_id'])
        
        metrics.hit_rates = data.get('hit_rates', [])
        metrics.accuracies = data.get('accuracies', [])
        metrics.confidence_scores = data.get('confidence_scores', [])
        metrics.resource_usage = data.get('resource_usage', [])
        metrics.prediction_times = data.get('prediction_times', [])
        metrics.timestamps = data.get('timestamps', [])
        metrics.prize_rates = data.get('prize_rates', [])
        metrics.front_matches = data.get('front_matches', [])
        metrics.back_matches = data.get('back_matches', [])
        metrics.overall_scores = data.get('overall_scores', [])
        
        return metrics


class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self, history_size: int = 100):
        """
        初始化性能跟踪器
        
        Args:
            history_size: 历史记录大小
        """
        self.history_size = history_size
        self.model_metrics = {}  # 模型性能指标字典
        self.cache_manager = cache_manager
        
        # 加载历史记录
        self._load_history()
        
        logger_manager.info(f"初始化性能跟踪器，历史记录大小: {history_size}")
    
    def track_performance(self, model_id: str, metrics: Dict[str, float]) -> None:
        """
        跟踪模型性能
        
        Args:
            model_id: 模型ID
            metrics: 性能指标字典
        """
        # 如果模型不存在，创建新的性能指标
        if model_id not in self.model_metrics:
            self.model_metrics[model_id] = ModelPerformanceMetrics(model_id)
        
        # 添加性能指标
        self.model_metrics[model_id].add_metrics(metrics)
        
        # 限制历史记录大小
        self._limit_history_size(model_id)
        
        # 保存历史记录
        self._save_history()
        
        logger_manager.info(f"跟踪模型 {model_id} 性能: {metrics}")
    
    def _limit_history_size(self, model_id: str) -> None:
        """
        限制历史记录大小
        
        Args:
            model_id: 模型ID
        """
        if model_id not in self.model_metrics:
            return
        
        metrics = self.model_metrics[model_id]
        
        # 限制各项指标的历史记录大小
        if len(metrics.hit_rates) > self.history_size:
            metrics.hit_rates = metrics.hit_rates[-self.history_size:]
        
        if len(metrics.accuracies) > self.history_size:
            metrics.accuracies = metrics.accuracies[-self.history_size:]
        
        if len(metrics.confidence_scores) > self.history_size:
            metrics.confidence_scores = metrics.confidence_scores[-self.history_size:]
        
        if len(metrics.resource_usage) > self.history_size:
            metrics.resource_usage = metrics.resource_usage[-self.history_size:]
        
        if len(metrics.prediction_times) > self.history_size:
            metrics.prediction_times = metrics.prediction_times[-self.history_size:]
        
        if len(metrics.timestamps) > self.history_size:
            metrics.timestamps = metrics.timestamps[-self.history_size:]
        
        if len(metrics.prize_rates) > self.history_size:
            metrics.prize_rates = metrics.prize_rates[-self.history_size:]
        
        if len(metrics.front_matches) > self.history_size:
            metrics.front_matches = metrics.front_matches[-self.history_size:]
        
        if len(metrics.back_matches) > self.history_size:
            metrics.back_matches = metrics.back_matches[-self.history_size:]
        
        if len(metrics.overall_scores) > self.history_size:
            metrics.overall_scores = metrics.overall_scores[-self.history_size:]
    
    def get_model_metrics(self, model_id: str) -> Optional[ModelPerformanceMetrics]:
        """
        获取模型性能指标
        
        Args:
            model_id: 模型ID
            
        Returns:
            模型性能指标，如果模型不存在则返回None
        """
        return self.model_metrics.get(model_id)
    
    def get_all_model_metrics(self) -> Dict[str, ModelPerformanceMetrics]:
        """
        获取所有模型性能指标
        
        Returns:
            所有模型性能指标字典
        """
        return self.model_metrics
    
    def get_best_model(self, metric_name: str = 'overall_score', window: Optional[int] = None) -> Optional[str]:
        """
        获取最佳模型
        
        Args:
            metric_name: 指标名称
            window: 窗口大小，如果为None则使用全部历史
            
        Returns:
            最佳模型ID，如果没有模型则返回None
        """
        if not self.model_metrics:
            return None
        
        # 计算各模型的平均指标
        model_scores = {}
        for model_id, metrics in self.model_metrics.items():
            avg_metrics = metrics.get_average_metrics(window)
            if metric_name in avg_metrics:
                model_scores[model_id] = avg_metrics[metric_name]
        
        if not model_scores:
            return None
        
        # 返回得分最高的模型
        return max(model_scores.items(), key=lambda x: x[1])[0]
    
    def compare_models(self, model_ids: List[str], metric_name: str = 'overall_score', 
                      window: Optional[int] = None) -> Dict[str, float]:
        """
        比较多个模型
        
        Args:
            model_ids: 模型ID列表
            metric_name: 指标名称
            window: 窗口大小，如果为None则使用全部历史
            
        Returns:
            模型得分字典
        """
        result = {}
        
        for model_id in model_ids:
            if model_id in self.model_metrics:
                avg_metrics = self.model_metrics[model_id].get_average_metrics(window)
                if metric_name in avg_metrics:
                    result[model_id] = avg_metrics[metric_name]
        
        return result
    
    def detect_performance_degradation(self, model_id: str, threshold: float = 0.1, 
                                     window: int = 10) -> bool:
        """
        检测性能退化
        
        Args:
            model_id: 模型ID
            threshold: 退化阈值
            window: 窗口大小
            
        Returns:
            是否检测到性能退化
        """
        if model_id not in self.model_metrics:
            return False
        
        metrics = self.model_metrics[model_id]
        
        # 获取整体得分趋势
        scores = metrics.get_trend('overall_score')
        
        if len(scores) < window * 2:
            return False
        
        # 计算最近窗口的平均得分
        recent_avg = sum(scores[-window:]) / window
        
        # 计算前一个窗口的平均得分
        previous_avg = sum(scores[-2*window:-window]) / window
        
        # 检测退化
        if previous_avg - recent_avg > threshold:
            logger_manager.warning(f"检测到模型 {model_id} 性能退化: {previous_avg:.4f} -> {recent_avg:.4f}")
            return True
        
        return False
    
    def _save_history(self) -> None:
        """保存历史记录"""
        try:
            # 转换为可序列化的字典
            data = {}
            for model_id, metrics in self.model_metrics.items():
                data[model_id] = metrics.to_dict()
            
            # 保存到缓存
            self.cache_manager.save_cache("data", "performance_history", data)
            
            logger_manager.debug("性能历史记录已保存")
        except Exception as e:
            logger_manager.error(f"保存性能历史记录失败: {e}")
    
    def _load_history(self) -> None:
        """加载历史记录"""
        try:
            # 从缓存加载
            data = self.cache_manager.load_cache("data", "performance_history")
            
            if data:
                # 转换为ModelPerformanceMetrics对象
                for model_id, metrics_data in data.items():
                    self.model_metrics[model_id] = ModelPerformanceMetrics.from_dict(metrics_data)
                
                logger_manager.info(f"加载了 {len(self.model_metrics)} 个模型的性能历史记录")
        except Exception as e:
            logger_manager.error(f"加载性能历史记录失败: {e}")
    
    def plot_performance_trend(self, model_id: str, metric_name: str = 'overall_score', 
                             save_path: Optional[str] = None) -> None:
        """
        绘制性能趋势图
        
        Args:
            model_id: 模型ID
            metric_name: 指标名称
            save_path: 保存路径，如果为None则显示图表
        """
        if model_id not in self.model_metrics:
            logger_manager.warning(f"模型 {model_id} 不存在")
            return
        
        metrics = self.model_metrics[model_id]
        values = metrics.get_trend(metric_name)
        timestamps = metrics.timestamps[-len(values):] if len(metrics.timestamps) >= len(values) else None
        
        if not values:
            logger_manager.warning(f"模型 {model_id} 没有 {metric_name} 指标数据")
            return
        
        try:
            plt.figure(figsize=(10, 6))
            
            if timestamps:
                # 转换时间戳为日期时间对象
                dates = [datetime.fromisoformat(ts) for ts in timestamps]
                plt.plot(dates, values)
                plt.gcf().autofmt_xdate()  # 自动格式化日期标签
            else:
                plt.plot(values)
            
            plt.title(f"{model_id} - {metric_name} 趋势")
            plt.xlabel("时间")
            plt.ylabel(metric_name)
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
                logger_manager.info(f"性能趋势图已保存到 {save_path}")
            else:
                plt.show()
        except Exception as e:
            logger_manager.error(f"绘制性能趋势图失败: {e}")


if __name__ == "__main__":
    # 测试性能跟踪器
    print("📊 测试性能跟踪器...")
    
    # 创建性能跟踪器
    tracker = PerformanceTracker()
    
    # 跟踪模型性能
    tracker.track_performance("model1", {
        'hit_rate': 0.6,
        'accuracy': 0.7,
        'confidence': 0.8,
        'prize_rate': 0.2,
        'front_matches': 2.5,
        'back_matches': 1.0
    })
    
    tracker.track_performance("model2", {
        'hit_rate': 0.7,
        'accuracy': 0.8,
        'confidence': 0.9,
        'prize_rate': 0.3,
        'front_matches': 3.0,
        'back_matches': 1.2
    })
    
    # 获取模型性能指标
    metrics1 = tracker.get_model_metrics("model1")
    metrics2 = tracker.get_model_metrics("model2")
    
    print(f"模型1平均指标: {metrics1.get_average_metrics()}")
    print(f"模型2平均指标: {metrics2.get_average_metrics()}")
    
    # 获取最佳模型
    best_model = tracker.get_best_model()
    print(f"最佳模型: {best_model}")
    
    # 比较模型
    comparison = tracker.compare_models(["model1", "model2"])
    print(f"模型比较: {comparison}")
    
    print("性能跟踪器测试完成")