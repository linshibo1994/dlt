#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型重训练管理器
检测性能退化并触发模型重训练
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
from collections import defaultdict
from datetime import datetime

from .performance_tracker import PerformanceTracker
from core_modules import logger_manager, cache_manager


class RetrainingManager:
    """重训练管理器"""
    
    def __init__(self, performance_tracker: Optional[PerformanceTracker] = None,
               degradation_threshold: float = 0.1, window_size: int = 10,
               cooldown_period: int = 5):
        """
        初始化重训练管理器
        
        Args:
            performance_tracker: 性能跟踪器，如果为None则创建新的
            degradation_threshold: 性能退化阈值
            window_size: 窗口大小
            cooldown_period: 冷却期（重训练后的等待期）
        """
        self.performance_tracker = performance_tracker or PerformanceTracker()
        self.degradation_threshold = degradation_threshold
        self.window_size = window_size
        self.cooldown_period = cooldown_period
        self.last_retrain = {}  # 上次重训练时间
        self.retrain_history = defaultdict(list)  # 重训练历史
        self.retrain_callbacks = {}  # 重训练回调函数
        
        logger_manager.info(f"初始化重训练管理器，退化阈值: {degradation_threshold}, 窗口大小: {window_size}")
    
    def register_model(self, model_id: str, retrain_callback: Callable[[], bool]) -> None:
        """
        注册模型
        
        Args:
            model_id: 模型ID
            retrain_callback: 重训练回调函数，返回重训练是否成功
        """
        self.retrain_callbacks[model_id] = retrain_callback
        self.last_retrain[model_id] = None
        
        logger_manager.info(f"注册模型 {model_id} 到重训练管理器")
    
    def unregister_model(self, model_id: str) -> bool:
        """
        取消注册模型
        
        Args:
            model_id: 模型ID
            
        Returns:
            是否成功取消注册
        """
        if model_id in self.retrain_callbacks:
            del self.retrain_callbacks[model_id]
            
            if model_id in self.last_retrain:
                del self.last_retrain[model_id]
            
            logger_manager.info(f"取消注册模型 {model_id}")
            return True
        else:
            logger_manager.warning(f"模型 {model_id} 未注册")
            return False
    
    def check_performance(self, model_id: str) -> Dict[str, Any]:
        """
        检查模型性能
        
        Args:
            model_id: 模型ID
            
        Returns:
            性能检查结果字典
        """
        if model_id not in self.retrain_callbacks:
            return {
                'model_id': model_id,
                'status': 'not_registered',
                'message': f"模型 {model_id} 未注册"
            }
        
        # 检测性能退化
        degraded = self.performance_tracker.detect_performance_degradation(
            model_id, self.degradation_threshold, self.window_size
        )
        
        # 获取冷却状态
        in_cooldown = self._is_in_cooldown(model_id)
        
        # 获取性能指标
        metrics = self.performance_tracker.get_model_metrics(model_id)
        avg_metrics = metrics.get_average_metrics(self.window_size) if metrics else {}
        
        result = {
            'model_id': model_id,
            'degraded': degraded,
            'in_cooldown': in_cooldown,
            'metrics': avg_metrics,
            'last_retrain': self.last_retrain[model_id],
            'status': 'normal'
        }
        
        if degraded:
            if in_cooldown:
                result['status'] = 'cooldown'
                result['message'] = f"模型 {model_id} 性能退化，但处于冷却期"
            else:
                result['status'] = 'degraded'
                result['message'] = f"模型 {model_id} 性能退化，需要重训练"
        
        return result
    
    def _is_in_cooldown(self, model_id: str) -> bool:
        """
        检查是否在冷却期
        
        Args:
            model_id: 模型ID
            
        Returns:
            是否在冷却期
        """
        if model_id not in self.last_retrain or self.last_retrain[model_id] is None:
            return False
        
        last_retrain = self.last_retrain[model_id]
        retrain_count = len(self.retrain_history[model_id])
        
        # 如果重训练次数小于冷却期，不在冷却期
        if retrain_count < self.cooldown_period:
            return False
        
        return True
    
    def trigger_retrain(self, model_id: str, force: bool = False) -> Dict[str, Any]:
        """
        触发模型重训练
        
        Args:
            model_id: 模型ID
            force: 是否强制重训练，忽略冷却期
            
        Returns:
            重训练结果字典
        """
        if model_id not in self.retrain_callbacks:
            return {
                'model_id': model_id,
                'success': False,
                'message': f"模型 {model_id} 未注册"
            }
        
        # 检查性能
        check_result = self.check_performance(model_id)
        
        # 如果没有性能退化且不强制重训练，不进行重训练
        if not check_result['degraded'] and not force:
            return {
                'model_id': model_id,
                'success': False,
                'message': f"模型 {model_id} 性能正常，不需要重训练"
            }
        
        # 如果在冷却期且不强制重训练，不进行重训练
        if check_result['in_cooldown'] and not force:
            return {
                'model_id': model_id,
                'success': False,
                'message': f"模型 {model_id} 处于冷却期，不进行重训练"
            }
        
        # 执行重训练
        logger_manager.info(f"触发模型 {model_id} 重训练")
        
        try:
            # 调用重训练回调函数
            success = self.retrain_callbacks[model_id]()
            
            # 更新重训练记录
            timestamp = datetime.now().isoformat()
            self.last_retrain[model_id] = timestamp
            
            self.retrain_history[model_id].append({
                'timestamp': timestamp,
                'success': success,
                'forced': force,
                'metrics_before': check_result['metrics']
            })
            
            if success:
                logger_manager.info(f"模型 {model_id} 重训练成功")
                return {
                    'model_id': model_id,
                    'success': True,
                    'message': f"模型 {model_id} 重训练成功",
                    'timestamp': timestamp
                }
            else:
                logger_manager.error(f"模型 {model_id} 重训练失败")
                return {
                    'model_id': model_id,
                    'success': False,
                    'message': f"模型 {model_id} 重训练失败",
                    'timestamp': timestamp
                }
        except Exception as e:
            logger_manager.error(f"模型 {model_id} 重训练异常: {e}")
            return {
                'model_id': model_id,
                'success': False,
                'message': f"模型 {model_id} 重训练异常: {e}"
            }
    
    def get_retrain_history(self, model_id: str) -> List[Dict[str, Any]]:
        """
        获取重训练历史
        
        Args:
            model_id: 模型ID
            
        Returns:
            重训练历史列表
        """
        return self.retrain_history.get(model_id, [])
    
    def optimize_training_params(self, model_id: str) -> Dict[str, Any]:
        """
        优化训练参数
        
        Args:
            model_id: 模型ID
            
        Returns:
            优化结果字典
        """
        if model_id not in self.retrain_callbacks:
            return {
                'model_id': model_id,
                'success': False,
                'message': f"模型 {model_id} 未注册"
            }
        
        # 获取重训练历史
        history = self.get_retrain_history(model_id)
        
        if len(history) < 2:
            return {
                'model_id': model_id,
                'success': False,
                'message': f"模型 {model_id} 重训练历史不足，无法优化参数"
            }
        
        # 分析重训练历史，优化参数
        # 这里只是一个示例，实际应用中可以根据历史性能变化调整学习率、批大小等参数
        
        # 计算重训练成功率
        success_count = sum(1 for record in history if record['success'])
        success_rate = success_count / len(history)
        
        # 根据成功率调整参数
        params = {}
        
        if success_rate < 0.5:
            # 如果成功率低，降低学习难度
            params['learning_rate'] = 0.001  # 降低学习率
            params['batch_size'] = 32  # 减小批大小
            params['epochs'] = 100  # 增加训练轮数
        else:
            # 如果成功率高，可以适当提高学习难度
            params['learning_rate'] = 0.01  # 提高学习率
            params['batch_size'] = 64  # 增加批大小
            params['epochs'] = 50  # 减少训练轮数
        
        logger_manager.info(f"优化模型 {model_id} 训练参数: {params}")
        
        return {
            'model_id': model_id,
            'success': True,
            'message': f"模型 {model_id} 训练参数优化完成",
            'params': params
        }
    
    def auto_check_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        自动检查所有模型
        
        Returns:
            检查结果字典，键为模型ID，值为检查结果
        """
        results = {}
        
        for model_id in self.retrain_callbacks:
            results[model_id] = self.check_performance(model_id)
        
        return results
    
    def auto_retrain_degraded_models(self) -> Dict[str, Dict[str, Any]]:
        """
        自动重训练性能退化的模型
        
        Returns:
            重训练结果字典，键为模型ID，值为重训练结果
        """
        results = {}
        
        # 检查所有模型
        check_results = self.auto_check_all_models()
        
        # 重训练性能退化的模型
        for model_id, check_result in check_results.items():
            if check_result['status'] == 'degraded':
                results[model_id] = self.trigger_retrain(model_id)
        
        return results


class AdaptiveRetrainingManager(RetrainingManager):
    """自适应重训练管理器"""
    
    def __init__(self, performance_tracker: Optional[PerformanceTracker] = None,
               initial_threshold: float = 0.1, window_size: int = 10,
               cooldown_period: int = 5, min_threshold: float = 0.05,
               max_threshold: float = 0.3):
        """
        初始化自适应重训练管理器
        
        Args:
            performance_tracker: 性能跟踪器，如果为None则创建新的
            initial_threshold: 初始退化阈值
            window_size: 窗口大小
            cooldown_period: 冷却期
            min_threshold: 最小退化阈值
            max_threshold: 最大退化阈值
        """
        super().__init__(performance_tracker, initial_threshold, window_size, cooldown_period)
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.threshold_history = defaultdict(list)  # 阈值历史
        
        logger_manager.info(f"初始化自适应重训练管理器，阈值范围: [{min_threshold}, {max_threshold}]")
    
    def adjust_threshold(self, model_id: str) -> float:
        """
        调整退化阈值
        
        Args:
            model_id: 模型ID
            
        Returns:
            调整后的阈值
        """
        if model_id not in self.retrain_callbacks:
            return self.degradation_threshold
        
        # 获取重训练历史
        history = self.get_retrain_history(model_id)
        
        if len(history) < 2:
            return self.degradation_threshold
        
        # 计算重训练频率
        timestamps = [datetime.fromisoformat(record['timestamp']) for record in history]
        intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]
        avg_interval = sum(intervals) / len(intervals)
        
        # 根据重训练频率调整阈值
        # 如果重训练太频繁，增加阈值；如果太少，减小阈值
        target_interval = 7 * 24 * 3600  # 目标间隔为一周
        
        if avg_interval < target_interval * 0.5:
            # 重训练太频繁，增加阈值
            new_threshold = min(self.degradation_threshold * 1.2, self.max_threshold)
        elif avg_interval > target_interval * 2:
            # 重训练太少，减小阈值
            new_threshold = max(self.degradation_threshold * 0.8, self.min_threshold)
        else:
            # 重训练频率适中，保持阈值
            new_threshold = self.degradation_threshold
        
        # 更新阈值
        self.degradation_threshold = new_threshold
        
        # 记录阈值历史
        self.threshold_history[model_id].append({
            'timestamp': datetime.now().isoformat(),
            'threshold': new_threshold,
            'avg_interval': avg_interval
        })
        
        logger_manager.info(f"调整模型 {model_id} 退化阈值: {new_threshold}")
        
        return new_threshold
    
    def check_performance(self, model_id: str) -> Dict[str, Any]:
        """
        检查模型性能
        
        Args:
            model_id: 模型ID
            
        Returns:
            性能检查结果字典
        """
        # 调整阈值
        self.adjust_threshold(model_id)
        
        # 调用父类方法检查性能
        return super().check_performance(model_id)
    
    def get_threshold_history(self, model_id: str) -> List[Dict[str, Any]]:
        """
        获取阈值历史
        
        Args:
            model_id: 模型ID
            
        Returns:
            阈值历史列表
        """
        return self.threshold_history.get(model_id, [])


if __name__ == "__main__":
    # 测试重训练管理器
    print("🔄 测试重训练管理器...")
    
    # 创建性能跟踪器
    tracker = PerformanceTracker()
    
    # 跟踪模型性能
    tracker.track_performance("model1", {
        'overall_score': 0.8
    })
    
    # 创建重训练管理器
    manager = RetrainingManager(tracker)
    
    # 注册模型
    def mock_retrain():
        print("模拟重训练模型1")
        return True
    
    manager.register_model("model1", mock_retrain)
    
    # 检查性能
    check_result = manager.check_performance("model1")
    print(f"性能检查结果: {check_result}")
    
    # 模拟性能退化
    for i in range(15):
        score = 0.8 - i * 0.05
        tracker.track_performance("model1", {
            'overall_score': max(0.1, score)
        })
    
    # 再次检查性能
    check_result = manager.check_performance("model1")
    print(f"性能退化后检查结果: {check_result}")
    
    # 触发重训练
    retrain_result = manager.trigger_retrain("model1")
    print(f"重训练结果: {retrain_result}")
    
    # 测试自适应重训练管理器
    print("\n🔄 测试自适应重训练管理器...")
    
    # 创建自适应重训练管理器
    adaptive_manager = AdaptiveRetrainingManager(tracker)
    
    # 注册模型
    adaptive_manager.register_model("model1", mock_retrain)
    
    # 检查性能
    check_result = adaptive_manager.check_performance("model1")
    print(f"自适应性能检查结果: {check_result}")
    
    # 触发重训练
    retrain_result = adaptive_manager.trigger_retrain("model1")
    print(f"自适应重训练结果: {retrain_result}")
    
    print("重训练管理器测试完成")