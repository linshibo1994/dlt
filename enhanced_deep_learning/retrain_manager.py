#!/usr/bin/env python3
"""
重训练管理器
自动管理模型的重训练过程，包括触发条件、数据准备和训练调度
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
import json
import os
from datetime import datetime, timedelta
import threading
import time
from enum import Enum
from dataclasses import dataclass

from core_modules import logger_manager, config_manager
from .model_base import BaseDeepLearningModel
from .model_registry import model_registry


class RetrainTrigger(Enum):
    """重训练触发条件"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    NEW_DATA_THRESHOLD = "new_data_threshold"


@dataclass
class RetrainTask:
    """重训练任务"""
    model_name: str
    trigger: RetrainTrigger
    priority: int
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    data_range: Optional[Tuple[str, str]] = None
    config_override: Optional[Dict[str, Any]] = None
    callback: Optional[Callable] = None


class RetrainManager:
    """重训练管理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化重训练管理器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 重训练参数
        self.performance_threshold = self.config.get('performance_threshold', 0.05)
        self.data_drift_threshold = self.config.get('data_drift_threshold', 0.1)
        self.min_retrain_interval = self.config.get('min_retrain_interval', 24)  # 小时
        self.max_concurrent_retrains = self.config.get('max_concurrent_retrains', 2)
        self.new_data_threshold = self.config.get('new_data_threshold', 100)  # 新数据条数
        
        # 任务队列和状态
        self.retrain_queue: List[RetrainTask] = []
        self.active_retrains: Dict[str, threading.Thread] = {}
        self.retrain_history: List[Dict[str, Any]] = []
        self.last_retrain_times: Dict[str, datetime] = {}
        
        # 性能监控
        self.model_performances: Dict[str, List[float]] = {}
        self.baseline_performances: Dict[str, float] = {}
        
        # 数据监控
        self.data_statistics: Dict[str, Dict[str, Any]] = {}
        self.baseline_statistics: Dict[str, Dict[str, Any]] = {}
        
        # 调度器状态
        self.scheduler_running = False
        self.scheduler_thread = None
        
        logger_manager.info("重训练管理器初始化完成")
    
    def start_scheduler(self) -> bool:
        """启动重训练调度器"""
        try:
            if self.scheduler_running:
                logger_manager.warning("重训练调度器已在运行")
                return True
            
            self.scheduler_running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()
            
            logger_manager.info("重训练调度器已启动")
            return True
            
        except Exception as e:
            logger_manager.error(f"启动重训练调度器失败: {e}")
            return False
    
    def stop_scheduler(self) -> bool:
        """停止重训练调度器"""
        try:
            if not self.scheduler_running:
                logger_manager.warning("重训练调度器未运行")
                return True
            
            self.scheduler_running = False
            
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=10)
            
            logger_manager.info("重训练调度器已停止")
            return True
            
        except Exception as e:
            logger_manager.error(f"停止重训练调度器失败: {e}")
            return False
    
    def schedule_retrain(self, model_name: str, trigger: RetrainTrigger, 
                        priority: int = 5, scheduled_at: datetime = None,
                        config_override: Dict[str, Any] = None,
                        callback: Callable = None) -> bool:
        """
        调度重训练任务
        
        Args:
            model_name: 模型名称
            trigger: 触发条件
            priority: 优先级 (1-10, 1最高)
            scheduled_at: 调度时间
            config_override: 配置覆盖
            callback: 回调函数
            
        Returns:
            是否调度成功
        """
        try:
            # 检查是否可以重训练
            if not self._can_retrain(model_name):
                logger_manager.warning(f"模型 {model_name} 暂时无法重训练")
                return False
            
            # 创建重训练任务
            task = RetrainTask(
                model_name=model_name,
                trigger=trigger,
                priority=priority,
                created_at=datetime.now(),
                scheduled_at=scheduled_at,
                config_override=config_override,
                callback=callback
            )
            
            # 添加到队列
            self.retrain_queue.append(task)
            
            # 按优先级排序
            self.retrain_queue.sort(key=lambda x: (x.priority, x.created_at))
            
            logger_manager.info(f"重训练任务已调度: {model_name}, 触发条件: {trigger.value}")
            
            return True
            
        except Exception as e:
            logger_manager.error(f"调度重训练任务失败: {e}")
            return False
    
    def check_retrain_conditions(self, model_name: str, data: pd.DataFrame, 
                                performance: float) -> List[RetrainTrigger]:
        """
        检查重训练条件
        
        Args:
            model_name: 模型名称
            data: 最新数据
            performance: 当前性能
            
        Returns:
            触发的重训练条件列表
        """
        triggers = []
        
        try:
            # 检查性能退化
            if self._check_performance_degradation(model_name, performance):
                triggers.append(RetrainTrigger.PERFORMANCE_DEGRADATION)
            
            # 检查数据漂移
            if self._check_data_drift(model_name, data):
                triggers.append(RetrainTrigger.DATA_DRIFT)
            
            # 检查新数据阈值
            if self._check_new_data_threshold(model_name, data):
                triggers.append(RetrainTrigger.NEW_DATA_THRESHOLD)
            
            # 检查定时重训练
            if self._check_scheduled_retrain(model_name):
                triggers.append(RetrainTrigger.SCHEDULED)
            
            return triggers
            
        except Exception as e:
            logger_manager.error(f"检查重训练条件失败: {e}")
            return []
    
    def execute_retrain(self, task: RetrainTask, data: pd.DataFrame) -> bool:
        """
        执行重训练任务
        
        Args:
            task: 重训练任务
            data: 训练数据
            
        Returns:
            是否执行成功
        """
        try:
            logger_manager.info(f"开始执行重训练: {task.model_name}")
            
            # 获取模型实例
            model = model_registry.create_model(task.model_name, task.config_override)
            if model is None:
                logger_manager.error(f"无法创建模型实例: {task.model_name}")
                return False
            
            # 准备训练数据
            train_data = self._prepare_training_data(data, task.data_range)
            
            # 执行训练
            start_time = datetime.now()
            training_result = model.train(train_data)
            end_time = datetime.now()
            
            # 保存模型
            model.save_models()
            
            # 记录重训练历史
            retrain_record = {
                'model_name': task.model_name,
                'trigger': task.trigger.value,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration': (end_time - start_time).total_seconds(),
                'data_size': len(train_data),
                'training_result': training_result,
                'success': True
            }
            
            self.retrain_history.append(retrain_record)
            self.last_retrain_times[task.model_name] = end_time
            
            # 执行回调
            if task.callback:
                task.callback(task, retrain_record)
            
            logger_manager.info(f"重训练完成: {task.model_name}")
            
            return True
            
        except Exception as e:
            logger_manager.error(f"执行重训练失败: {e}")
            
            # 记录失败
            retrain_record = {
                'model_name': task.model_name,
                'trigger': task.trigger.value,
                'start_time': datetime.now().isoformat(),
                'error': str(e),
                'success': False
            }
            
            self.retrain_history.append(retrain_record)
            
            return False
    
    def _scheduler_loop(self):
        """调度器主循环"""
        logger_manager.info("重训练调度器循环开始")
        
        while self.scheduler_running:
            try:
                # 处理队列中的任务
                self._process_retrain_queue()
                
                # 清理完成的重训练线程
                self._cleanup_completed_retrains()
                
                # 等待下次检查
                time.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                logger_manager.error(f"调度器循环错误: {e}")
                time.sleep(60)
        
        logger_manager.info("重训练调度器循环结束")
    
    def _process_retrain_queue(self):
        """处理重训练队列"""
        while (self.retrain_queue and 
               len(self.active_retrains) < self.max_concurrent_retrains):
            
            task = self.retrain_queue.pop(0)
            
            # 检查是否到了调度时间
            if task.scheduled_at and datetime.now() < task.scheduled_at:
                # 重新放回队列
                self.retrain_queue.insert(0, task)
                break
            
            # 启动重训练线程
            thread = threading.Thread(
                target=self._retrain_worker,
                args=(task,),
                daemon=True
            )
            
            thread.start()
            self.active_retrains[task.model_name] = thread
    
    def _retrain_worker(self, task: RetrainTask):
        """重训练工作线程"""
        try:
            # 这里需要获取训练数据，实际实现中应该从数据源获取
            # 暂时使用空DataFrame作为占位符
            data = pd.DataFrame()
            
            # 执行重训练
            self.execute_retrain(task, data)
            
        except Exception as e:
            logger_manager.error(f"重训练工作线程错误: {e}")
        
        finally:
            # 从活动重训练中移除
            if task.model_name in self.active_retrains:
                del self.active_retrains[task.model_name]
    
    def _cleanup_completed_retrains(self):
        """清理已完成的重训练线程"""
        completed = []
        
        for model_name, thread in self.active_retrains.items():
            if not thread.is_alive():
                completed.append(model_name)
        
        for model_name in completed:
            del self.active_retrains[model_name]
    
    def _can_retrain(self, model_name: str) -> bool:
        """检查是否可以重训练"""
        # 检查是否已在重训练
        if model_name in self.active_retrains:
            return False
        
        # 检查最小重训练间隔
        if model_name in self.last_retrain_times:
            last_retrain = self.last_retrain_times[model_name]
            min_interval = timedelta(hours=self.min_retrain_interval)
            if datetime.now() - last_retrain < min_interval:
                return False
        
        return True
    
    def _check_performance_degradation(self, model_name: str, performance: float) -> bool:
        """检查性能退化"""
        if model_name not in self.model_performances:
            self.model_performances[model_name] = []
        
        self.model_performances[model_name].append(performance)
        
        # 保持最近的性能记录
        if len(self.model_performances[model_name]) > 100:
            self.model_performances[model_name] = self.model_performances[model_name][-100:]
        
        # 检查是否有基准性能
        if model_name not in self.baseline_performances:
            if len(self.model_performances[model_name]) >= 10:
                self.baseline_performances[model_name] = np.mean(self.model_performances[model_name][:10])
            return False
        
        # 计算性能退化
        baseline = self.baseline_performances[model_name]
        degradation = (baseline - performance) / baseline if baseline > 0 else 0
        
        return degradation > self.performance_threshold
    
    def _check_data_drift(self, model_name: str, data: pd.DataFrame) -> bool:
        """检查数据漂移"""
        if data.empty:
            return False
        
        # 计算当前数据统计
        current_stats = self._calculate_data_statistics(data)
        
        # 检查是否有基准统计
        if model_name not in self.baseline_statistics:
            self.baseline_statistics[model_name] = current_stats
            return False
        
        # 计算统计差异
        baseline_stats = self.baseline_statistics[model_name]
        drift_score = self._calculate_drift_score(baseline_stats, current_stats)
        
        return drift_score > self.data_drift_threshold
    
    def _check_new_data_threshold(self, model_name: str, data: pd.DataFrame) -> bool:
        """检查新数据阈值"""
        # 简化实现：检查数据量是否超过阈值
        return len(data) >= self.new_data_threshold
    
    def _check_scheduled_retrain(self, model_name: str) -> bool:
        """检查定时重训练"""
        # 简化实现：每周重训练一次
        if model_name not in self.last_retrain_times:
            return True
        
        last_retrain = self.last_retrain_times[model_name]
        return datetime.now() - last_retrain > timedelta(days=7)
    
    def _prepare_training_data(self, data: pd.DataFrame, 
                              data_range: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
        """准备训练数据"""
        if data_range:
            start_date, end_date = data_range
            # 根据日期范围过滤数据
            if 'date' in data.columns:
                mask = (data['date'] >= start_date) & (data['date'] <= end_date)
                return data[mask]
        
        return data
    
    def _calculate_data_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """计算数据统计"""
        stats = {}
        
        for column in data.select_dtypes(include=[np.number]).columns:
            stats[column] = {
                'mean': data[column].mean(),
                'std': data[column].std(),
                'min': data[column].min(),
                'max': data[column].max()
            }
        
        return stats
    
    def _calculate_drift_score(self, baseline_stats: Dict[str, Any], 
                              current_stats: Dict[str, Any]) -> float:
        """计算漂移分数"""
        drift_scores = []
        
        for column in baseline_stats:
            if column in current_stats:
                baseline = baseline_stats[column]
                current = current_stats[column]
                
                # 计算均值和标准差的相对变化
                mean_change = abs(current['mean'] - baseline['mean']) / (abs(baseline['mean']) + 1e-8)
                std_change = abs(current['std'] - baseline['std']) / (abs(baseline['std']) + 1e-8)
                
                drift_scores.append(mean_change + std_change)
        
        return np.mean(drift_scores) if drift_scores else 0.0
    
    def get_retrain_status(self) -> Dict[str, Any]:
        """获取重训练状态"""
        return {
            'queue_size': len(self.retrain_queue),
            'active_retrains': list(self.active_retrains.keys()),
            'total_retrains': len(self.retrain_history),
            'successful_retrains': sum(1 for r in self.retrain_history if r.get('success', False)),
            'scheduler_running': self.scheduler_running,
            'last_retrain_times': {
                name: time.isoformat() for name, time in self.last_retrain_times.items()
            }
        }
