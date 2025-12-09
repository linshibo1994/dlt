#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
智能早停机制
Intelligent Early Stopping

提供通用的智能早停机制，支持多种训练场景和算法。
"""

import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Any, Callable
from abc import ABC, abstractmethod
import time

from core_modules import logger_manager


class IntelligentEarlyStopping(tf.keras.callbacks.Callback):
    """
    智能早停回调：当连续N次训练结果相同时停止训练
    
    特性：
    1. 连续相同结果检测（默认20次）
    2. 传统早停作为备用机制
    3. 自适应学习率调整
    4. 详细的训练日志
    """
    
    def __init__(self, 
                 patience: int = 20, 
                 min_delta: float = 1e-6, 
                 monitor: str = 'val_loss', 
                 verbose: int = 1,
                 restore_best_weights: bool = True,
                 traditional_patience: Optional[int] = None):
        """
        初始化智能早停机制
        
        Args:
            patience: 连续相同结果的容忍次数
            min_delta: 最小变化阈值
            monitor: 监控的指标
            verbose: 日志详细程度
            restore_best_weights: 是否恢复最佳权重
            traditional_patience: 传统早停的耐心度，默认为patience的一半
        """
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.traditional_patience = traditional_patience or max(1, patience // 2)
        
        # 状态变量
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.best_weights = None
        self.same_results_count = 0
        self.last_loss = None
        self.stop_reason = None
        
    def on_train_begin(self, logs=None):
        """训练开始时重置状态"""
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.best_weights = None
        self.same_results_count = 0
        self.last_loss = None
        self.stop_reason = None
        
        if self.verbose > 0:
            logger_manager.info(f"智能早停机制启动: 连续{self.patience}次相同结果或传统早停{self.traditional_patience}次无改善时停止")
        
    def on_epoch_end(self, epoch, logs=None):
        """每轮训练结束时检查是否需要停止"""
        current = logs.get(self.monitor)
        if current is None:
            if self.verbose > 0:
                logger_manager.warning(f"监控指标 {self.monitor} 不可用")
            return
            
        # 检查是否连续相同结果
        if self.last_loss is not None:
            if abs(current - self.last_loss) < self.min_delta:
                self.same_results_count += 1
                if self.verbose > 1:
                    logger_manager.info(f"连续相同结果: {self.same_results_count}/{self.patience}")
            else:
                self.same_results_count = 0
                
        self.last_loss = current
        
        # 智能早停检查：连续相同结果
        if self.same_results_count >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            self.stop_reason = "intelligent_early_stopping"
            if self.verbose > 0:
                logger_manager.info(f"🛑 智能早停：连续{self.patience}次相同结果，在第{epoch + 1}轮停止训练")
            return
                
        # 传统早停检查：性能不再改善
        if self.best is None:
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        elif current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.traditional_patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.stop_reason = "traditional_early_stopping"
                if self.verbose > 0:
                    logger_manager.info(f"🛑 传统早停：连续{self.traditional_patience}次无改善，在第{epoch + 1}轮停止训练")
                    
    def on_train_end(self, logs=None):
        """训练结束时的处理"""
        if self.stopped_epoch > 0:
            if self.verbose > 0:
                reason_text = {
                    "intelligent_early_stopping": "智能早停（连续相同结果）",
                    "traditional_early_stopping": "传统早停（性能无改善）"
                }.get(self.stop_reason, "未知原因")
                
                logger_manager.info(f"✅ 训练在第{self.stopped_epoch + 1}轮停止，原因: {reason_text}")
                
            # 恢复最佳权重
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    logger_manager.info("恢复最佳权重")
                self.model.set_weights(self.best_weights)


class GeneralIntelligentEarlyStopping:
    """
    通用智能早停机制（非TensorFlow特定）
    
    适用于传统机器学习算法、自定义训练循环等场景
    """
    
    def __init__(self, 
                 patience: int = 20, 
                 min_delta: float = 1e-6, 
                 verbose: int = 1):
        """
        初始化通用智能早停机制
        
        Args:
            patience: 连续相同结果的容忍次数
            min_delta: 最小变化阈值
            verbose: 日志详细程度
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        # 状态变量
        self.same_results_count = 0
        self.last_metric = None
        self.best_metric = None
        self.iteration_count = 0
        self.should_stop = False
        self.stop_reason = None
        
    def reset(self):
        """重置状态"""
        self.same_results_count = 0
        self.last_metric = None
        self.best_metric = None
        self.iteration_count = 0
        self.should_stop = False
        self.stop_reason = None
        
        if self.verbose > 0:
            logger_manager.info(f"通用智能早停机制启动: 连续{self.patience}次相同结果时停止")
    
    def update(self, metric_value: float) -> bool:
        """
        更新指标并检查是否应该停止
        
        Args:
            metric_value: 当前指标值
            
        Returns:
            是否应该停止训练
        """
        self.iteration_count += 1
        
        # 检查是否连续相同结果
        if self.last_metric is not None:
            if abs(metric_value - self.last_metric) < self.min_delta:
                self.same_results_count += 1
                if self.verbose > 1:
                    logger_manager.info(f"连续相同结果: {self.same_results_count}/{self.patience}")
            else:
                self.same_results_count = 0
                
        self.last_metric = metric_value
        
        # 更新最佳指标
        if self.best_metric is None or metric_value < self.best_metric:
            self.best_metric = metric_value
        
        # 检查是否应该停止
        if self.same_results_count >= self.patience:
            self.should_stop = True
            self.stop_reason = "consecutive_same_results"
            if self.verbose > 0:
                logger_manager.info(f"🛑 通用智能早停：连续{self.patience}次相同结果，在第{self.iteration_count}次迭代停止")
            return True
            
        return False
    
    def get_best_metric(self) -> Optional[float]:
        """获取最佳指标值"""
        return self.best_metric
    
    def get_stop_info(self) -> Dict[str, Any]:
        """获取停止信息"""
        return {
            'stopped': self.should_stop,
            'stop_reason': self.stop_reason,
            'iteration_count': self.iteration_count,
            'best_metric': self.best_metric,
            'same_results_count': self.same_results_count
        }


def create_intelligent_callbacks(patience: int = 20, 
                                min_delta: float = 1e-6,
                                monitor: str = 'val_loss',
                                reduce_lr_patience: int = 10) -> list:
    """
    创建包含智能早停的回调函数列表
    
    Args:
        patience: 智能早停的耐心度
        min_delta: 最小变化阈值
        monitor: 监控指标
        reduce_lr_patience: 学习率调整的耐心度
        
    Returns:
        回调函数列表
    """
    callbacks = [
        IntelligentEarlyStopping(
            patience=patience,
            min_delta=min_delta,
            monitor=monitor,
            verbose=1,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks
