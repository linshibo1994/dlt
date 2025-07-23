#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练工具模块
提供训练进度跟踪、可视化和回调函数
"""

import os
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import time

from tensorflow.keras.callbacks import Callback
from core_modules import logger_manager, task_manager


class TrainingUtils:
    """训练工具类"""

    @staticmethod
    def create_progress_callback(model_name: str, epochs: int, log_interval: int = 1):
        """创建训练进度回调"""
        return TrainingProgressCallback(model_name, epochs, log_interval)

    @staticmethod
    def create_visualization_callback(model_name: str, save_dir: str = "training_plots"):
        """创建可视化回调"""
        return TrainingVisualizationCallback(model_name, save_dir)

    @staticmethod
    def create_resource_callback(monitor_interval: int = 10):
        """创建资源监控回调"""
        return ResourceMonitoringCallback(monitor_interval)

    @staticmethod
    def get_default_callbacks(model_name: str, epochs: int):
        """获取默认回调列表"""
        return [
            TrainingUtils.create_progress_callback(model_name, epochs),
            TrainingUtils.create_visualization_callback(model_name),
            TrainingUtils.create_resource_callback()
        ]


class TrainingProgressCallback(Callback):
    """训练进度回调"""
    
    def __init__(self, model_name: str, epochs: int, log_interval: int = 1):
        """
        初始化训练进度回调
        
        Args:
            model_name: 模型名称
            epochs: 总训练轮数
            log_interval: 日志记录间隔（轮数）
        """
        super().__init__()
        self.model_name = model_name
        self.epochs = epochs
        self.log_interval = log_interval
        self.start_time = None
        self.progress_bar = None
    
    def on_train_begin(self, logs=None):
        """训练开始时的回调"""
        self.start_time = time.time()
        logger_manager.info(f"{self.model_name}模型开始训练: {self.epochs} 轮")
        
        # 创建进度条
        self.progress_bar = task_manager.create_progress_bar(
            total=self.epochs,
            description=f"训练{self.model_name}模型"
        )
    
    def on_epoch_end(self, epoch, logs=None):
        """每轮结束时的回调"""
        logs = logs or {}
        
        # 更新进度条
        if self.progress_bar:
            # 计算剩余时间
            elapsed_time = time.time() - self.start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            remaining_epochs = self.epochs - (epoch + 1)
            estimated_remaining_time = avg_time_per_epoch * remaining_epochs
            
            # 格式化指标
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
            
            # 更新进度条
            self.progress_bar.update(1, f"轮 {epoch+1}/{self.epochs}, {metrics_str}")
        
        # 记录日志
        if (epoch + 1) % self.log_interval == 0 or (epoch + 1) == self.epochs:
            metrics_log = ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
            logger_manager.info(f"{self.model_name} - 轮 {epoch+1}/{self.epochs}: {metrics_log}")
    
    def on_train_end(self, logs=None):
        """训练结束时的回调"""
        logs = logs or {}
        
        # 完成进度条
        if self.progress_bar:
            self.progress_bar.finish()
        
        # 计算总训练时间
        total_time = time.time() - self.start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # 记录最终指标
        final_metrics = ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        logger_manager.info(f"{self.model_name}模型训练完成，总时间: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        logger_manager.info(f"最终指标: {final_metrics}")


class ModelCheckpointCallback(Callback):
    """模型检查点回调"""
    
    def __init__(self, model_name: str, save_dir: str, save_best_only: bool = True, 
                 monitor: str = 'val_loss', mode: str = 'min'):
        """
        初始化模型检查点回调
        
        Args:
            model_name: 模型名称
            save_dir: 保存目录
            save_best_only: 是否只保存最佳模型
            monitor: 监控指标
            mode: 监控模式，'min'或'max'
        """
        super().__init__()
        self.model_name = model_name
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化最佳值
        if mode == 'min':
            self.best = float('inf')
            self.monitor_op = lambda current, best: current < best
        else:
            self.best = float('-inf')
            self.monitor_op = lambda current, best: current > best
        
        self.best_epoch = 0
    
    def on_epoch_end(self, epoch, logs=None):
        """每轮结束时的回调"""
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            logger_manager.warning(f"未找到监控指标: {self.monitor}")
            return
        
        # 检查是否是最佳模型
        if self.monitor_op(current, self.best):
            self.best = current
            self.best_epoch = epoch + 1
            
            # 保存模型
            if self.save_best_only:
                model_path = os.path.join(self.save_dir, f"{self.model_name}_best.h5")
                self.model.save(model_path)
                logger_manager.info(f"保存最佳模型 (轮 {epoch+1}, {self.monitor}: {current:.4f}): {model_path}")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0 and not self.save_best_only:
            model_path = os.path.join(self.save_dir, f"{self.model_name}_epoch_{epoch+1}.h5")
            self.model.save(model_path)
            logger_manager.info(f"保存检查点 (轮 {epoch+1}): {model_path}")
    
    def on_train_end(self, logs=None):
        """训练结束时的回调"""
        logger_manager.info(f"最佳模型: 轮 {self.best_epoch}, {self.monitor}: {self.best:.4f}")


class TrainingVisualizer:
    """训练可视化工具"""
    
    def __init__(self, model_name: str, save_dir: str = None):
        """
        初始化训练可视化工具
        
        Args:
            model_name: 模型名称
            save_dir: 图表保存目录
        """
        self.model_name = model_name
        self.save_dir = save_dir or os.path.join("output", "visualizations")
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
    
    def plot_training_history(self, history: Dict[str, List[float]], save: bool = True) -> Optional[Any]:
        """
        绘制训练历史
        
        Args:
            history: 训练历史字典
            save: 是否保存图表
            
        Returns:
            如果matplotlib可用，返回图表对象
        """
        try:
            import matplotlib.pyplot as plt
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # 绘制损失曲线
            ax1.plot(history['loss'], label='训练损失')
            if 'val_loss' in history:
                ax1.plot(history['val_loss'], label='验证损失')
            ax1.set_title(f'{self.model_name} - 损失曲线')
            ax1.set_xlabel('轮数')
            ax1.set_ylabel('损失')
            ax1.legend()
            ax1.grid(True)
            
            # 绘制指标曲线
            metrics = [m for m in history.keys() if m not in ['loss', 'val_loss']]
            for metric in metrics:
                if metric.startswith('val_'):
                    continue
                ax2.plot(history[metric], label=f'训练 {metric}')
                val_metric = f'val_{metric}'
                if val_metric in history:
                    ax2.plot(history[val_metric], label=f'验证 {metric}')
            
            ax2.set_title(f'{self.model_name} - 指标曲线')
            ax2.set_xlabel('轮数')
            ax2.set_ylabel('指标值')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            # 保存图表
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.save_dir, f"{self.model_name}_history_{timestamp}.png")
                plt.savefig(save_path)
                logger_manager.info(f"训练历史图表已保存: {save_path}")
            
            return fig
        
        except ImportError:
            logger_manager.warning("matplotlib未安装，无法绘制训练历史")
            return None
    
    def plot_prediction_comparison(self, actual: List[List[int]], predicted: List[List[int]], 
                                  save: bool = True) -> Optional[Any]:
        """
        绘制预测结果比较
        
        Args:
            actual: 实际号码列表
            predicted: 预测号码列表
            save: 是否保存图表
            
        Returns:
            如果matplotlib可用，返回图表对象
        """
        try:
            import matplotlib.pyplot as plt
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 绘制实际号码和预测号码
            x = np.arange(len(actual))
            width = 0.35
            
            # 绘制前区号码
            ax.bar(x - width/2, [sum(a[:5])/5 for a in actual], width, label='实际前区平均值')
            ax.bar(x + width/2, [sum(p[:5])/5 for p in predicted], width, label='预测前区平均值')
            
            ax.set_title(f'{self.model_name} - 预测结果比较')
            ax.set_xlabel('样本')
            ax.set_ylabel('号码平均值')
            ax.legend()
            ax.grid(True)
            
            plt.tight_layout()
            
            # 保存图表
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.save_dir, f"{self.model_name}_prediction_{timestamp}.png")
                plt.savefig(save_path)
                logger_manager.info(f"预测比较图表已保存: {save_path}")
            
            return fig
        
        except ImportError:
            logger_manager.warning("matplotlib未安装，无法绘制预测比较")
            return None


def get_callbacks(model_name: str, epochs: int) -> List[Callback]:
    """
    获取训练回调函数列表
    
    Args:
        model_name: 模型名称
        epochs: 训练轮数
        
    Returns:
        回调函数列表
    """
    callbacks = [
        # 训练进度回调
        TrainingProgressCallback(model_name, epochs),
        
        # 早停回调
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # 学习率调度回调
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
            verbose=1
        ),
        
        # 模型检查点回调
        ModelCheckpointCallback(
            model_name=model_name,
            save_dir=os.path.join("cache", "models"),
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
    ]
    
    return callbacks