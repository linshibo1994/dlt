#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练工具模块
Training Utilities Module

提供深度学习模型训练的通用工具和回调函数。
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import core_modules as cm

logger_manager = cm.logger_manager


def get_callbacks(model_name: str, model_dir: str, patience: int = 10) -> List[tf.keras.callbacks.Callback]:
    """
    获取训练回调函数
    
    Args:
        model_name: 模型名称
        model_dir: 模型保存目录
        patience: 早停耐心值
        
    Returns:
        回调函数列表
    """
    callbacks = []
    
    # 早停回调
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # 学习率衰减回调
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=patience//2,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # 模型检查点回调
    if model_dir and os.path.exists(model_dir):
        checkpoint_path = os.path.join(model_dir, f'{model_name}_best.h5')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
    
    return callbacks


class TrainingVisualizer:
    """训练可视化器"""
    
    def __init__(self, model_name: str = "model"):
        """
        初始化训练可视化器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.history = {}
        
    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None):
        """
        每个epoch结束时的回调
        
        Args:
            epoch: 当前epoch
            logs: 训练日志
        """
        if logs is None:
            logs = {}
            
        # 记录训练历史
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
        
        # 记录日志
        if epoch % 10 == 0:  # 每10个epoch记录一次
            loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            logger_manager.info(f"{self.model_name} Epoch {epoch}: loss={loss:.4f}, val_loss={val_loss:.4f}")
    
    def plot_history(self, save_path: Optional[str] = None):
        """
        绘制训练历史
        
        Args:
            save_path: 保存路径
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.history:
                logger_manager.warning("没有训练历史数据可绘制")
                return
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # 绘制损失
            if 'loss' in self.history:
                axes[0].plot(self.history['loss'], label='Training Loss')
                if 'val_loss' in self.history:
                    axes[0].plot(self.history['val_loss'], label='Validation Loss')
                axes[0].set_title(f'{self.model_name} Loss')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].legend()
            
            # 绘制准确率（如果有）
            if 'accuracy' in self.history:
                axes[1].plot(self.history['accuracy'], label='Training Accuracy')
                if 'val_accuracy' in self.history:
                    axes[1].plot(self.history['val_accuracy'], label='Validation Accuracy')
                axes[1].set_title(f'{self.model_name} Accuracy')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Accuracy')
                axes[1].legend()
            else:
                # 如果没有准确率，绘制学习率
                if 'lr' in self.history:
                    axes[1].plot(self.history['lr'], label='Learning Rate')
                    axes[1].set_title(f'{self.model_name} Learning Rate')
                    axes[1].set_xlabel('Epoch')
                    axes[1].set_ylabel('Learning Rate')
                    axes[1].legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger_manager.info(f"训练历史图表已保存到: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger_manager.warning("matplotlib未安装，无法绘制训练历史")
        except Exception as e:
            logger_manager.error(f"绘制训练历史失败: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取训练摘要
        
        Returns:
            训练摘要字典
        """
        summary = {
            'model_name': self.model_name,
            'total_epochs': len(self.history.get('loss', [])),
            'final_loss': self.history.get('loss', [])[-1] if self.history.get('loss') else None,
            'final_val_loss': self.history.get('val_loss', [])[-1] if self.history.get('val_loss') else None,
            'best_loss': min(self.history.get('loss', [])) if self.history.get('loss') else None,
            'best_val_loss': min(self.history.get('val_loss', [])) if self.history.get('val_loss') else None
        }
        
        return summary


class TrainingCallback(tf.keras.callbacks.Callback):
    """自定义训练回调"""

    def __init__(self, visualizer: TrainingVisualizer):
        """
        初始化回调

        Args:
            visualizer: 训练可视化器
        """
        super().__init__()
        self.visualizer = visualizer

    def on_epoch_end(self, epoch, logs=None):
        """每个epoch结束时调用"""
        self.visualizer.on_epoch_end(epoch, logs)


class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """训练进度回调"""

    def __init__(self, model_name: str = "model", log_frequency: int = 10):
        """
        初始化训练进度回调

        Args:
            model_name: 模型名称
            log_frequency: 日志记录频率（每多少个epoch记录一次）
        """
        super().__init__()
        self.model_name = model_name
        self.log_frequency = log_frequency
        self.epoch_count = 0
        self.start_time = None

    def on_train_begin(self, logs=None):
        """训练开始时调用"""
        import time
        self.start_time = time.time()
        logger_manager.info(f"{self.model_name} 开始训练")

    def on_epoch_end(self, epoch, logs=None):
        """每个epoch结束时调用"""
        self.epoch_count += 1

        if self.epoch_count % self.log_frequency == 0:
            if logs:
                loss = logs.get('loss', 0)
                val_loss = logs.get('val_loss', 0)
                logger_manager.info(f"{self.model_name} Epoch {epoch}: loss={loss:.4f}, val_loss={val_loss:.4f}")
            else:
                logger_manager.info(f"{self.model_name} Epoch {epoch} 完成")

    def on_train_end(self, logs=None):
        """训练结束时调用"""
        if self.start_time:
            import time
            total_time = time.time() - self.start_time
            logger_manager.info(f"{self.model_name} 训练完成，总耗时: {total_time:.2f}秒")


def create_training_config(model_type: str = "default") -> Dict[str, Any]:
    """
    创建训练配置
    
    Args:
        model_type: 模型类型
        
    Returns:
        训练配置字典
    """
    base_config = {
        'epochs': 50,
        'batch_size': 32,
        'validation_split': 0.2,
        'patience': 10,
        'learning_rate': 0.001,
        'optimizer': 'adam'
    }
    
    # 根据模型类型调整配置
    if model_type.lower() == 'transformer':
        base_config.update({
            'epochs': 30,
            'batch_size': 16,
            'learning_rate': 0.0001,
            'patience': 8
        })
    elif model_type.lower() == 'gan':
        base_config.update({
            'epochs': 100,
            'batch_size': 64,
            'learning_rate': 0.0002,
            'patience': 15
        })
    elif model_type.lower() == 'lstm':
        base_config.update({
            'epochs': 40,
            'batch_size': 32,
            'learning_rate': 0.001,
            'patience': 12
        })
    
    return base_config


def prepare_training_data(data: np.ndarray, sequence_length: int = 10, 
                         validation_split: float = 0.2) -> tuple:
    """
    准备训练数据
    
    Args:
        data: 原始数据
        sequence_length: 序列长度
        validation_split: 验证集比例
        
    Returns:
        (X_train, y_train, X_val, y_val)
    """
    try:
        # 创建序列数据
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # 分割训练集和验证集
        split_idx = int(len(X) * (1 - validation_split))
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger_manager.info(f"训练数据准备完成: 训练集{X_train.shape}, 验证集{X_val.shape}")
        
        return X_train, y_train, X_val, y_val
        
    except Exception as e:
        logger_manager.error(f"准备训练数据失败: {e}")
        raise


def save_training_history(history: Dict[str, List], save_path: str):
    """
    保存训练历史
    
    Args:
        history: 训练历史
        save_path: 保存路径
    """
    try:
        import json
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 转换numpy数组为列表
        serializable_history = {}
        for key, value in history.items():
            if isinstance(value, np.ndarray):
                serializable_history[key] = value.tolist()
            elif isinstance(value, list):
                serializable_history[key] = value
            else:
                serializable_history[key] = str(value)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2)
        
        logger_manager.info(f"训练历史已保存到: {save_path}")
        
    except Exception as e:
        logger_manager.error(f"保存训练历史失败: {e}")


def load_training_history(load_path: str) -> Dict[str, List]:
    """
    加载训练历史
    
    Args:
        load_path: 加载路径
        
    Returns:
        训练历史字典
    """
    try:
        import json
        
        with open(load_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        logger_manager.info(f"训练历史已从 {load_path} 加载")
        return history
        
    except Exception as e:
        logger_manager.error(f"加载训练历史失败: {e}")
        return {}
