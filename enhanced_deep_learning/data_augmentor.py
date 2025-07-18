#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据增强器
提供数据增强和合成功能
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional, Union
from sklearn.utils import resample
from datetime import datetime

from core_modules import logger_manager, data_manager, cache_manager, task_manager, with_progress


class DataAugmentor:
    """数据增强器"""
    
    def __init__(self):
        """初始化数据增强器"""
        self.df = data_manager.get_data()
        
        if self.df is None:
            logger_manager.error("数据未加载")
        else:
            logger_manager.info("数据增强器初始化完成")
    
    @with_progress(100, "数据增强")
    def augment_data(self, progress_bar, data: np.ndarray, factor: float = 1.5,
                   method: str = 'noise') -> np.ndarray:
        """
        数据增强
        
        Args:
            progress_bar: 进度条
            data: 输入数据
            factor: 增强因子，表示增强后的数据量是原始数据量的多少倍
            method: 增强方法，支持'noise', 'smote', 'bootstrap'
            
        Returns:
            增强后的数据
        """
        if data is None or len(data) == 0:
            logger_manager.warning("输入数据为空，无法增强")
            return np.array([])
        
        if factor <= 1.0:
            logger_manager.warning(f"增强因子 {factor} 小于等于1.0，不需要增强")
            return data
        
        original_size = len(data)
        augmented_size = int(original_size * factor)
        augmented_data = np.copy(data)
        
        # 计算需要生成的新样本数量
        new_samples_count = augmented_size - original_size
        
        # 更新进度条总数
        progress_bar.total = new_samples_count
        
        if method == 'noise':
            # 添加随机噪声
            for i in range(new_samples_count):
                # 随机选择一个样本
                idx = np.random.randint(0, original_size)
                sample = np.copy(data[idx])
                
                # 添加随机噪声
                noise_level = 0.05  # 5%的噪声
                noise = np.random.normal(0, noise_level, sample.shape)
                
                # 确保号码特征（前7个）在增强后仍然有效
                augmented_sample = sample + noise
                
                # 添加到增强数据集
                augmented_data = np.vstack([augmented_data, augmented_sample])
                
                # 更新进度
                progress_bar.update(1)
        
        elif method == 'bootstrap':
            # 使用自助法（Bootstrap）
            bootstrap_samples = resample(data, n_samples=new_samples_count, random_state=42)
            augmented_data = np.vstack([augmented_data, bootstrap_samples])
            progress_bar.update(new_samples_count)
        
        elif method == 'smote':
            try:
                # 使用SMOTE（合成少数类过采样技术）
                from imblearn.over_sampling import SMOTE
                
                # 创建标签（这里只是为了使用SMOTE，实际上没有意义）
                y = np.zeros(len(data))
                
                # 应用SMOTE
                smote = SMOTE(sampling_strategy='auto', random_state=42)
                X_resampled, _ = smote.fit_resample(data, y)
                
                # 如果SMOTE生成的样本数量不足，使用自助法补充
                if len(X_resampled) < augmented_size:
                    bootstrap_samples = resample(X_resampled, n_samples=augmented_size-len(X_resampled), random_state=42)
                    augmented_data = np.vstack([X_resampled, bootstrap_samples])
                else:
                    augmented_data = X_resampled[:augmented_size]
                
                progress_bar.update(new_samples_count)
            except Exception as e:
                logger_manager.error(f"SMOTE增强失败: {e}，使用噪声增强代替")
                return self.augment_data(progress_bar, data, factor, 'noise')
        
        else:
            logger_manager.warning(f"未知的增强方法: {method}，使用噪声增强代替")
            return self.augment_data(progress_bar, data, factor, 'noise')
        
        logger_manager.info(f"数据增强完成，方法: {method}，原始数据量: {original_size}，增强后数据量: {len(augmented_data)}")
        
        return augmented_data
    
    def generate_synthetic_data(self, count: int, feature_dim: int,
                              method: str = 'random') -> np.ndarray:
        """
        生成合成数据
        
        Args:
            count: 样本数量
            feature_dim: 特征维度
            method: 生成方法，支持'random', 'gaussian', 'uniform'
            
        Returns:
            合成数据
        """
        if count <= 0 or feature_dim <= 0:
            logger_manager.warning(f"无效的参数: count={count}, feature_dim={feature_dim}")
            return np.array([])
        
        if method == 'random':
            # 生成随机数据
            synthetic_data = np.random.random((count, feature_dim))
        elif method == 'gaussian':
            # 生成高斯分布数据
            synthetic_data = np.random.normal(0.5, 0.15, (count, feature_dim))
            # 裁剪到[0, 1]范围
            synthetic_data = np.clip(synthetic_data, 0, 1)
        elif method == 'uniform':
            # 生成均匀分布数据
            synthetic_data = np.random.uniform(0, 1, (count, feature_dim))
        else:
            logger_manager.warning(f"未知的生成方法: {method}，使用随机生成代替")
            synthetic_data = np.random.random((count, feature_dim))
        
        logger_manager.info(f"合成数据生成完成，方法: {method}，数据形状: {synthetic_data.shape}")
        
        return synthetic_data
    
    def balance_data(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        平衡数据
        
        Args:
            data: 输入数据
            labels: 标签
            
        Returns:
            平衡后的数据和标签
        """
        if data is None or len(data) == 0 or labels is None or len(labels) == 0:
            logger_manager.warning("输入数据或标签为空，无法平衡")
            return data, labels
        
        if len(data) != len(labels):
            logger_manager.warning(f"数据和标签长度不一致: {len(data)} vs {len(labels)}")
            return data, labels
        
        try:
            # 统计各类别样本数量
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            if len(unique_labels) <= 1:
                logger_manager.warning("只有一个类别，无需平衡")
                return data, labels
            
            # 找出样本数量最多的类别
            max_count = np.max(counts)
            
            # 平衡后的数据和标签
            balanced_data = []
            balanced_labels = []
            
            # 对每个类别进行过采样
            for label in unique_labels:
                # 获取当前类别的样本
                label_mask = labels == label
                label_data = data[label_mask]
                label_count = len(label_data)
                
                # 如果样本数量不足，进行过采样
                if label_count < max_count:
                    # 计算需要生成的样本数量
                    n_samples = max_count - label_count
                    
                    # 使用自助法生成新样本
                    bootstrap_samples = resample(label_data, n_samples=n_samples, random_state=42)
                    
                    # 合并原始样本和新样本
                    balanced_class_data = np.vstack([label_data, bootstrap_samples])
                    balanced_class_labels = np.array([label] * max_count)
                else:
                    # 如果样本数量足够，直接使用
                    balanced_class_data = label_data
                    balanced_class_labels = np.array([label] * label_count)
                
                # 添加到平衡后的数据和标签
                balanced_data.append(balanced_class_data)
                balanced_labels.append(balanced_class_labels)
            
            # 合并所有类别的数据和标签
            balanced_data = np.vstack(balanced_data)
            balanced_labels = np.concatenate(balanced_labels)
            
            logger_manager.info(f"数据平衡完成，原始数据量: {len(data)}，平衡后数据量: {len(balanced_data)}")
            
            return balanced_data, balanced_labels
        except Exception as e:
            logger_manager.error(f"数据平衡失败: {e}")
            return data, labels
    
    def mix_data(self, data1: np.ndarray, data2: np.ndarray, ratio: float = 0.5) -> np.ndarray:
        """
        混合数据
        
        Args:
            data1: 第一组数据
            data2: 第二组数据
            ratio: 混合比例，表示data1的比例
            
        Returns:
            混合后的数据
        """
        if data1 is None or len(data1) == 0:
            logger_manager.warning("第一组数据为空，返回第二组数据")
            return data2
        
        if data2 is None or len(data2) == 0:
            logger_manager.warning("第二组数据为空，返回第一组数据")
            return data1
        
        if data1.shape[1] != data2.shape[1]:
            logger_manager.warning(f"两组数据维度不一致: {data1.shape[1]} vs {data2.shape[1]}")
            return data1
        
        # 计算混合后的数据量
        mixed_size = len(data1) + len(data2)
        
        # 计算各组数据的样本数量
        data1_count = int(mixed_size * ratio)
        data2_count = mixed_size - data1_count
        
        # 如果某组数据样本不足，使用重采样
        if data1_count > len(data1):
            data1_extra = resample(data1, n_samples=data1_count-len(data1), random_state=42)
            data1_samples = np.vstack([data1, data1_extra])
        else:
            # 随机选择样本
            indices = np.random.choice(len(data1), data1_count, replace=False)
            data1_samples = data1[indices]
        
        if data2_count > len(data2):
            data2_extra = resample(data2, n_samples=data2_count-len(data2), random_state=42)
            data2_samples = np.vstack([data2, data2_extra])
        else:
            # 随机选择样本
            indices = np.random.choice(len(data2), data2_count, replace=False)
            data2_samples = data2[indices]
        
        # 合并两组数据
        mixed_data = np.vstack([data1_samples, data2_samples])
        
        # 随机打乱
        np.random.shuffle(mixed_data)
        
        logger_manager.info(f"数据混合完成，混合比例: {ratio}，混合后数据量: {len(mixed_data)}")
        
        return mixed_data


if __name__ == "__main__":
    # 测试数据增强器
    print("📊 测试数据增强器...")
    
    # 创建数据增强器
    augmentor = DataAugmentor()
    
    # 创建测试数据
    test_data = np.random.random((100, 10))
    
    # 测试数据增强
    augmented_data = augmentor.augment_data(test_data, factor=1.5)
    print(f"增强后数据形状: {augmented_data.shape}")
    
    # 测试合成数据生成
    synthetic_data = augmentor.generate_synthetic_data(50, 10)
    print(f"合成数据形状: {synthetic_data.shape}")
    
    # 测试数据平衡
    labels = np.random.randint(0, 3, 100)
    balanced_data, balanced_labels = augmentor.balance_data(test_data, labels)
    print(f"平衡后数据形状: {balanced_data.shape}")
    
    # 测试数据混合
    mixed_data = augmentor.mix_data(test_data, synthetic_data, ratio=0.7)
    print(f"混合后数据形状: {mixed_data.shape}")
    
    print("数据增强器测试完成")