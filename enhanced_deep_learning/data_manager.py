#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
深度学习数据管理器
负责数据预处理、批处理和增强
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime

from .config import DEFAULT_DATA_MANAGER_CONFIG
from .exceptions import TrainingDataError
from core_modules import logger_manager, data_manager, cache_manager, task_manager, with_progress


class DeepLearningDataManager:
    """深度学习训练数据管理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化数据管理器
        
        Args:
            config: 配置参数字典
        """
        # 合并默认配置和用户配置
        self.config = DEFAULT_DATA_MANAGER_CONFIG.copy()
        if config:
            self.config.update(config)
        
        self.df = data_manager.get_data()
        self.cache_manager = cache_manager
        self.cached_features = {}
        self.cached_sequences = {}
        
        if self.df is None:
            logger_manager.error("数据未加载")
            raise TrainingDataError("数据未加载")
    
    def get_training_window(self, window_size: Optional[int] = None) -> pd.DataFrame:
        """
        获取指定窗口大小的训练数据
        
        Args:
            window_size: 窗口大小，如果为None则使用默认值
            
        Returns:
            训练数据子集
        """
        if window_size is None:
            window_size = self.config.get('default_window', 500)
        
        if window_size > len(self.df):
            logger_manager.warning(f"请求的窗口大小 {window_size} 超过可用数据量 {len(self.df)}，将使用全部数据")
            return self.df
        
        return self.df.head(window_size)
    
    def incremental_update(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        增量更新训练数据
        
        Args:
            new_data: 新数据
            
        Returns:
            更新后的数据
        """
        if new_data is None or len(new_data) == 0:
            logger_manager.warning("没有新数据可更新")
            return self.df
        
        # 检查数据一致性
        if not self._check_data_consistency(new_data):
            raise TrainingDataError("新数据格式与现有数据不一致")
        
        # 合并数据并去重
        updated_df = pd.concat([new_data, self.df]).drop_duplicates(subset=['issue'])
        
        # 按期号排序
        updated_df = updated_df.sort_values(by='issue', ascending=False).reset_index(drop=True)
        
        # 更新缓存
        self._clear_feature_cache()
        
        logger_manager.info(f"数据增量更新完成，当前数据量: {len(updated_df)}")
        
        return updated_df
    
    def _check_data_consistency(self, new_data: pd.DataFrame) -> bool:
        """
        检查数据一致性
        
        Args:
            new_data: 新数据
            
        Returns:
            数据是否一致
        """
        # 检查必要列是否存在
        required_columns = ['issue', 'date', 'front_balls', 'back_balls']
        for col in required_columns:
            if col not in new_data.columns:
                logger_manager.error(f"新数据缺少必要列: {col}")
                return False
        
        # 检查数据类型
        try:
            # 尝试解析一行数据
            if len(new_data) > 0:
                row = new_data.iloc[0]
                front_balls, back_balls = data_manager.parse_balls(row)
                
                # 检查号码格式
                if len(front_balls) != 5 or len(back_balls) != 2:
                    logger_manager.error(f"号码格式错误: 前区 {len(front_balls)}/5, 后区 {len(back_balls)}/2")
                    return False
        except Exception as e:
            logger_manager.error(f"数据格式检查失败: {e}")
            return False
        
        return True
    
    def normalize_data(self, features: np.ndarray, method: str = None) -> np.ndarray:
        """
        数据标准化处理
        
        Args:
            features: 特征数组
            method: 标准化方法，'minmax'或'standard'
            
        Returns:
            标准化后的特征
        """
        if method is None:
            method = self.config.get('normalization', 'minmax')
        
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            logger_manager.warning(f"未知的标准化方法: {method}，使用MinMaxScaler")
            scaler = MinMaxScaler()
        
        return scaler.fit_transform(features)
    
    @with_progress(100, "数据增强")
    def augment_data(self, progress_bar, features: np.ndarray, factor: float = None) -> np.ndarray:
        """
        数据增强处理
        
        Args:
            progress_bar: 进度条
            features: 特征数组
            factor: 增强因子，表示增强后的数据量是原始数据量的多少倍
            
        Returns:
            增强后的特征
        """
        if factor is None:
            factor = self.config.get('augmentation_factor', 1.5)
        
        if factor <= 1.0:
            return features
        
        original_size = len(features)
        augmented_size = int(original_size * factor)
        augmented_features = np.copy(features)
        
        # 计算需要生成的新样本数量
        new_samples_count = augmented_size - original_size
        
        # 更新进度条总数
        progress_bar.total = new_samples_count
        
        # 生成新样本
        for i in range(new_samples_count):
            # 随机选择一个样本
            idx = np.random.randint(0, original_size)
            sample = np.copy(features[idx])
            
            # 添加随机噪声
            noise_level = 0.05  # 5%的噪声
            noise = np.random.normal(0, noise_level, sample.shape)
            
            # 确保号码特征（前7个）在增强后仍然有效
            augmented_sample = sample + noise
            augmented_sample[:7] = np.clip(np.round(augmented_sample[:7]), 
                                          [1, 1, 1, 1, 1, 1, 1], 
                                          [35, 35, 35, 35, 35, 12, 12])
            
            # 添加到增强数据集
            augmented_features = np.vstack([augmented_features, augmented_sample])
            
            # 更新进度
            progress_bar.update(1)
        
        logger_manager.info(f"数据增强完成: {original_size} -> {len(augmented_features)}")
        
        return augmented_features
    
    def detect_anomalies(self, features: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        检测并处理异常数据
        
        Args:
            features: 特征数组
            threshold: 异常阈值（标准差的倍数）
            
        Returns:
            正常数据和异常数据的元组
        """
        # 计算每个特征的均值和标准差
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        
        # 计算Z分数
        z_scores = np.abs((features - mean) / (std + 1e-10))
        
        # 识别异常样本（任何特征的Z分数超过阈值）
        anomalies_mask = np.any(z_scores > threshold, axis=1)
        
        # 分离正常数据和异常数据
        normal_features = features[~anomalies_mask]
        anomaly_features = features[anomalies_mask]
        
        anomaly_count = len(anomaly_features)
        if anomaly_count > 0:
            logger_manager.info(f"检测到 {anomaly_count} 个异常样本 ({anomaly_count/len(features)*100:.2f}%)")
        
        return normal_features, anomaly_features
    
    def prepare_batch_data(self, sequence_length: int, batch_size: int = 32, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        准备批处理数据
        
        Args:
            sequence_length: 序列长度
            batch_size: 批大小
            validation_split: 验证集比例
            
        Returns:
            包含训练和验证数据的字典
        """
        # 检查缓存
        cache_key = f"seq_{sequence_length}_batch_{batch_size}_val_{validation_split}"
        if cache_key in self.cached_sequences:
            logger_manager.info(f"使用缓存的序列数据: {cache_key}")
            return self.cached_sequences[cache_key]
        
        # 提取特征
        features = self._extract_features()
        
        # 标准化
        features_scaled = self.normalize_data(features)
        
        # 准备序列数据
        X, y = self._prepare_sequences(features_scaled, sequence_length)
        
        if len(X) == 0:
            raise TrainingDataError(f"序列数据不足，无法准备批处理数据，序列长度: {sequence_length}")
        
        # 分割训练集和验证集
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # 创建TensorFlow数据集
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size)
        
        # 准备返回数据
        batch_data = {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'feature_dim': features.shape[1],
            'sequence_length': sequence_length,
            'batch_size': batch_size
        }
        
        # 缓存数据
        if self.config.get('cache_enabled', True):
            self.cached_sequences[cache_key] = batch_data
        
        logger_manager.info(f"批处理数据准备完成: {len(X_train)} 训练样本, {len(X_val)} 验证样本")
        
        return batch_data
    
    def _extract_features(self) -> np.ndarray:
        """
        提取深度特征
        
        Returns:
            特征数组
        """
        # 检查缓存
        if 'features' in self.cached_features:
            return self.cached_features['features']
        
        features = []
        
        for _, row in self.df.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            
            # 基础特征 (7维)
            feature_vector = front_balls + back_balls
            
            # 统计特征 (8维)
            front_sum = sum(front_balls)
            back_sum = sum(back_balls)
            total_sum = front_sum + back_sum
            front_mean = np.mean(front_balls)
            back_mean = np.mean(back_balls)
            front_std = np.std(front_balls)
            back_std = np.std(back_balls)
            span = max(front_balls) - min(front_balls)
            
            # 模式特征 (6维)
            odd_count = sum(1 for x in front_balls if x % 2 == 1)
            even_count = 5 - odd_count
            big_count = sum(1 for x in front_balls if x > 17)  # 大号(18-35)
            small_count = 5 - big_count  # 小号(1-17)
            consecutive_count = self._count_consecutive(front_balls)
            prime_count = sum(1 for x in front_balls if self._is_prime(x))
            
            # 组合所有特征
            all_features = (
                feature_vector +  # 7维
                [front_sum, back_sum, total_sum, front_mean, back_mean, front_std, back_std, span] +  # 8维
                [odd_count, even_count, big_count, small_count, consecutive_count, prime_count]  # 6维
            )
            
            features.append(all_features)
        
        features_array = np.array(features)
        
        # 缓存特征
        if self.config.get('cache_enabled', True):
            self.cached_features['features'] = features_array
        
        return features_array
    
    def _prepare_sequences(self, features: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备时间序列数据
        
        Args:
            features: 特征数组
            sequence_length: 序列长度
            
        Returns:
            输入序列和目标值的元组
        """
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            X.append(features[i-sequence_length:i])
            y.append(features[i][:7])  # 只预测前5个前区号码和2个后区号码
        
        return np.array(X), np.array(y)
    
    def _count_consecutive(self, numbers):
        """计算连续号码数量"""
        sorted_nums = sorted(numbers)
        consecutive = 0
        for i in range(len(sorted_nums) - 1):
            if sorted_nums[i+1] - sorted_nums[i] == 1:
                consecutive += 1
        return consecutive
    
    def _is_prime(self, n):
        """判断是否为质数"""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def _clear_feature_cache(self):
        """清除特征缓存"""
        self.cached_features = {}
        self.cached_sequences = {}
        logger_manager.debug("特征缓存已清除")
    
    def use_cached_data(self):
        """使用缓存的上一版本数据"""
        try:
            cached_df = self.cache_manager.load_cache("data", "dlt_data_backup")
            if cached_df is not None and not cached_df.empty:
                self.df = cached_df
                logger_manager.info(f"已加载缓存的上一版本数据: {len(self.df)} 期")
                return True
            else:
                logger_manager.warning("没有可用的缓存数据")
                return False
        except Exception as e:
            logger_manager.error(f"加载缓存数据失败: {e}")
            return False