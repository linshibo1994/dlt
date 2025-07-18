#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
深度学习预测器基类
定义所有深度学习预测器的通用接口和方法
"""

import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

# 检查TensorFlow可用性
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from core_modules import logger_manager, data_manager, cache_manager


class BaseDeepPredictor(ABC):
    """深度学习预测器基类"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        初始化预测器
        
        Args:
            name: 预测器名称
            config: 配置参数字典
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow未安装，无法使用深度学习预测器")
        
        self.name = name
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.scaler = MinMaxScaler()
        self.feature_dim = 21  # 默认特征维度
        self.sequence_length = self.config.get('sequence_length', 20)
        
        # 获取数据
        self.df = data_manager.get_data()
        if self.df is None:
            logger_manager.error("数据未加载")
        
        # 模型路径
        self.model_dir = os.path.join("cache", "models")
        os.makedirs(self.model_dir, exist_ok=True)
    
    def _extract_features(self, df_subset) -> np.ndarray:
        """
        提取深度特征
        
        Args:
            df_subset: 数据子集
            
        Returns:
            特征数组
        """
        features = []
        
        for _, row in df_subset.iterrows():
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
        
        return np.array(features)
    
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
    
    def _prepare_sequences(self, features):
        """
        准备时间序列数据
        
        Args:
            features: 特征数组
            
        Returns:
            输入序列和目标值
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            y.append(features[i][:7])  # 只预测前5个前区号码和2个后区号码
        
        return np.array(X), np.array(y)
    
    @abstractmethod
    def _build_model(self):
        """构建模型"""
        pass
    
    def train(self, epochs=100, validation_split=0.2, batch_size=32) -> bool:
        """
        训练模型
        
        Args:
            epochs: 训练轮数
            validation_split: 验证集比例
            batch_size: 批处理大小
            
        Returns:
            训练是否成功
        """
        logger_manager.info(f"开始训练{self.name}模型...")
        
        # 提取特征
        features = self._extract_features(self.df)
        
        # 数据标准化
        features_scaled = self.scaler.fit_transform(features)
        
        # 准备序列数据
        X, y = self._prepare_sequences(features_scaled)
        
        if len(X) == 0:
            logger_manager.error("序列数据不足，无法训练模型")
            return False
        
        # 构建模型
        self.model = self._build_model()
        
        # 设置回调
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0001)
        ]
        
        # 训练模型
        history = self.model.fit(
            X, y,
            epochs=epochs,
            validation_split=validation_split,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        logger_manager.info(f"{self.name}模型训练完成")
        
        # 保存模型
        self._save_model()
        
        return True
    
    def _save_model(self):
        """保存模型"""
        try:
            # 保存模型
            model_path = os.path.join(self.model_dir, f"{self.name.lower()}_model.h5")
            self.model.save(model_path)
            
            # 保存缩放器
            scaler_path = os.path.join(self.model_dir, f"{self.name.lower()}_scaler.npy")
            np.save(scaler_path, {
                'scale_': self.scaler.scale_,
                'min_': self.scaler.min_,
                'data_min_': self.scaler.data_min_,
                'data_max_': self.scaler.data_max_,
                'data_range_': self.scaler.data_range_
            })
            
            logger_manager.info(f"{self.name}模型已保存到 {model_path}")
        except Exception as e:
            logger_manager.error(f"保存{self.name}模型失败: {e}")
    
    def _load_model(self):
        """加载模型"""
        try:
            # 模型路径
            model_path = os.path.join(self.model_dir, f"{self.name.lower()}_model.h5")
            scaler_path = os.path.join(self.model_dir, f"{self.name.lower()}_scaler.npy")
            
            # 检查文件是否存在
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                logger_manager.warning(f"{self.name}模型文件不存在，需要重新训练")
                return False
            
            # 加载模型
            self.model = tf.keras.models.load_model(model_path)
            
            # 加载缩放器
            scaler_data = np.load(scaler_path, allow_pickle=True).item()
            self.scaler.scale_ = scaler_data['scale_']
            self.scaler.min_ = scaler_data['min_']
            self.scaler.data_min_ = scaler_data['data_min_']
            self.scaler.data_max_ = scaler_data['data_max_']
            self.scaler.data_range_ = scaler_data['data_range_']
            
            self.is_trained = True
            logger_manager.info(f"{self.name}模型加载成功")
            return True
        except Exception as e:
            logger_manager.error(f"加载{self.name}模型失败: {e}")
            return False
    
    def predict(self, count=1) -> List[Tuple[List[int], List[int]]]:
        """
        生成预测结果
        
        Args:
            count: 预测注数
            
        Returns:
            预测结果列表，每个元素为(前区号码列表, 后区号码列表)
        """
        # 尝试加载已有模型
        if not self.is_trained:
            if not self._load_model():
                logger_manager.info(f"{self.name}模型未训练，开始训练...")
                if not self.train():
                    logger_manager.error(f"{self.name}模型训练失败")
                    return []
        
        # 获取最近的序列数据
        recent_features = self._extract_features(self.df.head(self.sequence_length))
        recent_scaled = self.scaler.transform(recent_features)
        
        # 准备输入序列
        input_sequence = recent_scaled.reshape(1, self.sequence_length, self.feature_dim)
        
        predictions = []
        
        for _ in range(count):
            # 预测
            pred_scaled = self.model.predict(input_sequence, verbose=0)
            
            # 反标准化
            # 创建完整特征向量用于反标准化
            full_pred = np.zeros((1, self.feature_dim))
            full_pred[0, :7] = pred_scaled[0]
            pred_original = self.scaler.inverse_transform(full_pred)[0, :7]
            
            # 转换为彩票号码
            front_balls = [max(1, min(35, int(round(x)))) for x in pred_original[:5]]
            back_balls = [max(1, min(12, int(round(x)))) for x in pred_original[5:7]]
            
            # 确保号码唯一性
            front_balls = self._ensure_unique_numbers(front_balls, 1, 35, 5)
            back_balls = self._ensure_unique_numbers(back_balls, 1, 12, 2)
            
            predictions.append((sorted(front_balls), sorted(back_balls)))
            
            # 更新输入序列用于下一次预测
            new_feature = np.concatenate([pred_original, recent_scaled[-1, 7:]])
            input_sequence = np.roll(input_sequence, -1, axis=1)
            input_sequence[0, -1] = new_feature
        
        return predictions
    
    def _ensure_unique_numbers(self, numbers, min_val, max_val, target_count):
        """确保号码唯一性"""
        unique_numbers = list(set(numbers))
        
        # 如果数量不足，随机补充
        while len(unique_numbers) < target_count:
            candidate = np.random.randint(min_val, max_val + 1)
            if candidate not in unique_numbers:
                unique_numbers.append(candidate)
        
        return unique_numbers[:target_count]
    
    def get_confidence(self) -> float:
        """
        获取预测置信度
        
        Returns:
            置信度分数 (0.0-1.0)
        """
        # 默认实现，子类可以重写
        return 0.7


# 兼容性检查
if not TENSORFLOW_AVAILABLE:
    class BaseDeepPredictor(ABC):
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow未安装，无法使用深度学习预测器")
        
        def predict(self, *args, **kwargs):
            return []