#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据预处理器
提供数据标准化、增强和异常检测功能
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from datetime import datetime

from core_modules import logger_manager, data_manager, cache_manager


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, cache_enabled: bool = True):
        """
        初始化数据预处理器
        
        Args:
            cache_enabled: 是否启用缓存
        """
        self.cache_enabled = cache_enabled
        self.df = data_manager.get_data()
        self.cache_manager = cache_manager
        self.scalers = {}
        
        if self.df is None:
            logger_manager.error("数据未加载")
        else:
            logger_manager.info("数据预处理器初始化完成")
    
    def normalize_data(self, data: np.ndarray, method: str = 'minmax',
                     feature_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """
        数据标准化
        
        Args:
            data: 输入数据
            method: 标准化方法，支持'minmax', 'standard', 'robust'
            feature_range: 特征范围，仅对'minmax'有效
            
        Returns:
            标准化后的数据
        """
        if data is None or len(data) == 0:
            logger_manager.warning("输入数据为空，无法标准化")
            return np.array([])
        
        # 检查缓存
        cache_key = f"{method}_{feature_range[0]}_{feature_range[1]}"
        if self.cache_enabled and cache_key in self.scalers:
            logger_manager.debug(f"使用缓存的标准化器: {cache_key}")
            scaler = self.scalers[cache_key]
        else:
            # 创建标准化器
            if method == 'minmax':
                scaler = MinMaxScaler(feature_range=feature_range)
            elif method == 'standard':
                scaler = StandardScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                logger_manager.warning(f"未知的标准化方法: {method}，使用MinMaxScaler")
                scaler = MinMaxScaler(feature_range=feature_range)
            
            # 缓存标准化器
            if self.cache_enabled:
                self.scalers[cache_key] = scaler
        
        # 标准化数据
        try:
            normalized_data = scaler.fit_transform(data)
            logger_manager.info(f"数据标准化完成，方法: {method}，数据形状: {normalized_data.shape}")
            return normalized_data
        except Exception as e:
            logger_manager.error(f"数据标准化失败: {e}")
            return data
    
    def denormalize_data(self, normalized_data: np.ndarray, method: str = 'minmax',
                       feature_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """
        数据反标准化
        
        Args:
            normalized_data: 标准化后的数据
            method: 标准化方法，支持'minmax', 'standard', 'robust'
            feature_range: 特征范围，仅对'minmax'有效
            
        Returns:
            原始数据
        """
        if normalized_data is None or len(normalized_data) == 0:
            logger_manager.warning("输入数据为空，无法反标准化")
            return np.array([])
        
        # 检查缓存
        cache_key = f"{method}_{feature_range[0]}_{feature_range[1]}"
        if self.cache_enabled and cache_key in self.scalers:
            logger_manager.debug(f"使用缓存的标准化器: {cache_key}")
            scaler = self.scalers[cache_key]
        else:
            logger_manager.warning(f"未找到缓存的标准化器: {cache_key}，无法反标准化")
            return normalized_data
        
        # 反标准化数据
        try:
            original_data = scaler.inverse_transform(normalized_data)
            logger_manager.info(f"数据反标准化完成，方法: {method}，数据形状: {original_data.shape}")
            return original_data
        except Exception as e:
            logger_manager.error(f"数据反标准化失败: {e}")
            return normalized_data
    
    def detect_anomalies(self, data: np.ndarray, contamination: float = 0.05,
                       method: str = 'isolation_forest') -> Tuple[np.ndarray, np.ndarray]:
        """
        异常检测
        
        Args:
            data: 输入数据
            contamination: 异常比例
            method: 异常检测方法，支持'isolation_forest', 'zscore'
            
        Returns:
            正常数据和异常数据的元组
        """
        if data is None or len(data) == 0:
            logger_manager.warning("输入数据为空，无法检测异常")
            return np.array([]), np.array([])
        
        if method == 'isolation_forest':
            try:
                # 使用隔离森林检测异常
                detector = IsolationForest(contamination=contamination, random_state=42)
                predictions = detector.fit_predict(data)
                
                # 1表示正常，-1表示异常
                normal_mask = predictions == 1
                anomaly_mask = predictions == -1
                
                normal_data = data[normal_mask]
                anomaly_data = data[anomaly_mask]
                
                logger_manager.info(f"异常检测完成，方法: {method}，正常数据: {len(normal_data)}，异常数据: {len(anomaly_data)}")
                return normal_data, anomaly_data
            except Exception as e:
                logger_manager.error(f"异常检测失败: {e}")
                return data, np.array([])
        elif method == 'zscore':
            try:
                # 使用Z分数检测异常
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                
                # 计算Z分数
                z_scores = np.abs((data - mean) / (std + 1e-10))
                
                # 如果任何特征的Z分数超过阈值，则认为是异常
                threshold = 3.0  # 3个标准差
                anomaly_mask = np.any(z_scores > threshold, axis=1)
                normal_mask = ~anomaly_mask
                
                normal_data = data[normal_mask]
                anomaly_data = data[anomaly_mask]
                
                logger_manager.info(f"异常检测完成，方法: {method}，正常数据: {len(normal_data)}，异常数据: {len(anomaly_data)}")
                return normal_data, anomaly_data
            except Exception as e:
                logger_manager.error(f"异常检测失败: {e}")
                return data, np.array([])
        else:
            logger_manager.warning(f"未知的异常检测方法: {method}")
            return data, np.array([])
    
    def reduce_dimensions(self, data: np.ndarray, n_components: int = 10) -> np.ndarray:
        """
        降维
        
        Args:
            data: 输入数据
            n_components: 目标维度
            
        Returns:
            降维后的数据
        """
        if data is None or len(data) == 0:
            logger_manager.warning("输入数据为空，无法降维")
            return np.array([])
        
        if n_components >= data.shape[1]:
            logger_manager.warning(f"目标维度 {n_components} 大于等于原始维度 {data.shape[1]}，不需要降维")
            return data
        
        try:
            # 使用PCA降维
            pca = PCA(n_components=n_components)
            reduced_data = pca.fit_transform(data)
            
            # 计算解释方差比例
            explained_variance_ratio = pca.explained_variance_ratio_.sum()
            
            logger_manager.info(f"降维完成，原始维度: {data.shape[1]}，目标维度: {n_components}，解释方差比例: {explained_variance_ratio:.4f}")
            return reduced_data
        except Exception as e:
            logger_manager.error(f"降维失败: {e}")
            return data