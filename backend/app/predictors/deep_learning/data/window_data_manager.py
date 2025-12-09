#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
窗口化数据管理器
提供可变窗口大小的数据管理功能
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional, Union
from collections import defaultdict
from datetime import datetime

from core_modules import logger_manager, data_manager, cache_manager


class WindowDataManager:
    """窗口化数据管理器"""
    
    def __init__(self, default_window_size: int = 500, cache_enabled: bool = True):
        """
        初始化窗口化数据管理器
        
        Args:
            default_window_size: 默认窗口大小
            cache_enabled: 是否启用缓存
        """
        self.default_window_size = default_window_size
        self.cache_enabled = cache_enabled
        self.df = data_manager.get_data()
        self.cache_manager = cache_manager
        self.window_cache = {}
        
        if self.df is None:
            logger_manager.error("数据未加载")
        else:
            logger_manager.info(f"窗口化数据管理器初始化完成，默认窗口大小: {default_window_size}")
    
    def get_window(self, window_size: Optional[int] = None) -> pd.DataFrame:
        """
        获取指定窗口大小的数据
        
        Args:
            window_size: 窗口大小，如果为None则使用默认值
            
        Returns:
            窗口数据
        """
        if window_size is None:
            window_size = self.default_window_size
        
        # 检查缓存
        if self.cache_enabled and window_size in self.window_cache:
            logger_manager.debug(f"从缓存获取窗口数据，窗口大小: {window_size}")
            return self.window_cache[window_size]
        
        if self.df is None:
            logger_manager.error("数据未加载，无法获取窗口")
            return pd.DataFrame()
        
        # 如果窗口大小大于数据量，返回全部数据
        if window_size >= len(self.df):
            logger_manager.warning(f"请求的窗口大小 {window_size} 超过可用数据量 {len(self.df)}，返回全部数据")
            result = self.df.copy()
        else:
            # 获取最新的window_size条数据
            result = self.df.head(window_size).copy()
        
        # 缓存结果
        if self.cache_enabled:
            self.window_cache[window_size] = result
        
        logger_manager.info(f"获取窗口数据，窗口大小: {window_size}，数据量: {len(result)}")
        
        return result
    
    def get_sliding_windows(self, window_size: int, step_size: int, count: int) -> List[pd.DataFrame]:
        """
        获取滑动窗口数据
        
        Args:
            window_size: 窗口大小
            step_size: 步长
            count: 窗口数量
            
        Returns:
            窗口数据列表
        """
        if self.df is None:
            logger_manager.error("数据未加载，无法获取滑动窗口")
            return []
        
        # 检查参数
        if window_size <= 0 or step_size <= 0 or count <= 0:
            logger_manager.error(f"无效的参数: window_size={window_size}, step_size={step_size}, count={count}")
            return []
        
        # 检查数据量是否足够
        required_data = (count - 1) * step_size + window_size
        if required_data > len(self.df):
            logger_manager.warning(f"请求的滑动窗口需要 {required_data} 条数据，但只有 {len(self.df)} 条可用")
            count = max(1, (len(self.df) - window_size) // step_size + 1)
            logger_manager.warning(f"调整窗口数量为 {count}")
        
        # 获取滑动窗口
        windows = []
        for i in range(count):
            start = i * step_size
            end = start + window_size
            if end > len(self.df):
                break
            windows.append(self.df.iloc[start:end].copy())
        
        logger_manager.info(f"获取滑动窗口，窗口大小: {window_size}，步长: {step_size}，窗口数量: {len(windows)}")
        
        return windows
    
    def get_expanding_windows(self, initial_size: int, step_size: int, count: int) -> List[pd.DataFrame]:
        """
        获取扩展窗口数据
        
        Args:
            initial_size: 初始窗口大小
            step_size: 步长
            count: 窗口数量
            
        Returns:
            窗口数据列表
        """
        if self.df is None:
            logger_manager.error("数据未加载，无法获取扩展窗口")
            return []
        
        # 检查参数
        if initial_size <= 0 or step_size <= 0 or count <= 0:
            logger_manager.error(f"无效的参数: initial_size={initial_size}, step_size={step_size}, count={count}")
            return []
        
        # 检查数据量是否足够
        required_data = initial_size + (count - 1) * step_size
        if required_data > len(self.df):
            logger_manager.warning(f"请求的扩展窗口需要 {required_data} 条数据，但只有 {len(self.df)} 条可用")
            count = max(1, (len(self.df) - initial_size) // step_size + 1)
            logger_manager.warning(f"调整窗口数量为 {count}")
        
        # 获取扩展窗口
        windows = []
        for i in range(count):
            size = initial_size + i * step_size
            if size > len(self.df):
                break
            windows.append(self.df.head(size).copy())
        
        logger_manager.info(f"获取扩展窗口，初始大小: {initial_size}，步长: {step_size}，窗口数量: {len(windows)}")
        
        return windows
    
    def get_rolling_windows(self, window_size: int, periods: List[int]) -> Dict[int, pd.DataFrame]:
        """
        获取滚动窗口数据
        
        Args:
            window_size: 窗口大小
            periods: 期号列表
            
        Returns:
            窗口数据字典，键为期号，值为窗口数据
        """
        if self.df is None:
            logger_manager.error("数据未加载，无法获取滚动窗口")
            return {}
        
        # 检查参数
        if window_size <= 0 or not periods:
            logger_manager.error(f"无效的参数: window_size={window_size}, periods={periods}")
            return {}
        
        # 获取滚动窗口
        windows = {}
        for period in periods:
            # 找到期号对应的索引
            try:
                period_idx = self.df[self.df['issue'] == period].index[0]
            except (IndexError, KeyError):
                logger_manager.warning(f"找不到期号 {period}")
                continue
            
            # 获取窗口数据
            start_idx = max(0, period_idx - window_size + 1)
            window_data = self.df.iloc[start_idx:period_idx+1].copy()
            
            windows[period] = window_data
        
        logger_manager.info(f"获取滚动窗口，窗口大小: {window_size}，期号数量: {len(windows)}")
        
        return windows
    
    def clear_cache(self) -> None:
        """清除缓存"""
        self.window_cache = {}
        logger_manager.info("窗口缓存已清除")

    def create_sequence_windows(self, data: np.ndarray, window_size: int, step_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建序列窗口数据（用于深度学习）

        Args:
            data: 输入数据
            window_size: 窗口大小
            step_size: 步长

        Returns:
            窗口数据和标签
        """
        try:
            if len(data) < window_size + 1:
                logger_manager.warning(f"数据长度 {len(data)} 小于窗口大小 {window_size} + 1")
                return np.array([]), np.array([])

            X, y = [], []

            for i in range(0, len(data) - window_size, step_size):
                # 输入窗口
                window = data[i:i + window_size]
                # 目标值（下一个值）
                target = data[i + window_size]

                X.append(window)
                y.append(target)

            X = np.array(X)
            y = np.array(y)

            logger_manager.info(f"创建序列窗口数据: X.shape={X.shape}, y.shape={y.shape}")
            return X, y

        except Exception as e:
            logger_manager.error(f"创建序列窗口数据失败: {e}")
            return np.array([]), np.array([])

    def create_multi_step_windows(self, data: np.ndarray, window_size: int,
                                 prediction_steps: int = 1, step_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建多步预测窗口数据

        Args:
            data: 输入数据
            window_size: 窗口大小
            prediction_steps: 预测步数
            step_size: 步长

        Returns:
            窗口数据和多步标签
        """
        try:
            if len(data) < window_size + prediction_steps:
                logger_manager.warning(f"数据长度不足以创建多步预测窗口")
                return np.array([]), np.array([])

            X, y = [], []

            for i in range(0, len(data) - window_size - prediction_steps + 1, step_size):
                # 输入窗口
                window = data[i:i + window_size]
                # 多步目标值
                targets = data[i + window_size:i + window_size + prediction_steps]

                X.append(window)
                y.append(targets)

            X = np.array(X)
            y = np.array(y)

            logger_manager.info(f"创建多步窗口数据: X.shape={X.shape}, y.shape={y.shape}")
            return X, y

        except Exception as e:
            logger_manager.error(f"创建多步窗口数据失败: {e}")
            return np.array([]), np.array([])

    def create_adaptive_windows(self, data: np.ndarray, min_window_size: int = 10,
                               max_window_size: int = 100, volatility_threshold: float = 0.1) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        创建自适应窗口数据（根据数据波动性调整窗口大小）

        Args:
            data: 输入数据
            min_window_size: 最小窗口大小
            max_window_size: 最大窗口大小
            volatility_threshold: 波动性阈值

        Returns:
            自适应窗口数据列表
        """
        try:
            windows = []
            i = 0

            while i < len(data) - min_window_size:
                # 计算当前区域的波动性
                current_window = data[i:i + min_window_size]
                volatility = np.std(current_window) / np.mean(current_window) if np.mean(current_window) != 0 else 0

                # 根据波动性调整窗口大小
                if volatility > volatility_threshold:
                    window_size = min_window_size
                else:
                    window_size = min(max_window_size, len(data) - i - 1)

                # 确保有足够的数据
                if i + window_size + 1 <= len(data):
                    window = data[i:i + window_size]
                    target = data[i + window_size]
                    windows.append((window, target))

                i += window_size // 2  # 移动半个窗口大小

            logger_manager.info(f"创建自适应窗口数据: {len(windows)} 个窗口")
            return windows

        except Exception as e:
            logger_manager.error(f"创建自适应窗口数据失败: {e}")
            return []
    
    def get_data_stats(self, window_size: Optional[int] = None) -> Dict[str, Any]:
        """
        获取数据统计信息
        
        Args:
            window_size: 窗口大小，如果为None则使用全部数据
            
        Returns:
            统计信息字典
        """
        if self.df is None:
            logger_manager.error("数据未加载，无法获取统计信息")
            return {}
        
        # 获取窗口数据
        if window_size is None:
            data = self.df
        else:
            data = self.get_window(window_size)
        
        if data.empty:
            return {}
        
        # 计算统计信息
        stats = {
            'data_count': len(data),
            'date_range': {
                'start': data['date'].min(),
                'end': data['date'].max()
            },
            'issue_range': {
                'start': data['issue'].min(),
                'end': data['issue'].max()
            }
        }
        
        # 计算号码统计
        front_balls = []
        back_balls = []
        
        for _, row in data.iterrows():
            front, back = data_manager.parse_balls(row)
            front_balls.extend(front)
            back_balls.extend(back)
        
        # 前区号码频率
        front_freq = {}
        for num in range(1, 36):
            front_freq[num] = front_balls.count(num)
        
        # 后区号码频率
        back_freq = {}
        for num in range(1, 13):
            back_freq[num] = back_balls.count(num)
        
        stats['front_frequency'] = front_freq
        stats['back_frequency'] = back_freq
        
        # 计算热门号码
        front_hot = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        back_hot = sorted(back_freq.items(), key=lambda x: x[1], reverse=True)[:2]
        
        stats['front_hot'] = [num for num, _ in front_hot]
        stats['back_hot'] = [num for num, _ in back_hot]
        
        logger_manager.info(f"获取数据统计信息，数据量: {stats['data_count']}")
        
        return stats
    
    def save_window_data(self, window_size: int, file_path: str) -> bool:
        """
        保存窗口数据到文件
        
        Args:
            window_size: 窗口大小
            file_path: 文件路径
            
        Returns:
            是否保存成功
        """
        # 获取窗口数据
        data = self.get_window(window_size)
        
        if data.empty:
            logger_manager.error("窗口数据为空，无法保存")
            return False
        
        try:
            # 保存到文件
            data.to_csv(file_path, index=False)
            logger_manager.info(f"窗口数据已保存到 {file_path}，窗口大小: {window_size}，数据量: {len(data)}")
            return True
        except Exception as e:
            logger_manager.error(f"保存窗口数据失败: {e}")
            return False

    def create_expanding_window(self, min_size: int = 10, max_size: Optional[int] = None) -> List[pd.DataFrame]:
        """
        创建扩展窗口

        Args:
            min_size: 最小窗口大小
            max_size: 最大窗口大小

        Returns:
            扩展窗口列表
        """
        if self.df is None:
            logger_manager.error("数据未加载，无法创建扩展窗口")
            return []

        if max_size is None:
            max_size = len(self.df)

        windows = []
        for size in range(min_size, min(max_size + 1, len(self.df) + 1)):
            window = self.df.tail(size)  # 取最近的数据
            windows.append(window)

        logger_manager.info(f"创建了 {len(windows)} 个扩展窗口，大小范围: {min_size}-{max_size}")
        return windows

    def create_variable_window(self, size_strategy: str = "adaptive") -> List[pd.DataFrame]:
        """
        创建可变大小窗口

        Args:
            size_strategy: 窗口大小策略 ('adaptive', 'fibonacci', 'exponential')

        Returns:
            可变窗口列表
        """
        if self.df is None:
            logger_manager.error("数据未加载，无法创建可变窗口")
            return []

        windows = []
        data_len = len(self.df)

        if size_strategy == "adaptive":
            # 自适应窗口大小
            sizes = [10, 20, 30, 50, 100, min(200, data_len)]
        elif size_strategy == "fibonacci":
            # 斐波那契序列窗口大小
            sizes = self._generate_fibonacci_sizes(data_len)
        elif size_strategy == "exponential":
            # 指数增长窗口大小
            sizes = self._generate_exponential_sizes(data_len)
        else:
            sizes = [10, 20, 30]

        for size in sizes:
            if size <= data_len:
                window = self.df.tail(size)  # 取最近的数据
                windows.append(window)

        logger_manager.info(f"创建了 {len(windows)} 个可变窗口，策略: {size_strategy}")
        return windows

    def _generate_fibonacci_sizes(self, max_size: int) -> List[int]:
        """生成斐波那契序列窗口大小"""
        sizes = [1, 1]
        while sizes[-1] < max_size:
            next_size = sizes[-1] + sizes[-2]
            if next_size > max_size:
                break
            sizes.append(next_size)
        return sizes[2:]  # 跳过前两个1

    def _generate_exponential_sizes(self, max_size: int) -> List[int]:
        """生成指数增长窗口大小"""
        sizes = []
        size = 10
        while size <= max_size:
            sizes.append(size)
            size *= 2
        return sizes


class CachedWindowDataManager(WindowDataManager):
    """缓存窗口化数据管理器"""
    
    def __init__(self, default_window_size: int = 500, cache_dir: str = "artifacts/cache/windows"):
        """
        初始化缓存窗口化数据管理器
        
        Args:
            default_window_size: 默认窗口大小
            cache_dir: 缓存目录
        """
        super().__init__(default_window_size, True)
        self.cache_dir = cache_dir
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        logger_manager.info(f"缓存窗口化数据管理器初始化完成，缓存目录: {cache_dir}")
    
    def get_window(self, window_size: Optional[int] = None) -> pd.DataFrame:
        """
        获取指定窗口大小的数据
        
        Args:
            window_size: 窗口大小，如果为None则使用默认值
            
        Returns:
            窗口数据
        """
        if window_size is None:
            window_size = self.default_window_size
        
        # 检查内存缓存
        if window_size in self.window_cache:
            logger_manager.debug(f"从内存缓存获取窗口数据，窗口大小: {window_size}")
            return self.window_cache[window_size]
        
        # 检查文件缓存
        cache_file = os.path.join(self.cache_dir, f"window_{window_size}.csv")
        if os.path.exists(cache_file):
            try:
                # 从文件缓存加载
                data = pd.read_csv(cache_file)
                
                # 检查缓存是否过期
                if self._is_cache_valid(data):
                    logger_manager.debug(f"从文件缓存获取窗口数据，窗口大小: {window_size}")
                    
                    # 更新内存缓存
                    self.window_cache[window_size] = data
                    
                    return data
            except Exception as e:
                logger_manager.warning(f"加载文件缓存失败: {e}")
        
        # 从原始数据获取
        data = super().get_window(window_size)
        
        # 保存到文件缓存
        if not data.empty:
            try:
                data.to_csv(cache_file, index=False)
                logger_manager.debug(f"窗口数据已保存到文件缓存，窗口大小: {window_size}")
            except Exception as e:
                logger_manager.warning(f"保存文件缓存失败: {e}")
        
        return data
    
    def _is_cache_valid(self, cached_data: pd.DataFrame) -> bool:
        """
        检查缓存是否有效
        
        Args:
            cached_data: 缓存数据
            
        Returns:
            缓存是否有效
        """
        if self.df is None or cached_data.empty:
            return False
        
        # 检查最新期号是否一致
        try:
            latest_issue_cache = cached_data['issue'].iloc[0]
            latest_issue_data = self.df['issue'].iloc[0]
            
            return latest_issue_cache == latest_issue_data
        except (IndexError, KeyError):
            return False
    
    def clear_cache(self) -> None:
        """清除缓存"""
        # 清除内存缓存
        super().clear_cache()
        
        # 清除文件缓存
        try:
            for file in os.listdir(self.cache_dir):
                if file.startswith("window_") and file.endswith(".csv"):
                    os.remove(os.path.join(self.cache_dir, file))
            
            logger_manager.info("文件缓存已清除")
        except Exception as e:
            logger_manager.error(f"清除文件缓存失败: {e}")


if __name__ == "__main__":
    # 测试窗口化数据管理器
    print("📊 测试窗口化数据管理器...")
    
    # 创建窗口化数据管理器
    window_manager = WindowDataManager()
    
    # 获取窗口数据
    window_data = window_manager.get_window(100)
    print(f"窗口数据大小: {len(window_data)}")
    
    # 获取滑动窗口
    sliding_windows = window_manager.get_sliding_windows(50, 10, 3)
    print(f"滑动窗口数量: {len(sliding_windows)}")
    
    # 获取扩展窗口
    expanding_windows = window_manager.get_expanding_windows(50, 25, 3)
    print(f"扩展窗口数量: {len(expanding_windows)}")
    
    # 获取数据统计信息
    stats = window_manager.get_data_stats(100)
    print(f"数据统计信息: {stats}")
    
    # 测试缓存窗口化数据管理器
    print("\n📊 测试缓存窗口化数据管理器...")
    
    # 创建缓存窗口化数据管理器
    cached_manager = CachedWindowDataManager()
    
    # 获取窗口数据
    window_data = cached_manager.get_window(100)
    print(f"窗口数据大小: {len(window_data)}")
    
    # 再次获取窗口数据（应该从缓存获取）
    window_data = cached_manager.get_window(100)
    print(f"从缓存获取窗口数据大小: {len(window_data)}")
    
    # 清除缓存
    cached_manager.clear_cache()
    
    print("窗口化数据管理器测试完成")