#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
窗口化数据管理器测试
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from enhanced_deep_learning.window_data_manager import WindowDataManager, CachedWindowDataManager


class TestWindowDataManager(unittest.TestCase):
    """窗口化数据管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建窗口化数据管理器
        self.manager = WindowDataManager(default_window_size=100)
        
        # 检查数据是否加载成功
        if self.manager.df is None or len(self.manager.df) == 0:
            self.skipTest("数据未加载，跳过测试")
    
    def test_get_window(self):
        """测试获取窗口数据"""
        # 获取默认窗口数据
        default_window = self.manager.get_window()
        
        # 检查窗口大小
        self.assertLessEqual(len(default_window), 100)
        
        # 获取自定义窗口数据
        custom_window = self.manager.get_window(50)
        
        # 检查窗口大小
        self.assertLessEqual(len(custom_window), 50)
        
        # 获取超大窗口数据
        large_window = self.manager.get_window(10000)
        
        # 检查窗口大小（应该返回全部数据）
        self.assertEqual(len(large_window), len(self.manager.df))
    
    def test_get_sliding_windows(self):
        """测试获取滑动窗口"""
        # 获取滑动窗口
        sliding_windows = self.manager.get_sliding_windows(50, 10, 3)
        
        # 检查窗口数量
        self.assertLessEqual(len(sliding_windows), 3)
        
        if sliding_windows:
            # 检查窗口大小
            self.assertLessEqual(len(sliding_windows[0]), 50)
            
            # 检查窗口间隔
            if len(sliding_windows) > 1:
                first_issue = sliding_windows[0]['issue'].iloc[0]
                second_issue = sliding_windows[1]['issue'].iloc[0]
                self.assertNotEqual(first_issue, second_issue)
    
    def test_get_expanding_windows(self):
        """测试获取扩展窗口"""
        # 获取扩展窗口
        expanding_windows = self.manager.get_expanding_windows(50, 25, 3)
        
        # 检查窗口数量
        self.assertLessEqual(len(expanding_windows), 3)
        
        if expanding_windows:
            # 检查窗口大小递增
            sizes = [len(window) for window in expanding_windows]
            for i in range(1, len(sizes)):
                self.assertGreater(sizes[i], sizes[i-1])
    
    def test_get_data_stats(self):
        """测试获取数据统计信息"""
        # 获取统计信息
        stats = self.manager.get_data_stats(100)
        
        # 检查统计信息
        self.assertIn('data_count', stats)
        self.assertIn('date_range', stats)
        self.assertIn('issue_range', stats)
        self.assertIn('front_frequency', stats)
        self.assertIn('back_frequency', stats)
        self.assertIn('front_hot', stats)
        self.assertIn('back_hot', stats)
        
        # 检查数据量
        self.assertLessEqual(stats['data_count'], 100)
        
        # 检查热门号码
        self.assertEqual(len(stats['front_hot']), 5)
        self.assertEqual(len(stats['back_hot']), 2)
    
    def test_save_window_data(self):
        """测试保存窗口数据"""
        # 保存窗口数据
        temp_file = "temp_window_data.csv"
        result = self.manager.save_window_data(50, temp_file)
        
        # 检查保存结果
        self.assertTrue(result)
        self.assertTrue(os.path.exists(temp_file))
        
        # 加载保存的数据
        saved_data = pd.read_csv(temp_file)
        
        # 检查数据量
        self.assertLessEqual(len(saved_data), 50)
        
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)


class TestCachedWindowDataManager(unittest.TestCase):
    """缓存窗口化数据管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时缓存目录
        self.cache_dir = "temp_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 创建缓存窗口化数据管理器
        self.manager = CachedWindowDataManager(default_window_size=100, cache_dir=self.cache_dir)
        
        # 检查数据是否加载成功
        if self.manager.df is None or len(self.manager.df) == 0:
            self.skipTest("数据未加载，跳过测试")
    
    def tearDown(self):
        """测试后清理"""
        # 清理临时缓存目录
        if os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))
            os.rmdir(self.cache_dir)
    
    def test_cache_mechanism(self):
        """测试缓存机制"""
        # 获取窗口数据（应该从原始数据获取）
        window_data1 = self.manager.get_window(50)
        
        # 检查窗口大小
        self.assertLessEqual(len(window_data1), 50)
        
        # 检查缓存文件是否创建
        cache_file = os.path.join(self.cache_dir, "window_50.csv")
        self.assertTrue(os.path.exists(cache_file))
        
        # 再次获取窗口数据（应该从缓存获取）
        window_data2 = self.manager.get_window(50)
        
        # 检查两次获取的数据是否相同
        self.assertEqual(len(window_data1), len(window_data2))
        
        # 清除缓存
        self.manager.clear_cache()
        
        # 检查缓存文件是否被删除
        self.assertFalse(os.path.exists(cache_file))


if __name__ == "__main__":
    unittest.main()