#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增量数据更新器测试
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import shutil

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from enhanced_deep_learning.incremental_data_updater import IncrementalDataUpdater, AutoIncrementalUpdater


class TestIncrementalDataUpdater(unittest.TestCase):
    """增量数据更新器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = "temp_incremental"
        self.temp_data_dir = os.path.join(self.temp_dir, "data")
        self.temp_cache_dir = os.path.join(self.temp_dir, "cache")
        
        os.makedirs(self.temp_data_dir, exist_ok=True)
        os.makedirs(self.temp_cache_dir, exist_ok=True)
        
        # 创建增量数据更新器
        self.updater = IncrementalDataUpdater(data_dir=self.temp_data_dir, cache_dir=self.temp_cache_dir)
        
        # 检查数据是否加载成功
        if self.updater.df is None or len(self.updater.df) == 0:
            self.skipTest("数据未加载，跳过测试")
    
    def tearDown(self):
        """测试后清理"""
        # 清理临时目录
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_calculate_fingerprints(self):
        """测试计算数据指纹"""
        # 创建测试数据
        test_data = pd.DataFrame({
            'issue': ['24001', '24002'],
            'date': ['2024-01-01', '2024-01-03'],
            'front_balls': ['1,2,3,4,5', '6,7,8,9,10'],
            'back_balls': ['1,2', '3,4']
        })
        
        # 计算指纹
        fingerprints = self.updater._calculate_fingerprints(test_data)
        
        # 检查指纹
        self.assertEqual(len(fingerprints), 2)
        self.assertIn('24001', fingerprints)
        self.assertIn('24002', fingerprints)
        self.assertIsInstance(fingerprints['24001'], str)
        self.assertEqual(len(fingerprints['24001']), 32)  # MD5哈希长度
    
    def test_update_data(self):
        """测试更新数据"""
        # 创建模拟新数据
        mock_new_data = self.updater.df.head(5).copy()
        
        # 修改期号，模拟新数据
        for i, row in mock_new_data.iterrows():
            mock_new_data.at[i, 'issue'] = f"test_{row['issue']}"
        
        # 更新数据
        original_count = len(self.updater.df)
        updated_df, update_info = self.updater.update_data(mock_new_data)
        
        # 检查更新结果
        self.assertEqual(update_info['status'], 'success')
        self.assertEqual(update_info['added_count'], 5)
        self.assertEqual(len(updated_df), original_count + 5)
        
        # 检查是否有新期号
        for issue in [f"test_{row['issue']}" for _, row in mock_new_data.iterrows()]:
            self.assertTrue(any(updated_df['issue'] == issue))
    
    def test_data_consistency_check(self):
        """测试数据一致性检查"""
        # 创建格式不一致的数据
        invalid_data = pd.DataFrame({
            'issue': ['24001', '24002'],
            'date': ['2024-01-01', '2024-01-03'],
            'invalid_column': ['1,2,3,4,5', '6,7,8,9,10']
        })
        
        # 更新数据
        _, update_info = self.updater.update_data(invalid_data)
        
        # 检查更新结果
        self.assertEqual(update_info['status'], 'error')
        self.assertEqual(update_info['message'], '数据格式不一致')
    
    def test_backup_and_restore(self):
        """测试备份和恢复"""
        # 备份原始数据
        self.updater._backup_data()
        
        # 创建模拟新数据
        mock_new_data = self.updater.df.head(5).copy()
        
        # 修改期号，模拟新数据
        for i, row in mock_new_data.iterrows():
            mock_new_data.at[i, 'issue'] = f"backup_{row['issue']}"
        
        # 更新数据
        original_count = len(self.updater.df)
        updated_df, _ = self.updater.update_data(mock_new_data)
        
        # 检查更新后的数据量
        self.assertEqual(len(updated_df), original_count + 5)
        
        # 恢复备份
        restored_df = self.updater.restore_backup()
        
        # 检查恢复后的数据量
        self.assertEqual(len(restored_df), original_count)
        
        # 检查是否没有新期号
        for issue in [f"backup_{row['issue']}" for _, row in mock_new_data.iterrows()]:
            self.assertFalse(any(restored_df['issue'] == issue))
    
    def test_update_trigger(self):
        """测试更新触发器"""
        # 创建触发器文件
        trigger_file = os.path.join(self.temp_dir, "update_trigger.txt")
        result = self.updater.create_update_trigger(trigger_file)
        
        # 检查创建结果
        self.assertTrue(result)
        self.assertTrue(os.path.exists(trigger_file))
        
        # 检查触发器
        result = self.updater.check_update_trigger(trigger_file)
        
        # 检查检查结果
        self.assertTrue(result)
        
        # 检查触发器文件是否被删除
        self.assertFalse(os.path.exists(trigger_file))


class TestAutoIncrementalUpdater(unittest.TestCase):
    """自动增量更新器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = "temp_auto_incremental"
        self.temp_data_dir = os.path.join(self.temp_dir, "data")
        self.temp_cache_dir = os.path.join(self.temp_dir, "cache")
        
        os.makedirs(self.temp_data_dir, exist_ok=True)
        os.makedirs(self.temp_cache_dir, exist_ok=True)
        
        # 创建自动增量更新器
        self.updater = AutoIncrementalUpdater(
            data_dir=self.temp_data_dir,
            cache_dir=self.temp_cache_dir,
            update_interval=24
        )
        
        # 检查数据是否加载成功
        if self.updater.df is None or len(self.updater.df) == 0:
            self.skipTest("数据未加载，跳过测试")
    
    def tearDown(self):
        """测试后清理"""
        # 清理临时目录
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_should_update(self):
        """测试是否应该更新"""
        # 初始状态应该更新
        self.assertTrue(self.updater.should_update())
        
        # 设置上次更新时间为现在
        self.updater._save_last_update()
        
        # 现在不应该更新
        self.assertFalse(self.updater.should_update())
        
        # 设置上次更新时间为25小时前
        self.updater._save_last_update(datetime.now() - timedelta(hours=25))
        
        # 现在应该更新
        self.assertTrue(self.updater.should_update())
    
    def test_auto_update(self):
        """测试自动更新"""
        # 创建模拟数据源
        def mock_data_source():
            mock_data = self.updater.df.head(3).copy()
            for i, row in mock_data.iterrows():
                mock_data.at[i, 'issue'] = f"auto_{row['issue']}"
            return mock_data
        
        # 自动更新
        update_result = self.updater.auto_update(mock_data_source)
        
        # 检查更新结果
        self.assertEqual(update_result['status'], 'success')
        self.assertEqual(update_result['added_count'], 3)
        
        # 检查更新历史
        history = self.updater.get_update_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['status'], 'success')
        self.assertEqual(history[0]['added_count'], 3)
        
        # 再次更新（不应该更新）
        self.updater._save_last_update()  # 设置上次更新时间为现在
        update_result = self.updater.auto_update(mock_data_source)
        
        # 检查更新结果
        self.assertEqual(update_result['status'], 'skipped')
    
    def test_update_history(self):
        """测试更新历史"""
        # 添加更新历史
        self.updater._add_update_history({
            'status': 'success',
            'added_count': 5,
            'total_count': 100
        })
        
        # 检查更新历史
        history = self.updater.get_update_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['status'], 'success')
        self.assertEqual(history[0]['added_count'], 5)
        self.assertEqual(history[0]['total_count'], 100)
        self.assertIn('timestamp', history[0])


if __name__ == "__main__":
    unittest.main()