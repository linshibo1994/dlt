#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批处理系统测试
"""

import unittest
import os
import sys
import numpy as np
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from enhanced_deep_learning.batch_processor import BatchProcessor
from enhanced_deep_learning.gpu_accelerator import GPUMemoryManager


class TestBatchProcessor(unittest.TestCase):
    """批处理系统测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建批处理系统
        self.processor = BatchProcessor(initial_batch_size=32)
        
        # 创建测试数据
        self.test_data = np.random.rand(100, 10)
    
    def test_batch_size_setting(self):
        """测试批处理大小设置"""
        # 设置批处理大小
        self.processor.set_batch_size(64)
        
        # 检查批处理大小
        self.assertEqual(self.processor.current_batch_size, 64)
        
        # 测试设置无效值
        self.processor.set_batch_size(-10)
        self.assertEqual(self.processor.current_batch_size, 1)  # 应该被限制为最小值1
    
    def test_batch_process(self):
        """测试批量处理"""
        # 定义处理函数
        def process_function(batch):
            return batch * 2
        
        # 批量处理
        result = self.processor.batch_process(self.test_data, process_function)
        
        # 检查结果
        self.assertEqual(result.shape, self.test_data.shape)
        np.testing.assert_allclose(result, self.test_data * 2)
    
    def test_parallel_batch_process(self):
        """测试并行批量处理"""
        # 定义处理函数
        def process_function(batch):
            return batch * 2
        
        # 并行处理（线程）
        result1 = self.processor.parallel_batch_process(self.test_data, process_function, use_processes=False)
        
        # 检查结果
        self.assertEqual(result1.shape, self.test_data.shape)
        np.testing.assert_allclose(result1, self.test_data * 2)
        
        # 并行处理（进程）
        result2 = self.processor.parallel_batch_process(self.test_data, process_function, use_processes=True)
        
        # 检查结果
        self.assertEqual(result2.shape, self.test_data.shape)
        np.testing.assert_allclose(result2, self.test_data * 2)
    
    def test_dynamic_batch_processing(self):
        """测试动态批处理"""
        # 定义处理函数（处理时间与批大小成正比）
        def process_function(batch):
            time.sleep(0.001 * len(batch))  # 模拟处理时间
            return batch * 2
        
        # 动态批处理
        result = self.processor.dynamic_batch_processing(
            self.test_data, process_function, initial_batch_size=10, target_time_per_batch=0.05)
        
        # 检查结果
        self.assertEqual(result.shape, self.test_data.shape)
        np.testing.assert_allclose(result, self.test_data * 2)
    
    def test_optimize_batch_size(self):
        """测试优化批处理大小"""
        # 启用自适应批处理大小
        self.processor.enable_adaptive_batch_size(True)
        
        # 优化批处理大小
        optimized_size = self.processor.optimize_batch_size(500)
        
        # 检查优化结果
        self.assertGreaterEqual(optimized_size, 1)
        
        # 禁用自适应批处理大小
        self.processor.enable_adaptive_batch_size(False)
        
        # 设置批处理大小
        self.processor.set_batch_size(128)
        
        # 优化批处理大小（应该保持不变）
        optimized_size = self.processor.optimize_batch_size(500)
        self.assertEqual(optimized_size, 128)
    
    def test_empty_data(self):
        """测试空数据处理"""
        # 定义处理函数
        def process_function(batch):
            return batch * 2
        
        # 创建空数据
        empty_data = np.array([])
        
        # 批量处理
        result1 = self.processor.batch_process(empty_data, process_function)
        self.assertEqual(len(result1), 0)
        
        # 并行处理
        result2 = self.processor.parallel_batch_process(empty_data, process_function)
        self.assertEqual(len(result2), 0)
        
        # 动态批处理
        result3 = self.processor.dynamic_batch_processing(empty_data, process_function)
        self.assertEqual(len(result3), 0)
    
    def test_error_handling(self):
        """测试错误处理"""
        # 定义会失败的处理函数
        def failing_process_function(batch):
            raise ValueError("测试错误")
        
        # 批量处理（应该返回空数组）
        result = self.processor.batch_process(self.test_data, failing_process_function)
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()