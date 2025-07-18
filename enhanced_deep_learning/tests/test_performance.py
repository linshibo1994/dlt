#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
性能基准测试
测试各组件的性能
"""

import unittest
import os
import sys
import time
import numpy as np
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入测试模块
from enhanced_deep_learning.gpu_accelerator import GPUAccelerator, GPUMemoryManager
from enhanced_deep_learning.batch_processor import BatchProcessor
from enhanced_deep_learning.model_optimizer import ModelOptimizer, CachedModel


class TestPerformanceBenchmarks(unittest.TestCase):
    """性能基准测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建组件
        self.accelerator = GPUAccelerator()
        self.memory_manager = GPUMemoryManager(self.accelerator)
        self.batch_processor = BatchProcessor(memory_manager=self.memory_manager)
        self.optimizer = ModelOptimizer()
        
        # 创建测试数据
        self.small_data = np.random.rand(100, 10)
        self.medium_data = np.random.rand(1000, 10)
        self.large_data = np.random.rand(10000, 10)
    
    def test_batch_processor_performance(self):
        """测试批处理系统性能"""
        # 定义处理函数
        def process_function(batch):
            # 模拟处理时间
            time.sleep(0.001)
            return batch * 2
        
        # 测试不同大小的数据
        for name, data in [('small', self.small_data), 
                          ('medium', self.medium_data), 
                          ('large', self.large_data)]:
            # 测试批量处理
            start_time = time.time()
            result1 = self.batch_processor.batch_process(data, process_function)
            batch_time = time.time() - start_time
            
            # 测试并行处理
            start_time = time.time()
            result2 = self.batch_processor.parallel_batch_process(data, process_function)
            parallel_time = time.time() - start_time
            
            # 测试动态批处理
            start_time = time.time()
            result3 = self.batch_processor.dynamic_batch_processing(data, process_function)
            dynamic_time = time.time() - start_time
            
            # 输出性能结果
            print(f"\n{name.upper()} 数据性能测试 ({len(data)} 样本):")
            print(f"批量处理时间: {batch_time:.4f}秒")
            print(f"并行处理时间: {parallel_time:.4f}秒")
            print(f"动态批处理时间: {dynamic_time:.4f}秒")
            print(f"加速比 (批量 vs 并行): {batch_time / parallel_time:.2f}x")
            
            # 检查结果
            np.testing.assert_allclose(result1, data * 2)
            np.testing.assert_allclose(result2, data * 2)
            np.testing.assert_allclose(result3, data * 2)
    
    def test_cached_model_performance(self):
        """测试缓存模型性能"""
        # 创建模拟模型
        class MockModel:
            def __init__(self, delay=0.01):
                self.delay = delay
                self.call_count = 0
            
            def predict(self, inputs):
                self.call_count += 1
                time.sleep(self.delay)  # 模拟预测时间
                return inputs * 2
        
        # 创建模型
        mock_model = MockModel()
        
        # 创建缓存模型
        cached_model = CachedModel(mock_model, self.optimizer)
        
        # 准备测试数据
        num_samples = 100
        num_repeats = 5
        
        # 生成随机输入
        inputs = [np.random.rand(10, 5) for _ in range(num_samples)]
        
        # 测试无缓存性能
        start_time = time.time()
        for _ in range(num_repeats):
            for x in inputs:
                mock_model.predict(x)
        no_cache_time = time.time() - start_time
        
        # 重置模型
        mock_model.call_count = 0
        
        # 测试有缓存性能
        start_time = time.time()
        for _ in range(num_repeats):
            for x in inputs:
                cached_model.predict(x)
        cache_time = time.time() - start_time
        
        # 获取缓存统计信息
        stats = cached_model.get_cache_stats()
        
        # 输出性能结果
        print("\n缓存模型性能测试:")
        print(f"无缓存时间: {no_cache_time:.4f}秒")
        print(f"有缓存时间: {cache_time:.4f}秒")
        print(f"加速比: {no_cache_time / cache_time:.2f}x")
        print(f"缓存命中率: {stats['hit_rate']:.2%}")
        print(f"缓存命中次数: {stats['cache_hits']}")
        print(f"缓存未命中次数: {stats['cache_misses']}")
        
        # 检查结果
        self.assertEqual(mock_model.call_count, num_samples)  # 每个输入只应该调用一次
        self.assertEqual(stats['cache_hits'], num_samples * (num_repeats - 1))  # 后续重复应该命中缓存
    
    @patch('enhanced_deep_learning.model_optimizer.ModelOptimizer.quantize_tensorflow_model')
    def test_model_optimization_performance(self, mock_quantize):
        """测试模型优化性能"""
        # 设置模拟返回值
        mock_quantize.side_effect = lambda model, type: model
        
        # 创建模拟模型
        class MockModel:
            def __init__(self, size=1000):
                self.weights = np.random.rand(size, size)
            
            def predict(self, inputs):
                return np.dot(inputs, self.weights)
        
        # 创建不同大小的模型
        small_model = MockModel(100)
        medium_model = MockModel(500)
        large_model = MockModel(1000)
        
        # 测试不同大小的模型
        for name, model in [('small', small_model), 
                           ('medium', medium_model), 
                           ('large', large_model)]:
            # 测试不同量化类型
            for q_type in ['float16', 'int8', 'dynamic']:
                # 测试量化性能
                start_time = time.time()
                quantized_model = self.optimizer.quantize_model(model, 'tensorflow', q_type)
                quantize_time = time.time() - start_time
                
                # 输出性能结果
                print(f"\n{name.upper()} 模型量化性能测试 ({q_type}):")
                print(f"量化时间: {quantize_time:.4f}秒")
    
    @unittest.skipIf(not GPUAccelerator().is_gpu_available(), "需要GPU才能运行此测试")
    def test_gpu_acceleration_performance(self):
        """测试GPU加速性能"""
        # 检查GPU可用性
        if not self.accelerator.is_gpu_available():
            self.skipTest("没有可用的GPU")
        
        # 创建大型测试数据
        data = np.random.rand(5000, 100)
        
        # 定义CPU处理函数
        def cpu_process(batch):
            return np.dot(batch, batch.T)
        
        # 测试CPU性能
        start_time = time.time()
        cpu_result = cpu_process(data)
        cpu_time = time.time() - start_time
        
        # 尝试导入TensorFlow
        try:
            import tensorflow as tf
            
            # 定义GPU处理函数
            def gpu_process(batch):
                with tf.device('/GPU:0'):
                    tensor = tf.constant(batch, dtype=tf.float32)
                    result = tf.matmul(tensor, tf.transpose(tensor))
                    return result.numpy()
            
            # 测试GPU性能
            start_time = time.time()
            gpu_result = gpu_process(data)
            gpu_time = time.time() - start_time
            
            # 输出性能结果
            print("\nGPU加速性能测试:")
            print(f"CPU处理时间: {cpu_time:.4f}秒")
            print(f"GPU处理时间: {gpu_time:.4f}秒")
            print(f"加速比: {cpu_time / gpu_time:.2f}x")
            
            # 检查结果
            np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-5)
        
        except ImportError:
            print("\nGPU加速性能测试:")
            print("无法导入TensorFlow，跳过GPU测试")
            print(f"CPU处理时间: {cpu_time:.4f}秒")


if __name__ == "__main__":
    unittest.main()