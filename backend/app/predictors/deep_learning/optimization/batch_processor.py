#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批处理系统
提供批量预测和动态批大小调整功能
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

from core_modules import logger_manager, task_manager, with_progress
from .gpu_accelerator import GPUAccelerator, GPUMemoryManager


class BatchProcessor:
    """批处理系统"""
    
    def __init__(self, initial_batch_size: int = 32, 
               memory_manager: Optional[GPUMemoryManager] = None,
               max_workers: Optional[int] = None):
        """
        初始化批处理系统
        
        Args:
            initial_batch_size: 初始批处理大小
            memory_manager: GPU内存管理器，如果为None则创建新的
            max_workers: 最大工作线程数，如果为None则使用CPU核心数
        """
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.memory_manager = memory_manager or GPUMemoryManager()
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.adaptive_batch_size = True
        
        logger_manager.info(f"批处理系统初始化完成，初始批大小: {initial_batch_size}，最大工作线程: {self.max_workers}")
    
    def set_batch_size(self, batch_size: int) -> None:
        """
        设置批处理大小
        
        Args:
            batch_size: 批处理大小
        """
        self.current_batch_size = max(1, batch_size)
        logger_manager.info(f"批处理大小已设置为 {self.current_batch_size}")
    
    def enable_adaptive_batch_size(self, enabled: bool = True) -> None:
        """
        启用自适应批处理大小
        
        Args:
            enabled: 是否启用
        """
        self.adaptive_batch_size = enabled
        logger_manager.info(f"自适应批处理大小已{'启用' if enabled else '禁用'}")
    
    def optimize_batch_size(self, model_size_mb: int) -> int:
        """
        优化批处理大小
        
        Args:
            model_size_mb: 模型大小（MB）
            
        Returns:
            优化后的批处理大小
        """
        if not self.adaptive_batch_size:
            return self.current_batch_size
        
        # 使用内存管理器优化批处理大小
        optimized_batch_size = self.memory_manager.optimize_batch_size(
            self.current_batch_size, model_size_mb)
        
        # 更新当前批处理大小
        self.current_batch_size = optimized_batch_size
        
        return optimized_batch_size
    
    @with_progress(100, "批量处理")
    def batch_process(self, progress_bar, data: np.ndarray, 
                    process_func: Callable[[np.ndarray], np.ndarray],
                    model_size_mb: int = 500) -> np.ndarray:
        """
        批量处理数据
        
        Args:
            progress_bar: 进度条
            data: 输入数据
            process_func: 处理函数，接受一个批次数据，返回处理结果
            model_size_mb: 模型大小（MB），用于优化批处理大小
            
        Returns:
            处理结果
        """
        if data is None or len(data) == 0:
            logger_manager.warning("没有数据可处理")
            return np.array([])
        
        # 优化批处理大小
        if self.adaptive_batch_size:
            self.optimize_batch_size(model_size_mb)
        
        # 计算批次数
        n_samples = len(data)
        batch_size = min(self.current_batch_size, n_samples)
        n_batches = (n_samples + batch_size - 1) // batch_size  # 向上取整
        
        # 更新进度条总数
        progress_bar.total = n_batches
        
        # 批量处理
        results = []
        
        for i in range(0, n_samples, batch_size):
            # 获取当前批次
            batch_end = min(i + batch_size, n_samples)
            batch_data = data[i:batch_end]
            
            # 处理当前批次
            try:
                batch_result = process_func(batch_data)
                results.append(batch_result)
            except Exception as e:
                logger_manager.error(f"批处理失败: {e}")
                
                # 如果是第一个批次就失败，尝试减小批处理大小
                if i == 0 and self.adaptive_batch_size:
                    reduced_batch_size = self.current_batch_size // 2
                    if reduced_batch_size >= 1:
                        logger_manager.info(f"减小批处理大小并重试: {self.current_batch_size} -> {reduced_batch_size}")
                        self.current_batch_size = reduced_batch_size
                        
                        # 递归调用自身重试
                        return self.batch_process(progress_bar, data, process_func, model_size_mb)
            
            # 清理GPU内存（如果需要）
            self.memory_manager.clear_if_needed()
            
            # 更新进度
            progress_bar.update(1)
        
        # 合并结果
        if results:
            try:
                combined_results = np.vstack(results)
            except:
                # 如果无法垂直堆叠，尝试水平连接
                try:
                    combined_results = np.concatenate(results)
                except:
                    # 如果仍然失败，返回列表
                    combined_results = results
        else:
            combined_results = np.array([])
        
        logger_manager.info(f"批处理完成，处理 {n_samples} 个样本，批大小: {batch_size}，批次数: {n_batches}")
        
        return combined_results
    
    def parallel_batch_process(self, data: np.ndarray, 
                             process_func: Callable[[np.ndarray], np.ndarray],
                             use_processes: bool = False) -> np.ndarray:
        """
        并行批量处理数据
        
        Args:
            data: 输入数据
            process_func: 处理函数，接受一个批次数据，返回处理结果
            use_processes: 是否使用进程池（True）或线程池（False）
            
        Returns:
            处理结果
        """
        if data is None or len(data) == 0:
            logger_manager.warning("没有数据可处理")
            return np.array([])
        
        # 计算每个工作线程的数据量
        n_samples = len(data)
        n_workers = min(self.max_workers, n_samples)
        chunk_size = (n_samples + n_workers - 1) // n_workers  # 向上取整
        
        # 准备数据块
        data_chunks = [data[i:min(i + chunk_size, n_samples)] 
                      for i in range(0, n_samples, chunk_size)]
        
        # 选择执行器
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        # 并行处理
        results = []
        start_time = time.time()
        
        with executor_class(max_workers=n_workers) as executor:
            # 提交任务
            futures = [executor.submit(process_func, chunk) for chunk in data_chunks]
            
            # 收集结果
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger_manager.error(f"并行处理失败: {e}")
        
        # 计算处理时间
        elapsed_time = time.time() - start_time
        
        # 合并结果
        if results:
            try:
                combined_results = np.vstack(results)
            except:
                # 如果无法垂直堆叠，尝试水平连接
                try:
                    combined_results = np.concatenate(results)
                except:
                    # 如果仍然失败，返回列表
                    combined_results = results
        else:
            combined_results = np.array([])
        
        logger_manager.info(f"并行处理完成，处理 {n_samples} 个样本，工作线程: {n_workers}，用时: {elapsed_time:.2f}秒")
        
        return combined_results
    
    def dynamic_batch_processing(self, data: np.ndarray, 
                               process_func: Callable[[np.ndarray], np.ndarray],
                               initial_batch_size: Optional[int] = None,
                               target_time_per_batch: float = 1.0) -> np.ndarray:
        """
        动态批处理
        
        Args:
            data: 输入数据
            process_func: 处理函数，接受一个批次数据，返回处理结果
            initial_batch_size: 初始批处理大小，如果为None则使用当前批处理大小
            target_time_per_batch: 每批次目标处理时间（秒）
            
        Returns:
            处理结果
        """
        if data is None or len(data) == 0:
            logger_manager.warning("没有数据可处理")
            return np.array([])
        
        # 设置初始批处理大小
        batch_size = initial_batch_size or self.current_batch_size
        
        # 计算批次数
        n_samples = len(data)
        n_batches = (n_samples + batch_size - 1) // batch_size  # 向上取整
        
        # 批量处理
        results = []
        processed_samples = 0
        
        with task_manager.progress_bar(total=n_samples, desc="动态批处理") as progress_bar:
            while processed_samples < n_samples:
                # 获取当前批次
                batch_end = min(processed_samples + batch_size, n_samples)
                batch_data = data[processed_samples:batch_end]
                batch_size_actual = len(batch_data)
                
                # 处理当前批次并计时
                start_time = time.time()
                try:
                    batch_result = process_func(batch_data)
                    results.append(batch_result)
                except Exception as e:
                    logger_manager.error(f"批处理失败: {e}")
                    
                    # 减小批处理大小并重试
                    if batch_size > 1:
                        batch_size = max(1, batch_size // 2)
                        logger_manager.info(f"减小批处理大小并重试: {batch_size}")
                        continue
                
                # 计算处理时间
                elapsed_time = time.time() - start_time
                
                # 动态调整批处理大小
                if elapsed_time > 0:
                    # 计算每个样本的处理时间
                    time_per_sample = elapsed_time / batch_size_actual
                    
                    # 计算新的批处理大小，使每批次处理时间接近目标时间
                    new_batch_size = int(target_time_per_batch / time_per_sample)
                    
                    # 限制批处理大小变化幅度
                    new_batch_size = max(batch_size // 2, min(batch_size * 2, new_batch_size))
                    
                    # 确保批处理大小至少为1
                    new_batch_size = max(1, new_batch_size)
                    
                    if new_batch_size != batch_size:
                        logger_manager.debug(f"动态调整批处理大小: {batch_size} -> {new_batch_size}")
                        batch_size = new_batch_size
                
                # 清理GPU内存（如果需要）
                self.memory_manager.clear_if_needed()
                
                # 更新进度
                processed_samples = batch_end
                progress_bar.update(batch_size_actual)
        
        # 合并结果
        if results:
            try:
                combined_results = np.vstack(results)
            except:
                # 如果无法垂直堆叠，尝试水平连接
                try:
                    combined_results = np.concatenate(results)
                except:
                    # 如果仍然失败，返回列表
                    combined_results = results
        else:
            combined_results = np.array([])
        
        logger_manager.info(f"动态批处理完成，处理 {n_samples} 个样本，最终批大小: {batch_size}")
        
        return combined_results


if __name__ == "__main__":
    # 测试批处理系统
    print("🚀 测试批处理系统...")
    
    # 创建批处理系统
    processor = BatchProcessor()
    
    # 创建测试数据
    test_data = np.random.rand(1000, 10)
    
    # 定义处理函数
    def process_function(batch):
        # 模拟处理时间
        time.sleep(0.01)
        return batch * 2
    
    # 测试批量处理
    print("\n测试批量处理...")
    result1 = processor.batch_process(test_data, process_function)
    print(f"批量处理结果形状: {result1.shape}")
    
    # 测试并行处理
    print("\n测试并行处理...")
    result2 = processor.parallel_batch_process(test_data, process_function)
    print(f"并行处理结果形状: {result2.shape}")
    
    # 测试动态批处理
    print("\n测试动态批处理...")
    result3 = processor.dynamic_batch_processing(test_data, process_function)
    print(f"动态批处理结果形状: {result3.shape}")
    
    print("批处理系统测试完成")