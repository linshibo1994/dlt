#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分布式计算管理器模块
Distributed Computing Manager Module

提供分布式计算、节点管理、任务调度等功能。
"""

import os
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException


class ComputingStrategy(Enum):
    """计算策略枚举"""
    SINGLE_THREAD = "single_thread"
    MULTI_THREAD = "multi_thread"
    MULTI_PROCESS = "multi_process"
    DISTRIBUTED = "distributed"


@dataclass
class ComputingNode:
    """计算节点"""
    node_id: str
    host: str
    port: int
    cpu_count: int
    memory_gb: float
    gpu_count: int = 0
    status: str = "idle"
    last_heartbeat: float = 0.0


@dataclass
class ComputingTask:
    """计算任务"""
    task_id: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: int = 0
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3


class NodeManager:
    """节点管理器"""
    
    def __init__(self):
        """初始化节点管理器"""
        self.nodes: Dict[str, ComputingNode] = {}
        self._lock = threading.RLock()
        
        logger_manager.debug("节点管理器初始化完成")
    
    def register_node(self, node: ComputingNode) -> bool:
        """
        注册计算节点
        
        Args:
            node: 计算节点
            
        Returns:
            是否注册成功
        """
        try:
            with self._lock:
                self.nodes[node.node_id] = node
                node.last_heartbeat = time.time()
                
                logger_manager.info(f"计算节点注册成功: {node.node_id} ({node.host}:{node.port})")
                return True
                
        except Exception as e:
            logger_manager.error(f"注册计算节点失败: {e}")
            return False
    
    def unregister_node(self, node_id: str) -> bool:
        """
        注销计算节点
        
        Args:
            node_id: 节点ID
            
        Returns:
            是否注销成功
        """
        try:
            with self._lock:
                if node_id in self.nodes:
                    del self.nodes[node_id]
                    logger_manager.info(f"计算节点注销成功: {node_id}")
                    return True
                return False
                
        except Exception as e:
            logger_manager.error(f"注销计算节点失败: {e}")
            return False
    
    def update_heartbeat(self, node_id: str) -> bool:
        """
        更新节点心跳
        
        Args:
            node_id: 节点ID
            
        Returns:
            是否更新成功
        """
        try:
            with self._lock:
                if node_id in self.nodes:
                    self.nodes[node_id].last_heartbeat = time.time()
                    return True
                return False
                
        except Exception as e:
            logger_manager.error(f"更新节点心跳失败: {e}")
            return False
    
    def get_available_nodes(self) -> List[ComputingNode]:
        """获取可用节点列表"""
        try:
            with self._lock:
                current_time = time.time()
                available_nodes = []
                
                for node in self.nodes.values():
                    # 检查心跳超时（30秒）
                    if current_time - node.last_heartbeat < 30:
                        if node.status in ["idle", "running"]:
                            available_nodes.append(node)
                
                return available_nodes
                
        except Exception as e:
            logger_manager.error(f"获取可用节点失败: {e}")
            return []
    
    def get_node_statistics(self) -> Dict[str, Any]:
        """获取节点统计信息"""
        try:
            with self._lock:
                total_nodes = len(self.nodes)
                available_nodes = len(self.get_available_nodes())
                
                total_cpu = sum(node.cpu_count for node in self.nodes.values())
                total_memory = sum(node.memory_gb for node in self.nodes.values())
                total_gpu = sum(node.gpu_count for node in self.nodes.values())
                
                return {
                    "total_nodes": total_nodes,
                    "available_nodes": available_nodes,
                    "total_cpu_cores": total_cpu,
                    "total_memory_gb": total_memory,
                    "total_gpu_count": total_gpu
                }
                
        except Exception as e:
            logger_manager.error(f"获取节点统计信息失败: {e}")
            return {}


class DistributedComputingManager:
    """分布式计算管理器"""
    
    def __init__(self, max_workers: int = None):
        """
        初始化分布式计算管理器
        
        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.node_manager = NodeManager()
        self.task_queue: List[ComputingTask] = []
        self.running_tasks: Dict[str, ComputingTask] = {}
        self._lock = threading.RLock()
        
        # 执行器
        self.thread_executor = None
        self.process_executor = None
        
        # 注册本地节点
        self._register_local_node()
        
        logger_manager.info("分布式计算管理器初始化完成")
    
    def _register_local_node(self):
        """注册本地节点"""
        try:
            import psutil
            
            local_node = ComputingNode(
                node_id="local",
                host="localhost",
                port=0,
                cpu_count=mp.cpu_count(),
                memory_gb=psutil.virtual_memory().total / (1024**3),
                gpu_count=self._get_gpu_count()
            )
            
            self.node_manager.register_node(local_node)
            
        except Exception as e:
            logger_manager.error(f"注册本地节点失败: {e}")
    
    def _get_gpu_count(self) -> int:
        """获取GPU数量"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except ImportError:
            pass
        
        return 0
    
    def submit_task(self, task: ComputingTask) -> str:
        """
        提交计算任务
        
        Args:
            task: 计算任务
            
        Returns:
            任务ID
        """
        try:
            with self._lock:
                self.task_queue.append(task)
                # 按优先级排序
                self.task_queue.sort(key=lambda t: t.priority, reverse=True)
                
                logger_manager.debug(f"计算任务提交成功: {task.task_id}")
                return task.task_id
                
        except Exception as e:
            logger_manager.error(f"提交计算任务失败: {e}")
            raise DeepLearningException(f"提交计算任务失败: {e}")
    
    def execute_parallel(self, function: Callable, args_list: List[tuple], 
                         strategy: ComputingStrategy = ComputingStrategy.MULTI_THREAD) -> List[Any]:
        """
        并行执行函数
        
        Args:
            function: 要执行的函数
            args_list: 参数列表
            strategy: 计算策略
            
        Returns:
            执行结果列表
        """
        try:
            if strategy == ComputingStrategy.SINGLE_THREAD:
                return self._execute_single_thread(function, args_list)
            elif strategy == ComputingStrategy.MULTI_THREAD:
                return self._execute_multi_thread(function, args_list)
            elif strategy == ComputingStrategy.MULTI_PROCESS:
                return self._execute_multi_process(function, args_list)
            elif strategy == ComputingStrategy.DISTRIBUTED:
                return self._execute_distributed(function, args_list)
            else:
                raise DeepLearningException(f"不支持的计算策略: {strategy}")
                
        except Exception as e:
            logger_manager.error(f"并行执行失败: {e}")
            raise
    
    def _execute_single_thread(self, function: Callable, args_list: List[tuple]) -> List[Any]:
        """单线程执行"""
        results = []
        
        for args in args_list:
            try:
                result = function(*args)
                results.append(result)
            except Exception as e:
                logger_manager.error(f"单线程执行失败: {e}")
                results.append(None)
        
        return results
    
    def _execute_multi_thread(self, function: Callable, args_list: List[tuple]) -> List[Any]:
        """多线程执行"""
        results = [None] * len(args_list)
        
        if not self.thread_executor:
            self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        try:
            # 提交任务
            future_to_index = {}
            for i, args in enumerate(args_list):
                future = self.thread_executor.submit(function, *args)
                future_to_index[future] = i
            
            # 收集结果
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger_manager.error(f"多线程任务执行失败: {e}")
                    results[index] = None
            
            return results
            
        except Exception as e:
            logger_manager.error(f"多线程执行失败: {e}")
            return results
    
    def _execute_multi_process(self, function: Callable, args_list: List[tuple]) -> List[Any]:
        """多进程执行"""
        results = [None] * len(args_list)
        
        if not self.process_executor:
            self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers)
        
        try:
            # 提交任务
            future_to_index = {}
            for i, args in enumerate(args_list):
                future = self.process_executor.submit(function, *args)
                future_to_index[future] = i
            
            # 收集结果
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger_manager.error(f"多进程任务执行失败: {e}")
                    results[index] = None
            
            return results
            
        except Exception as e:
            logger_manager.error(f"多进程执行失败: {e}")
            return results
    
    def _execute_distributed(self, function: Callable, args_list: List[tuple]) -> List[Any]:
        """分布式执行"""
        # 简化的分布式执行，实际应该使用如Ray、Dask等框架
        logger_manager.warning("分布式执行暂未完全实现，回退到多进程执行")
        return self._execute_multi_process(function, args_list)
    
    def map_reduce(self, map_function: Callable, reduce_function: Callable, 
                   data_list: List[Any], strategy: ComputingStrategy = ComputingStrategy.MULTI_THREAD) -> Any:
        """
        MapReduce操作
        
        Args:
            map_function: Map函数
            reduce_function: Reduce函数
            data_list: 数据列表
            strategy: 计算策略
            
        Returns:
            最终结果
        """
        try:
            # Map阶段
            args_list = [(item,) for item in data_list]
            map_results = self.execute_parallel(map_function, args_list, strategy)
            
            # 过滤None结果
            valid_results = [result for result in map_results if result is not None]
            
            # Reduce阶段
            if not valid_results:
                return None
            
            result = valid_results[0]
            for item in valid_results[1:]:
                result = reduce_function(result, item)
            
            return result
            
        except Exception as e:
            logger_manager.error(f"MapReduce操作失败: {e}")
            raise
    
    def get_computing_statistics(self) -> Dict[str, Any]:
        """获取计算统计信息"""
        try:
            with self._lock:
                node_stats = self.node_manager.get_node_statistics()
                
                stats = {
                    "max_workers": self.max_workers,
                    "queued_tasks": len(self.task_queue),
                    "running_tasks": len(self.running_tasks),
                    "thread_executor_active": self.thread_executor is not None,
                    "process_executor_active": self.process_executor is not None
                }
                
                stats.update(node_stats)
                return stats
                
        except Exception as e:
            logger_manager.error(f"获取计算统计信息失败: {e}")
            return {}
    
    def shutdown(self):
        """关闭计算管理器"""
        try:
            if self.thread_executor:
                self.thread_executor.shutdown(wait=True)
                self.thread_executor = None
            
            if self.process_executor:
                self.process_executor.shutdown(wait=True)
                self.process_executor = None
            
            logger_manager.info("分布式计算管理器已关闭")
            
        except Exception as e:
            logger_manager.error(f"关闭计算管理器失败: {e}")


# 全局分布式计算管理器实例
distributed_computing_manager = DistributedComputingManager()


if __name__ == "__main__":
    # 测试分布式计算管理器功能
    print("🖥️ 测试分布式计算管理器功能...")
    
    try:
        manager = DistributedComputingManager()
        
        # 测试函数
        def square(x):
            return x * x
        
        def add(a, b):
            return a + b
        
        # 测试并行执行
        args_list = [(i,) for i in range(10)]
        
        # 多线程执行
        results_thread = manager.execute_parallel(square, args_list, ComputingStrategy.MULTI_THREAD)
        print(f"✅ 多线程执行: {len(results_thread)} 个结果")
        
        # 多进程执行
        results_process = manager.execute_parallel(square, args_list, ComputingStrategy.MULTI_PROCESS)
        print(f"✅ 多进程执行: {len(results_process)} 个结果")
        
        # MapReduce测试
        data_list = list(range(1, 11))
        sum_result = manager.map_reduce(square, add, data_list)
        print(f"✅ MapReduce结果: {sum_result}")
        
        # 获取统计信息
        stats = manager.get_computing_statistics()
        print(f"✅ 计算统计: {stats}")
        
        # 关闭管理器
        manager.shutdown()
        
        print("✅ 分布式计算管理器功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("分布式计算管理器功能测试完成")
