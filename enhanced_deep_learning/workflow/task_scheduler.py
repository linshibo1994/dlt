#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
任务调度器模块
Task Scheduler Module

提供任务调度、优先级管理、资源分配等功能。
"""

import os
import time
import threading
import heapq
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
from concurrent.futures import ThreadPoolExecutor, Future

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskPriority(Enum):
    """任务优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class ScheduledTask:
    """调度任务"""
    task_id: str
    name: str
    handler: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    scheduled_time: Optional[datetime] = None
    deadline: Optional[datetime] = None
    max_execution_time: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_time: datetime = field(default_factory=datetime.now)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Any = None
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """用于优先级队列排序"""
        if self.priority != other.priority:
            return self.priority.value > other.priority.value
        
        if self.scheduled_time and other.scheduled_time:
            return self.scheduled_time < other.scheduled_time
        
        return self.created_time < other.created_time


class ResourceManager:
    """资源管理器"""
    
    def __init__(self, max_cpu_tasks: int = 4, max_memory_mb: int = 1024):
        """
        初始化资源管理器
        
        Args:
            max_cpu_tasks: 最大CPU密集型任务数
            max_memory_mb: 最大内存使用量(MB)
        """
        self.max_cpu_tasks = max_cpu_tasks
        self.max_memory_mb = max_memory_mb
        self.current_cpu_tasks = 0
        self.current_memory_mb = 0
        self.resource_lock = threading.RLock()
        
        logger_manager.debug(f"资源管理器初始化完成，CPU任务限制: {max_cpu_tasks}, 内存限制: {max_memory_mb}MB")
    
    def acquire_resources(self, cpu_intensive: bool = False, memory_mb: int = 0) -> bool:
        """
        获取资源
        
        Args:
            cpu_intensive: 是否CPU密集型任务
            memory_mb: 需要的内存量(MB)
            
        Returns:
            是否成功获取资源
        """
        with self.resource_lock:
            # 检查CPU资源
            if cpu_intensive and self.current_cpu_tasks >= self.max_cpu_tasks:
                return False
            
            # 检查内存资源
            if self.current_memory_mb + memory_mb > self.max_memory_mb:
                return False
            
            # 分配资源
            if cpu_intensive:
                self.current_cpu_tasks += 1
            
            self.current_memory_mb += memory_mb
            
            return True
    
    def release_resources(self, cpu_intensive: bool = False, memory_mb: int = 0):
        """
        释放资源
        
        Args:
            cpu_intensive: 是否CPU密集型任务
            memory_mb: 释放的内存量(MB)
        """
        with self.resource_lock:
            if cpu_intensive and self.current_cpu_tasks > 0:
                self.current_cpu_tasks -= 1
            
            if memory_mb > 0:
                self.current_memory_mb = max(0, self.current_memory_mb - memory_mb)
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """获取资源使用情况"""
        with self.resource_lock:
            return {
                'cpu_tasks': {
                    'current': self.current_cpu_tasks,
                    'max': self.max_cpu_tasks,
                    'usage_rate': self.current_cpu_tasks / self.max_cpu_tasks if self.max_cpu_tasks > 0 else 0
                },
                'memory': {
                    'current_mb': self.current_memory_mb,
                    'max_mb': self.max_memory_mb,
                    'usage_rate': self.current_memory_mb / self.max_memory_mb if self.max_memory_mb > 0 else 0
                }
            }


class TaskScheduler:
    """任务调度器"""
    
    def __init__(self, max_workers: int = 4, max_memory_mb: int = 1024):
        """
        初始化任务调度器
        
        Args:
            max_workers: 最大工作线程数
            max_memory_mb: 最大内存使用量
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.resource_manager = ResourceManager(max_workers, max_memory_mb)
        
        # 任务队列和状态管理
        self.pending_tasks = []  # 优先级队列
        self.scheduled_tasks = {}  # task_id -> ScheduledTask
        self.running_tasks = {}  # task_id -> Future
        self.completed_tasks = []
        self.failed_tasks = []
        
        # 线程同步
        self.scheduler_lock = threading.RLock()
        self.running = False
        self.scheduler_thread = None
        
        logger_manager.info(f"任务调度器初始化完成，最大工作线程: {max_workers}")
    
    def schedule_task(self, task: ScheduledTask) -> bool:
        """
        调度任务
        
        Args:
            task: 调度任务
            
        Returns:
            是否调度成功
        """
        try:
            with self.scheduler_lock:
                # 检查任务ID是否已存在
                if task.task_id in self.scheduled_tasks:
                    logger_manager.warning(f"任务ID已存在: {task.task_id}")
                    return False
                
                # 添加到调度队列
                task.status = TaskStatus.SCHEDULED
                self.scheduled_tasks[task.task_id] = task
                heapq.heappush(self.pending_tasks, task)
                
                logger_manager.info(f"任务调度成功: {task.name}")
                return True
                
        except Exception as e:
            logger_manager.error(f"调度任务失败: {e}")
            return False
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否取消成功
        """
        try:
            with self.scheduler_lock:
                # 检查是否在运行中
                if task_id in self.running_tasks:
                    future = self.running_tasks[task_id]
                    if future.cancel():
                        task = self.scheduled_tasks[task_id]
                        task.status = TaskStatus.CANCELLED
                        task.end_time = datetime.now()
                        
                        del self.running_tasks[task_id]
                        self.failed_tasks.append(task)
                        
                        logger_manager.info(f"运行中任务取消成功: {task_id}")
                        return True
                
                # 检查是否在待执行队列中
                if task_id in self.scheduled_tasks:
                    task = self.scheduled_tasks[task_id]
                    if task.status in [TaskStatus.PENDING, TaskStatus.SCHEDULED]:
                        task.status = TaskStatus.CANCELLED
                        task.end_time = datetime.now()
                        
                        # 从待执行队列中移除
                        self.pending_tasks = [t for t in self.pending_tasks if t.task_id != task_id]
                        heapq.heapify(self.pending_tasks)
                        
                        self.failed_tasks.append(task)
                        
                        logger_manager.info(f"待执行任务取消成功: {task_id}")
                        return True
                
                logger_manager.warning(f"任务不存在或无法取消: {task_id}")
                return False
                
        except Exception as e:
            logger_manager.error(f"取消任务失败: {e}")
            return False
    
    def start_scheduler(self):
        """启动调度器"""
        if self.running:
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger_manager.info("任务调度器已启动")
    
    def stop_scheduler(self):
        """停止调度器"""
        self.running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        logger_manager.info("任务调度器已停止")
    
    def _scheduler_loop(self):
        """调度器主循环"""
        while self.running:
            try:
                self._process_pending_tasks()
                self._check_running_tasks()
                self._cleanup_completed_tasks()
                
                time.sleep(0.1)  # 短暂休眠
                
            except Exception as e:
                logger_manager.error(f"调度器循环失败: {e}")
                time.sleep(1)
    
    def _process_pending_tasks(self):
        """处理待执行任务"""
        try:
            with self.scheduler_lock:
                current_time = datetime.now()
                
                while self.pending_tasks:
                    task = self.pending_tasks[0]
                    
                    # 检查是否到达执行时间
                    if task.scheduled_time and task.scheduled_time > current_time:
                        break
                    
                    # 检查依赖是否满足
                    if not self._check_dependencies(task):
                        # 重新排队等待依赖满足
                        heapq.heappop(self.pending_tasks)
                        task.scheduled_time = current_time + timedelta(seconds=10)
                        heapq.heappush(self.pending_tasks, task)
                        continue
                    
                    # 检查资源是否可用
                    cpu_intensive = task.metadata.get('cpu_intensive', False)
                    memory_mb = task.metadata.get('memory_mb', 0)
                    
                    if not self.resource_manager.acquire_resources(cpu_intensive, memory_mb):
                        break  # 资源不足，等待下次循环
                    
                    # 从队列中移除并执行
                    heapq.heappop(self.pending_tasks)
                    self._execute_task(task)
                    
        except Exception as e:
            logger_manager.error(f"处理待执行任务失败: {e}")
    
    def _check_dependencies(self, task: ScheduledTask) -> bool:
        """检查任务依赖是否满足"""
        try:
            for dep_id in task.dependencies:
                if dep_id in self.scheduled_tasks:
                    dep_task = self.scheduled_tasks[dep_id]
                    if dep_task.status != TaskStatus.COMPLETED:
                        return False
                else:
                    # 依赖任务不存在
                    return False
            
            return True
            
        except Exception as e:
            logger_manager.error(f"检查任务依赖失败: {e}")
            return False
    
    def _execute_task(self, task: ScheduledTask):
        """执行任务"""
        try:
            task.status = TaskStatus.RUNNING
            task.start_time = datetime.now()
            
            # 提交到线程池执行
            future = self.executor.submit(self._task_wrapper, task)
            self.running_tasks[task.task_id] = future
            
            logger_manager.info(f"任务开始执行: {task.name}")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.end_time = datetime.now()
            
            # 释放资源
            cpu_intensive = task.metadata.get('cpu_intensive', False)
            memory_mb = task.metadata.get('memory_mb', 0)
            self.resource_manager.release_resources(cpu_intensive, memory_mb)
            
            self.failed_tasks.append(task)
            
            logger_manager.error(f"任务执行失败: {task.name}, 错误: {e}")
    
    def _task_wrapper(self, task: ScheduledTask) -> Any:
        """任务包装器"""
        try:
            # 执行任务处理器
            result = task.handler(**task.parameters)
            
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.end_time = datetime.now()
            
            logger_manager.info(f"任务执行完成: {task.name}")
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.end_time = datetime.now()
            
            logger_manager.error(f"任务执行异常: {task.name}, 错误: {e}")
            
            # 重试逻辑
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.SCHEDULED
                task.scheduled_time = datetime.now() + timedelta(seconds=30)  # 30秒后重试
                
                with self.scheduler_lock:
                    heapq.heappush(self.pending_tasks, task)
                
                logger_manager.info(f"任务重试: {task.name}, 第 {task.retry_count} 次")
            
            raise
        
        finally:
            # 释放资源
            cpu_intensive = task.metadata.get('cpu_intensive', False)
            memory_mb = task.metadata.get('memory_mb', 0)
            self.resource_manager.release_resources(cpu_intensive, memory_mb)
    
    def _check_running_tasks(self):
        """检查运行中的任务"""
        try:
            with self.scheduler_lock:
                completed_task_ids = []
                
                for task_id, future in self.running_tasks.items():
                    if future.done():
                        completed_task_ids.append(task_id)
                        
                        task = self.scheduled_tasks[task_id]
                        
                        try:
                            result = future.result()
                            if task.status != TaskStatus.COMPLETED:
                                task.status = TaskStatus.COMPLETED
                                task.result = result
                                task.end_time = datetime.now()
                            
                            self.completed_tasks.append(task)
                            
                        except Exception as e:
                            if task.status != TaskStatus.FAILED:
                                task.status = TaskStatus.FAILED
                                task.error_message = str(e)
                                task.end_time = datetime.now()
                            
                            self.failed_tasks.append(task)
                
                # 清理已完成的任务
                for task_id in completed_task_ids:
                    del self.running_tasks[task_id]
                    
        except Exception as e:
            logger_manager.error(f"检查运行中任务失败: {e}")
    
    def _cleanup_completed_tasks(self):
        """清理已完成的任务"""
        try:
            # 保留最近1000个已完成任务
            if len(self.completed_tasks) > 1000:
                self.completed_tasks = self.completed_tasks[-1000:]
            
            # 保留最近1000个失败任务
            if len(self.failed_tasks) > 1000:
                self.failed_tasks = self.failed_tasks[-1000:]
                
        except Exception as e:
            logger_manager.error(f"清理已完成任务失败: {e}")
    
    def get_task_status(self, task_id: str) -> Optional[ScheduledTask]:
        """获取任务状态"""
        with self.scheduler_lock:
            return self.scheduled_tasks.get(task_id)
    
    def list_tasks(self, status: TaskStatus = None) -> List[ScheduledTask]:
        """列出任务"""
        try:
            with self.scheduler_lock:
                all_tasks = []
                
                # 待执行任务
                all_tasks.extend(self.pending_tasks)
                
                # 运行中任务
                for task_id in self.running_tasks:
                    if task_id in self.scheduled_tasks:
                        all_tasks.append(self.scheduled_tasks[task_id])
                
                # 已完成任务
                all_tasks.extend(self.completed_tasks)
                
                # 失败任务
                all_tasks.extend(self.failed_tasks)
                
                # 状态过滤
                if status:
                    all_tasks = [task for task in all_tasks if task.status == status]
                
                return all_tasks
                
        except Exception as e:
            logger_manager.error(f"列出任务失败: {e}")
            return []
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        try:
            with self.scheduler_lock:
                stats = {
                    'pending_tasks': len(self.pending_tasks),
                    'running_tasks': len(self.running_tasks),
                    'completed_tasks': len(self.completed_tasks),
                    'failed_tasks': len(self.failed_tasks),
                    'total_tasks': len(self.scheduled_tasks),
                    'resource_usage': self.resource_manager.get_resource_usage(),
                    'scheduler_running': self.running
                }
                
                return stats
                
        except Exception as e:
            logger_manager.error(f"获取调度器统计信息失败: {e}")
            return {}


# 全局任务调度器实例
task_scheduler = TaskScheduler()


if __name__ == "__main__":
    # 测试任务调度器功能
    print("📅 测试任务调度器功能...")
    
    try:
        scheduler = TaskScheduler(max_workers=2)
        scheduler.start_scheduler()
        
        # 创建测试任务
        def test_task(name: str, duration: int = 1):
            print(f"执行任务: {name}")
            time.sleep(duration)
            return f"任务 {name} 完成"
        
        # 调度任务
        tasks = []
        for i in range(5):
            task = ScheduledTask(
                task_id=f"task_{i}",
                name=f"测试任务 {i}",
                handler=test_task,
                parameters={"name": f"Task-{i}", "duration": 1},
                priority=TaskPriority.NORMAL if i % 2 == 0 else TaskPriority.HIGH
            )
            
            if scheduler.schedule_task(task):
                tasks.append(task)
                print(f"✅ 任务调度成功: {task.name}")
        
        # 等待任务完成
        time.sleep(8)
        
        # 获取统计信息
        stats = scheduler.get_scheduler_stats()
        print(f"✅ 调度器统计: 完成 {stats['completed_tasks']}, 失败 {stats['failed_tasks']}")
        
        # 列出任务
        completed_tasks = scheduler.list_tasks(TaskStatus.COMPLETED)
        print(f"✅ 已完成任务: {len(completed_tasks)} 个")
        
        # 停止调度器
        scheduler.stop_scheduler()
        print("✅ 调度器已停止")
        
        print("✅ 任务调度器功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("任务调度器功能测试完成")
