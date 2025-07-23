#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
工作流管理器模块
Workflow Manager Module

提供工作流定义、执行、监控和管理功能。
"""

import os
import json
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException


class WorkflowStatus(Enum):
    """工作流状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskType(Enum):
    """任务类型枚举"""
    DATA_PROCESSING = "data_processing"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    PREDICTION = "prediction"
    VISUALIZATION = "visualization"
    NOTIFICATION = "notification"
    CUSTOM = "custom"


@dataclass
class WorkflowTask:
    """工作流任务"""
    task_id: str
    name: str
    task_type: TaskType
    handler: Callable
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Any = None
    error_message: str = ""


@dataclass
class WorkflowConfig:
    """工作流配置"""
    workflow_id: str
    name: str
    description: str = ""
    tasks: List[WorkflowTask] = field(default_factory=list)
    parallel_execution: bool = False
    max_parallel_tasks: int = 4
    timeout: Optional[int] = None
    auto_retry: bool = True
    notification_enabled: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """工作流执行记录"""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_tasks: int = 0
    progress: float = 0.0
    error_message: str = ""
    results: Dict[str, Any] = field(default_factory=dict)


class WorkflowExecutor:
    """工作流执行器"""
    
    def __init__(self, max_workers: int = 4):
        """
        初始化工作流执行器
        
        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers
        self.active_executions = {}
        self.execution_history = []
        self.lock = threading.RLock()
        
        logger_manager.info(f"工作流执行器初始化完成，最大工作线程: {max_workers}")
    
    def execute_workflow(self, config: WorkflowConfig) -> str:
        """
        执行工作流
        
        Args:
            config: 工作流配置
            
        Returns:
            执行ID
        """
        try:
            execution_id = str(uuid.uuid4())
            
            # 创建执行记录
            execution = WorkflowExecution(
                execution_id=execution_id,
                workflow_id=config.workflow_id,
                status=WorkflowStatus.RUNNING,
                start_time=datetime.now(),
                total_tasks=len(config.tasks)
            )
            
            with self.lock:
                self.active_executions[execution_id] = execution
            
            # 启动执行线程
            execution_thread = threading.Thread(
                target=self._execute_workflow_thread,
                args=(config, execution)
            )
            execution_thread.daemon = True
            execution_thread.start()
            
            logger_manager.info(f"工作流执行已启动: {execution_id}")
            return execution_id
            
        except Exception as e:
            logger_manager.error(f"启动工作流执行失败: {e}")
            raise DeepLearningException(f"启动工作流执行失败: {e}")
    
    def _execute_workflow_thread(self, config: WorkflowConfig, execution: WorkflowExecution):
        """工作流执行线程"""
        try:
            logger_manager.info(f"开始执行工作流: {config.name}")
            
            # 构建任务依赖图
            task_graph = self._build_task_graph(config.tasks)
            
            # 执行任务
            if config.parallel_execution:
                self._execute_parallel(task_graph, execution)
            else:
                self._execute_sequential(task_graph, execution)
            
            # 更新执行状态
            execution.status = WorkflowStatus.COMPLETED
            execution.end_time = datetime.now()
            execution.progress = 100.0
            
            logger_manager.info(f"工作流执行完成: {config.name}")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.end_time = datetime.now()
            
            logger_manager.error(f"工作流执行失败: {e}")
        
        finally:
            # 移动到历史记录
            with self.lock:
                if execution.execution_id in self.active_executions:
                    del self.active_executions[execution.execution_id]
                self.execution_history.append(execution)
    
    def _build_task_graph(self, tasks: List[WorkflowTask]) -> Dict[str, WorkflowTask]:
        """构建任务依赖图"""
        task_graph = {}
        
        for task in tasks:
            task_graph[task.task_id] = task
        
        return task_graph
    
    def _execute_sequential(self, task_graph: Dict[str, WorkflowTask], 
                          execution: WorkflowExecution):
        """顺序执行任务"""
        try:
            executed_tasks = set()
            
            while len(executed_tasks) < len(task_graph):
                # 找到可以执行的任务（依赖已满足）
                ready_tasks = []
                
                for task_id, task in task_graph.items():
                    if task_id not in executed_tasks:
                        dependencies_met = all(dep in executed_tasks for dep in task.dependencies)
                        if dependencies_met:
                            ready_tasks.append(task)
                
                if not ready_tasks:
                    raise DeepLearningException("检测到循环依赖或无法满足的依赖")
                
                # 执行第一个就绪任务
                task = ready_tasks[0]
                success = self._execute_task(task, execution)
                
                if success:
                    executed_tasks.add(task.task_id)
                    execution.completed_tasks += 1
                else:
                    execution.failed_tasks += 1
                    if not task.handler:  # 如果任务失败且不允许继续
                        raise DeepLearningException(f"任务执行失败: {task.name}")
                
                # 更新进度
                execution.progress = (len(executed_tasks) / len(task_graph)) * 100
                
        except Exception as e:
            logger_manager.error(f"顺序执行任务失败: {e}")
            raise
    
    def _execute_parallel(self, task_graph: Dict[str, WorkflowTask], 
                         execution: WorkflowExecution):
        """并行执行任务"""
        try:
            executed_tasks = set()
            running_tasks = {}
            
            while len(executed_tasks) < len(task_graph):
                # 启动新任务
                ready_tasks = []
                
                for task_id, task in task_graph.items():
                    if (task_id not in executed_tasks and 
                        task_id not in running_tasks and
                        len(running_tasks) < self.max_workers):
                        
                        dependencies_met = all(dep in executed_tasks for dep in task.dependencies)
                        if dependencies_met:
                            ready_tasks.append(task)
                
                # 启动就绪任务
                for task in ready_tasks[:self.max_workers - len(running_tasks)]:
                    task_thread = threading.Thread(
                        target=self._execute_task_thread,
                        args=(task, execution, running_tasks, executed_tasks)
                    )
                    task_thread.daemon = True
                    task_thread.start()
                    
                    running_tasks[task.task_id] = task_thread
                
                # 等待任务完成
                time.sleep(0.1)
                
                # 清理完成的任务
                completed_task_ids = []
                for task_id, thread in running_tasks.items():
                    if not thread.is_alive():
                        completed_task_ids.append(task_id)
                
                for task_id in completed_task_ids:
                    del running_tasks[task_id]
                
                # 更新进度
                execution.progress = (len(executed_tasks) / len(task_graph)) * 100
            
            # 等待所有任务完成
            for thread in running_tasks.values():
                thread.join()
                
        except Exception as e:
            logger_manager.error(f"并行执行任务失败: {e}")
            raise
    
    def _execute_task_thread(self, task: WorkflowTask, execution: WorkflowExecution,
                           running_tasks: Dict, executed_tasks: set):
        """任务执行线程"""
        try:
            success = self._execute_task(task, execution)
            
            if success:
                executed_tasks.add(task.task_id)
                execution.completed_tasks += 1
            else:
                execution.failed_tasks += 1
                
        except Exception as e:
            logger_manager.error(f"任务线程执行失败: {e}")
            execution.failed_tasks += 1
    
    def _execute_task(self, task: WorkflowTask, execution: WorkflowExecution) -> bool:
        """执行单个任务"""
        try:
            task.status = WorkflowStatus.RUNNING
            task.start_time = datetime.now()
            
            logger_manager.info(f"开始执行任务: {task.name}")
            
            # 执行任务处理器
            if task.handler:
                result = task.handler(**task.parameters)
                task.result = result
                execution.results[task.task_id] = result
            
            task.status = WorkflowStatus.COMPLETED
            task.end_time = datetime.now()
            
            logger_manager.info(f"任务执行完成: {task.name}")
            return True
            
        except Exception as e:
            task.status = WorkflowStatus.FAILED
            task.error_message = str(e)
            task.end_time = datetime.now()
            
            logger_manager.error(f"任务执行失败: {task.name}, 错误: {e}")
            
            # 重试逻辑
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                logger_manager.info(f"任务重试: {task.name}, 第 {task.retry_count} 次")
                return self._execute_task(task, execution)
            
            return False
    
    def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """获取执行状态"""
        with self.lock:
            # 检查活跃执行
            if execution_id in self.active_executions:
                return self.active_executions[execution_id]
            
            # 检查历史记录
            for execution in self.execution_history:
                if execution.execution_id == execution_id:
                    return execution
            
            return None
    
    def cancel_execution(self, execution_id: str) -> bool:
        """取消执行"""
        try:
            with self.lock:
                if execution_id in self.active_executions:
                    execution = self.active_executions[execution_id]
                    execution.status = WorkflowStatus.CANCELLED
                    execution.end_time = datetime.now()
                    
                    logger_manager.info(f"工作流执行已取消: {execution_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger_manager.error(f"取消工作流执行失败: {e}")
            return False


class WorkflowManager:
    """工作流管理器"""
    
    def __init__(self):
        """初始化工作流管理器"""
        self.workflows = {}
        self.executor = WorkflowExecutor()
        self.lock = threading.RLock()
        
        logger_manager.info("工作流管理器初始化完成")
    
    def register_workflow(self, config: WorkflowConfig) -> bool:
        """
        注册工作流
        
        Args:
            config: 工作流配置
            
        Returns:
            是否注册成功
        """
        try:
            with self.lock:
                self.workflows[config.workflow_id] = config
                
            logger_manager.info(f"工作流注册成功: {config.name}")
            return True
            
        except Exception as e:
            logger_manager.error(f"注册工作流失败: {e}")
            return False
    
    def execute_workflow(self, workflow_id: str) -> Optional[str]:
        """
        执行工作流
        
        Args:
            workflow_id: 工作流ID
            
        Returns:
            执行ID
        """
        try:
            with self.lock:
                if workflow_id not in self.workflows:
                    raise DeepLearningException(f"工作流不存在: {workflow_id}")
                
                config = self.workflows[workflow_id]
            
            execution_id = self.executor.execute_workflow(config)
            return execution_id
            
        except Exception as e:
            logger_manager.error(f"执行工作流失败: {e}")
            return None
    
    def get_workflow_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """获取工作流状态"""
        return self.executor.get_execution_status(execution_id)
    
    def cancel_workflow(self, execution_id: str) -> bool:
        """取消工作流"""
        return self.executor.cancel_execution(execution_id)
    
    def list_workflows(self) -> List[WorkflowConfig]:
        """列出所有工作流"""
        with self.lock:
            return list(self.workflows.values())
    
    def get_execution_history(self) -> List[WorkflowExecution]:
        """获取执行历史"""
        return self.executor.execution_history.copy()


# 全局工作流管理器实例
workflow_manager = WorkflowManager()


if __name__ == "__main__":
    # 测试工作流管理器功能
    print("🔄 测试工作流管理器功能...")
    
    try:
        manager = WorkflowManager()
        
        # 创建测试任务
        def task1():
            print("执行任务1")
            return "task1_result"
        
        def task2():
            print("执行任务2")
            return "task2_result"
        
        def task3():
            print("执行任务3")
            return "task3_result"
        
        # 创建工作流配置
        tasks = [
            WorkflowTask(
                task_id="task1",
                name="数据处理任务",
                task_type=TaskType.DATA_PROCESSING,
                handler=task1
            ),
            WorkflowTask(
                task_id="task2",
                name="模型训练任务",
                task_type=TaskType.MODEL_TRAINING,
                handler=task2,
                dependencies=["task1"]
            ),
            WorkflowTask(
                task_id="task3",
                name="模型评估任务",
                task_type=TaskType.MODEL_EVALUATION,
                handler=task3,
                dependencies=["task2"]
            )
        ]
        
        config = WorkflowConfig(
            workflow_id="test_workflow",
            name="测试工作流",
            description="用于测试的工作流",
            tasks=tasks
        )
        
        # 注册工作流
        if manager.register_workflow(config):
            print("✅ 工作流注册成功")
        
        # 执行工作流
        execution_id = manager.execute_workflow("test_workflow")
        if execution_id:
            print(f"✅ 工作流执行已启动: {execution_id}")
            
            # 等待执行完成
            import time
            time.sleep(2)
            
            # 检查状态
            status = manager.get_workflow_status(execution_id)
            if status:
                print(f"✅ 工作流状态: {status.status.value}")
                print(f"   完成任务: {status.completed_tasks}/{status.total_tasks}")
        
        # 列出工作流
        workflows = manager.list_workflows()
        print(f"✅ 工作流列表获取成功: {len(workflows)} 个工作流")
        
        print("✅ 工作流管理器功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("工作流管理器功能测试完成")
