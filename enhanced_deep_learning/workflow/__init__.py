"""
工作流模块
Workflow Module

提供自动化工作流、任务调度、流程管理等功能。
"""

from .workflow_manager import (
    WorkflowManager, WorkflowConfig, WorkflowStatus,
    workflow_manager
)
from .automation_engine import (
    AutomationEngine, AutomationRule, AutomationTrigger,
    automation_engine
)
from .task_scheduler import (
    TaskScheduler, ScheduledTask, TaskStatus,
    task_scheduler
)

__all__ = [
    # 工作流管理器
    'WorkflowManager',
    'WorkflowConfig',
    'WorkflowStatus',
    'workflow_manager',
    
    # 自动化引擎
    'AutomationEngine',
    'AutomationRule',
    'AutomationTrigger',
    'automation_engine',
    
    # 任务调度器
    'TaskScheduler',
    'ScheduledTask',
    'TaskStatus',
    'task_scheduler'
]
