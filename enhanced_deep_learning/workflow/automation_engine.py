#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动化引擎模块
Automation Engine Module

提供自动化规则、触发器、动作执行等功能。
"""

import os
import json
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import schedule

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException


class TriggerType(Enum):
    """触发器类型枚举"""
    TIME_BASED = "time_based"
    EVENT_BASED = "event_based"
    CONDITION_BASED = "condition_based"
    MANUAL = "manual"


class ActionType(Enum):
    """动作类型枚举"""
    EXECUTE_WORKFLOW = "execute_workflow"
    SEND_NOTIFICATION = "send_notification"
    UPDATE_MODEL = "update_model"
    BACKUP_DATA = "backup_data"
    GENERATE_REPORT = "generate_report"
    CUSTOM = "custom"


@dataclass
class AutomationTrigger:
    """自动化触发器"""
    trigger_id: str
    trigger_type: TriggerType
    name: str
    description: str = ""
    schedule_expression: str = ""  # cron表达式或时间间隔
    condition: Optional[Callable] = None
    event_pattern: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class AutomationAction:
    """自动化动作"""
    action_id: str
    action_type: ActionType
    name: str
    handler: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class AutomationRule:
    """自动化规则"""
    rule_id: str
    name: str
    description: str
    trigger: AutomationTrigger
    actions: List[AutomationAction]
    enabled: bool = True
    created_time: datetime = field(default_factory=datetime.now)
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0


@dataclass
class AutomationExecution:
    """自动化执行记录"""
    execution_id: str
    rule_id: str
    trigger_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    actions_executed: int = 0
    actions_failed: int = 0
    error_message: str = ""
    results: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    """事件总线"""
    
    def __init__(self):
        """初始化事件总线"""
        self.subscribers = {}
        self.lock = threading.RLock()
        
        logger_manager.debug("事件总线初始化完成")
    
    def subscribe(self, event_type: str, callback: Callable):
        """
        订阅事件
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        with self.lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            
            self.subscribers[event_type].append(callback)
            
        logger_manager.debug(f"事件订阅成功: {event_type}")
    
    def publish(self, event_type: str, event_data: Dict[str, Any]):
        """
        发布事件
        
        Args:
            event_type: 事件类型
            event_data: 事件数据
        """
        try:
            with self.lock:
                callbacks = self.subscribers.get(event_type, [])
            
            for callback in callbacks:
                try:
                    callback(event_data)
                except Exception as e:
                    logger_manager.error(f"事件回调执行失败: {e}")
            
            logger_manager.debug(f"事件发布成功: {event_type}")
            
        except Exception as e:
            logger_manager.error(f"事件发布失败: {e}")
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """取消订阅事件"""
        try:
            with self.lock:
                if event_type in self.subscribers:
                    if callback in self.subscribers[event_type]:
                        self.subscribers[event_type].remove(callback)
                        
            logger_manager.debug(f"事件取消订阅成功: {event_type}")
            
        except Exception as e:
            logger_manager.error(f"取消事件订阅失败: {e}")


class ScheduleManager:
    """调度管理器"""
    
    def __init__(self):
        """初始化调度管理器"""
        self.scheduled_jobs = {}
        self.running = False
        self.scheduler_thread = None
        self.lock = threading.RLock()
        
        logger_manager.debug("调度管理器初始化完成")
    
    def add_job(self, job_id: str, schedule_expression: str, job_func: Callable):
        """
        添加定时任务
        
        Args:
            job_id: 任务ID
            schedule_expression: 调度表达式
            job_func: 任务函数
        """
        try:
            with self.lock:
                # 解析调度表达式
                if schedule_expression.startswith("every"):
                    # 简单的间隔调度
                    parts = schedule_expression.split()
                    if len(parts) >= 3:
                        interval = int(parts[1])
                        unit = parts[2]
                        
                        if unit == "seconds":
                            job = schedule.every(interval).seconds.do(job_func)
                        elif unit == "minutes":
                            job = schedule.every(interval).minutes.do(job_func)
                        elif unit == "hours":
                            job = schedule.every(interval).hours.do(job_func)
                        elif unit == "days":
                            job = schedule.every(interval).days.do(job_func)
                        else:
                            job = schedule.every(interval).minutes.do(job_func)
                        
                        self.scheduled_jobs[job_id] = job
                        
                elif ":" in schedule_expression:
                    # 每日定时调度
                    job = schedule.every().day.at(schedule_expression).do(job_func)
                    self.scheduled_jobs[job_id] = job
                
                else:
                    # 默认每分钟执行
                    job = schedule.every().minute.do(job_func)
                    self.scheduled_jobs[job_id] = job
                
            logger_manager.info(f"定时任务添加成功: {job_id}")
            
        except Exception as e:
            logger_manager.error(f"添加定时任务失败: {e}")
    
    def remove_job(self, job_id: str):
        """移除定时任务"""
        try:
            with self.lock:
                if job_id in self.scheduled_jobs:
                    job = self.scheduled_jobs[job_id]
                    schedule.cancel_job(job)
                    del self.scheduled_jobs[job_id]
                    
            logger_manager.info(f"定时任务移除成功: {job_id}")
            
        except Exception as e:
            logger_manager.error(f"移除定时任务失败: {e}")
    
    def start_scheduler(self):
        """启动调度器"""
        if self.running:
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger_manager.info("调度器已启动")
    
    def stop_scheduler(self):
        """停止调度器"""
        self.running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger_manager.info("调度器已停止")
    
    def _scheduler_loop(self):
        """调度器循环"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger_manager.error(f"调度器循环失败: {e}")
                time.sleep(5)


class AutomationEngine:
    """自动化引擎"""
    
    def __init__(self):
        """初始化自动化引擎"""
        self.rules = {}
        self.executions = []
        self.event_bus = EventBus()
        self.schedule_manager = ScheduleManager()
        self.lock = threading.RLock()
        
        # 启动调度器
        self.schedule_manager.start_scheduler()
        
        logger_manager.info("自动化引擎初始化完成")
    
    def register_rule(self, rule: AutomationRule) -> bool:
        """
        注册自动化规则
        
        Args:
            rule: 自动化规则
            
        Returns:
            是否注册成功
        """
        try:
            with self.lock:
                self.rules[rule.rule_id] = rule
            
            # 设置触发器
            self._setup_trigger(rule)
            
            logger_manager.info(f"自动化规则注册成功: {rule.name}")
            return True
            
        except Exception as e:
            logger_manager.error(f"注册自动化规则失败: {e}")
            return False
    
    def _setup_trigger(self, rule: AutomationRule):
        """设置触发器"""
        try:
            trigger = rule.trigger
            
            if trigger.trigger_type == TriggerType.TIME_BASED:
                # 时间触发器
                job_func = lambda: self._execute_rule(rule.rule_id)
                self.schedule_manager.add_job(
                    f"rule_{rule.rule_id}",
                    trigger.schedule_expression,
                    job_func
                )
                
            elif trigger.trigger_type == TriggerType.EVENT_BASED:
                # 事件触发器
                event_type = trigger.event_pattern.get("type", "default")
                callback = lambda event_data: self._handle_event_trigger(rule.rule_id, event_data)
                self.event_bus.subscribe(event_type, callback)
                
            elif trigger.trigger_type == TriggerType.CONDITION_BASED:
                # 条件触发器（需要定期检查）
                if trigger.condition:
                    job_func = lambda: self._check_condition_trigger(rule.rule_id)
                    self.schedule_manager.add_job(
                        f"condition_{rule.rule_id}",
                        "every 1 minutes",  # 每分钟检查一次条件
                        job_func
                    )
            
        except Exception as e:
            logger_manager.error(f"设置触发器失败: {e}")
    
    def _execute_rule(self, rule_id: str):
        """执行规则"""
        try:
            with self.lock:
                if rule_id not in self.rules:
                    return
                
                rule = self.rules[rule_id]
                
                if not rule.enabled:
                    return
            
            # 创建执行记录
            execution = AutomationExecution(
                execution_id=f"exec_{int(time.time())}_{rule_id}",
                rule_id=rule_id,
                trigger_id=rule.trigger.trigger_id,
                start_time=datetime.now()
            )
            
            logger_manager.info(f"开始执行自动化规则: {rule.name}")
            
            # 执行动作
            success = True
            for action in rule.actions:
                try:
                    result = self._execute_action(action)
                    execution.results[action.action_id] = result
                    execution.actions_executed += 1
                    
                except Exception as e:
                    execution.actions_failed += 1
                    execution.error_message += f"动作 {action.name} 失败: {e}; "
                    success = False
                    
                    logger_manager.error(f"动作执行失败: {action.name}, 错误: {e}")
            
            # 更新执行记录
            execution.end_time = datetime.now()
            execution.success = success
            
            # 更新规则统计
            rule.last_executed = datetime.now()
            rule.execution_count += 1
            
            if success:
                rule.success_count += 1
            else:
                rule.failure_count += 1
            
            # 更新触发器统计
            rule.trigger.last_triggered = datetime.now()
            rule.trigger.trigger_count += 1
            
            # 保存执行记录
            self.executions.append(execution)
            
            logger_manager.info(f"自动化规则执行完成: {rule.name}, 成功: {success}")
            
        except Exception as e:
            logger_manager.error(f"执行自动化规则失败: {e}")
    
    def _execute_action(self, action: AutomationAction) -> Any:
        """执行动作"""
        try:
            logger_manager.info(f"执行动作: {action.name}")
            
            # 执行动作处理器
            result = action.handler(**action.parameters)
            
            logger_manager.info(f"动作执行成功: {action.name}")
            return result
            
        except Exception as e:
            logger_manager.error(f"动作执行失败: {action.name}, 错误: {e}")
            
            # 重试逻辑
            if action.retry_count < action.max_retries:
                action.retry_count += 1
                logger_manager.info(f"动作重试: {action.name}, 第 {action.retry_count} 次")
                return self._execute_action(action)
            
            raise
    
    def _handle_event_trigger(self, rule_id: str, event_data: Dict[str, Any]):
        """处理事件触发器"""
        try:
            with self.lock:
                if rule_id not in self.rules:
                    return
                
                rule = self.rules[rule_id]
                
                # 检查事件模式匹配
                if self._match_event_pattern(rule.trigger.event_pattern, event_data):
                    self._execute_rule(rule_id)
                    
        except Exception as e:
            logger_manager.error(f"处理事件触发器失败: {e}")
    
    def _check_condition_trigger(self, rule_id: str):
        """检查条件触发器"""
        try:
            with self.lock:
                if rule_id not in self.rules:
                    return
                
                rule = self.rules[rule_id]
                
                # 检查条件
                if rule.trigger.condition and rule.trigger.condition():
                    self._execute_rule(rule_id)
                    
        except Exception as e:
            logger_manager.error(f"检查条件触发器失败: {e}")
    
    def _match_event_pattern(self, pattern: Dict[str, Any], event_data: Dict[str, Any]) -> bool:
        """匹配事件模式"""
        try:
            for key, value in pattern.items():
                if key not in event_data or event_data[key] != value:
                    return False
            return True
            
        except Exception:
            return False
    
    def trigger_event(self, event_type: str, event_data: Dict[str, Any]):
        """触发事件"""
        self.event_bus.publish(event_type, event_data)
    
    def manual_trigger(self, rule_id: str) -> bool:
        """手动触发规则"""
        try:
            self._execute_rule(rule_id)
            return True
            
        except Exception as e:
            logger_manager.error(f"手动触发规则失败: {e}")
            return False
    
    def get_rule_status(self, rule_id: str) -> Optional[AutomationRule]:
        """获取规则状态"""
        with self.lock:
            return self.rules.get(rule_id)
    
    def list_rules(self) -> List[AutomationRule]:
        """列出所有规则"""
        with self.lock:
            return list(self.rules.values())
    
    def get_execution_history(self) -> List[AutomationExecution]:
        """获取执行历史"""
        return self.executions.copy()
    
    def shutdown(self):
        """关闭自动化引擎"""
        try:
            self.schedule_manager.stop_scheduler()
            logger_manager.info("自动化引擎已关闭")
            
        except Exception as e:
            logger_manager.error(f"关闭自动化引擎失败: {e}")


# 全局自动化引擎实例
automation_engine = AutomationEngine()


if __name__ == "__main__":
    # 测试自动化引擎功能
    print("🤖 测试自动化引擎功能...")
    
    try:
        engine = AutomationEngine()
        
        # 创建测试动作
        def test_action(message: str = "Hello"):
            print(f"执行测试动作: {message}")
            return f"动作结果: {message}"
        
        # 创建自动化规则
        trigger = AutomationTrigger(
            trigger_id="test_trigger",
            trigger_type=TriggerType.MANUAL,
            name="测试触发器"
        )
        
        action = AutomationAction(
            action_id="test_action",
            action_type=ActionType.CUSTOM,
            name="测试动作",
            handler=test_action,
            parameters={"message": "自动化测试"}
        )
        
        rule = AutomationRule(
            rule_id="test_rule",
            name="测试规则",
            description="用于测试的自动化规则",
            trigger=trigger,
            actions=[action]
        )
        
        # 注册规则
        if engine.register_rule(rule):
            print("✅ 自动化规则注册成功")
        
        # 手动触发规则
        if engine.manual_trigger("test_rule"):
            print("✅ 规则手动触发成功")
        
        # 测试事件触发
        event_trigger = AutomationTrigger(
            trigger_id="event_trigger",
            trigger_type=TriggerType.EVENT_BASED,
            name="事件触发器",
            event_pattern={"type": "test_event", "source": "test"}
        )
        
        event_rule = AutomationRule(
            rule_id="event_rule",
            name="事件规则",
            description="事件触发的规则",
            trigger=event_trigger,
            actions=[action]
        )
        
        engine.register_rule(event_rule)
        
        # 触发事件
        engine.trigger_event("test_event", {"type": "test_event", "source": "test", "data": "test_data"})
        print("✅ 事件触发成功")
        
        # 获取规则列表
        rules = engine.list_rules()
        print(f"✅ 规则列表获取成功: {len(rules)} 个规则")
        
        # 获取执行历史
        history = engine.get_execution_history()
        print(f"✅ 执行历史获取成功: {len(history)} 条记录")
        
        print("✅ 自动化引擎功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("自动化引擎功能测试完成")
