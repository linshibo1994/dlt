#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
命令注册表模块
Command Registry Module

提供命令定义、注册、发现和执行功能。
"""

import os
import json
import threading
import argparse
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import inspect

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException


class CommandType(Enum):
    """命令类型枚举"""
    SYSTEM = "system"
    MODEL = "model"
    DATA = "data"
    TRAINING = "training"
    PREDICTION = "prediction"
    EVALUATION = "evaluation"
    UTILITY = "utility"


class ParameterType(Enum):
    """参数类型枚举"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    FILE = "file"
    DIRECTORY = "directory"


@dataclass
class CommandParameter:
    """命令参数"""
    name: str
    param_type: ParameterType
    description: str = ""
    required: bool = False
    default: Any = None
    choices: List[Any] = field(default_factory=list)
    validation: Optional[Callable] = None
    help_text: str = ""


@dataclass
class CommandDefinition:
    """命令定义"""
    name: str
    command_type: CommandType
    description: str
    handler: Callable
    parameters: List[CommandParameter] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    usage: str = ""
    examples: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    author: str = ""
    tags: List[str] = field(default_factory=list)
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommandExecution:
    """命令执行记录"""
    command_name: str
    parameters: Dict[str, Any]
    execution_time: datetime
    duration: float
    success: bool
    result: Any = None
    error_message: str = ""
    user: str = ""


class CommandHandler:
    """命令处理器"""
    
    def __init__(self, definition: CommandDefinition):
        """
        初始化命令处理器
        
        Args:
            definition: 命令定义
        """
        self.definition = definition
        self.execution_history = []
        self.lock = threading.RLock()
        
        logger_manager.debug(f"命令处理器初始化: {definition.name}")
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """
        验证参数
        
        Args:
            parameters: 参数字典
            
        Returns:
            (是否有效, 错误信息)
        """
        try:
            for param in self.definition.parameters:
                param_name = param.name
                param_value = parameters.get(param_name)
                
                # 检查必需参数
                if param.required and param_value is None:
                    return False, f"缺少必需参数: {param_name}"
                
                # 使用默认值
                if param_value is None and param.default is not None:
                    parameters[param_name] = param.default
                    param_value = param.default
                
                # 类型验证
                if param_value is not None:
                    if not self._validate_parameter_type(param_value, param.param_type):
                        return False, f"参数类型错误: {param_name} 应为 {param.param_type.value}"
                    
                    # 选择验证
                    if param.choices and param_value not in param.choices:
                        return False, f"参数值无效: {param_name} 必须是 {param.choices} 中的一个"
                    
                    # 自定义验证
                    if param.validation and not param.validation(param_value):
                        return False, f"参数验证失败: {param_name}"
            
            return True, ""
            
        except Exception as e:
            return False, f"参数验证异常: {e}"
    
    def _validate_parameter_type(self, value: Any, param_type: ParameterType) -> bool:
        """验证参数类型"""
        try:
            if param_type == ParameterType.STRING:
                return isinstance(value, str)
            elif param_type == ParameterType.INTEGER:
                return isinstance(value, int)
            elif param_type == ParameterType.FLOAT:
                return isinstance(value, (int, float))
            elif param_type == ParameterType.BOOLEAN:
                return isinstance(value, bool)
            elif param_type == ParameterType.LIST:
                return isinstance(value, list)
            elif param_type == ParameterType.DICT:
                return isinstance(value, dict)
            elif param_type == ParameterType.FILE:
                return isinstance(value, str) and os.path.isfile(value)
            elif param_type == ParameterType.DIRECTORY:
                return isinstance(value, str) and os.path.isdir(value)
            else:
                return True
                
        except Exception:
            return False
    
    def execute(self, parameters: Dict[str, Any], user: str = "") -> CommandExecution:
        """
        执行命令
        
        Args:
            parameters: 参数字典
            user: 用户标识
            
        Returns:
            命令执行记录
        """
        start_time = datetime.now()
        
        execution = CommandExecution(
            command_name=self.definition.name,
            parameters=parameters.copy(),
            execution_time=start_time,
            duration=0.0,
            success=False,
            user=user
        )
        
        try:
            # 验证参数
            valid, error_msg = self.validate_parameters(parameters)
            if not valid:
                execution.error_message = error_msg
                return execution
            
            # 执行命令
            result = self.definition.handler(**parameters)
            
            # 记录成功执行
            execution.success = True
            execution.result = result
            
            logger_manager.info(f"命令执行成功: {self.definition.name}")
            
        except Exception as e:
            execution.error_message = str(e)
            logger_manager.error(f"命令执行失败: {self.definition.name}, 错误: {e}")
        
        finally:
            # 计算执行时间
            execution.duration = (datetime.now() - start_time).total_seconds()
            
            # 记录执行历史
            with self.lock:
                self.execution_history.append(execution)
                # 保留最近100次执行记录
                if len(self.execution_history) > 100:
                    self.execution_history = self.execution_history[-100:]
        
        return execution
    
    def get_usage_help(self) -> str:
        """获取使用帮助"""
        try:
            help_text = f"命令: {self.definition.name}\n"
            help_text += f"描述: {self.definition.description}\n"
            
            if self.definition.aliases:
                help_text += f"别名: {', '.join(self.definition.aliases)}\n"
            
            if self.definition.usage:
                help_text += f"用法: {self.definition.usage}\n"
            
            if self.definition.parameters:
                help_text += "\n参数:\n"
                for param in self.definition.parameters:
                    required_text = " (必需)" if param.required else ""
                    default_text = f" [默认: {param.default}]" if param.default is not None else ""
                    choices_text = f" 选择: {param.choices}" if param.choices else ""
                    
                    help_text += f"  --{param.name}: {param.description}{required_text}{default_text}{choices_text}\n"
            
            if self.definition.examples:
                help_text += "\n示例:\n"
                for example in self.definition.examples:
                    help_text += f"  {example}\n"
            
            return help_text
            
        except Exception as e:
            return f"获取帮助失败: {e}"
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        try:
            with self.lock:
                total_executions = len(self.execution_history)
                successful_executions = sum(1 for ex in self.execution_history if ex.success)
                
                if total_executions == 0:
                    return {
                        'total_executions': 0,
                        'success_rate': 0.0,
                        'average_duration': 0.0
                    }
                
                success_rate = successful_executions / total_executions
                average_duration = sum(ex.duration for ex in self.execution_history) / total_executions
                
                return {
                    'total_executions': total_executions,
                    'successful_executions': successful_executions,
                    'failed_executions': total_executions - successful_executions,
                    'success_rate': success_rate,
                    'average_duration': average_duration,
                    'last_execution': self.execution_history[-1].execution_time.isoformat() if self.execution_history else None
                }
                
        except Exception as e:
            logger_manager.error(f"获取执行统计失败: {e}")
            return {}


class CommandRegistry:
    """命令注册表"""
    
    def __init__(self):
        """初始化命令注册表"""
        self.commands = {}  # command_name -> CommandHandler
        self.aliases = {}   # alias -> command_name
        self.categories = {}  # category -> List[command_name]
        self.lock = threading.RLock()
        
        logger_manager.info("命令注册表初始化完成")
    
    def register_command(self, definition: CommandDefinition) -> bool:
        """
        注册命令
        
        Args:
            definition: 命令定义
            
        Returns:
            是否注册成功
        """
        try:
            with self.lock:
                command_name = definition.name
                
                # 检查命令是否已存在
                if command_name in self.commands:
                    logger_manager.warning(f"命令已存在，将覆盖: {command_name}")
                
                # 创建命令处理器
                handler = CommandHandler(definition)
                self.commands[command_name] = handler
                
                # 注册别名
                for alias in definition.aliases:
                    self.aliases[alias] = command_name
                
                # 分类管理
                category = definition.command_type.value
                if category not in self.categories:
                    self.categories[category] = []
                
                if command_name not in self.categories[category]:
                    self.categories[category].append(command_name)
                
                logger_manager.info(f"命令注册成功: {command_name}")
                return True
                
        except Exception as e:
            logger_manager.error(f"注册命令失败: {e}")
            return False
    
    def unregister_command(self, command_name: str) -> bool:
        """
        注销命令
        
        Args:
            command_name: 命令名称
            
        Returns:
            是否注销成功
        """
        try:
            with self.lock:
                if command_name not in self.commands:
                    logger_manager.warning(f"命令不存在: {command_name}")
                    return False
                
                handler = self.commands[command_name]
                definition = handler.definition
                
                # 删除命令
                del self.commands[command_name]
                
                # 删除别名
                for alias in definition.aliases:
                    if alias in self.aliases:
                        del self.aliases[alias]
                
                # 从分类中删除
                category = definition.command_type.value
                if category in self.categories and command_name in self.categories[category]:
                    self.categories[category].remove(command_name)
                
                logger_manager.info(f"命令注销成功: {command_name}")
                return True
                
        except Exception as e:
            logger_manager.error(f"注销命令失败: {e}")
            return False
    
    def get_command(self, command_name: str) -> Optional[CommandHandler]:
        """
        获取命令处理器
        
        Args:
            command_name: 命令名称或别名
            
        Returns:
            命令处理器
        """
        with self.lock:
            # 直接查找
            if command_name in self.commands:
                return self.commands[command_name]
            
            # 通过别名查找
            if command_name in self.aliases:
                actual_name = self.aliases[command_name]
                return self.commands.get(actual_name)
            
            return None
    
    def execute_command(self, command_name: str, parameters: Dict[str, Any] = None,
                       user: str = "") -> CommandExecution:
        """
        执行命令
        
        Args:
            command_name: 命令名称
            parameters: 参数字典
            user: 用户标识
            
        Returns:
            命令执行记录
        """
        try:
            handler = self.get_command(command_name)
            if not handler:
                raise DeepLearningException(f"命令不存在: {command_name}")
            
            if not handler.definition.enabled:
                raise DeepLearningException(f"命令已禁用: {command_name}")
            
            parameters = parameters or {}
            return handler.execute(parameters, user)
            
        except Exception as e:
            # 创建失败的执行记录
            execution = CommandExecution(
                command_name=command_name,
                parameters=parameters or {},
                execution_time=datetime.now(),
                duration=0.0,
                success=False,
                error_message=str(e),
                user=user
            )
            return execution
    
    def list_commands(self, category: CommandType = None, 
                     enabled_only: bool = True) -> List[CommandDefinition]:
        """
        列出命令
        
        Args:
            category: 命令类型过滤
            enabled_only: 只列出启用的命令
            
        Returns:
            命令定义列表
        """
        try:
            with self.lock:
                commands = []
                
                for handler in self.commands.values():
                    definition = handler.definition
                    
                    # 类型过滤
                    if category and definition.command_type != category:
                        continue
                    
                    # 启用状态过滤
                    if enabled_only and not definition.enabled:
                        continue
                    
                    commands.append(definition)
                
                # 按名称排序
                commands.sort(key=lambda x: x.name)
                return commands
                
        except Exception as e:
            logger_manager.error(f"列出命令失败: {e}")
            return []
    
    def get_command_help(self, command_name: str = None) -> str:
        """
        获取命令帮助
        
        Args:
            command_name: 命令名称，如果为None则返回所有命令帮助
            
        Returns:
            帮助信息
        """
        try:
            if command_name:
                handler = self.get_command(command_name)
                if handler:
                    return handler.get_usage_help()
                else:
                    return f"命令不存在: {command_name}"
            else:
                # 返回所有命令的简要帮助
                help_text = "可用命令:\n\n"
                
                for category, command_names in self.categories.items():
                    if command_names:
                        help_text += f"{category.upper()}:\n"
                        for cmd_name in sorted(command_names):
                            handler = self.commands.get(cmd_name)
                            if handler and handler.definition.enabled:
                                help_text += f"  {cmd_name}: {handler.definition.description}\n"
                        help_text += "\n"
                
                help_text += "使用 'help <command>' 获取特定命令的详细帮助。"
                return help_text
                
        except Exception as e:
            return f"获取帮助失败: {e}"
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """获取注册表统计信息"""
        try:
            with self.lock:
                stats = {
                    'total_commands': len(self.commands),
                    'enabled_commands': sum(1 for h in self.commands.values() if h.definition.enabled),
                    'total_aliases': len(self.aliases),
                    'categories': {},
                    'execution_stats': {}
                }
                
                # 分类统计
                for category, command_names in self.categories.items():
                    enabled_count = sum(1 for name in command_names 
                                      if name in self.commands and self.commands[name].definition.enabled)
                    stats['categories'][category] = {
                        'total': len(command_names),
                        'enabled': enabled_count
                    }
                
                # 执行统计
                for cmd_name, handler in self.commands.items():
                    cmd_stats = handler.get_execution_stats()
                    if cmd_stats['total_executions'] > 0:
                        stats['execution_stats'][cmd_name] = cmd_stats
                
                return stats
                
        except Exception as e:
            logger_manager.error(f"获取注册表统计失败: {e}")
            return {}
    
    def save_registry(self, file_path: str) -> bool:
        """
        保存注册表到文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否保存成功
        """
        try:
            registry_data = {
                'commands': [],
                'generated_time': datetime.now().isoformat()
            }
            
            with self.lock:
                for handler in self.commands.values():
                    definition = handler.definition
                    
                    # 序列化命令定义（不包括handler函数）
                    cmd_data = {
                        'name': definition.name,
                        'command_type': definition.command_type.value,
                        'description': definition.description,
                        'parameters': [
                            {
                                'name': p.name,
                                'param_type': p.param_type.value,
                                'description': p.description,
                                'required': p.required,
                                'default': p.default,
                                'choices': p.choices,
                                'help_text': p.help_text
                            } for p in definition.parameters
                        ],
                        'aliases': definition.aliases,
                        'usage': definition.usage,
                        'examples': definition.examples,
                        'version': definition.version,
                        'author': definition.author,
                        'tags': definition.tags,
                        'enabled': definition.enabled,
                        'metadata': definition.metadata
                    }
                    
                    registry_data['commands'].append(cmd_data)
            
            # 保存到文件
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)
            
            logger_manager.info(f"注册表保存成功: {file_path}")
            return True
            
        except Exception as e:
            logger_manager.error(f"保存注册表失败: {e}")
            return False


# 全局命令注册表实例
command_registry = CommandRegistry()


# 装饰器函数
def register_command(name: str = None, command_type: CommandType = CommandType.UTILITY,
                    description: str = "", usage: str = "", aliases: List[str] = None,
                    examples: List[str] = None, version: str = "1.0.0", author: str = "",
                    tags: List[str] = None, enabled: bool = True):
    """命令注册装饰器"""
    def decorator(func):
        # 自动解析函数参数
        sig = inspect.signature(func)
        parameters = []
        
        for param_name, param in sig.parameters.items():
            param_type = ParameterType.STRING  # 默认类型
            required = param.default == inspect.Parameter.empty
            default = None if required else param.default
            
            # 根据类型注解推断参数类型
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = ParameterType.INTEGER
                elif param.annotation == float:
                    param_type = ParameterType.FLOAT
                elif param.annotation == bool:
                    param_type = ParameterType.BOOLEAN
                elif param.annotation == list:
                    param_type = ParameterType.LIST
                elif param.annotation == dict:
                    param_type = ParameterType.DICT
            
            cmd_param = CommandParameter(
                name=param_name,
                param_type=param_type,
                required=required,
                default=default
            )
            parameters.append(cmd_param)
        
        # 创建命令定义
        definition = CommandDefinition(
            name=name or func.__name__,
            command_type=command_type,
            description=description,
            handler=func,
            parameters=parameters,
            aliases=aliases or [],
            usage=usage,
            examples=examples or [],
            version=version,
            author=author,
            tags=tags or [],
            enabled=enabled
        )
        
        # 注册命令
        command_registry.register_command(definition)
        
        return func
    return decorator


if __name__ == "__main__":
    # 测试命令注册表功能
    print("📋 测试命令注册表功能...")
    
    try:
        registry = CommandRegistry()
        
        # 创建测试命令
        def test_command(message: str, count: int = 1, verbose: bool = False):
            """测试命令"""
            result = []
            for i in range(count):
                result.append(f"{message} #{i+1}")
            
            if verbose:
                return {"messages": result, "count": count}
            else:
                return result
        
        # 创建命令定义
        definition = CommandDefinition(
            name="test",
            command_type=CommandType.UTILITY,
            description="测试命令",
            handler=test_command,
            parameters=[
                CommandParameter("message", ParameterType.STRING, "消息内容", required=True),
                CommandParameter("count", ParameterType.INTEGER, "重复次数", default=1),
                CommandParameter("verbose", ParameterType.BOOLEAN, "详细输出", default=False)
            ],
            aliases=["t"],
            usage="test <message> [--count N] [--verbose]",
            examples=["test 'Hello World'", "test 'Hello' --count 3 --verbose"]
        )
        
        # 注册命令
        if registry.register_command(definition):
            print("✅ 命令注册成功")
        
        # 执行命令
        execution = registry.execute_command("test", {
            "message": "Hello World",
            "count": 2,
            "verbose": True
        })
        
        if execution.success:
            print(f"✅ 命令执行成功: {execution.result}")
        else:
            print(f"❌ 命令执行失败: {execution.error_message}")
        
        # 测试别名
        alias_execution = registry.execute_command("t", {"message": "Alias Test"})
        if alias_execution.success:
            print("✅ 别名执行成功")
        
        # 获取帮助
        help_text = registry.get_command_help("test")
        print("✅ 帮助信息获取成功")
        
        # 获取统计信息
        stats = registry.get_registry_stats()
        print(f"✅ 统计信息获取成功: {stats['total_commands']} 个命令")
        
        # 测试装饰器
        @register_command(name="decorated_test", description="装饰器测试命令")
        def decorated_command(name: str, age: int = 18):
            return f"Hello {name}, you are {age} years old"
        
        print("✅ 装饰器注册成功")
        
        print("✅ 命令注册表功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("命令注册表功能测试完成")
