#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
系统集成器模块
System Integrator Module

提供与主系统的集成、命令注册、接口适配等功能。
"""

import os
import json
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import importlib
import inspect

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException


class IntegrationStatus(Enum):
    """集成状态枚举"""
    PENDING = "pending"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DISABLED = "disabled"


class IntegrationType(Enum):
    """集成类型枚举"""
    COMMAND = "command"
    API = "api"
    PLUGIN = "plugin"
    SERVICE = "service"
    WEBHOOK = "webhook"


@dataclass
class IntegrationConfig:
    """集成配置"""
    name: str
    integration_type: IntegrationType
    target_system: str
    version: str = "1.0.0"
    description: str = ""
    enabled: bool = True
    auto_register: bool = True
    dependencies: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationPoint:
    """集成点"""
    integration_id: str
    config: IntegrationConfig
    status: IntegrationStatus
    handler: Optional[Callable] = None
    registered_time: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    error_message: str = ""


class SystemAdapter:
    """系统适配器基类"""
    
    def __init__(self, system_name: str):
        """
        初始化系统适配器
        
        Args:
            system_name: 系统名称
        """
        self.system_name = system_name
        self.integrations = {}
        
        logger_manager.debug(f"系统适配器初始化: {system_name}")
    
    def register_integration(self, integration_point: IntegrationPoint) -> bool:
        """注册集成点"""
        try:
            self.integrations[integration_point.integration_id] = integration_point
            logger_manager.info(f"集成点注册成功: {integration_point.integration_id}")
            return True
            
        except Exception as e:
            logger_manager.error(f"注册集成点失败: {e}")
            return False
    
    def unregister_integration(self, integration_id: str) -> bool:
        """注销集成点"""
        try:
            if integration_id in self.integrations:
                del self.integrations[integration_id]
                logger_manager.info(f"集成点注销成功: {integration_id}")
                return True
            else:
                logger_manager.warning(f"集成点不存在: {integration_id}")
                return False
                
        except Exception as e:
            logger_manager.error(f"注销集成点失败: {e}")
            return False
    
    def get_integration(self, integration_id: str) -> Optional[IntegrationPoint]:
        """获取集成点"""
        return self.integrations.get(integration_id)
    
    def list_integrations(self) -> List[IntegrationPoint]:
        """列出所有集成点"""
        return list(self.integrations.values())


class CommandAdapter(SystemAdapter):
    """命令适配器"""
    
    def __init__(self):
        """初始化命令适配器"""
        super().__init__("command_system")
        self.command_handlers = {}
        
        logger_manager.info("命令适配器初始化完成")
    
    def register_command(self, command_name: str, handler: Callable,
                        description: str = "", usage: str = "") -> bool:
        """
        注册命令
        
        Args:
            command_name: 命令名称
            handler: 命令处理器
            description: 命令描述
            usage: 使用说明
            
        Returns:
            是否注册成功
        """
        try:
            # 创建集成配置
            config = IntegrationConfig(
                name=command_name,
                integration_type=IntegrationType.COMMAND,
                target_system="cli",
                description=description,
                configuration={
                    'usage': usage,
                    'handler_name': handler.__name__
                }
            )
            
            # 创建集成点
            integration_point = IntegrationPoint(
                integration_id=f"cmd_{command_name}",
                config=config,
                status=IntegrationStatus.ACTIVE,
                handler=handler
            )
            
            # 注册集成点
            if self.register_integration(integration_point):
                self.command_handlers[command_name] = handler
                logger_manager.info(f"命令注册成功: {command_name}")
                return True
            else:
                return False
                
        except Exception as e:
            logger_manager.error(f"注册命令失败: {e}")
            return False
    
    def execute_command(self, command_name: str, *args, **kwargs) -> Any:
        """
        执行命令
        
        Args:
            command_name: 命令名称
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            命令执行结果
        """
        try:
            if command_name not in self.command_handlers:
                raise DeepLearningException(f"命令不存在: {command_name}")
            
            handler = self.command_handlers[command_name]
            integration_point = self.get_integration(f"cmd_{command_name}")
            
            if integration_point:
                integration_point.last_used = datetime.now()
                integration_point.usage_count += 1
            
            # 执行命令
            result = handler(*args, **kwargs)
            
            logger_manager.info(f"命令执行成功: {command_name}")
            return result
            
        except Exception as e:
            logger_manager.error(f"命令执行失败: {e}")
            raise DeepLearningException(f"命令执行失败: {e}")
    
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
                integration_point = self.get_integration(f"cmd_{command_name}")
                if integration_point:
                    config = integration_point.config
                    help_text = f"命令: {command_name}\n"
                    help_text += f"描述: {config.description}\n"
                    help_text += f"用法: {config.configuration.get('usage', '无')}\n"
                    return help_text
                else:
                    return f"命令不存在: {command_name}"
            else:
                # 返回所有命令帮助
                help_text = "可用命令:\n"
                for integration_point in self.list_integrations():
                    if integration_point.config.integration_type == IntegrationType.COMMAND:
                        help_text += f"  {integration_point.config.name}: {integration_point.config.description}\n"
                return help_text
                
        except Exception as e:
            logger_manager.error(f"获取命令帮助失败: {e}")
            return f"获取帮助失败: {e}"


class APIAdapter(SystemAdapter):
    """API适配器"""
    
    def __init__(self):
        """初始化API适配器"""
        super().__init__("api_system")
        self.api_endpoints = {}
        
        logger_manager.info("API适配器初始化完成")
    
    def register_endpoint(self, path: str, method: str, handler: Callable,
                         description: str = "") -> bool:
        """
        注册API端点
        
        Args:
            path: API路径
            method: HTTP方法
            handler: 处理器函数
            description: 描述
            
        Returns:
            是否注册成功
        """
        try:
            endpoint_key = f"{method.upper()}:{path}"
            
            # 创建集成配置
            config = IntegrationConfig(
                name=endpoint_key,
                integration_type=IntegrationType.API,
                target_system="web_api",
                description=description,
                configuration={
                    'path': path,
                    'method': method.upper(),
                    'handler_name': handler.__name__
                }
            )
            
            # 创建集成点
            integration_point = IntegrationPoint(
                integration_id=f"api_{endpoint_key}",
                config=config,
                status=IntegrationStatus.ACTIVE,
                handler=handler
            )
            
            # 注册集成点
            if self.register_integration(integration_point):
                self.api_endpoints[endpoint_key] = handler
                logger_manager.info(f"API端点注册成功: {endpoint_key}")
                return True
            else:
                return False
                
        except Exception as e:
            logger_manager.error(f"注册API端点失败: {e}")
            return False
    
    def handle_request(self, path: str, method: str, *args, **kwargs) -> Any:
        """
        处理API请求
        
        Args:
            path: API路径
            method: HTTP方法
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            处理结果
        """
        try:
            endpoint_key = f"{method.upper()}:{path}"
            
            if endpoint_key not in self.api_endpoints:
                raise DeepLearningException(f"API端点不存在: {endpoint_key}")
            
            handler = self.api_endpoints[endpoint_key]
            integration_point = self.get_integration(f"api_{endpoint_key}")
            
            if integration_point:
                integration_point.last_used = datetime.now()
                integration_point.usage_count += 1
            
            # 处理请求
            result = handler(*args, **kwargs)
            
            logger_manager.info(f"API请求处理成功: {endpoint_key}")
            return result
            
        except Exception as e:
            logger_manager.error(f"API请求处理失败: {e}")
            raise DeepLearningException(f"API请求处理失败: {e}")


class SystemIntegrator:
    """系统集成器"""
    
    def __init__(self):
        """初始化系统集成器"""
        self.adapters = {}
        self.integration_configs = {}
        self.lock = threading.RLock()
        
        # 初始化默认适配器
        self.command_adapter = CommandAdapter()
        self.api_adapter = APIAdapter()
        
        self.adapters['command'] = self.command_adapter
        self.adapters['api'] = self.api_adapter
        
        logger_manager.info("系统集成器初始化完成")
    
    def register_adapter(self, adapter_name: str, adapter: SystemAdapter) -> bool:
        """
        注册适配器
        
        Args:
            adapter_name: 适配器名称
            adapter: 适配器实例
            
        Returns:
            是否注册成功
        """
        try:
            with self.lock:
                self.adapters[adapter_name] = adapter
                logger_manager.info(f"适配器注册成功: {adapter_name}")
                return True
                
        except Exception as e:
            logger_manager.error(f"注册适配器失败: {e}")
            return False
    
    def get_adapter(self, adapter_name: str) -> Optional[SystemAdapter]:
        """获取适配器"""
        return self.adapters.get(adapter_name)
    
    def register_command(self, command_name: str, handler: Callable,
                        description: str = "", usage: str = "") -> bool:
        """注册命令"""
        return self.command_adapter.register_command(command_name, handler, description, usage)
    
    def execute_command(self, command_name: str, *args, **kwargs) -> Any:
        """执行命令"""
        return self.command_adapter.execute_command(command_name, *args, **kwargs)
    
    def register_api_endpoint(self, path: str, method: str, handler: Callable,
                             description: str = "") -> bool:
        """注册API端点"""
        return self.api_adapter.register_endpoint(path, method, handler, description)
    
    def handle_api_request(self, path: str, method: str, *args, **kwargs) -> Any:
        """处理API请求"""
        return self.api_adapter.handle_request(path, method, *args, **kwargs)
    
    def load_integration_config(self, config_file: str) -> bool:
        """
        加载集成配置
        
        Args:
            config_file: 配置文件路径
            
        Returns:
            是否加载成功
        """
        try:
            if not os.path.exists(config_file):
                logger_manager.warning(f"配置文件不存在: {config_file}")
                return False
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            for config_item in config_data.get('integrations', []):
                config = IntegrationConfig(**config_item)
                self.integration_configs[config.name] = config
            
            logger_manager.info(f"集成配置加载成功: {len(self.integration_configs)} 个配置")
            return True
            
        except Exception as e:
            logger_manager.error(f"加载集成配置失败: {e}")
            return False
    
    def save_integration_config(self, config_file: str) -> bool:
        """
        保存集成配置
        
        Args:
            config_file: 配置文件路径
            
        Returns:
            是否保存成功
        """
        try:
            # 收集所有集成配置
            all_configs = []
            
            for adapter in self.adapters.values():
                for integration_point in adapter.list_integrations():
                    config_dict = {
                        'name': integration_point.config.name,
                        'integration_type': integration_point.config.integration_type.value,
                        'target_system': integration_point.config.target_system,
                        'version': integration_point.config.version,
                        'description': integration_point.config.description,
                        'enabled': integration_point.config.enabled,
                        'auto_register': integration_point.config.auto_register,
                        'dependencies': integration_point.config.dependencies,
                        'configuration': integration_point.config.configuration,
                        'metadata': integration_point.config.metadata
                    }
                    all_configs.append(config_dict)
            
            # 保存配置
            config_data = {
                'integrations': all_configs,
                'generated_time': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger_manager.info(f"集成配置保存成功: {config_file}")
            return True
            
        except Exception as e:
            logger_manager.error(f"保存集成配置失败: {e}")
            return False
    
    def get_integration_status(self) -> Dict[str, Any]:
        """获取集成状态"""
        try:
            with self.lock:
                status = {
                    'total_adapters': len(self.adapters),
                    'total_integrations': 0,
                    'adapters': {},
                    'integration_summary': {
                        'active': 0,
                        'inactive': 0,
                        'error': 0
                    }
                }
                
                for adapter_name, adapter in self.adapters.items():
                    integrations = adapter.list_integrations()
                    status['total_integrations'] += len(integrations)
                    
                    adapter_status = {
                        'system_name': adapter.system_name,
                        'integration_count': len(integrations),
                        'integrations': []
                    }
                    
                    for integration_point in integrations:
                        integration_info = {
                            'id': integration_point.integration_id,
                            'name': integration_point.config.name,
                            'type': integration_point.config.integration_type.value,
                            'status': integration_point.status.value,
                            'usage_count': integration_point.usage_count,
                            'last_used': integration_point.last_used.isoformat() if integration_point.last_used else None
                        }
                        adapter_status['integrations'].append(integration_info)
                        
                        # 更新汇总统计
                        if integration_point.status == IntegrationStatus.ACTIVE:
                            status['integration_summary']['active'] += 1
                        elif integration_point.status == IntegrationStatus.INACTIVE:
                            status['integration_summary']['inactive'] += 1
                        elif integration_point.status == IntegrationStatus.ERROR:
                            status['integration_summary']['error'] += 1
                    
                    status['adapters'][adapter_name] = adapter_status
                
                return status
                
        except Exception as e:
            logger_manager.error(f"获取集成状态失败: {e}")
            return {}
    
    def auto_discover_integrations(self, module_path: str) -> int:
        """
        自动发现集成
        
        Args:
            module_path: 模块路径
            
        Returns:
            发现的集成数量
        """
        try:
            discovered_count = 0
            
            # 动态导入模块
            module = importlib.import_module(module_path)
            
            # 查找带有特定装饰器或属性的函数
            for name in dir(module):
                obj = getattr(module, name)
                
                if callable(obj) and hasattr(obj, '_integration_config'):
                    config = obj._integration_config
                    
                    if config.get('type') == 'command':
                        self.register_command(
                            config.get('name', name),
                            obj,
                            config.get('description', ''),
                            config.get('usage', '')
                        )
                        discovered_count += 1
                    elif config.get('type') == 'api':
                        self.register_api_endpoint(
                            config.get('path', f'/{name}'),
                            config.get('method', 'GET'),
                            obj,
                            config.get('description', '')
                        )
                        discovered_count += 1
            
            logger_manager.info(f"自动发现集成完成: {discovered_count} 个")
            return discovered_count
            
        except Exception as e:
            logger_manager.error(f"自动发现集成失败: {e}")
            return 0


# 全局系统集成器实例
system_integrator = SystemIntegrator()


# 装饰器函数
def command_integration(name: str = None, description: str = "", usage: str = ""):
    """命令集成装饰器"""
    def decorator(func):
        func._integration_config = {
            'type': 'command',
            'name': name or func.__name__,
            'description': description,
            'usage': usage
        }
        return func
    return decorator


def api_integration(path: str = None, method: str = "GET", description: str = ""):
    """API集成装饰器"""
    def decorator(func):
        func._integration_config = {
            'type': 'api',
            'path': path or f'/{func.__name__}',
            'method': method,
            'description': description
        }
        return func
    return decorator


if __name__ == "__main__":
    # 测试系统集成器功能
    print("🔗 测试系统集成器功能...")
    
    try:
        integrator = SystemIntegrator()
        
        # 测试命令注册
        def test_command(message: str = "Hello"):
            return f"测试命令执行: {message}"
        
        if integrator.register_command("test", test_command, "测试命令", "test [message]"):
            print("✅ 命令注册成功")
        
        # 测试命令执行
        result = integrator.execute_command("test", "World")
        print(f"✅ 命令执行成功: {result}")
        
        # 测试API端点注册
        def test_api(data: dict = None):
            return {"status": "success", "data": data}
        
        if integrator.register_api_endpoint("/test", "POST", test_api, "测试API"):
            print("✅ API端点注册成功")
        
        # 测试API请求处理
        api_result = integrator.handle_api_request("/test", "POST", {"key": "value"})
        print(f"✅ API请求处理成功: {api_result}")
        
        # 测试集成状态
        status = integrator.get_integration_status()
        print(f"✅ 集成状态获取成功: {status['total_integrations']} 个集成")
        
        # 测试装饰器
        @command_integration(name="decorated_test", description="装饰器测试命令")
        def decorated_command():
            return "装饰器命令执行成功"
        
        print("✅ 装饰器定义成功")
        
        print("✅ 系统集成器功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("系统集成器功能测试完成")
