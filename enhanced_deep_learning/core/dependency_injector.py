#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
依赖注入器模块
Dependency Injector Module

提供依赖注入、服务注册、生命周期管理等功能。
"""

import inspect
from typing import Dict, Any, Optional, Type, Callable, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum
import threading

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException

T = TypeVar('T')


class ServiceScope(Enum):
    """服务作用域枚举"""
    SINGLETON = "singleton"    # 单例
    TRANSIENT = "transient"    # 瞬态
    SCOPED = "scoped"         # 作用域


@dataclass
class ServiceDescriptor:
    """服务描述符"""
    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    scope: ServiceScope = ServiceScope.TRANSIENT


class ServiceRegistry:
    """服务注册表"""
    
    def __init__(self):
        """初始化服务注册表"""
        self.services: Dict[Type, ServiceDescriptor] = {}
        self._lock = threading.RLock()
        
        logger_manager.debug("服务注册表初始化完成")
    
    def register_singleton(self, service_type: Type[T], implementation_type: Type[T] = None) -> 'ServiceRegistry':
        """
        注册单例服务
        
        Args:
            service_type: 服务类型
            implementation_type: 实现类型
            
        Returns:
            服务注册表实例
        """
        with self._lock:
            descriptor = ServiceDescriptor(
                service_type=service_type,
                implementation_type=implementation_type or service_type,
                scope=ServiceScope.SINGLETON
            )
            self.services[service_type] = descriptor
            
            logger_manager.debug(f"单例服务注册成功: {service_type.__name__}")
            return self
    
    def register_transient(self, service_type: Type[T], implementation_type: Type[T] = None) -> 'ServiceRegistry':
        """
        注册瞬态服务
        
        Args:
            service_type: 服务类型
            implementation_type: 实现类型
            
        Returns:
            服务注册表实例
        """
        with self._lock:
            descriptor = ServiceDescriptor(
                service_type=service_type,
                implementation_type=implementation_type or service_type,
                scope=ServiceScope.TRANSIENT
            )
            self.services[service_type] = descriptor
            
            logger_manager.debug(f"瞬态服务注册成功: {service_type.__name__}")
            return self
    
    def register_instance(self, service_type: Type[T], instance: T) -> 'ServiceRegistry':
        """
        注册服务实例
        
        Args:
            service_type: 服务类型
            instance: 服务实例
            
        Returns:
            服务注册表实例
        """
        with self._lock:
            descriptor = ServiceDescriptor(
                service_type=service_type,
                instance=instance,
                scope=ServiceScope.SINGLETON
            )
            self.services[service_type] = descriptor
            
            logger_manager.debug(f"服务实例注册成功: {service_type.__name__}")
            return self
    
    def register_factory(self, service_type: Type[T], factory: Callable[[], T]) -> 'ServiceRegistry':
        """
        注册服务工厂
        
        Args:
            service_type: 服务类型
            factory: 工厂函数
            
        Returns:
            服务注册表实例
        """
        with self._lock:
            descriptor = ServiceDescriptor(
                service_type=service_type,
                factory=factory,
                scope=ServiceScope.TRANSIENT
            )
            self.services[service_type] = descriptor
            
            logger_manager.debug(f"服务工厂注册成功: {service_type.__name__}")
            return self
    
    def is_registered(self, service_type: Type) -> bool:
        """
        检查服务是否已注册
        
        Args:
            service_type: 服务类型
            
        Returns:
            是否已注册
        """
        with self._lock:
            return service_type in self.services
    
    def get_descriptor(self, service_type: Type) -> Optional[ServiceDescriptor]:
        """
        获取服务描述符
        
        Args:
            service_type: 服务类型
            
        Returns:
            服务描述符
        """
        with self._lock:
            return self.services.get(service_type)
    
    def unregister(self, service_type: Type) -> bool:
        """
        取消注册服务
        
        Args:
            service_type: 服务类型
            
        Returns:
            是否取消成功
        """
        with self._lock:
            if service_type in self.services:
                del self.services[service_type]
                logger_manager.debug(f"服务取消注册成功: {service_type.__name__}")
                return True
            return False
    
    def clear(self):
        """清空所有注册的服务"""
        with self._lock:
            self.services.clear()
            logger_manager.debug("服务注册表已清空")


class DependencyInjector:
    """依赖注入器"""
    
    def __init__(self):
        """初始化依赖注入器"""
        self.registry = ServiceRegistry()
        self._singletons: Dict[Type, Any] = {}
        self._lock = threading.RLock()
        
        # 注册自身
        self.registry.register_instance(DependencyInjector, self)
        self.registry.register_instance(ServiceRegistry, self.registry)
        
        logger_manager.info("依赖注入器初始化完成")
    
    def get_service(self, service_type: Type[T]) -> T:
        """
        获取服务实例
        
        Args:
            service_type: 服务类型
            
        Returns:
            服务实例
        """
        try:
            with self._lock:
                descriptor = self.registry.get_descriptor(service_type)
                
                if not descriptor:
                    raise DeepLearningException(f"服务未注册: {service_type.__name__}")
                
                # 如果已有实例，直接返回
                if descriptor.instance is not None:
                    return descriptor.instance
                
                # 单例模式检查
                if descriptor.scope == ServiceScope.SINGLETON:
                    if service_type in self._singletons:
                        return self._singletons[service_type]
                
                # 创建实例
                instance = self._create_instance(descriptor)
                
                # 单例模式缓存
                if descriptor.scope == ServiceScope.SINGLETON:
                    self._singletons[service_type] = instance
                
                return instance
                
        except Exception as e:
            logger_manager.error(f"获取服务实例失败: {e}")
            raise DeepLearningException(f"获取服务实例失败: {e}")
    
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """创建服务实例"""
        try:
            # 使用工厂函数
            if descriptor.factory:
                return descriptor.factory()
            
            # 使用实现类型
            if descriptor.implementation_type:
                return self._create_with_dependencies(descriptor.implementation_type)
            
            # 使用服务类型
            return self._create_with_dependencies(descriptor.service_type)
            
        except Exception as e:
            logger_manager.error(f"创建服务实例失败: {e}")
            raise
    
    def _create_with_dependencies(self, implementation_type: Type) -> Any:
        """创建带依赖注入的实例"""
        try:
            # 获取构造函数签名
            signature = inspect.signature(implementation_type.__init__)
            
            # 准备构造参数
            kwargs = {}
            
            for param_name, param in signature.parameters.items():
                if param_name == 'self':
                    continue
                
                # 获取参数类型
                param_type = param.annotation
                
                if param_type != inspect.Parameter.empty:
                    # 尝试解析依赖
                    if self.registry.is_registered(param_type):
                        kwargs[param_name] = self.get_service(param_type)
                    elif param.default != inspect.Parameter.empty:
                        # 使用默认值
                        kwargs[param_name] = param.default
                    else:
                        logger_manager.warning(f"无法解析依赖: {param_name} ({param_type})")
            
            # 创建实例
            instance = implementation_type(**kwargs)
            
            logger_manager.debug(f"服务实例创建成功: {implementation_type.__name__}")
            return instance
            
        except Exception as e:
            logger_manager.error(f"创建带依赖注入的实例失败: {e}")
            raise
    
    def try_get_service(self, service_type: Type[T]) -> Optional[T]:
        """
        尝试获取服务实例
        
        Args:
            service_type: 服务类型
            
        Returns:
            服务实例或None
        """
        try:
            return self.get_service(service_type)
        except Exception:
            return None
    
    def create_scope(self) -> 'DependencyInjector':
        """
        创建作用域
        
        Returns:
            新的依赖注入器实例
        """
        scoped_injector = DependencyInjector()
        
        # 复制服务注册
        with self._lock:
            scoped_injector.registry.services = self.registry.services.copy()
        
        logger_manager.debug("依赖注入作用域创建成功")
        return scoped_injector
    
    def inject_dependencies(self, target: Any):
        """
        为现有对象注入依赖
        
        Args:
            target: 目标对象
        """
        try:
            # 获取对象的所有属性
            for attr_name in dir(target):
                if attr_name.startswith('_'):
                    continue
                
                attr_value = getattr(target, attr_name)
                
                # 检查是否为类型注解
                if hasattr(target.__class__, '__annotations__'):
                    annotations = target.__class__.__annotations__
                    
                    if attr_name in annotations:
                        attr_type = annotations[attr_name]
                        
                        if self.registry.is_registered(attr_type) and attr_value is None:
                            # 注入依赖
                            injected_value = self.get_service(attr_type)
                            setattr(target, attr_name, injected_value)
                            
                            logger_manager.debug(f"依赖注入成功: {attr_name} -> {attr_type.__name__}")
            
        except Exception as e:
            logger_manager.error(f"注入依赖失败: {e}")
    
    def get_service_info(self) -> Dict[str, Any]:
        """获取服务信息"""
        try:
            with self._lock:
                services_info = {}
                
                for service_type, descriptor in self.registry.services.items():
                    services_info[service_type.__name__] = {
                        "service_type": service_type.__name__,
                        "implementation_type": descriptor.implementation_type.__name__ if descriptor.implementation_type else None,
                        "scope": descriptor.scope.value,
                        "has_instance": descriptor.instance is not None,
                        "has_factory": descriptor.factory is not None,
                        "is_singleton_cached": service_type in self._singletons
                    }
                
                return {
                    "total_services": len(services_info),
                    "singleton_count": len(self._singletons),
                    "services": services_info
                }
                
        except Exception as e:
            logger_manager.error(f"获取服务信息失败: {e}")
            return {"error": str(e)}


# 全局依赖注入器实例
dependency_injector = DependencyInjector()


if __name__ == "__main__":
    # 测试依赖注入器功能
    print("💉 测试依赖注入器功能...")
    
    try:
        # 定义测试服务
        class ITestService:
            def get_message(self) -> str:
                pass
        
        class TestService(ITestService):
            def get_message(self) -> str:
                return "Hello from TestService"
        
        class TestClient:
            def __init__(self, service: ITestService):
                self.service = service
            
            def do_work(self) -> str:
                return self.service.get_message()
        
        # 创建依赖注入器
        injector = DependencyInjector()
        
        # 注册服务
        injector.registry.register_singleton(ITestService, TestService)
        injector.registry.register_transient(TestClient)
        
        # 获取服务
        client = injector.get_service(TestClient)
        message = client.do_work()
        print(f"✅ 依赖注入: {message}")
        
        # 测试单例
        client2 = injector.get_service(TestClient)
        same_service = client.service is client2.service
        print(f"✅ 单例模式: {same_service}")
        
        # 获取服务信息
        info = injector.get_service_info()
        print(f"✅ 服务信息: {info['total_services']} 个服务")
        
        print("✅ 依赖注入器功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("依赖注入器功能测试完成")
