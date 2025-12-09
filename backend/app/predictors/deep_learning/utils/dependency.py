#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
依赖注入容器
Dependency Injection Container

提供服务注册、解析和生命周期管理功能。
"""

from typing import Type, Any, Dict, Optional, Callable
from enum import Enum
import inspect
import threading
from core_modules import logger_manager


class ServiceLifetime(Enum):
    """服务生命周期枚举"""
    TRANSIENT = "transient"  # 瞬态：每次请求都创建新实例
    SINGLETON = "singleton"  # 单例：全局唯一实例
    SCOPED = "scoped"       # 作用域：在特定作用域内唯一


class ServiceDescriptor:
    """服务描述符"""
    
    def __init__(self, service_type: Type, implementation: Optional[Type] = None, 
                 instance: Optional[Any] = None, factory: Optional[Callable] = None,
                 lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT):
        self.service_type = service_type
        self.implementation = implementation or service_type
        self.instance = instance
        self.factory = factory
        self.lifetime = lifetime


class DependencyContainer:
    """依赖注入容器"""
    
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._scoped_instances: Dict[Type, Any] = {}
        self._lock = threading.RLock()
        
        logger_manager.info("依赖注入容器初始化完成")
    
    def register(self, service_type: Type, implementation: Type = None, 
                instance: Any = None) -> 'DependencyContainer':
        """注册瞬态服务"""
        return self._register_service(service_type, implementation, instance, 
                                    None, ServiceLifetime.TRANSIENT)
    
    def register_singleton(self, service_type: Type, implementation: Type = None, 
                          instance: Any = None) -> 'DependencyContainer':
        """注册单例服务"""
        return self._register_service(service_type, implementation, instance, 
                                    None, ServiceLifetime.SINGLETON)
    
    def register_scoped(self, service_type: Type, implementation: Type = None) -> 'DependencyContainer':
        """注册作用域服务"""
        return self._register_service(service_type, implementation, None, 
                                    None, ServiceLifetime.SCOPED)
    
    def register_factory(self, service_type: Type, factory: Callable, 
                        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT) -> 'DependencyContainer':
        """注册工厂方法"""
        return self._register_service(service_type, None, None, factory, lifetime)
    
    def _register_service(self, service_type: Type, implementation: Optional[Type], 
                         instance: Optional[Any], factory: Optional[Callable],
                         lifetime: ServiceLifetime) -> 'DependencyContainer':
        """内部服务注册方法"""
        with self._lock:
            descriptor = ServiceDescriptor(service_type, implementation, 
                                         instance, factory, lifetime)
            self._services[service_type] = descriptor
            
            # 如果提供了实例且是单例，直接存储
            if instance is not None and lifetime == ServiceLifetime.SINGLETON:
                self._singletons[service_type] = instance
            
            logger_manager.debug(f"注册服务: {service_type.__name__} -> {lifetime.value}")
            return self
    
    def resolve(self, service_type: Type) -> Any:
        """解析服务实例"""
        with self._lock:
            if service_type not in self._services:
                raise ValueError(f"服务 {service_type.__name__} 未注册")
            
            descriptor = self._services[service_type]
            
            # 单例模式
            if descriptor.lifetime == ServiceLifetime.SINGLETON:
                if service_type in self._singletons:
                    return self._singletons[service_type]
                
                instance = self._create_instance(descriptor)
                self._singletons[service_type] = instance
                return instance
            
            # 作用域模式
            elif descriptor.lifetime == ServiceLifetime.SCOPED:
                if service_type in self._scoped_instances:
                    return self._scoped_instances[service_type]
                
                instance = self._create_instance(descriptor)
                self._scoped_instances[service_type] = instance
                return instance
            
            # 瞬态模式
            else:
                return self._create_instance(descriptor)
    
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """创建服务实例"""
        try:
            # 如果有预设实例，直接返回
            if descriptor.instance is not None:
                return descriptor.instance
            
            # 如果有工厂方法，使用工厂创建
            if descriptor.factory is not None:
                return descriptor._inject_dependencies(descriptor.factory)
            
            # 使用构造函数创建
            return self._create_with_constructor(descriptor.implementation)
            
        except Exception as e:
            logger_manager.error(f"创建服务实例失败: {descriptor.service_type.__name__}", e)
            raise
    
    def _create_with_constructor(self, implementation: Type) -> Any:
        """使用构造函数创建实例，自动注入依赖"""
        try:
            # 获取构造函数签名
            signature = inspect.signature(implementation.__init__)
            parameters = signature.parameters
            
            # 准备构造函数参数
            kwargs = {}
            for param_name, param in parameters.items():
                if param_name == 'self':
                    continue
                
                # 如果参数有类型注解，尝试解析依赖
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation in self._services:
                        kwargs[param_name] = self.resolve(param.annotation)
                    elif param.default != inspect.Parameter.empty:
                        # 有默认值，跳过
                        continue
                    else:
                        logger_manager.warning(f"无法解析依赖: {param_name}: {param.annotation}")
            
            return implementation(**kwargs)
            
        except Exception as e:
            logger_manager.error(f"构造实例失败: {implementation.__name__}", e)
            # 尝试无参数构造
            try:
                return implementation()
            except:
                raise e
    
    def _inject_dependencies(self, func: Callable) -> Any:
        """为函数注入依赖"""
        try:
            signature = inspect.signature(func)
            parameters = signature.parameters
            
            kwargs = {}
            for param_name, param in parameters.items():
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation in self._services:
                        kwargs[param_name] = self.resolve(param.annotation)
            
            return func(**kwargs)
            
        except Exception as e:
            logger_manager.error(f"依赖注入失败: {func.__name__}", e)
            return func()
    
    def is_registered(self, service_type: Type) -> bool:
        """检查服务是否已注册"""
        return service_type in self._services
    
    def unregister(self, service_type: Type) -> bool:
        """注销服务"""
        with self._lock:
            if service_type in self._services:
                del self._services[service_type]
                
                # 清理实例缓存
                if service_type in self._singletons:
                    del self._singletons[service_type]
                if service_type in self._scoped_instances:
                    del self._scoped_instances[service_type]
                
                logger_manager.debug(f"注销服务: {service_type.__name__}")
                return True
            return False
    
    def clear_scoped(self) -> None:
        """清理作用域实例"""
        with self._lock:
            self._scoped_instances.clear()
            logger_manager.debug("清理作用域实例")
    
    def get_registered_services(self) -> Dict[Type, ServiceDescriptor]:
        """获取所有已注册的服务"""
        return self._services.copy()
    
    def get_service_info(self, service_type: Type) -> Optional[Dict[str, Any]]:
        """获取服务信息"""
        if service_type not in self._services:
            return None
        
        descriptor = self._services[service_type]
        return {
            'service_type': descriptor.service_type.__name__,
            'implementation': descriptor.implementation.__name__ if descriptor.implementation else None,
            'lifetime': descriptor.lifetime.value,
            'has_instance': descriptor.instance is not None,
            'has_factory': descriptor.factory is not None,
            'is_singleton_cached': service_type in self._singletons,
            'is_scoped_cached': service_type in self._scoped_instances
        }


# 全局容器实例
container = DependencyContainer()


def get_container() -> DependencyContainer:
    """获取全局依赖注入容器"""
    return container


# 装饰器支持
def injectable(lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT):
    """可注入服务装饰器"""
    def decorator(cls: Type) -> Type:
        container.register(cls, cls) if lifetime == ServiceLifetime.TRANSIENT else \
        container.register_singleton(cls, cls) if lifetime == ServiceLifetime.SINGLETON else \
        container.register_scoped(cls, cls)
        return cls
    return decorator


def inject(service_type: Type):
    """依赖注入装饰器"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            service = container.resolve(service_type)
            return func(service, *args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # 测试代码
    class IRepository:
        def get_data(self):
            pass
    
    class DatabaseRepository(IRepository):
        def get_data(self):
            return "Database data"
    
    class Service:
        def __init__(self, repo: IRepository):
            self.repo = repo
        
        def process(self):
            return f"Processing: {self.repo.get_data()}"
    
    # 注册服务
    container.register_singleton(IRepository, DatabaseRepository)
    container.register(Service)
    
    # 解析服务
    service = container.resolve(Service)
    print(service.process())
