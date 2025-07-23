#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
微服务管理器模块
Microservice Manager Module

提供微服务架构、服务注册发现、API网关等功能。
"""

import os
import time
import json
import threading
import requests
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
import socket
from urllib.parse import urljoin

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException


class ServiceStatus(Enum):
    """服务状态枚举"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"


@dataclass
class ServiceInstance:
    """服务实例"""
    instance_id: str
    service_name: str
    host: str
    port: int
    status: ServiceStatus
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    health_check_url: str = ""
    last_heartbeat: datetime = field(default_factory=datetime.now)
    registered_time: datetime = field(default_factory=datetime.now)


@dataclass
class ServiceRoute:
    """服务路由"""
    path: str
    service_name: str
    method: str = "GET"
    load_balancer: str = "round_robin"  # round_robin, random, least_connections
    timeout: int = 30
    retry_count: int = 3
    circuit_breaker: bool = False


class ServiceRegistry:
    """服务注册中心"""
    
    def __init__(self, cleanup_interval: int = 60):
        """
        初始化服务注册中心
        
        Args:
            cleanup_interval: 清理间隔（秒）
        """
        self.services = {}  # service_name -> List[ServiceInstance]
        self.cleanup_interval = cleanup_interval
        self.heartbeat_timeout = 30  # 心跳超时时间
        self.lock = threading.RLock()
        self.cleanup_thread = None
        self.running = False
        
        logger_manager.info("服务注册中心初始化完成")
    
    def register_service(self, instance: ServiceInstance) -> bool:
        """注册服务实例"""
        try:
            with self.lock:
                if instance.service_name not in self.services:
                    self.services[instance.service_name] = []
                
                # 检查是否已存在相同实例
                existing_instances = self.services[instance.service_name]
                for existing in existing_instances:
                    if (existing.host == instance.host and 
                        existing.port == instance.port):
                        # 更新现有实例
                        existing.status = instance.status
                        existing.version = instance.version
                        existing.metadata = instance.metadata
                        existing.last_heartbeat = datetime.now()
                        logger_manager.info(f"更新服务实例: {instance.service_name}@{instance.host}:{instance.port}")
                        return True
                
                # 添加新实例
                self.services[instance.service_name].append(instance)
                
                # 启动清理线程
                if not self.running:
                    self.start_cleanup()
                
                logger_manager.info(f"注册服务实例: {instance.service_name}@{instance.host}:{instance.port}")
                return True
                
        except Exception as e:
            logger_manager.error(f"注册服务实例失败: {e}")
            return False
    
    def deregister_service(self, service_name: str, host: str, port: int) -> bool:
        """注销服务实例"""
        try:
            with self.lock:
                if service_name not in self.services:
                    return False
                
                instances = self.services[service_name]
                for i, instance in enumerate(instances):
                    if instance.host == host and instance.port == port:
                        del instances[i]
                        logger_manager.info(f"注销服务实例: {service_name}@{host}:{port}")
                        
                        # 如果没有实例了，删除服务
                        if not instances:
                            del self.services[service_name]
                        
                        return True
                
                return False
                
        except Exception as e:
            logger_manager.error(f"注销服务实例失败: {e}")
            return False
    
    def discover_service(self, service_name: str) -> List[ServiceInstance]:
        """发现服务实例"""
        try:
            with self.lock:
                instances = self.services.get(service_name, [])
                # 只返回健康的实例
                healthy_instances = [
                    instance for instance in instances 
                    if instance.status == ServiceStatus.HEALTHY
                ]
                
                logger_manager.debug(f"发现服务实例: {service_name}, 健康实例数: {len(healthy_instances)}")
                return healthy_instances
                
        except Exception as e:
            logger_manager.error(f"发现服务实例失败: {e}")
            return []
    
    def heartbeat(self, service_name: str, host: str, port: int) -> bool:
        """服务心跳"""
        try:
            with self.lock:
                if service_name not in self.services:
                    return False
                
                for instance in self.services[service_name]:
                    if instance.host == host and instance.port == port:
                        instance.last_heartbeat = datetime.now()
                        instance.status = ServiceStatus.HEALTHY
                        logger_manager.debug(f"服务心跳: {service_name}@{host}:{port}")
                        return True
                
                return False
                
        except Exception as e:
            logger_manager.error(f"服务心跳失败: {e}")
            return False
    
    def start_cleanup(self):
        """启动清理线程"""
        if self.running:
            return
        
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop)
        self.cleanup_thread.daemon = True
        self.cleanup_thread.start()
        
        logger_manager.info("服务清理线程已启动")
    
    def stop_cleanup(self):
        """停止清理线程"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        
        logger_manager.info("服务清理线程已停止")
    
    def _cleanup_loop(self):
        """清理循环"""
        while self.running:
            try:
                self._cleanup_unhealthy_instances()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                logger_manager.error(f"服务清理失败: {e}")
                time.sleep(self.cleanup_interval)
    
    def _cleanup_unhealthy_instances(self):
        """清理不健康的实例"""
        try:
            with self.lock:
                current_time = datetime.now()
                services_to_remove = []
                
                for service_name, instances in self.services.items():
                    instances_to_remove = []
                    
                    for i, instance in enumerate(instances):
                        # 检查心跳超时
                        time_since_heartbeat = (current_time - instance.last_heartbeat).total_seconds()
                        
                        if time_since_heartbeat > self.heartbeat_timeout:
                            instance.status = ServiceStatus.UNHEALTHY
                            instances_to_remove.append(i)
                            logger_manager.warning(f"服务实例心跳超时: {service_name}@{instance.host}:{instance.port}")
                    
                    # 移除不健康的实例
                    for i in reversed(instances_to_remove):
                        del instances[i]
                    
                    # 如果没有实例了，标记服务待删除
                    if not instances:
                        services_to_remove.append(service_name)
                
                # 删除空服务
                for service_name in services_to_remove:
                    del self.services[service_name]
                    logger_manager.info(f"删除空服务: {service_name}")
                    
        except Exception as e:
            logger_manager.error(f"清理不健康实例失败: {e}")
    
    def get_service_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        try:
            with self.lock:
                stats = {
                    'total_services': len(self.services),
                    'total_instances': sum(len(instances) for instances in self.services.values()),
                    'healthy_instances': 0,
                    'unhealthy_instances': 0,
                    'services': {}
                }
                
                for service_name, instances in self.services.items():
                    healthy_count = sum(1 for instance in instances if instance.status == ServiceStatus.HEALTHY)
                    unhealthy_count = len(instances) - healthy_count
                    
                    stats['healthy_instances'] += healthy_count
                    stats['unhealthy_instances'] += unhealthy_count
                    
                    stats['services'][service_name] = {
                        'total_instances': len(instances),
                        'healthy_instances': healthy_count,
                        'unhealthy_instances': unhealthy_count
                    }
                
                return stats
                
        except Exception as e:
            logger_manager.error(f"获取服务统计信息失败: {e}")
            return {}


class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self):
        """初始化负载均衡器"""
        self.round_robin_counters = {}
        self.lock = threading.RLock()
        
        logger_manager.info("负载均衡器初始化完成")
    
    def select_instance(self, instances: List[ServiceInstance], 
                       strategy: str = "round_robin") -> Optional[ServiceInstance]:
        """选择服务实例"""
        try:
            if not instances:
                return None
            
            if strategy == "round_robin":
                return self._round_robin_select(instances)
            elif strategy == "random":
                return self._random_select(instances)
            elif strategy == "least_connections":
                return self._least_connections_select(instances)
            else:
                logger_manager.warning(f"未知的负载均衡策略: {strategy}，使用轮询")
                return self._round_robin_select(instances)
                
        except Exception as e:
            logger_manager.error(f"选择服务实例失败: {e}")
            return instances[0] if instances else None
    
    def _round_robin_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """轮询选择"""
        with self.lock:
            service_name = instances[0].service_name
            
            if service_name not in self.round_robin_counters:
                self.round_robin_counters[service_name] = 0
            
            index = self.round_robin_counters[service_name] % len(instances)
            self.round_robin_counters[service_name] += 1
            
            return instances[index]
    
    def _random_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """随机选择"""
        import random
        return random.choice(instances)
    
    def _least_connections_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """最少连接选择（简化实现）"""
        # 简化实现，实际应该跟踪每个实例的连接数
        return min(instances, key=lambda x: x.metadata.get('connections', 0))


class APIGateway:
    """API网关"""
    
    def __init__(self, service_registry: ServiceRegistry, port: int = 8080):
        """
        初始化API网关
        
        Args:
            service_registry: 服务注册中心
            port: 网关端口
        """
        self.service_registry = service_registry
        self.port = port
        self.routes = {}  # path -> ServiceRoute
        self.load_balancer = LoadBalancer()
        self.running = False
        self.server_thread = None
        
        logger_manager.info(f"API网关初始化完成，端口: {port}")
    
    def add_route(self, route: ServiceRoute):
        """添加路由"""
        self.routes[route.path] = route
        logger_manager.info(f"添加路由: {route.path} -> {route.service_name}")
    
    def remove_route(self, path: str):
        """移除路由"""
        if path in self.routes:
            del self.routes[path]
            logger_manager.info(f"移除路由: {path}")
    
    def start_gateway(self):
        """启动网关"""
        if self.running:
            return
        
        self.running = True
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        logger_manager.info(f"API网关已启动，监听端口: {self.port}")
    
    def stop_gateway(self):
        """停止网关"""
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=5)
        
        logger_manager.info("API网关已停止")
    
    def _run_server(self):
        """运行服务器（简化实现）"""
        try:
            # 这里应该使用真正的HTTP服务器，如Flask、FastAPI等
            # 简化实现只是一个占位符
            while self.running:
                time.sleep(1)
                
        except Exception as e:
            logger_manager.error(f"API网关服务器运行失败: {e}")
    
    def route_request(self, path: str, method: str = "GET") -> Optional[str]:
        """路由请求"""
        try:
            # 查找匹配的路由
            route = self._find_route(path, method)
            if not route:
                logger_manager.warning(f"未找到路由: {path}")
                return None
            
            # 发现服务实例
            instances = self.service_registry.discover_service(route.service_name)
            if not instances:
                logger_manager.warning(f"未找到服务实例: {route.service_name}")
                return None
            
            # 选择实例
            instance = self.load_balancer.select_instance(instances, route.load_balancer)
            if not instance:
                logger_manager.warning(f"无法选择服务实例: {route.service_name}")
                return None
            
            # 构建目标URL
            target_url = f"http://{instance.host}:{instance.port}{path}"
            
            logger_manager.debug(f"路由请求: {path} -> {target_url}")
            return target_url
            
        except Exception as e:
            logger_manager.error(f"路由请求失败: {e}")
            return None
    
    def _find_route(self, path: str, method: str) -> Optional[ServiceRoute]:
        """查找路由"""
        # 精确匹配
        if path in self.routes:
            route = self.routes[path]
            if route.method == method or route.method == "*":
                return route
        
        # 前缀匹配
        for route_path, route in self.routes.items():
            if path.startswith(route_path) and (route.method == method or route.method == "*"):
                return route
        
        return None
    
    def proxy_request(self, path: str, method: str = "GET", 
                     headers: Dict[str, str] = None, data: Any = None) -> Tuple[int, str]:
        """代理请求"""
        try:
            target_url = self.route_request(path, method)
            if not target_url:
                return 404, "Service not found"
            
            # 发送请求
            response = requests.request(
                method=method,
                url=target_url,
                headers=headers,
                json=data if method in ['POST', 'PUT', 'PATCH'] else None,
                timeout=30
            )
            
            return response.status_code, response.text
            
        except requests.RequestException as e:
            logger_manager.error(f"代理请求失败: {e}")
            return 500, f"Proxy error: {e}"
        except Exception as e:
            logger_manager.error(f"代理请求失败: {e}")
            return 500, f"Internal error: {e}"


class MicroserviceManager:
    """微服务管理器"""
    
    def __init__(self, gateway_port: int = 8080):
        """
        初始化微服务管理器
        
        Args:
            gateway_port: API网关端口
        """
        self.service_registry = ServiceRegistry()
        self.api_gateway = APIGateway(self.service_registry, gateway_port)
        self.services = {}
        
        logger_manager.info("微服务管理器初始化完成")
    
    def register_service(self, service_name: str, host: str, port: int,
                        version: str = "1.0.0", metadata: Dict[str, Any] = None) -> str:
        """注册服务"""
        try:
            instance_id = str(uuid.uuid4())
            
            instance = ServiceInstance(
                instance_id=instance_id,
                service_name=service_name,
                host=host,
                port=port,
                status=ServiceStatus.HEALTHY,
                version=version,
                metadata=metadata or {},
                health_check_url=f"http://{host}:{port}/health"
            )
            
            success = self.service_registry.register_service(instance)
            
            if success:
                self.services[instance_id] = instance
                logger_manager.info(f"服务注册成功: {service_name}@{host}:{port}")
                return instance_id
            else:
                return ""
                
        except Exception as e:
            logger_manager.error(f"注册服务失败: {e}")
            return ""
    
    def deregister_service(self, instance_id: str) -> bool:
        """注销服务"""
        try:
            if instance_id not in self.services:
                return False
            
            instance = self.services[instance_id]
            success = self.service_registry.deregister_service(
                instance.service_name, instance.host, instance.port
            )
            
            if success:
                del self.services[instance_id]
                logger_manager.info(f"服务注销成功: {instance_id}")
                return True
            else:
                return False
                
        except Exception as e:
            logger_manager.error(f"注销服务失败: {e}")
            return False
    
    def add_service_route(self, path: str, service_name: str, method: str = "GET"):
        """添加服务路由"""
        route = ServiceRoute(
            path=path,
            service_name=service_name,
            method=method
        )
        
        self.api_gateway.add_route(route)
        logger_manager.info(f"添加服务路由: {path} -> {service_name}")
    
    def start_gateway(self):
        """启动API网关"""
        self.api_gateway.start_gateway()
    
    def stop_gateway(self):
        """停止API网关"""
        self.api_gateway.stop_gateway()
    
    def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """获取服务健康状态"""
        try:
            instances = self.service_registry.discover_service(service_name)
            
            health_info = {
                'service_name': service_name,
                'total_instances': len(instances),
                'healthy_instances': len([i for i in instances if i.status == ServiceStatus.HEALTHY]),
                'instances': []
            }
            
            for instance in instances:
                instance_info = {
                    'instance_id': instance.instance_id,
                    'host': instance.host,
                    'port': instance.port,
                    'status': instance.status.value,
                    'version': instance.version,
                    'last_heartbeat': instance.last_heartbeat.isoformat()
                }
                health_info['instances'].append(instance_info)
            
            return health_info
            
        except Exception as e:
            logger_manager.error(f"获取服务健康状态失败: {e}")
            return {}
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """获取集群状态"""
        try:
            registry_stats = self.service_registry.get_service_stats()
            
            cluster_status = {
                'registry_stats': registry_stats,
                'gateway_running': self.api_gateway.running,
                'gateway_port': self.api_gateway.port,
                'total_routes': len(self.api_gateway.routes),
                'services': {}
            }
            
            # 获取每个服务的详细信息
            for service_name in registry_stats.get('services', {}):
                cluster_status['services'][service_name] = self.get_service_health(service_name)
            
            return cluster_status
            
        except Exception as e:
            logger_manager.error(f"获取集群状态失败: {e}")
            return {}


# 全局微服务管理器实例
microservice_manager = MicroserviceManager()


if __name__ == "__main__":
    # 测试微服务管理功能
    print("🔧 测试微服务管理功能...")
    
    try:
        manager = MicroserviceManager()
        
        # 测试服务注册
        instance_id = manager.register_service(
            service_name="test-service",
            host="localhost",
            port=8001,
            version="1.0.0",
            metadata={"environment": "test"}
        )
        
        if instance_id:
            print("✅ 服务注册成功")
            
            # 测试服务发现
            instances = manager.service_registry.discover_service("test-service")
            if instances:
                print(f"✅ 服务发现成功: {len(instances)} 个实例")
            
            # 测试添加路由
            manager.add_service_route("/api/test", "test-service", "GET")
            print("✅ 服务路由添加成功")
            
            # 测试服务健康状态
            health = manager.get_service_health("test-service")
            if health:
                print("✅ 服务健康状态获取成功")
            
            # 测试集群状态
            cluster_status = manager.get_cluster_status()
            if cluster_status:
                print("✅ 集群状态获取成功")
            
            # 测试服务注销
            if manager.deregister_service(instance_id):
                print("✅ 服务注销成功")
        
        print("✅ 微服务管理功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("微服务管理功能测试完成")
