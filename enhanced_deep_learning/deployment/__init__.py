"""
部署模块
Deployment Module

提供容器化部署、云平台集成、微服务架构等功能。
"""

from .docker_manager import (
    DockerManager, DockerImage, DockerContainer, ContainerStatus,
    docker_manager
)
from .kubernetes_manager import (
    KubernetesManager, KubernetesDeployment, KubernetesService,
    kubernetes_manager
)
from .cloud_integration import (
    CloudProvider, AWSIntegration, AzureIntegration, GCPIntegration,
    cloud_manager
)
from .microservice_manager import (
    MicroserviceManager, ServiceRegistry, APIGateway,
    microservice_manager
)

__all__ = [
    # Docker容器化
    'DockerManager',
    'DockerImage',
    'DockerContainer',
    'ContainerStatus',
    'docker_manager',
    
    # Kubernetes编排
    'KubernetesManager',
    'KubernetesDeployment',
    'KubernetesService',
    'kubernetes_manager',
    
    # 云平台集成
    'CloudProvider',
    'AWSIntegration',
    'AzureIntegration',
    'GCPIntegration',
    'cloud_manager',
    
    # 微服务架构
    'MicroserviceManager',
    'ServiceRegistry',
    'APIGateway',
    'microservice_manager'
]
