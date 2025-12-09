#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kubernetes管理器模块
Kubernetes Manager Module

提供Kubernetes集群管理、部署编排、服务发现等功能。
"""

import os
import time
import json
import subprocess
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
# 尝试导入yaml，如果不可用则跳过
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException


class DeploymentStatus(Enum):
    """部署状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class KubernetesDeployment:
    """Kubernetes部署"""
    name: str
    namespace: str
    image: str
    replicas: int
    status: DeploymentStatus
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    ports: List[int] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    created: datetime = field(default_factory=datetime.now)


@dataclass
class KubernetesService:
    """Kubernetes服务"""
    name: str
    namespace: str
    service_type: str  # ClusterIP, NodePort, LoadBalancer
    selector: Dict[str, str]
    ports: List[Dict[str, Any]]
    cluster_ip: Optional[str] = None
    external_ip: Optional[str] = None
    created: datetime = field(default_factory=datetime.now)


class KubernetesManifestGenerator:
    """Kubernetes清单生成器"""
    
    def __init__(self):
        """初始化Kubernetes清单生成器"""
        logger_manager.info("Kubernetes清单生成器初始化完成")
    
    def generate_deployment_manifest(self, deployment: KubernetesDeployment) -> Dict[str, Any]:
        """生成部署清单"""
        try:
            manifest = {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'metadata': {
                    'name': deployment.name,
                    'namespace': deployment.namespace,
                    'labels': deployment.labels,
                    'annotations': deployment.annotations
                },
                'spec': {
                    'replicas': deployment.replicas,
                    'selector': {
                        'matchLabels': {
                            'app': deployment.name
                        }
                    },
                    'template': {
                        'metadata': {
                            'labels': {
                                'app': deployment.name,
                                **deployment.labels
                            }
                        },
                        'spec': {
                            'containers': [{
                                'name': deployment.name,
                                'image': deployment.image,
                                'ports': [{'containerPort': port} for port in deployment.ports],
                                'env': [{'name': k, 'value': v} for k, v in deployment.environment.items()]
                            }]
                        }
                    }
                }
            }
            
            # 添加资源限制
            if deployment.resources:
                manifest['spec']['template']['spec']['containers'][0]['resources'] = deployment.resources
            
            logger_manager.debug(f"生成部署清单: {deployment.name}")
            return manifest
            
        except Exception as e:
            logger_manager.error(f"生成部署清单失败: {e}")
            return {}
    
    def generate_service_manifest(self, service: KubernetesService) -> Dict[str, Any]:
        """生成服务清单"""
        try:
            manifest = {
                'apiVersion': 'v1',
                'kind': 'Service',
                'metadata': {
                    'name': service.name,
                    'namespace': service.namespace
                },
                'spec': {
                    'selector': service.selector,
                    'ports': service.ports,
                    'type': service.service_type
                }
            }
            
            logger_manager.debug(f"生成服务清单: {service.name}")
            return manifest
            
        except Exception as e:
            logger_manager.error(f"生成服务清单失败: {e}")
            return {}
    
    def generate_configmap_manifest(self, name: str, namespace: str, 
                                  data: Dict[str, str]) -> Dict[str, Any]:
        """生成ConfigMap清单"""
        try:
            manifest = {
                'apiVersion': 'v1',
                'kind': 'ConfigMap',
                'metadata': {
                    'name': name,
                    'namespace': namespace
                },
                'data': data
            }
            
            logger_manager.debug(f"生成ConfigMap清单: {name}")
            return manifest
            
        except Exception as e:
            logger_manager.error(f"生成ConfigMap清单失败: {e}")
            return {}
    
    def generate_secret_manifest(self, name: str, namespace: str,
                                data: Dict[str, str], secret_type: str = "Opaque") -> Dict[str, Any]:
        """生成Secret清单"""
        try:
            import base64
            
            # 对数据进行base64编码
            encoded_data = {}
            for key, value in data.items():
                encoded_data[key] = base64.b64encode(value.encode()).decode()
            
            manifest = {
                'apiVersion': 'v1',
                'kind': 'Secret',
                'metadata': {
                    'name': name,
                    'namespace': namespace
                },
                'type': secret_type,
                'data': encoded_data
            }
            
            logger_manager.debug(f"生成Secret清单: {name}")
            return manifest
            
        except Exception as e:
            logger_manager.error(f"生成Secret清单失败: {e}")
            return {}
    
    def generate_ingress_manifest(self, name: str, namespace: str,
                                 rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成Ingress清单"""
        try:
            manifest = {
                'apiVersion': 'networking.k8s.io/v1',
                'kind': 'Ingress',
                'metadata': {
                    'name': name,
                    'namespace': namespace,
                    'annotations': {
                        'nginx.ingress.kubernetes.io/rewrite-target': '/'
                    }
                },
                'spec': {
                    'rules': rules
                }
            }
            
            logger_manager.debug(f"生成Ingress清单: {name}")
            return manifest
            
        except Exception as e:
            logger_manager.error(f"生成Ingress清单失败: {e}")
            return {}


class KubernetesManager:
    """Kubernetes管理器"""
    
    def __init__(self):
        """初始化Kubernetes管理器"""
        self.manifest_generator = KubernetesManifestGenerator()
        self.deployments = {}
        self.services = {}
        self.kubectl_available = self._check_kubectl_availability()
        
        logger_manager.info(f"Kubernetes管理器初始化完成，kubectl可用: {self.kubectl_available}")
    
    def _check_kubectl_availability(self) -> bool:
        """检查kubectl是否可用"""
        try:
            result = subprocess.run(['kubectl', 'version', '--client'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger_manager.warning("kubectl不可用")
            return False
    
    def _run_kubectl_command(self, command: List[str]) -> Tuple[bool, str]:
        """运行kubectl命令"""
        try:
            if not self.kubectl_available:
                return False, "kubectl不可用"
            
            result = subprocess.run(['kubectl'] + command, 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                return False, result.stderr.strip()
                
        except subprocess.TimeoutExpired:
            return False, "命令执行超时"
        except Exception as e:
            return False, str(e)
    
    def apply_manifest(self, manifest: Dict[str, Any], namespace: str = "default") -> bool:
        """应用Kubernetes清单"""
        try:
            # 将清单写入临时文件
            import tempfile

            if not YAML_AVAILABLE:
                logger_manager.error("PyYAML未安装，无法应用Kubernetes清单")
                return False

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(manifest, f, default_flow_style=False)
                temp_file = f.name
            
            try:
                # 应用清单
                command = ['apply', '-f', temp_file, '-n', namespace]
                success, output = self._run_kubectl_command(command)
                
                if success:
                    logger_manager.info(f"清单应用成功: {manifest.get('metadata', {}).get('name', 'unknown')}")
                    return True
                else:
                    logger_manager.error(f"清单应用失败: {output}")
                    return False
                    
            finally:
                # 清理临时文件
                os.unlink(temp_file)
                
        except Exception as e:
            logger_manager.error(f"应用清单失败: {e}")
            return False
    
    def create_deployment(self, deployment: KubernetesDeployment) -> bool:
        """创建部署"""
        try:
            manifest = self.manifest_generator.generate_deployment_manifest(deployment)
            if not manifest:
                return False
            
            success = self.apply_manifest(manifest, deployment.namespace)
            
            if success:
                self.deployments[f"{deployment.namespace}/{deployment.name}"] = deployment
                logger_manager.info(f"部署创建成功: {deployment.name}")
                return True
            else:
                return False
                
        except Exception as e:
            logger_manager.error(f"创建部署失败: {e}")
            return False
    
    def create_service(self, service: KubernetesService) -> bool:
        """创建服务"""
        try:
            manifest = self.manifest_generator.generate_service_manifest(service)
            if not manifest:
                return False
            
            success = self.apply_manifest(manifest, service.namespace)
            
            if success:
                self.services[f"{service.namespace}/{service.name}"] = service
                logger_manager.info(f"服务创建成功: {service.name}")
                return True
            else:
                return False
                
        except Exception as e:
            logger_manager.error(f"创建服务失败: {e}")
            return False
    
    def delete_deployment(self, name: str, namespace: str = "default") -> bool:
        """删除部署"""
        try:
            command = ['delete', 'deployment', name, '-n', namespace]
            success, output = self._run_kubectl_command(command)
            
            if success:
                deployment_key = f"{namespace}/{name}"
                if deployment_key in self.deployments:
                    del self.deployments[deployment_key]
                
                logger_manager.info(f"部署删除成功: {name}")
                return True
            else:
                logger_manager.error(f"部署删除失败: {output}")
                return False
                
        except Exception as e:
            logger_manager.error(f"删除部署失败: {e}")
            return False
    
    def delete_service(self, name: str, namespace: str = "default") -> bool:
        """删除服务"""
        try:
            command = ['delete', 'service', name, '-n', namespace]
            success, output = self._run_kubectl_command(command)
            
            if success:
                service_key = f"{namespace}/{name}"
                if service_key in self.services:
                    del self.services[service_key]
                
                logger_manager.info(f"服务删除成功: {name}")
                return True
            else:
                logger_manager.error(f"服务删除失败: {output}")
                return False
                
        except Exception as e:
            logger_manager.error(f"删除服务失败: {e}")
            return False
    
    def get_deployment_status(self, name: str, namespace: str = "default") -> Optional[DeploymentStatus]:
        """获取部署状态"""
        try:
            command = ['get', 'deployment', name, '-n', namespace, '-o', 'json']
            success, output = self._run_kubectl_command(command)
            
            if success:
                deployment_info = json.loads(output)
                status = deployment_info.get('status', {})
                
                # 判断部署状态
                ready_replicas = status.get('readyReplicas', 0)
                replicas = status.get('replicas', 0)
                
                if ready_replicas == replicas and replicas > 0:
                    return DeploymentStatus.RUNNING
                elif ready_replicas > 0:
                    return DeploymentStatus.PENDING
                else:
                    return DeploymentStatus.FAILED
            else:
                return DeploymentStatus.UNKNOWN
                
        except Exception as e:
            logger_manager.error(f"获取部署状态失败: {e}")
            return DeploymentStatus.UNKNOWN
    
    def scale_deployment(self, name: str, replicas: int, namespace: str = "default") -> bool:
        """扩缩容部署"""
        try:
            command = ['scale', 'deployment', name, f'--replicas={replicas}', '-n', namespace]
            success, output = self._run_kubectl_command(command)
            
            if success:
                # 更新本地记录
                deployment_key = f"{namespace}/{name}"
                if deployment_key in self.deployments:
                    self.deployments[deployment_key].replicas = replicas
                
                logger_manager.info(f"部署扩缩容成功: {name} -> {replicas} 副本")
                return True
            else:
                logger_manager.error(f"部署扩缩容失败: {output}")
                return False
                
        except Exception as e:
            logger_manager.error(f"扩缩容部署失败: {e}")
            return False
    
    def get_pod_logs(self, deployment_name: str, namespace: str = "default", 
                    tail: int = 100) -> str:
        """获取Pod日志"""
        try:
            # 首先获取Pod名称
            command = ['get', 'pods', '-l', f'app={deployment_name}', '-n', namespace, '-o', 'name']
            success, output = self._run_kubectl_command(command)
            
            if not success or not output:
                return "无法找到相关Pod"
            
            pod_name = output.split('\n')[0].replace('pod/', '')
            
            # 获取日志
            command = ['logs', pod_name, '-n', namespace, '--tail', str(tail)]
            success, output = self._run_kubectl_command(command)
            
            if success:
                return output
            else:
                return f"获取日志失败: {output}"
                
        except Exception as e:
            logger_manager.error(f"获取Pod日志失败: {e}")
            return f"获取日志失败: {e}"
    
    def list_deployments(self, namespace: str = "default") -> List[KubernetesDeployment]:
        """列出部署"""
        try:
            command = ['get', 'deployments', '-n', namespace, '-o', 'json']
            success, output = self._run_kubectl_command(command)
            
            if success:
                deployments_info = json.loads(output)
                deployments = []
                
                for item in deployments_info.get('items', []):
                    metadata = item.get('metadata', {})
                    spec = item.get('spec', {})
                    status = item.get('status', {})
                    
                    deployment = KubernetesDeployment(
                        name=metadata.get('name', ''),
                        namespace=metadata.get('namespace', ''),
                        image='',  # 需要从容器规格中提取
                        replicas=spec.get('replicas', 0),
                        status=DeploymentStatus.UNKNOWN,  # 需要根据状态判断
                        labels=metadata.get('labels', {}),
                        annotations=metadata.get('annotations', {})
                    )
                    
                    deployments.append(deployment)
                
                return deployments
            else:
                return []
                
        except Exception as e:
            logger_manager.error(f"列出部署失败: {e}")
            return []
    
    def deploy_application(self, app_name: str, image: str, port: int = 8000,
                          replicas: int = 1, namespace: str = "default") -> bool:
        """部署应用"""
        try:
            # 1. 创建部署
            deployment = KubernetesDeployment(
                name=app_name,
                namespace=namespace,
                image=image,
                replicas=replicas,
                status=DeploymentStatus.PENDING,
                labels={'app': app_name},
                ports=[port],
                resources={
                    'requests': {'memory': '256Mi', 'cpu': '250m'},
                    'limits': {'memory': '512Mi', 'cpu': '500m'}
                }
            )
            
            if not self.create_deployment(deployment):
                return False
            
            # 2. 创建服务
            service = KubernetesService(
                name=f"{app_name}-service",
                namespace=namespace,
                service_type="ClusterIP",
                selector={'app': app_name},
                ports=[{
                    'port': 80,
                    'targetPort': port,
                    'protocol': 'TCP'
                }]
            )
            
            if not self.create_service(service):
                return False
            
            logger_manager.info(f"应用部署成功: {app_name}")
            return True
            
        except Exception as e:
            logger_manager.error(f"部署应用失败: {e}")
            return False
    
    def create_namespace(self, namespace: str) -> bool:
        """创建命名空间"""
        try:
            manifest = {
                'apiVersion': 'v1',
                'kind': 'Namespace',
                'metadata': {
                    'name': namespace
                }
            }
            
            success = self.apply_manifest(manifest)
            
            if success:
                logger_manager.info(f"命名空间创建成功: {namespace}")
                return True
            else:
                return False
                
        except Exception as e:
            logger_manager.error(f"创建命名空间失败: {e}")
            return False


# 全局Kubernetes管理器实例
kubernetes_manager = KubernetesManager()


if __name__ == "__main__":
    # 测试Kubernetes管理功能
    print("☸️ 测试Kubernetes管理功能...")
    
    try:
        manager = KubernetesManager()
        
        if manager.kubectl_available:
            print("✅ kubectl可用")
            
            # 测试清单生成
            deployment = KubernetesDeployment(
                name="test-app",
                namespace="default",
                image="nginx:latest",
                replicas=2,
                status=DeploymentStatus.PENDING,
                ports=[80]
            )
            
            manifest = manager.manifest_generator.generate_deployment_manifest(deployment)
            if manifest and manifest.get('kind') == 'Deployment':
                print("✅ 部署清单生成成功")
            
            # 测试部署列表
            deployments = manager.list_deployments()
            print(f"✅ 部署列表获取成功，发现 {len(deployments)} 个部署")
            
        else:
            print("⚠️ kubectl不可用，跳过Kubernetes操作测试")
        
        print("✅ Kubernetes管理功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("Kubernetes管理功能测试完成")
