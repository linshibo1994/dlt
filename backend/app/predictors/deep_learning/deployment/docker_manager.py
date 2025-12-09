#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Docker管理器模块
Docker Manager Module

提供Docker容器化、镜像管理、容器编排等功能。
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
import shutil

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException


class ContainerStatus(Enum):
    """容器状态枚举"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    RESTARTING = "restarting"
    REMOVING = "removing"
    EXITED = "exited"
    DEAD = "dead"


@dataclass
class DockerImage:
    """Docker镜像"""
    image_id: str
    repository: str
    tag: str
    size: int
    created: datetime
    dockerfile_path: Optional[str] = None
    build_args: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class DockerContainer:
    """Docker容器"""
    container_id: str
    name: str
    image: str
    status: ContainerStatus
    ports: Dict[str, str] = field(default_factory=dict)
    volumes: Dict[str, str] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    created: datetime = field(default_factory=datetime.now)
    started: Optional[datetime] = None
    finished: Optional[datetime] = None


class DockerfileGenerator:
    """Dockerfile生成器"""
    
    def __init__(self):
        """初始化Dockerfile生成器"""
        self.base_images = {
            'python': 'python:3.9-slim',
            'tensorflow': 'tensorflow/tensorflow:2.12.0',
            'pytorch': 'pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime',
            'ubuntu': 'ubuntu:20.04'
        }
        
        logger_manager.info("Dockerfile生成器初始化完成")
    
    def generate_python_dockerfile(self, requirements_file: str = "requirements.txt",
                                 app_file: str = "main.py",
                                 port: int = 8000) -> str:
        """生成Python应用的Dockerfile"""
        try:
            dockerfile_content = f"""
# 使用官方Python运行时作为基础镜像
FROM {self.base_images['python']}

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY {requirements_file} .

# 安装Python依赖
RUN pip install --no-cache-dir -r {requirements_file}

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE {port}

# 创建非root用户
RUN adduser --disabled-password --gecos '' appuser
RUN chown -R appuser:appuser /app
USER appuser

# 启动命令
CMD ["python", "{app_file}"]
"""
            
            logger_manager.info("生成Python Dockerfile完成")
            return dockerfile_content.strip()
            
        except Exception as e:
            logger_manager.error(f"生成Python Dockerfile失败: {e}")
            return ""
    
    def generate_tensorflow_dockerfile(self, model_path: str = "model",
                                     port: int = 8501) -> str:
        """生成TensorFlow Serving的Dockerfile"""
        try:
            dockerfile_content = f"""
# 使用TensorFlow Serving镜像
FROM {self.base_images['tensorflow']}

# 设置环境变量
ENV MODEL_NAME=lottery_model
ENV MODEL_BASE_PATH=/models

# 创建模型目录
RUN mkdir -p $MODEL_BASE_PATH/$MODEL_NAME

# 复制模型文件
COPY {model_path} $MODEL_BASE_PATH/$MODEL_NAME/1/

# 暴露端口
EXPOSE {port}

# 启动TensorFlow Serving
CMD ["tensorflow_model_server", "--rest_api_port={port}", "--model_name=$MODEL_NAME", "--model_base_path=$MODEL_BASE_PATH"]
"""
            
            logger_manager.info("生成TensorFlow Dockerfile完成")
            return dockerfile_content.strip()
            
        except Exception as e:
            logger_manager.error(f"生成TensorFlow Dockerfile失败: {e}")
            return ""
    
    def generate_multi_stage_dockerfile(self, build_stage: str = "builder",
                                      runtime_stage: str = "runtime") -> str:
        """生成多阶段构建的Dockerfile"""
        try:
            dockerfile_content = f"""
# 构建阶段
FROM {self.base_images['python']} AS {build_stage}

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    make \\
    && rm -rf /var/lib/apt/lists/*

# 复制并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源代码
COPY . .

# 运行阶段
FROM {self.base_images['python']} AS {runtime_stage}

WORKDIR /app

# 从构建阶段复制已安装的包
COPY --from={build_stage} /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from={build_stage} /usr/local/bin /usr/local/bin

# 复制应用代码
COPY --from={build_stage} /app .

# 创建非root用户
RUN adduser --disabled-password --gecos '' appuser
RUN chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "main.py"]
"""
            
            logger_manager.info("生成多阶段Dockerfile完成")
            return dockerfile_content.strip()
            
        except Exception as e:
            logger_manager.error(f"生成多阶段Dockerfile失败: {e}")
            return ""


class DockerManager:
    """Docker管理器"""
    
    def __init__(self):
        """初始化Docker管理器"""
        self.dockerfile_generator = DockerfileGenerator()
        self.images = {}
        self.containers = {}
        self.docker_available = self._check_docker_availability()
        
        logger_manager.info(f"Docker管理器初始化完成，Docker可用: {self.docker_available}")
    
    def _check_docker_availability(self) -> bool:
        """检查Docker是否可用"""
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger_manager.warning("Docker不可用")
            return False
    
    def _run_docker_command(self, command: List[str]) -> Tuple[bool, str]:
        """运行Docker命令"""
        try:
            if not self.docker_available:
                return False, "Docker不可用"
            
            result = subprocess.run(['docker'] + command, 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                return False, result.stderr.strip()
                
        except subprocess.TimeoutExpired:
            return False, "命令执行超时"
        except Exception as e:
            return False, str(e)
    
    def build_image(self, dockerfile_path: str, image_name: str, 
                   tag: str = "latest", build_args: Dict[str, str] = None) -> bool:
        """构建Docker镜像"""
        try:
            if not os.path.exists(dockerfile_path):
                logger_manager.error(f"Dockerfile不存在: {dockerfile_path}")
                return False
            
            # 构建命令
            command = ['build', '-t', f'{image_name}:{tag}']
            
            # 添加构建参数
            if build_args:
                for key, value in build_args.items():
                    command.extend(['--build-arg', f'{key}={value}'])
            
            # 添加构建上下文
            build_context = os.path.dirname(dockerfile_path)
            command.append(build_context)
            
            logger_manager.info(f"开始构建镜像: {image_name}:{tag}")
            
            success, output = self._run_docker_command(command)
            
            if success:
                # 记录镜像信息
                image_id = self._get_image_id(f'{image_name}:{tag}')
                if image_id:
                    image = DockerImage(
                        image_id=image_id,
                        repository=image_name,
                        tag=tag,
                        size=0,  # 需要单独获取
                        created=datetime.now(),
                        dockerfile_path=dockerfile_path,
                        build_args=build_args or {}
                    )
                    self.images[f'{image_name}:{tag}'] = image
                
                logger_manager.info(f"镜像构建成功: {image_name}:{tag}")
                return True
            else:
                logger_manager.error(f"镜像构建失败: {output}")
                return False
                
        except Exception as e:
            logger_manager.error(f"构建镜像失败: {e}")
            return False
    
    def _get_image_id(self, image_name: str) -> Optional[str]:
        """获取镜像ID"""
        try:
            success, output = self._run_docker_command(['images', '-q', image_name])
            if success and output:
                return output.split('\n')[0]
            return None
        except Exception as e:
            logger_manager.error(f"获取镜像ID失败: {e}")
            return None
    
    def create_container(self, image_name: str, container_name: str,
                        ports: Dict[str, str] = None,
                        volumes: Dict[str, str] = None,
                        environment: Dict[str, str] = None,
                        detach: bool = True) -> Optional[str]:
        """创建容器"""
        try:
            command = ['create', '--name', container_name]
            
            # 添加端口映射
            if ports:
                for host_port, container_port in ports.items():
                    command.extend(['-p', f'{host_port}:{container_port}'])
            
            # 添加卷挂载
            if volumes:
                for host_path, container_path in volumes.items():
                    command.extend(['-v', f'{host_path}:{container_path}'])
            
            # 添加环境变量
            if environment:
                for key, value in environment.items():
                    command.extend(['-e', f'{key}={value}'])
            
            # 添加分离模式
            if detach:
                command.append('-d')
            
            command.append(image_name)
            
            success, output = self._run_docker_command(command)
            
            if success:
                container_id = output.strip()
                
                # 记录容器信息
                container = DockerContainer(
                    container_id=container_id,
                    name=container_name,
                    image=image_name,
                    status=ContainerStatus.CREATED,
                    ports=ports or {},
                    volumes=volumes or {},
                    environment=environment or {}
                )
                
                self.containers[container_name] = container
                
                logger_manager.info(f"容器创建成功: {container_name}")
                return container_id
            else:
                logger_manager.error(f"容器创建失败: {output}")
                return None
                
        except Exception as e:
            logger_manager.error(f"创建容器失败: {e}")
            return None
    
    def start_container(self, container_name: str) -> bool:
        """启动容器"""
        try:
            success, output = self._run_docker_command(['start', container_name])
            
            if success:
                if container_name in self.containers:
                    self.containers[container_name].status = ContainerStatus.RUNNING
                    self.containers[container_name].started = datetime.now()
                
                logger_manager.info(f"容器启动成功: {container_name}")
                return True
            else:
                logger_manager.error(f"容器启动失败: {output}")
                return False
                
        except Exception as e:
            logger_manager.error(f"启动容器失败: {e}")
            return False
    
    def stop_container(self, container_name: str) -> bool:
        """停止容器"""
        try:
            success, output = self._run_docker_command(['stop', container_name])
            
            if success:
                if container_name in self.containers:
                    self.containers[container_name].status = ContainerStatus.EXITED
                    self.containers[container_name].finished = datetime.now()
                
                logger_manager.info(f"容器停止成功: {container_name}")
                return True
            else:
                logger_manager.error(f"容器停止失败: {output}")
                return False
                
        except Exception as e:
            logger_manager.error(f"停止容器失败: {e}")
            return False
    
    def remove_container(self, container_name: str, force: bool = False) -> bool:
        """删除容器"""
        try:
            command = ['rm']
            if force:
                command.append('-f')
            command.append(container_name)
            
            success, output = self._run_docker_command(command)
            
            if success:
                if container_name in self.containers:
                    del self.containers[container_name]
                
                logger_manager.info(f"容器删除成功: {container_name}")
                return True
            else:
                logger_manager.error(f"容器删除失败: {output}")
                return False
                
        except Exception as e:
            logger_manager.error(f"删除容器失败: {e}")
            return False
    
    def get_container_logs(self, container_name: str, tail: int = 100) -> str:
        """获取容器日志"""
        try:
            command = ['logs', '--tail', str(tail), container_name]
            success, output = self._run_docker_command(command)
            
            if success:
                return output
            else:
                logger_manager.error(f"获取容器日志失败: {output}")
                return ""
                
        except Exception as e:
            logger_manager.error(f"获取容器日志失败: {e}")
            return ""
    
    def list_containers(self, all_containers: bool = False) -> List[DockerContainer]:
        """列出容器"""
        try:
            command = ['ps']
            if all_containers:
                command.append('-a')
            command.extend(['--format', 'json'])
            
            success, output = self._run_docker_command(command)
            
            if success and output:
                containers = []
                for line in output.split('\n'):
                    if line.strip():
                        try:
                            container_info = json.loads(line)
                            container = DockerContainer(
                                container_id=container_info.get('ID', ''),
                                name=container_info.get('Names', ''),
                                image=container_info.get('Image', ''),
                                status=ContainerStatus(container_info.get('State', 'unknown').lower())
                            )
                            containers.append(container)
                        except (json.JSONDecodeError, ValueError):
                            continue
                
                return containers
            else:
                return []
                
        except Exception as e:
            logger_manager.error(f"列出容器失败: {e}")
            return []
    
    def create_dockerfile(self, project_path: str, dockerfile_type: str = "python",
                         **kwargs) -> str:
        """创建Dockerfile"""
        try:
            dockerfile_path = os.path.join(project_path, "Dockerfile")
            
            if dockerfile_type == "python":
                content = self.dockerfile_generator.generate_python_dockerfile(**kwargs)
            elif dockerfile_type == "tensorflow":
                content = self.dockerfile_generator.generate_tensorflow_dockerfile(**kwargs)
            elif dockerfile_type == "multi-stage":
                content = self.dockerfile_generator.generate_multi_stage_dockerfile(**kwargs)
            else:
                logger_manager.error(f"不支持的Dockerfile类型: {dockerfile_type}")
                return ""
            
            with open(dockerfile_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger_manager.info(f"Dockerfile创建成功: {dockerfile_path}")
            return dockerfile_path
            
        except Exception as e:
            logger_manager.error(f"创建Dockerfile失败: {e}")
            return ""
    
    def create_docker_compose(self, project_path: str, services: Dict[str, Any]) -> str:
        """创建docker-compose.yml"""
        try:
            compose_content = {
                'version': '3.8',
                'services': services
            }
            
            compose_path = os.path.join(project_path, "docker-compose.yml")
            
            try:
                import yaml
                with open(compose_path, 'w', encoding='utf-8') as f:
                    yaml.dump(compose_content, f, default_flow_style=False)
            except ImportError:
                logger_manager.error("PyYAML未安装，无法创建docker-compose.yml")
                return ""
            
            logger_manager.info(f"docker-compose.yml创建成功: {compose_path}")
            return compose_path
            
        except ImportError:
            logger_manager.error("PyYAML未安装，无法创建docker-compose.yml")
            return ""
        except Exception as e:
            logger_manager.error(f"创建docker-compose.yml失败: {e}")
            return ""
    
    def deploy_application(self, project_path: str, app_name: str,
                          dockerfile_type: str = "python",
                          port_mapping: Dict[str, str] = None) -> bool:
        """部署应用"""
        try:
            # 1. 创建Dockerfile
            dockerfile_path = self.create_dockerfile(project_path, dockerfile_type)
            if not dockerfile_path:
                return False
            
            # 2. 构建镜像
            image_name = f"{app_name}-app"
            if not self.build_image(dockerfile_path, image_name):
                return False
            
            # 3. 创建并启动容器
            container_name = f"{app_name}-container"
            container_id = self.create_container(
                image_name=f"{image_name}:latest",
                container_name=container_name,
                ports=port_mapping or {"8000": "8000"}
            )
            
            if not container_id:
                return False
            
            # 4. 启动容器
            if not self.start_container(container_name):
                return False
            
            logger_manager.info(f"应用部署成功: {app_name}")
            return True
            
        except Exception as e:
            logger_manager.error(f"部署应用失败: {e}")
            return False
    
    def get_container_status(self, container_name: str) -> Optional[ContainerStatus]:
        """获取容器状态"""
        try:
            success, output = self._run_docker_command(['inspect', container_name, '--format', '{{.State.Status}}'])
            
            if success:
                status_str = output.strip().lower()
                try:
                    return ContainerStatus(status_str)
                except ValueError:
                    return ContainerStatus.DEAD
            else:
                return None
                
        except Exception as e:
            logger_manager.error(f"获取容器状态失败: {e}")
            return None


# 全局Docker管理器实例
docker_manager = DockerManager()


if __name__ == "__main__":
    # 测试Docker管理功能
    print("🐳 测试Docker管理功能...")
    
    try:
        manager = DockerManager()
        
        if manager.docker_available:
            print("✅ Docker可用")
            
            # 测试Dockerfile生成
            dockerfile_content = manager.dockerfile_generator.generate_python_dockerfile()
            if dockerfile_content and "FROM python" in dockerfile_content:
                print("✅ Dockerfile生成成功")
            
            # 测试容器列表
            containers = manager.list_containers(all_containers=True)
            print(f"✅ 容器列表获取成功，发现 {len(containers)} 个容器")
            
        else:
            print("⚠️ Docker不可用，跳过容器操作测试")
        
        print("✅ Docker管理功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("Docker管理功能测试完成")
