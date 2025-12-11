#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API文档生成器模块
API Documentation Generator Module

提供REST API文档生成、OpenAPI规范生成等功能。
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException


class APIDocFormat(Enum):
    """API文档格式枚举"""
    OPENAPI = "openapi"
    SWAGGER = "swagger"
    MARKDOWN = "markdown"
    HTML = "html"


@dataclass
class APIEndpoint:
    """API端点"""
    path: str
    method: str
    summary: str
    description: str = ""
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False


@dataclass
class APIDocConfig:
    """API文档配置"""
    title: str
    description: str = ""
    version: str = "1.0.0"
    base_url: str = "http://localhost:6000"
    contact: Dict[str, str] = field(default_factory=dict)
    license: Dict[str, str] = field(default_factory=dict)
    servers: List[Dict[str, str]] = field(default_factory=list)
    security: List[Dict[str, Any]] = field(default_factory=list)


class OpenAPIGenerator:
    """OpenAPI规范生成器"""
    
    def __init__(self):
        """初始化OpenAPI生成器"""
        logger_manager.debug("OpenAPI生成器初始化完成")
    
    def generate_spec(self, config: APIDocConfig, endpoints: List[APIEndpoint]) -> Dict[str, Any]:
        """
        生成OpenAPI规范
        
        Args:
            config: API文档配置
            endpoints: API端点列表
            
        Returns:
            OpenAPI规范字典
        """
        try:
            spec = {
                "openapi": "3.0.3",
                "info": {
                    "title": config.title,
                    "description": config.description,
                    "version": config.version
                },
                "servers": config.servers or [{"url": config.base_url}],
                "paths": {},
                "components": {
                    "schemas": {},
                    "securitySchemes": {}
                }
            }
            
            # 添加联系信息
            if config.contact:
                spec["info"]["contact"] = config.contact
            
            # 添加许可证信息
            if config.license:
                spec["info"]["license"] = config.license
            
            # 添加安全配置
            if config.security:
                spec["security"] = config.security
            
            # 处理端点
            for endpoint in endpoints:
                self._add_endpoint_to_spec(spec, endpoint)
            
            return spec
            
        except Exception as e:
            logger_manager.error(f"生成OpenAPI规范失败: {e}")
            return {}
    
    def _add_endpoint_to_spec(self, spec: Dict[str, Any], endpoint: APIEndpoint):
        """添加端点到规范"""
        try:
            path = endpoint.path
            method = endpoint.method.lower()
            
            if path not in spec["paths"]:
                spec["paths"][path] = {}
            
            operation = {
                "summary": endpoint.summary,
                "description": endpoint.description,
                "tags": endpoint.tags,
                "deprecated": endpoint.deprecated
            }
            
            # 添加参数
            if endpoint.parameters:
                operation["parameters"] = endpoint.parameters
            
            # 添加请求体
            if endpoint.request_body:
                operation["requestBody"] = endpoint.request_body
            
            # 添加响应
            if endpoint.responses:
                operation["responses"] = endpoint.responses
            else:
                operation["responses"] = {
                    "200": {
                        "description": "成功响应",
                        "content": {
                            "application/json": {
                                "schema": {"type": "object"}
                            }
                        }
                    }
                }
            
            spec["paths"][path][method] = operation
            
        except Exception as e:
            logger_manager.error(f"添加端点到规范失败: {e}")


class APIDocGenerator:
    """API文档生成器"""
    
    def __init__(self):
        """初始化API文档生成器"""
        self.openapi_generator = OpenAPIGenerator()
        
        logger_manager.info("API文档生成器初始化完成")
    
    def generate_openapi_spec(self, config: APIDocConfig, endpoints: List[APIEndpoint],
                             output_file: str) -> bool:
        """
        生成OpenAPI规范文件
        
        Args:
            config: API文档配置
            endpoints: API端点列表
            output_file: 输出文件路径
            
        Returns:
            是否生成成功
        """
        try:
            # 生成规范
            spec = self.openapi_generator.generate_spec(config, endpoints)
            
            if not spec:
                return False
            
            # 保存到文件
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(spec, f, indent=2, ensure_ascii=False)
            
            logger_manager.info(f"OpenAPI规范生成成功: {output_file}")
            return True
            
        except Exception as e:
            logger_manager.error(f"生成OpenAPI规范失败: {e}")
            return False
    
    def generate_markdown_docs(self, config: APIDocConfig, endpoints: List[APIEndpoint],
                              output_dir: str) -> bool:
        """
        生成Markdown格式的API文档
        
        Args:
            config: API文档配置
            endpoints: API端点列表
            output_dir: 输出目录
            
        Returns:
            是否生成成功
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 生成主文档
            main_doc = self._generate_main_markdown(config, endpoints)
            
            with open(output_path / "README.md", 'w', encoding='utf-8') as f:
                f.write(main_doc)
            
            # 按标签分组生成文档
            tag_groups = self._group_endpoints_by_tag(endpoints)
            
            for tag, tag_endpoints in tag_groups.items():
                tag_doc = self._generate_tag_markdown(tag, tag_endpoints)
                
                with open(output_path / f"{tag.lower().replace(' ', '_')}.md", 'w', encoding='utf-8') as f:
                    f.write(tag_doc)
            
            logger_manager.info(f"Markdown API文档生成成功: {output_dir}")
            return True
            
        except Exception as e:
            logger_manager.error(f"生成Markdown API文档失败: {e}")
            return False
    
    def _generate_main_markdown(self, config: APIDocConfig, endpoints: List[APIEndpoint]) -> str:
        """生成主Markdown文档"""
        try:
            doc = f"# {config.title}\n\n"
            
            if config.description:
                doc += f"{config.description}\n\n"
            
            doc += f"**版本**: {config.version}\n\n"
            doc += f"**基础URL**: {config.base_url}\n\n"
            
            if config.contact:
                doc += "## 联系信息\n\n"
                for key, value in config.contact.items():
                    doc += f"- **{key}**: {value}\n"
                doc += "\n"
            
            # 端点概览
            doc += "## API端点概览\n\n"
            doc += "| 方法 | 路径 | 描述 |\n"
            doc += "|------|------|------|\n"
            
            for endpoint in endpoints:
                doc += f"| {endpoint.method.upper()} | `{endpoint.path}` | {endpoint.summary} |\n"
            
            doc += "\n"
            
            # 标签分组
            tag_groups = self._group_endpoints_by_tag(endpoints)
            
            if len(tag_groups) > 1:
                doc += "## 按功能分类\n\n"
                for tag in tag_groups.keys():
                    doc += f"- [{tag}]({tag.lower().replace(' ', '_')}.md)\n"
                doc += "\n"
            
            return doc
            
        except Exception as e:
            logger_manager.error(f"生成主Markdown文档失败: {e}")
            return ""
    
    def _generate_tag_markdown(self, tag: str, endpoints: List[APIEndpoint]) -> str:
        """生成标签Markdown文档"""
        try:
            doc = f"# {tag}\n\n"
            
            for endpoint in endpoints:
                doc += f"## {endpoint.method.upper()} {endpoint.path}\n\n"
                doc += f"{endpoint.summary}\n\n"
                
                if endpoint.description:
                    doc += f"{endpoint.description}\n\n"
                
                # 参数
                if endpoint.parameters:
                    doc += "### 参数\n\n"
                    doc += "| 名称 | 位置 | 类型 | 必需 | 描述 |\n"
                    doc += "|------|------|------|------|------|\n"
                    
                    for param in endpoint.parameters:
                        name = param.get('name', '')
                        location = param.get('in', '')
                        param_type = param.get('schema', {}).get('type', '')
                        required = '是' if param.get('required', False) else '否'
                        description = param.get('description', '')
                        
                        doc += f"| {name} | {location} | {param_type} | {required} | {description} |\n"
                    
                    doc += "\n"
                
                # 请求体
                if endpoint.request_body:
                    doc += "### 请求体\n\n"
                    doc += "```json\n"
                    doc += json.dumps(endpoint.request_body, indent=2, ensure_ascii=False)
                    doc += "\n```\n\n"
                
                # 响应
                if endpoint.responses:
                    doc += "### 响应\n\n"
                    for status_code, response in endpoint.responses.items():
                        doc += f"#### {status_code}\n\n"
                        if 'description' in response:
                            doc += f"{response['description']}\n\n"
                        
                        if 'content' in response:
                            doc += "```json\n"
                            # 简化显示响应结构
                            content = response['content']
                            if 'application/json' in content:
                                schema = content['application/json'].get('schema', {})
                                doc += json.dumps(schema, indent=2, ensure_ascii=False)
                            doc += "\n```\n\n"
                
                doc += "---\n\n"
            
            return doc
            
        except Exception as e:
            logger_manager.error(f"生成标签Markdown文档失败: {e}")
            return ""
    
    def _group_endpoints_by_tag(self, endpoints: List[APIEndpoint]) -> Dict[str, List[APIEndpoint]]:
        """按标签分组端点"""
        try:
            groups = {}
            
            for endpoint in endpoints:
                tags = endpoint.tags or ['默认']
                
                for tag in tags:
                    if tag not in groups:
                        groups[tag] = []
                    groups[tag].append(endpoint)
            
            return groups
            
        except Exception as e:
            logger_manager.error(f"按标签分组端点失败: {e}")
            return {}
    
    def auto_discover_endpoints(self, app_module: str) -> List[APIEndpoint]:
        """
        自动发现API端点
        
        Args:
            app_module: 应用模块路径
            
        Returns:
            发现的端点列表
        """
        try:
            endpoints = []
            
            # 这里可以实现自动发现逻辑
            # 例如扫描Flask/FastAPI路由、Django URL配置等
            
            logger_manager.info(f"自动发现API端点: {len(endpoints)} 个")
            return endpoints
            
        except Exception as e:
            logger_manager.error(f"自动发现API端点失败: {e}")
            return []


# 全局API文档生成器实例
api_doc_generator = APIDocGenerator()


if __name__ == "__main__":
    # 测试API文档生成器功能
    print("📖 测试API文档生成器功能...")
    
    try:
        generator = APIDocGenerator()
        
        # 创建测试配置
        config = APIDocConfig(
            title="深度学习预测平台 API",
            description="提供深度学习模型训练、预测和管理的REST API",
            version="1.0.0",
            base_url="http://localhost:6000/api/v1",
            contact={
                "name": "API Support",
                "email": "support@example.com"
            }
        )
        
        # 创建测试端点
        endpoints = [
            APIEndpoint(
                path="/models",
                method="GET",
                summary="获取模型列表",
                description="返回所有可用的深度学习模型",
                tags=["模型管理"],
                responses={
                    "200": {
                        "description": "成功返回模型列表",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"type": "object"}
                                }
                            }
                        }
                    }
                }
            ),
            APIEndpoint(
                path="/models/{model_id}/predict",
                method="POST",
                summary="执行预测",
                description="使用指定模型执行预测",
                tags=["预测"],
                parameters=[
                    {
                        "name": "model_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "模型ID"
                    }
                ],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "data": {"type": "array"}
                                }
                            }
                        }
                    }
                }
            )
        ]
        
        # 测试OpenAPI规范生成
        if generator.generate_openapi_spec(config, endpoints, "test_openapi.json"):
            print("✅ OpenAPI规范生成成功")
        
        # 测试Markdown文档生成
        if generator.generate_markdown_docs(config, endpoints, "test_api_docs"):
            print("✅ Markdown API文档生成成功")
        
        print("✅ API文档生成器功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("API文档生成器功能测试完成")
