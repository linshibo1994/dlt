#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型元数据定义
Model Metadata Definition

定义模型元数据类，避免循环导入问题。
"""

from datetime import datetime
from typing import List, Dict, Any


class ModelMetadata:
    """模型元数据"""
    
    def __init__(self, name: str, version: str = "1.0.0", description: str = ""):
        self.name = name
        self.version = version
        self.description = description
        self.created_at = datetime.now()
        self.tags = []
        self.parameters = {}
        self.metrics = {}
        self.dependencies = []  # 添加依赖属性
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'tags': self.tags,
            'parameters': self.parameters,
            'metrics': self.metrics,
            'dependencies': self.dependencies
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """从字典创建实例"""
        metadata = cls(
            name=data['name'],
            version=data.get('version', '1.0.0'),
            description=data.get('description', '')
        )
        
        if 'created_at' in data:
            metadata.created_at = datetime.fromisoformat(data['created_at'])
        
        metadata.tags = data.get('tags', [])
        metadata.parameters = data.get('parameters', {})
        metadata.metrics = data.get('metrics', {})
        metadata.dependencies = data.get('dependencies', [])
        
        return metadata
    
    def copy(self) -> 'ModelMetadata':
        """创建副本"""
        return ModelMetadata.from_dict(self.to_dict())
