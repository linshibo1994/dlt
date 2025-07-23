#!/usr/bin/env python3
"""
模型注册表
管理所有深度学习模型的注册、发现和实例化
"""

from typing import Dict, List, Any, Type, Optional, Callable
import inspect
import json
import os
from datetime import datetime
from collections import defaultdict

from core_modules import logger_manager

# 创建基础模型类和元数据类
class BaseDeepLearningModel:
    """基础深度学习模型"""
    def __init__(self, name: str = "BaseModel"):
        self.name = name

    def predict(self, data):
        """预测方法"""
        raise NotImplementedError("子类必须实现predict方法")

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


class ModelRegistry:
    """模型注册表"""
    
    def __init__(self):
        """初始化模型注册表"""
        self._models: Dict[str, Type[BaseDeepLearningModel]] = {}
        self._metadata: Dict[str, ModelMetadata] = {}
        self._factories: Dict[str, Callable] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._versions: Dict[str, List[str]] = defaultdict(list)
        
        logger_manager.info("模型注册表初始化完成")
    
    def register_model(self, model_class: Type[BaseDeepLearningModel], 
                      name: str = None, metadata: ModelMetadata = None,
                      factory: Callable = None) -> bool:
        """
        注册模型
        
        Args:
            model_class: 模型类
            name: 模型名称
            metadata: 模型元数据
            factory: 模型工厂函数
            
        Returns:
            是否注册成功
        """
        try:
            # 确定模型名称
            if name is None:
                name = model_class.__name__
            
            # 检查是否已注册
            if name in self._models:
                logger_manager.warning(f"模型 {name} 已存在，将被覆盖")
            
            # 验证模型类
            if not self._validate_model_class(model_class):
                raise ValueError(f"模型类 {model_class.__name__} 验证失败")
            
            # 创建或使用提供的元数据
            if metadata is None:
                metadata = ModelMetadata(
                    name=name,
                    description=model_class.__doc__ or ""
                )
                # 添加依赖信息
                metadata.dependencies = self._extract_dependencies(model_class)
            
            # 注册模型
            self._models[name] = model_class
            self._metadata[name] = metadata
            
            # 注册工厂函数
            if factory is not None:
                self._factories[name] = factory
            else:
                self._factories[name] = self._create_default_factory(model_class)
            
            # 记录依赖关系
            self._dependencies[name] = metadata.dependencies
            
            # 记录版本
            self._versions[name].append(metadata.version)
            
            logger_manager.info(f"模型注册成功: {name} v{metadata.version}")
            
            return True
            
        except Exception as e:
            logger_manager.error(f"模型注册失败: {e}")
            return False
    
    def unregister_model(self, name: str) -> bool:
        """
        注销模型
        
        Args:
            name: 模型名称
            
        Returns:
            是否注销成功
        """
        try:
            if name not in self._models:
                logger_manager.warning(f"模型 {name} 不存在")
                return False
            
            # 检查依赖关系
            dependents = self._find_dependents(name)
            if dependents:
                logger_manager.warning(f"模型 {name} 被以下模型依赖: {dependents}")
                return False
            
            # 注销模型
            del self._models[name]
            del self._metadata[name]
            del self._factories[name]
            del self._dependencies[name]
            del self._versions[name]
            
            logger_manager.info(f"模型注销成功: {name}")
            
            return True
            
        except Exception as e:
            logger_manager.error(f"模型注销失败: {e}")
            return False
    
    def get_model_class(self, name: str) -> Optional[Type[BaseDeepLearningModel]]:
        """
        获取模型类
        
        Args:
            name: 模型名称
            
        Returns:
            模型类或None
        """
        return self._models.get(name)
    
    def create_model(self, name: str, config: Dict[str, Any] = None, 
                    **kwargs) -> Optional[BaseDeepLearningModel]:
        """
        创建模型实例
        
        Args:
            name: 模型名称
            config: 模型配置
            **kwargs: 额外参数
            
        Returns:
            模型实例或None
        """
        try:
            if name not in self._models:
                logger_manager.error(f"模型 {name} 未注册")
                return None
            
            # 检查依赖
            if not self._check_dependencies(name):
                logger_manager.error(f"模型 {name} 依赖检查失败")
                return None
            
            # 使用工厂函数创建实例
            factory = self._factories[name]
            metadata = self._metadata[name].copy() if hasattr(self._metadata[name], 'copy') else self._metadata[name]
            
            # 合并配置
            final_config = config or {}
            final_config.update(kwargs)
            
            # 创建实例
            instance = factory(config=final_config, metadata=metadata)
            
            logger_manager.info(f"模型实例创建成功: {name}")
            
            return instance
            
        except Exception as e:
            logger_manager.error(f"模型实例创建失败: {e}")
            return None

    def get_model(self, name: str, config: Dict[str, Any] = None) -> Optional[BaseDeepLearningModel]:
        """
        获取模型实例（create_model的别名，用于兼容性）

        Args:
            name: 模型名称
            config: 模型配置

        Returns:
            模型实例，如果获取失败返回None
        """
        return self.create_model(name, config)

    def list_models(self) -> List[str]:
        """
        列出所有注册的模型
        
        Returns:
            模型名称列表
        """
        return list(self._models.keys())
    
    def get_model_metadata(self, name: str) -> Optional[ModelMetadata]:
        """
        获取模型元数据
        
        Args:
            name: 模型名称
            
        Returns:
            模型元数据或None
        """
        return self._metadata.get(name)
    
    def find_models_by_dependency(self, dependency: str) -> List[str]:
        """
        根据依赖查找模型
        
        Args:
            dependency: 依赖名称
            
        Returns:
            模型名称列表
        """
        models = []
        for name, deps in self._dependencies.items():
            if dependency in deps:
                models.append(name)
        return models
    
    def get_model_versions(self, name: str) -> List[str]:
        """
        获取模型版本列表
        
        Args:
            name: 模型名称
            
        Returns:
            版本列表
        """
        return self._versions.get(name, [])
    
    def validate_model_compatibility(self, name: str, version: str = None) -> bool:
        """
        验证模型兼容性
        
        Args:
            name: 模型名称
            version: 版本号
            
        Returns:
            是否兼容
        """
        try:
            if name not in self._models:
                return False
            
            # 检查版本兼容性
            if version is not None:
                available_versions = self._versions.get(name, [])
                if version not in available_versions:
                    return False
            
            # 检查依赖兼容性
            return self._check_dependencies(name)
            
        except Exception as e:
            logger_manager.error(f"兼容性检查失败: {e}")
            return False
    
    def export_registry(self, file_path: str) -> bool:
        """
        导出注册表
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否导出成功
        """
        try:
            registry_data = {
                'models': {name: {
                    'class_name': cls.__name__,
                    'module': cls.__module__,
                    'metadata': self._metadata[name].to_dict(),
                    'dependencies': self._dependencies[name],
                    'versions': self._versions[name]
                } for name, cls in self._models.items()},
                'exported_at': datetime.now().isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)
            
            logger_manager.info(f"注册表导出成功: {file_path}")
            
            return True
            
        except Exception as e:
            logger_manager.error(f"注册表导出失败: {e}")
            return False
    
    def import_registry(self, file_path: str) -> bool:
        """
        导入注册表
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否导入成功
        """
        try:
            if not os.path.exists(file_path):
                logger_manager.error(f"注册表文件不存在: {file_path}")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                registry_data = json.load(f)
            
            # 导入模型信息（不包括实际的类，需要手动注册）
            for name, model_info in registry_data.get('models', {}).items():
                metadata = ModelMetadata.from_dict(model_info['metadata'])
                self._metadata[name] = metadata
                self._dependencies[name] = model_info['dependencies']
                self._versions[name] = model_info['versions']
            
            logger_manager.info(f"注册表导入成功: {file_path}")
            
            return True
            
        except Exception as e:
            logger_manager.error(f"注册表导入失败: {e}")
            return False
    
    def _validate_model_class(self, model_class: Type[BaseDeepLearningModel]) -> bool:
        """验证模型类"""
        try:
            # 导入BaseModel用于验证
            try:
                from .base_model import BaseModel
                # 检查是否继承自BaseModel或BaseDeepLearningModel
                if not (issubclass(model_class, BaseDeepLearningModel) or issubclass(model_class, BaseModel)):
                    logger_manager.warning(f"模型类 {model_class.__name__} 未继承自BaseModel或BaseDeepLearningModel")
                    return False
            except ImportError:
                # 如果无法导入BaseModel，只检查BaseDeepLearningModel
                if not issubclass(model_class, BaseDeepLearningModel):
                    return False

            # 检查必需的方法是否实现（兼容两种模式）
            # 新版BaseModel的方法
            new_methods = ['build_model', 'train', 'predict', 'evaluate']
            # 旧版BaseDeepLearningModel的方法
            old_methods = ['_build_model', '_prepare_data', '_train_model',
                         '_predict_model', '_save_model_file', '_load_model_file']

            # 检查是否实现了新版方法
            has_new_methods = all(hasattr(model_class, method) for method in new_methods)
            # 检查是否实现了旧版方法
            has_old_methods = all(hasattr(model_class, method) for method in old_methods)

            if not (has_new_methods or has_old_methods):
                logger_manager.warning(f"模型类 {model_class.__name__} 缺少必需的方法")
                return False

            # 检查方法是否为抽象方法（如果是，说明未实现）
            if has_new_methods:
                for method_name in new_methods:
                    method = getattr(model_class, method_name)
                    if getattr(method, '__isabstractmethod__', False):
                        logger_manager.warning(f"模型类 {model_class.__name__} 的方法 {method_name} 未实现")
                        return False

            return True

        except Exception as e:
            logger_manager.error(f"模型类验证失败: {e}")
            return False
    
    def _extract_dependencies(self, model_class: Type[BaseDeepLearningModel]) -> List[str]:
        """提取模型依赖"""
        dependencies = []
        
        try:
            # 从类的__init__方法签名中提取依赖
            init_signature = inspect.signature(model_class.__init__)
            for param_name, param in init_signature.parameters.items():
                if param_name != 'self' and param.annotation != inspect.Parameter.empty:
                    dependencies.append(str(param.annotation))
            
            # 从类的文档字符串中提取依赖信息
            if hasattr(model_class, '__dependencies__'):
                dependencies.extend(model_class.__dependencies__)
            
        except Exception as e:
            logger_manager.warning(f"依赖提取失败: {e}")
        
        return dependencies
    
    def _create_default_factory(self, model_class: Type[BaseDeepLearningModel]) -> Callable:
        """创建默认工厂函数"""
        def factory(config: Dict[str, Any] = None, metadata: ModelMetadata = None):
            return model_class(config=config, metadata=metadata)
        
        return factory
    
    def _check_dependencies(self, name: str) -> bool:
        """检查模型依赖"""
        try:
            dependencies = self._dependencies.get(name, [])
            
            for dep in dependencies:
                # 这里可以添加具体的依赖检查逻辑
                # 例如检查是否安装了必需的包
                pass
            
            return True
            
        except Exception as e:
            logger_manager.error(f"依赖检查失败: {e}")
            return False
    
    def _find_dependents(self, name: str) -> List[str]:
        """查找依赖于指定模型的其他模型"""
        dependents = []
        
        for model_name, deps in self._dependencies.items():
            if name in deps:
                dependents.append(model_name)
        
        return dependents
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """获取注册表统计信息"""
        return {
            'total_models': len(self._models),
            'models_by_dependency': {
                dep: len(models) for dep, models in 
                self._group_by_dependency().items()
            },
            'total_versions': sum(len(versions) for versions in self._versions.values()),
            'models': list(self._models.keys())
        }
    
    def _group_by_dependency(self) -> Dict[str, List[str]]:
        """按依赖分组模型"""
        groups = defaultdict(list)
        
        for name, deps in self._dependencies.items():
            for dep in deps:
                groups[dep].append(name)
        
        return dict(groups)


# 全局模型注册表实例
model_registry = ModelRegistry()

# 注册默认模型
def _register_default_models():
    """注册默认的深度学习模型"""
    try:
        # 导入模型类
        from . import LSTMPredictor, TransformerPredictor, GANPredictor, EnsembleManager

        # 注册LSTM模型
        lstm_metadata = ModelMetadata(
            name="lstm",
            version="1.0.0",
            description="长短期记忆网络预测模型"
        )
        model_registry.register_model(
            model_class=LSTMPredictor,
            name="lstm",
            metadata=lstm_metadata,
            factory=lambda config=None, metadata=None: LSTMPredictor()
        )

        # 注册Transformer模型
        transformer_metadata = ModelMetadata(
            name="transformer",
            version="1.0.0",
            description="Transformer注意力机制预测模型"
        )
        model_registry.register_model(
            model_class=TransformerPredictor,
            name="transformer",
            metadata=transformer_metadata,
            factory=lambda config=None, metadata=None: TransformerPredictor()
        )

        # 注册GAN模型
        gan_metadata = ModelMetadata(
            name="gan",
            version="1.0.0",
            description="生成对抗网络预测模型"
        )
        model_registry.register_model(
            model_class=GANPredictor,
            name="gan",
            metadata=gan_metadata,
            factory=lambda config=None, metadata=None: GANPredictor()
        )

        # 注册集成模型
        ensemble_metadata = ModelMetadata(
            name="ensemble",
            version="1.0.0",
            description="集成学习预测模型"
        )
        model_registry.register_model(
            model_class=EnsembleManager,
            name="ensemble",
            metadata=ensemble_metadata,
            factory=lambda config=None, metadata=None: EnsembleManager()
        )

        logger_manager.info("默认模型注册完成")

    except Exception as e:
        logger_manager.error(f"默认模型注册失败: {e}")

# 自动注册默认模型
_register_default_models()
