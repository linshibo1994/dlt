#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文档生成器模块
Documentation Generator Module

提供自动化文档生成、格式转换、模板管理等功能。
"""

import os
import json
import inspect
import importlib
from typing import Dict, List, Any, Optional, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import ast

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException


class DocFormat(Enum):
    """文档格式枚举"""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    RST = "rst"
    JSON = "json"


@dataclass
class DocConfig:
    """文档配置"""
    title: str
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    output_dir: str = "docs"
    format: DocFormat = DocFormat.MARKDOWN
    include_private: bool = False
    include_source: bool = True
    template_dir: str = ""
    custom_css: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModuleAnalyzer:
    """模块分析器"""
    
    def __init__(self):
        """初始化模块分析器"""
        logger_manager.debug("模块分析器初始化完成")
    
    def analyze_module(self, module_path: str) -> Dict[str, Any]:
        """
        分析模块
        
        Args:
            module_path: 模块路径
            
        Returns:
            模块分析结果
        """
        try:
            # 动态导入模块
            module = importlib.import_module(module_path)
            
            analysis = {
                'name': module.__name__,
                'doc': inspect.getdoc(module) or "",
                'file': inspect.getfile(module) if hasattr(module, '__file__') else "",
                'classes': [],
                'functions': [],
                'constants': [],
                'imports': []
            }
            
            # 分析模块成员
            for name, obj in inspect.getmembers(module):
                if name.startswith('_'):
                    continue
                
                if inspect.isclass(obj) and obj.__module__ == module.__name__:
                    analysis['classes'].append(self._analyze_class(obj))
                elif inspect.isfunction(obj) and obj.__module__ == module.__name__:
                    analysis['functions'].append(self._analyze_function(obj))
                elif not callable(obj) and not inspect.ismodule(obj):
                    analysis['constants'].append({
                        'name': name,
                        'value': str(obj),
                        'type': type(obj).__name__
                    })
            
            return analysis
            
        except Exception as e:
            logger_manager.error(f"分析模块失败: {e}")
            return {}
    
    def _analyze_class(self, cls: Type) -> Dict[str, Any]:
        """分析类"""
        try:
            class_info = {
                'name': cls.__name__,
                'doc': inspect.getdoc(cls) or "",
                'bases': [base.__name__ for base in cls.__bases__],
                'methods': [],
                'properties': [],
                'class_variables': []
            }
            
            # 分析类成员
            for name, obj in inspect.getmembers(cls):
                if name.startswith('_') and name not in ['__init__']:
                    continue
                
                if inspect.ismethod(obj) or inspect.isfunction(obj):
                    class_info['methods'].append(self._analyze_method(obj))
                elif isinstance(obj, property):
                    class_info['properties'].append({
                        'name': name,
                        'doc': inspect.getdoc(obj) or "",
                        'getter': obj.fget is not None,
                        'setter': obj.fset is not None,
                        'deleter': obj.fdel is not None
                    })
                elif not callable(obj):
                    class_info['class_variables'].append({
                        'name': name,
                        'value': str(obj),
                        'type': type(obj).__name__
                    })
            
            return class_info
            
        except Exception as e:
            logger_manager.error(f"分析类失败: {e}")
            return {}
    
    def _analyze_function(self, func: callable) -> Dict[str, Any]:
        """分析函数"""
        return self._analyze_method(func)
    
    def _analyze_method(self, method: callable) -> Dict[str, Any]:
        """分析方法"""
        try:
            sig = inspect.signature(method)
            
            method_info = {
                'name': method.__name__,
                'doc': inspect.getdoc(method) or "",
                'signature': str(sig),
                'parameters': [],
                'return_annotation': str(sig.return_annotation) if sig.return_annotation != inspect.Signature.empty else None
            }
            
            # 分析参数
            for param_name, param in sig.parameters.items():
                param_info = {
                    'name': param_name,
                    'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
                    'default': str(param.default) if param.default != inspect.Parameter.empty else None,
                    'kind': param.kind.name
                }
                method_info['parameters'].append(param_info)
            
            return method_info
            
        except Exception as e:
            logger_manager.error(f"分析方法失败: {e}")
            return {}


class MarkdownGenerator:
    """Markdown生成器"""
    
    def __init__(self):
        """初始化Markdown生成器"""
        logger_manager.debug("Markdown生成器初始化完成")
    
    def generate_module_doc(self, analysis: Dict[str, Any]) -> str:
        """生成模块文档"""
        try:
            doc = f"# {analysis['name']}\n\n"
            
            if analysis['doc']:
                doc += f"{analysis['doc']}\n\n"
            
            # 常量
            if analysis['constants']:
                doc += "## 常量\n\n"
                for const in analysis['constants']:
                    doc += f"- **{const['name']}**: `{const['value']}` ({const['type']})\n"
                doc += "\n"
            
            # 函数
            if analysis['functions']:
                doc += "## 函数\n\n"
                for func in analysis['functions']:
                    doc += self._generate_function_doc(func)
                doc += "\n"
            
            # 类
            if analysis['classes']:
                doc += "## 类\n\n"
                for cls in analysis['classes']:
                    doc += self._generate_class_doc(cls)
                doc += "\n"
            
            return doc
            
        except Exception as e:
            logger_manager.error(f"生成模块文档失败: {e}")
            return ""
    
    def _generate_function_doc(self, func_info: Dict[str, Any]) -> str:
        """生成函数文档"""
        try:
            doc = f"### {func_info['name']}\n\n"
            
            if func_info['doc']:
                doc += f"{func_info['doc']}\n\n"
            
            doc += f"**签名**: `{func_info['signature']}`\n\n"
            
            if func_info['parameters']:
                doc += "**参数**:\n\n"
                for param in func_info['parameters']:
                    param_doc = f"- **{param['name']}"
                    if param['annotation']:
                        param_doc += f" ({param['annotation']})"
                    if param['default']:
                        param_doc += f" = {param['default']}"
                    param_doc += "**\n"
                    doc += param_doc
                doc += "\n"
            
            if func_info['return_annotation']:
                doc += f"**返回**: {func_info['return_annotation']}\n\n"
            
            return doc
            
        except Exception as e:
            logger_manager.error(f"生成函数文档失败: {e}")
            return ""
    
    def _generate_class_doc(self, class_info: Dict[str, Any]) -> str:
        """生成类文档"""
        try:
            doc = f"### {class_info['name']}\n\n"
            
            if class_info['doc']:
                doc += f"{class_info['doc']}\n\n"
            
            if class_info['bases']:
                doc += f"**继承**: {', '.join(class_info['bases'])}\n\n"
            
            # 类变量
            if class_info['class_variables']:
                doc += "**类变量**:\n\n"
                for var in class_info['class_variables']:
                    doc += f"- **{var['name']}**: `{var['value']}` ({var['type']})\n"
                doc += "\n"
            
            # 属性
            if class_info['properties']:
                doc += "**属性**:\n\n"
                for prop in class_info['properties']:
                    prop_doc = f"- **{prop['name']}**"
                    if prop['doc']:
                        prop_doc += f": {prop['doc']}"
                    prop_doc += "\n"
                    doc += prop_doc
                doc += "\n"
            
            # 方法
            if class_info['methods']:
                doc += "**方法**:\n\n"
                for method in class_info['methods']:
                    doc += f"#### {method['name']}\n\n"
                    if method['doc']:
                        doc += f"{method['doc']}\n\n"
                    doc += f"**签名**: `{method['signature']}`\n\n"
                    
                    if method['parameters']:
                        doc += "**参数**:\n\n"
                        for param in method['parameters']:
                            param_doc = f"- **{param['name']}"
                            if param['annotation']:
                                param_doc += f" ({param['annotation']})"
                            if param['default']:
                                param_doc += f" = {param['default']}"
                            param_doc += "**\n"
                            doc += param_doc
                        doc += "\n"
                    
                    if method['return_annotation']:
                        doc += f"**返回**: {method['return_annotation']}\n\n"
            
            return doc
            
        except Exception as e:
            logger_manager.error(f"生成类文档失败: {e}")
            return ""


class DocumentationGenerator:
    """文档生成器"""
    
    def __init__(self):
        """初始化文档生成器"""
        self.analyzer = ModuleAnalyzer()
        self.markdown_generator = MarkdownGenerator()
        
        logger_manager.info("文档生成器初始化完成")
    
    def generate_project_docs(self, project_path: str, config: DocConfig) -> bool:
        """
        生成项目文档
        
        Args:
            project_path: 项目路径
            config: 文档配置
            
        Returns:
            是否生成成功
        """
        try:
            # 创建输出目录
            output_dir = Path(config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成主文档
            self._generate_main_doc(config, output_dir)
            
            # 扫描Python模块
            python_files = list(Path(project_path).rglob("*.py"))
            
            for py_file in python_files:
                if py_file.name.startswith('__'):
                    continue
                
                try:
                    # 转换文件路径为模块路径
                    relative_path = py_file.relative_to(Path(project_path))
                    module_path = str(relative_path.with_suffix('')).replace(os.sep, '.')
                    
                    # 分析模块
                    analysis = self.analyzer.analyze_module(module_path)
                    
                    if analysis:
                        # 生成文档
                        doc_content = self.markdown_generator.generate_module_doc(analysis)
                        
                        # 保存文档
                        doc_file = output_dir / f"{module_path}.md"
                        doc_file.parent.mkdir(parents=True, exist_ok=True)
                        
                        with open(doc_file, 'w', encoding='utf-8') as f:
                            f.write(doc_content)
                        
                        logger_manager.debug(f"模块文档生成成功: {module_path}")
                
                except Exception as e:
                    logger_manager.warning(f"生成模块文档失败 {py_file}: {e}")
                    continue
            
            logger_manager.info(f"项目文档生成完成: {config.output_dir}")
            return True
            
        except Exception as e:
            logger_manager.error(f"生成项目文档失败: {e}")
            return False
    
    def _generate_main_doc(self, config: DocConfig, output_dir: Path):
        """生成主文档"""
        try:
            main_doc = f"# {config.title}\n\n"
            
            if config.description:
                main_doc += f"{config.description}\n\n"
            
            main_doc += f"**版本**: {config.version}\n\n"
            
            if config.author:
                main_doc += f"**作者**: {config.author}\n\n"
            
            main_doc += f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            main_doc += "## 目录结构\n\n"
            main_doc += "本文档包含以下模块的详细说明：\n\n"
            
            # 保存主文档
            with open(output_dir / "README.md", 'w', encoding='utf-8') as f:
                f.write(main_doc)
            
        except Exception as e:
            logger_manager.error(f"生成主文档失败: {e}")
    
    def generate_module_doc(self, module_path: str, output_file: str) -> bool:
        """
        生成单个模块文档
        
        Args:
            module_path: 模块路径
            output_file: 输出文件
            
        Returns:
            是否生成成功
        """
        try:
            # 分析模块
            analysis = self.analyzer.analyze_module(module_path)
            
            if not analysis:
                return False
            
            # 生成文档
            doc_content = self.markdown_generator.generate_module_doc(analysis)
            
            # 保存文档
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(doc_content)
            
            logger_manager.info(f"模块文档生成成功: {output_file}")
            return True
            
        except Exception as e:
            logger_manager.error(f"生成模块文档失败: {e}")
            return False
    
    def generate_api_reference(self, modules: List[str], output_dir: str) -> bool:
        """
        生成API参考文档
        
        Args:
            modules: 模块列表
            output_dir: 输出目录
            
        Returns:
            是否生成成功
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 生成API索引
            index_content = "# API 参考\n\n"
            index_content += "## 模块列表\n\n"
            
            for module_path in modules:
                try:
                    analysis = self.analyzer.analyze_module(module_path)
                    
                    if analysis:
                        # 生成模块文档
                        doc_content = self.markdown_generator.generate_module_doc(analysis)
                        
                        # 保存模块文档
                        module_file = output_path / f"{module_path.replace('.', '_')}.md"
                        with open(module_file, 'w', encoding='utf-8') as f:
                            f.write(doc_content)
                        
                        # 添加到索引
                        index_content += f"- [{analysis['name']}]({module_file.name})\n"
                        
                        if analysis['doc']:
                            index_content += f"  - {analysis['doc'].split('.')[0]}\n"
                
                except Exception as e:
                    logger_manager.warning(f"处理模块失败 {module_path}: {e}")
                    continue
            
            # 保存索引文件
            with open(output_path / "index.md", 'w', encoding='utf-8') as f:
                f.write(index_content)
            
            logger_manager.info(f"API参考文档生成完成: {output_dir}")
            return True
            
        except Exception as e:
            logger_manager.error(f"生成API参考文档失败: {e}")
            return False


# 全局文档生成器实例
documentation_generator = DocumentationGenerator()


if __name__ == "__main__":
    # 测试文档生成器功能
    print("📚 测试文档生成器功能...")
    
    try:
        generator = DocumentationGenerator()
        
        # 创建测试配置
        config = DocConfig(
            title="测试项目文档",
            description="这是一个测试项目的文档",
            version="1.0.0",
            author="Test Author",
            output_dir="test_docs"
        )
        
        # 测试模块分析
        try:
            analysis = generator.analyzer.analyze_module("enhanced_deep_learning.utils.exceptions")
            if analysis:
                print("✅ 模块分析成功")
                
                # 测试文档生成
                doc_content = generator.markdown_generator.generate_module_doc(analysis)
                if doc_content:
                    print("✅ 文档生成成功")
        
        except Exception as e:
            print(f"⚠️ 模块分析测试失败: {e}")
        
        # 测试API参考生成
        modules = [
            "enhanced_deep_learning.utils.exceptions"
        ]

        if generator.generate_api_reference(modules, "test_api_docs"):
            print("✅ API参考文档生成成功")
        
        print("✅ 文档生成器功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("文档生成器功能测试完成")
