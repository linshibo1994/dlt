#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用户指南生成器模块
User Guide Generator Module

提供用户指南、教程、示例代码生成等功能。
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


class GuideType(Enum):
    """指南类型枚举"""
    QUICK_START = "quick_start"
    TUTORIAL = "tutorial"
    HOW_TO = "how_to"
    REFERENCE = "reference"
    FAQ = "faq"


@dataclass
class GuideSection:
    """指南章节"""
    title: str
    content: str
    code_examples: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    links: List[Dict[str, str]] = field(default_factory=list)
    order: int = 0


@dataclass
class GuideConfig:
    """指南配置"""
    title: str
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    guide_type: GuideType = GuideType.TUTORIAL
    target_audience: str = ""
    prerequisites: List[str] = field(default_factory=list)
    estimated_time: str = ""
    difficulty: str = "初级"


class CodeExampleGenerator:
    """代码示例生成器"""
    
    def __init__(self):
        """初始化代码示例生成器"""
        logger_manager.debug("代码示例生成器初始化完成")
    
    def generate_basic_usage_example(self) -> str:
        """生成基础使用示例"""
        return '''```python
# 导入必要的模块
from enhanced_deep_learning import DeepLearningPlatform
from enhanced_deep_learning.models import LSTMPredictor
from enhanced_deep_learning.data import DataProcessor

# 创建平台实例
platform = DeepLearningPlatform()

# 加载数据
data_processor = DataProcessor()
data = data_processor.load_data("data/lottery_data.csv")

# 预处理数据
processed_data = data_processor.preprocess(data)

# 创建LSTM模型
model = LSTMPredictor(
    input_size=10,
    hidden_size=128,
    num_layers=2,
    output_size=2
)

# 训练模型
model.train(processed_data)

# 进行预测
predictions = model.predict(processed_data[-100:])

print(f"预测结果: {predictions}")
```'''
    
    def generate_advanced_example(self) -> str:
        """生成高级使用示例"""
        return '''```python
# 高级配置示例
from enhanced_deep_learning import DeepLearningPlatform
from enhanced_deep_learning.models import EnsembleModel
from enhanced_deep_learning.optimization import HyperparameterOptimizer
from enhanced_deep_learning.visualization import Dashboard

# 创建集成模型
ensemble = EnsembleModel([
    LSTMPredictor(input_size=10, hidden_size=128),
    TransformerPredictor(d_model=128, nhead=8),
    GANPredictor(latent_dim=100)
])

# 超参数优化
optimizer = HyperparameterOptimizer()
best_params = optimizer.optimize(
    model=ensemble,
    data=processed_data,
    search_space={
        'learning_rate': (0.001, 0.1),
        'batch_size': [16, 32, 64, 128],
        'dropout': (0.1, 0.5)
    }
)

# 使用最佳参数训练
ensemble.set_params(**best_params)
ensemble.train(processed_data)

# 启动可视化仪表板
dashboard = Dashboard()
dashboard.add_model(ensemble)
dashboard.start_server(port=8080)
```'''
    
    def generate_deployment_example(self) -> str:
        """生成部署示例"""
        return '''```bash
# Docker部署示例
docker build -t deep-learning-platform .
docker run -p 8000:8000 deep-learning-platform

# Kubernetes部署
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# 使用API
curl -X POST http://localhost:8000/api/v1/predict \\
  -H "Content-Type: application/json" \\
  -d '{"data": [[1, 2, 3, 4, 5]]}'
```'''


class UserGuideGenerator:
    """用户指南生成器"""
    
    def __init__(self):
        """初始化用户指南生成器"""
        self.code_generator = CodeExampleGenerator()
        
        logger_manager.info("用户指南生成器初始化完成")
    
    def generate_quick_start_guide(self, output_dir: str) -> bool:
        """
        生成快速开始指南
        
        Args:
            output_dir: 输出目录
            
        Returns:
            是否生成成功
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 快速开始指南内容
            guide_content = self._generate_quick_start_content()
            
            with open(output_path / "quick_start.md", 'w', encoding='utf-8') as f:
                f.write(guide_content)
            
            logger_manager.info(f"快速开始指南生成成功: {output_dir}")
            return True
            
        except Exception as e:
            logger_manager.error(f"生成快速开始指南失败: {e}")
            return False
    
    def _generate_quick_start_content(self) -> str:
        """生成快速开始内容"""
        content = """# 快速开始指南

欢迎使用深度学习预测平台！本指南将帮助您快速上手使用平台的核心功能。

## 安装

### 系统要求

- Python 3.8+
- 8GB+ RAM
- GPU支持（可选，推荐）

### 安装步骤

```bash
# 克隆项目
git clone https://github.com/your-repo/deep-learning-platform.git
cd deep-learning-platform

# 安装依赖
pip install -r requirements.txt

# 安装平台
pip install -e .
```

## 基础使用

### 1. 数据准备

首先准备您的数据文件，支持CSV、JSON等格式：

```python
import pandas as pd

# 加载彩票历史数据
data = pd.read_csv("lottery_data.csv")
print(data.head())
```

### 2. 创建和训练模型

"""

        # 添加基础使用示例
        content += self.code_generator.generate_basic_usage_example()
        
        content += """

### 3. 模型评估

```python
# 评估模型性能
from enhanced_deep_learning.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.evaluate(model, test_data)

print(f"准确率: {metrics['accuracy']:.4f}")
print(f"损失: {metrics['loss']:.4f}")
```

### 4. 可视化结果

```python
# 创建可视化图表
from enhanced_deep_learning.visualization import InteractiveVisualizer

visualizer = InteractiveVisualizer()
visualizer.plot_predictions(predictions, actual_values)
visualizer.show()
```

## 下一步

- 查看[完整教程](tutorial.md)了解更多功能
- 阅读[API参考](api_reference.md)了解详细接口
- 访问[示例项目](examples/)获取更多示例

## 获取帮助

如果遇到问题，可以：

- 查看[常见问题](faq.md)
- 提交[GitHub Issue](https://github.com/your-repo/issues)
- 联系技术支持

祝您使用愉快！
"""
        
        return content
    
    def generate_tutorial(self, config: GuideConfig, sections: List[GuideSection],
                         output_file: str) -> bool:
        """
        生成教程
        
        Args:
            config: 指南配置
            sections: 章节列表
            output_file: 输出文件
            
        Returns:
            是否生成成功
        """
        try:
            content = self._generate_tutorial_header(config)
            
            # 生成目录
            content += "## 目录\n\n"
            for i, section in enumerate(sorted(sections, key=lambda x: x.order), 1):
                content += f"{i}. [{section.title}](#{section.title.lower().replace(' ', '-')})\n"
            content += "\n"
            
            # 生成章节内容
            for section in sorted(sections, key=lambda x: x.order):
                content += self._generate_section_content(section)
            
            # 保存文件
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger_manager.info(f"教程生成成功: {output_file}")
            return True
            
        except Exception as e:
            logger_manager.error(f"生成教程失败: {e}")
            return False
    
    def _generate_tutorial_header(self, config: GuideConfig) -> str:
        """生成教程头部"""
        header = f"# {config.title}\n\n"
        
        if config.description:
            header += f"{config.description}\n\n"
        
        # 元信息
        header += "## 教程信息\n\n"
        header += f"- **版本**: {config.version}\n"
        header += f"- **难度**: {config.difficulty}\n"
        
        if config.estimated_time:
            header += f"- **预计时间**: {config.estimated_time}\n"
        
        if config.target_audience:
            header += f"- **目标读者**: {config.target_audience}\n"
        
        if config.author:
            header += f"- **作者**: {config.author}\n"
        
        header += "\n"
        
        # 前置条件
        if config.prerequisites:
            header += "## 前置条件\n\n"
            for prereq in config.prerequisites:
                header += f"- {prereq}\n"
            header += "\n"
        
        return header
    
    def _generate_section_content(self, section: GuideSection) -> str:
        """生成章节内容"""
        content = f"## {section.title}\n\n"
        content += f"{section.content}\n\n"
        
        # 代码示例
        for example in section.code_examples:
            content += f"{example}\n\n"
        
        # 图片
        for image in section.images:
            content += f"![{section.title}]({image})\n\n"
        
        # 相关链接
        if section.links:
            content += "### 相关链接\n\n"
            for link in section.links:
                content += f"- [{link['title']}]({link['url']})\n"
            content += "\n"
        
        return content
    
    def generate_faq(self, faqs: List[Dict[str, str]], output_file: str) -> bool:
        """
        生成常见问题文档
        
        Args:
            faqs: 问答列表
            output_file: 输出文件
            
        Returns:
            是否生成成功
        """
        try:
            content = "# 常见问题 (FAQ)\n\n"
            content += "以下是用户经常遇到的问题和解答：\n\n"
            
            for i, faq in enumerate(faqs, 1):
                content += f"## {i}. {faq['question']}\n\n"
                content += f"{faq['answer']}\n\n"
                
                if 'code' in faq:
                    content += f"```python\n{faq['code']}\n```\n\n"
                
                content += "---\n\n"
            
            # 保存文件
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger_manager.info(f"FAQ文档生成成功: {output_file}")
            return True
            
        except Exception as e:
            logger_manager.error(f"生成FAQ文档失败: {e}")
            return False
    
    def generate_complete_user_guide(self, output_dir: str) -> bool:
        """
        生成完整用户指南
        
        Args:
            output_dir: 输出目录
            
        Returns:
            是否生成成功
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 生成快速开始指南
            self.generate_quick_start_guide(output_dir)
            
            # 生成完整教程
            tutorial_config = GuideConfig(
                title="深度学习预测平台完整教程",
                description="从入门到精通的完整学习路径",
                version="1.0.0",
                difficulty="中级",
                estimated_time="2-3小时",
                target_audience="数据科学家、机器学习工程师",
                prerequisites=[
                    "Python基础知识",
                    "机器学习基本概念",
                    "深度学习框架使用经验"
                ]
            )
            
            tutorial_sections = [
                GuideSection(
                    title="平台架构概览",
                    content="了解平台的整体架构和核心组件",
                    order=1
                ),
                GuideSection(
                    title="数据处理和预处理",
                    content="学习如何处理和预处理您的数据",
                    code_examples=[self.code_generator.generate_basic_usage_example()],
                    order=2
                ),
                GuideSection(
                    title="模型训练和优化",
                    content="掌握模型训练和超参数优化技巧",
                    code_examples=[self.code_generator.generate_advanced_example()],
                    order=3
                ),
                GuideSection(
                    title="部署和生产环境",
                    content="了解如何将模型部署到生产环境",
                    code_examples=[self.code_generator.generate_deployment_example()],
                    order=4
                )
            ]
            
            self.generate_tutorial(tutorial_config, tutorial_sections, 
                                 str(output_path / "complete_tutorial.md"))
            
            # 生成FAQ
            faqs = [
                {
                    "question": "如何选择合适的模型？",
                    "answer": "选择模型需要考虑数据类型、数据量、计算资源等因素。对于时序数据，推荐使用LSTM或Transformer；对于生成任务，可以考虑GAN。"
                },
                {
                    "question": "训练过程中出现内存不足怎么办？",
                    "answer": "可以尝试减小批大小、使用梯度累积、启用混合精度训练或使用模型并行。",
                    "code": "# 减小批大小\nmodel.set_batch_size(16)\n\n# 启用混合精度\nmodel.enable_mixed_precision()"
                },
                {
                    "question": "如何提高预测准确率？",
                    "answer": "可以尝试数据增强、集成学习、超参数优化、特征工程等方法。"
                }
            ]
            
            self.generate_faq(faqs, str(output_path / "faq.md"))
            
            # 生成索引文件
            self._generate_guide_index(output_path)
            
            logger_manager.info(f"完整用户指南生成成功: {output_dir}")
            return True
            
        except Exception as e:
            logger_manager.error(f"生成完整用户指南失败: {e}")
            return False
    
    def _generate_guide_index(self, output_path: Path):
        """生成指南索引"""
        try:
            index_content = """# 用户指南

欢迎使用深度学习预测平台用户指南！

## 文档导航

### 🚀 快速开始
- [快速开始指南](quick_start.md) - 5分钟快速上手

### 📚 完整教程
- [完整教程](complete_tutorial.md) - 深入学习平台功能

### ❓ 帮助支持
- [常见问题](faq.md) - 常见问题解答

### 📖 参考文档
- [API参考](../api_reference/) - 详细API文档
- [示例代码](../examples/) - 实用示例集合

## 学习路径

### 初学者
1. 阅读快速开始指南
2. 运行基础示例
3. 查看常见问题

### 进阶用户
1. 学习完整教程
2. 研究高级示例
3. 参考API文档

### 开发者
1. 查看架构文档
2. 阅读源码注释
3. 参与社区讨论

## 获取帮助

如果您在使用过程中遇到问题：

1. 首先查看[常见问题](faq.md)
2. 搜索[GitHub Issues](https://github.com/your-repo/issues)
3. 提交新的Issue或联系技术支持

祝您学习愉快！
"""
            
            with open(output_path / "README.md", 'w', encoding='utf-8') as f:
                f.write(index_content)
                
        except Exception as e:
            logger_manager.error(f"生成指南索引失败: {e}")


# 全局用户指南生成器实例
user_guide_generator = UserGuideGenerator()


if __name__ == "__main__":
    # 测试用户指南生成器功能
    print("📖 测试用户指南生成器功能...")
    
    try:
        generator = UserGuideGenerator()
        
        # 测试快速开始指南生成
        if generator.generate_quick_start_guide("test_user_guide"):
            print("✅ 快速开始指南生成成功")
        
        # 测试完整用户指南生成
        if generator.generate_complete_user_guide("test_complete_guide"):
            print("✅ 完整用户指南生成成功")
        
        print("✅ 用户指南生成器功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("用户指南生成器功能测试完成")
