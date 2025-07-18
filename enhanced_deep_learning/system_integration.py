#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
系统集成
提供与现有系统的集成功能
"""

import os
import sys
from typing import List, Dict, Tuple, Any, Optional, Union

from core_modules import logger_manager
from .cli_handler import cli_handler


class SystemIntegration:
    """系统集成"""
    
    def __init__(self):
        """初始化系统集成"""
        self.cli_handler = cli_handler
        logger_manager.info("系统集成初始化完成")
    
    def register_commands(self, main_parser) -> None:
        """
        注册命令到主CLI系统
        
        Args:
            main_parser: 主命令解析器
        """
        self.cli_handler.register_with_main_cli(main_parser)
    
    def handle_command(self, args) -> int:
        """
        处理命令
        
        Args:
            args: 命令参数
            
        Returns:
            命令执行结果代码
        """
        return self.cli_handler.execute_main_cli_command(args)
    
    def get_prediction_methods(self) -> List[Dict[str, Any]]:
        """
        获取预测方法列表
        
        Returns:
            预测方法列表
        """
        # 定义深度学习预测方法
        methods = [
            {
                'name': 'transformer',
                'display_name': 'Transformer深度学习',
                'description': '基于Transformer架构的深度学习预测',
                'category': 'deep_learning',
                'command': 'dl predict -m transformer',
                'options': [
                    {'name': 'count', 'flag': '-c', 'type': 'int', 'default': 5, 'description': '生成的预测结果数量'},
                    {'name': 'compound', 'flag': '--compound', 'type': 'bool', 'default': False, 'description': '是否生成复式投注'},
                    {'name': 'front_count', 'flag': '--front-count', 'type': 'int', 'default': 7, 'description': '前区号码数量（复式投注）'},
                    {'name': 'back_count', 'flag': '--back-count', 'type': 'int', 'default': 3, 'description': '后区号码数量（复式投注）'},
                    {'name': 'confidence', 'flag': '--confidence', 'type': 'bool', 'default': True, 'description': '是否显示置信度'},
                    {'name': 'report', 'flag': '--report', 'type': 'bool', 'default': False, 'description': '是否生成详细报告'}
                ]
            },
            {
                'name': 'gan',
                'display_name': 'GAN深度学习',
                'description': '基于生成对抗网络的深度学习预测',
                'category': 'deep_learning',
                'command': 'dl predict -m gan',
                'options': [
                    {'name': 'count', 'flag': '-c', 'type': 'int', 'default': 5, 'description': '生成的预测结果数量'},
                    {'name': 'compound', 'flag': '--compound', 'type': 'bool', 'default': False, 'description': '是否生成复式投注'},
                    {'name': 'front_count', 'flag': '--front-count', 'type': 'int', 'default': 7, 'description': '前区号码数量（复式投注）'},
                    {'name': 'back_count', 'flag': '--back-count', 'type': 'int', 'default': 3, 'description': '后区号码数量（复式投注）'},
                    {'name': 'confidence', 'flag': '--confidence', 'type': 'bool', 'default': True, 'description': '是否显示置信度'},
                    {'name': 'report', 'flag': '--report', 'type': 'bool', 'default': False, 'description': '是否生成详细报告'}
                ]
            },
            {
                'name': 'ensemble',
                'display_name': '深度学习集成',
                'description': '多种深度学习模型的集成预测',
                'category': 'deep_learning',
                'command': 'dl predict -m ensemble',
                'options': [
                    {'name': 'count', 'flag': '-c', 'type': 'int', 'default': 5, 'description': '生成的预测结果数量'},
                    {'name': 'compound', 'flag': '--compound', 'type': 'bool', 'default': False, 'description': '是否生成复式投注'},
                    {'name': 'front_count', 'flag': '--front-count', 'type': 'int', 'default': 7, 'description': '前区号码数量（复式投注）'},
                    {'name': 'back_count', 'flag': '--back-count', 'type': 'int', 'default': 3, 'description': '后区号码数量（复式投注）'},
                    {'name': 'confidence', 'flag': '--confidence', 'type': 'bool', 'default': True, 'description': '是否显示置信度'},
                    {'name': 'report', 'flag': '--report', 'type': 'bool', 'default': False, 'description': '是否生成详细报告'}
                ]
            }
        ]
        
        return methods
    
    def get_help_content(self) -> Dict[str, str]:
        """
        获取帮助内容
        
        Returns:
            帮助内容字典
        """
        help_content = {
            'deep_learning': """
# 深度学习预测系统

## 概述
深度学习预测系统使用先进的神经网络模型对大乐透号码进行预测，包括Transformer模型、GAN模型和集成模型。

## 可用命令

### 训练模型
```bash
python3 dlt_main.py dl train -m [transformer|gan|all] -p 1000 -e 100 -b 32 [--gpu] [--save-model]
```

### 预测号码
```bash
python3 dlt_main.py dl predict -m [transformer|gan|ensemble] -c 5 [--compound] [--front-count 7] [--back-count 3] [--confidence] [--report]
```

### 管理模型集成
```bash
python3 dlt_main.py dl ensemble list
python3 dlt_main.py dl ensemble add -m model_name -w 1.0
python3 dlt_main.py dl ensemble remove -m model_name
python3 dlt_main.py dl ensemble update -m model_name -w 0.8
python3 dlt_main.py dl ensemble weights
```

### 管理元学习系统
```bash
python3 dlt_main.py dl metalearning status
python3 dlt_main.py dl metalearning enable --auto-weight --auto-retrain
python3 dlt_main.py dl metalearning disable --auto-weight --auto-retrain
python3 dlt_main.py dl metalearning retrain
```

### 优化模型
```bash
python3 dlt_main.py dl optimize quantize -m [transformer|gan|all] --type [float16|int8|dynamic]
python3 dlt_main.py dl optimize cache [--clear-cache]
python3 dlt_main.py dl optimize monitor [--report]
```

### 显示系统信息
```bash
python3 dlt_main.py dl info [--gpu] [--models] [--performance] [--all]
```

## 示例

### 训练Transformer模型
```bash
python3 dlt_main.py dl train -m transformer -p 2000 -e 200 --gpu --save-model
```

### 使用集成模型预测
```bash
python3 dlt_main.py dl predict -m ensemble -c 10 --confidence --report
```

### 生成复式投注
```bash
python3 dlt_main.py dl predict -m ensemble --compound --front-count 8 --back-count 4
```
""",
            'transformer': """
# Transformer深度学习预测

## 概述
Transformer模型是一种基于注意力机制的深度学习模型，能够有效捕捉数据中的长期依赖关系和模式。

## 工作原理
1. 将历史开奖数据转换为序列
2. 使用自注意力机制学习号码之间的关系
3. 通过多头注意力和前馈网络提取特征
4. 生成预测结果和置信度

## 使用方法
```bash
python3 dlt_main.py dl predict -m transformer -c 5 [--compound] [--confidence] [--report]
```

## 优势
- 能够捕捉长期依赖关系
- 自注意力机制有效提取特征
- 适合处理序列数据
- 预测结果稳定性高

## 适用场景
- 寻找历史数据中的长期规律
- 需要高置信度预测
- 作为集成系统的重要组成部分
""",
            'gan': """
# GAN深度学习预测

## 概述
GAN（生成对抗网络）模型由生成器和判别器组成，通过对抗训练生成符合历史分布的预测结果。

## 工作原理
1. 生成器尝试生成逼真的号码组合
2. 判别器区分真实开奖号码和生成的号码
3. 通过对抗训练提高生成质量
4. 从生成的多个结果中筛选最优预测

## 使用方法
```bash
python3 dlt_main.py dl predict -m gan -c 5 [--compound] [--confidence] [--report]
```

## 优势
- 能够生成多样化的预测结果
- 学习数据的概率分布
- 适合发现新的号码组合模式
- 创新性强

## 适用场景
- 寻找创新的号码组合
- 需要多样化预测结果
- 作为集成系统的补充部分
""",
            'ensemble': """
# 深度学习集成预测

## 概述
集成预测系统结合多种深度学习模型的优势，通过加权平均或堆叠集成方法生成更准确的预测结果。

## 工作原理
1. 使用多个预训练模型进行独立预测
2. 根据模型性能动态调整权重
3. 通过加权平均或元模型组合结果
4. 生成最终预测和综合置信度

## 使用方法
```bash
python3 dlt_main.py dl predict -m ensemble -c 5 [--compound] [--confidence] [--report]
```

## 优势
- 结合多种模型的优点
- 自适应权重调整
- 更稳定的预测性能
- 更高的整体准确率

## 适用场景
- 需要综合多种预测方法
- 追求稳定性和准确性
- 作为主要预测系统使用
"""
        }
        
        return help_content
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        获取系统信息
        
        Returns:
            系统信息字典
        """
        # 导入必要模块
        try:
            from .gpu_accelerator import GPUAccelerator
            from .metalearning_manager import MetaLearningManager
        except ImportError:
            return {'status': 'error', 'message': '导入模块失败'}
        
        # 创建GPU加速器
        accelerator = GPUAccelerator()
        
        # 创建元学习管理器
        meta = MetaLearningManager()
        
        # 获取系统信息
        info = {
            'status': 'ok',
            'gpu': {
                'available': accelerator.is_gpu_available(),
                'devices': accelerator.get_gpu_devices(),
                'memory': accelerator.get_gpu_memory_info()
            },
            'metalearning': {
                'status': meta.get_status(),
                'performance': meta.get_performance_tracking()
            },
            'models': {
                'transformer': {'available': self._check_model_available('transformer')},
                'gan': {'available': self._check_model_available('gan')},
                'ensemble': {'available': self._check_model_available('ensemble')}
            }
        }
        
        return info
    
    def _check_model_available(self, model_type: str) -> bool:
        """
        检查模型是否可用
        
        Args:
            model_type: 模型类型
            
        Returns:
            模型是否可用
        """
        model_dir = "models"
        
        if not os.path.exists(model_dir):
            return False
        
        if model_type == 'transformer':
            return any(f.startswith('transformer') and f.endswith('.h5') for f in os.listdir(model_dir))
        elif model_type == 'gan':
            return any(f.startswith('gan') and f.endswith('.h5') for f in os.listdir(model_dir))
        elif model_type == 'ensemble':
            # 检查是否有配置文件
            return os.path.exists(os.path.join(model_dir, 'ensemble_config.json'))
        
        return False


# 创建系统集成实例
system_integration = SystemIntegration()


def register_commands(main_parser):
    """
    注册命令到主CLI系统
    
    Args:
        main_parser: 主命令解析器
    """
    system_integration.register_commands(main_parser)


def handle_command(args):
    """
    处理命令
    
    Args:
        args: 命令参数
        
    Returns:
        命令执行结果代码
    """
    return system_integration.handle_command(args)


def get_prediction_methods():
    """
    获取预测方法列表
    
    Returns:
        预测方法列表
    """
    return system_integration.get_prediction_methods()


def get_help_content():
    """
    获取帮助内容
    
    Returns:
        帮助内容字典
    """
    return system_integration.get_help_content()


def get_system_info():
    """
    获取系统信息
    
    Returns:
        系统信息字典
    """
    return system_integration.get_system_info()


if __name__ == "__main__":
    # 测试系统集成
    print("🚀 测试系统集成...")
    
    # 创建系统集成
    integration = SystemIntegration()
    
    # 获取预测方法
    methods = integration.get_prediction_methods()
    print(f"可用预测方法: {len(methods)}")
    
    # 获取帮助内容
    help_content = integration.get_help_content()
    print(f"帮助内容主题: {list(help_content.keys())}")
    
    # 获取系统信息
    info = integration.get_system_info()
    print(f"系统状态: {info.get('status', 'unknown')}")
    
    print("系统集成测试完成")