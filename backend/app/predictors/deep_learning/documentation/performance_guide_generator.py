#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
性能优化指南生成器模块
Performance Guide Generator Module

提供性能优化指南、最佳实践、调优建议等文档生成功能。
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException


@dataclass
class PerformanceTip:
    """性能优化技巧"""
    title: str
    description: str
    category: str
    difficulty: str = "中级"
    impact: str = "中等"
    code_example: str = ""
    before_after: Dict[str, str] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class PerformanceGuideGenerator:
    """性能优化指南生成器"""
    
    def __init__(self):
        """初始化性能优化指南生成器"""
        self.performance_tips = []
        self._initialize_tips()
        
        logger_manager.info("性能优化指南生成器初始化完成")
    
    def _initialize_tips(self):
        """初始化性能优化技巧"""
        try:
            # CPU优化技巧
            cpu_tips = [
                PerformanceTip(
                    title="使用向量化操作",
                    description="使用NumPy和PyTorch的向量化操作替代Python循环，可以显著提升计算性能。",
                    category="CPU优化",
                    difficulty="初级",
                    impact="高",
                    code_example="""# 避免使用Python循环
# 慢速方式
result = []
for i in range(len(data)):
    result.append(data[i] * 2)

# 快速方式 - 使用向量化
import numpy as np
data = np.array(data)
result = data * 2

# PyTorch中的向量化
import torch
data = torch.tensor(data)
result = data * 2""",
                    before_after={
                        "before": "Python循环: 100ms",
                        "after": "向量化操作: 5ms (20x提升)"
                    }
                ),
                
                PerformanceTip(
                    title="优化批处理大小",
                    description="合理设置批处理大小可以平衡内存使用和计算效率。",
                    category="CPU优化",
                    difficulty="中级",
                    impact="中等",
                    code_example="""# 动态调整批处理大小
def get_optimal_batch_size(model, input_shape, max_memory_gb=8):
    import psutil
    available_memory = psutil.virtual_memory().available / (1024**3)
    
    # 估算单个样本的内存使用
    sample_memory = np.prod(input_shape) * 4 / (1024**3)  # float32
    
    # 考虑模型参数和梯度
    model_memory = sum(p.numel() * 4 for p in model.parameters()) / (1024**3)
    
    # 计算最优批大小
    optimal_batch = int((available_memory * 0.8 - model_memory) / sample_memory)
    
    return min(optimal_batch, 128)  # 限制最大批大小""",
                    prerequisites=["了解内存管理", "熟悉模型结构"]
                ),
                
                PerformanceTip(
                    title="使用多进程数据加载",
                    description="使用多进程并行加载数据，避免数据加载成为训练瓶颈。",
                    category="CPU优化",
                    difficulty="中级",
                    impact="高",
                    code_example="""# PyTorch数据加载优化
from torch.utils.data import DataLoader
import multiprocessing

# 获取CPU核心数
num_workers = min(multiprocessing.cpu_count(), 8)

# 创建优化的数据加载器
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,  # 如果使用GPU
    persistent_workers=True,  # 保持worker进程
    prefetch_factor=2  # 预取因子
)""",
                    warnings=["过多的worker可能导致内存不足", "Windows系统需要特殊处理"]
                )
            ]
            
            # GPU优化技巧
            gpu_tips = [
                PerformanceTip(
                    title="启用混合精度训练",
                    description="使用FP16混合精度训练可以减少内存使用并提升训练速度。",
                    category="GPU优化",
                    difficulty="中级",
                    impact="高",
                    code_example="""# PyTorch混合精度训练
import torch
from torch.cuda.amp import autocast, GradScaler

# 创建梯度缩放器
scaler = GradScaler()

# 训练循环
for batch in dataloader:
    optimizer.zero_grad()
    
    # 使用autocast进行前向传播
    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, targets)
    
    # 缩放损失并反向传播
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()""",
                    before_after={
                        "before": "FP32训练: 12GB显存, 100s/epoch",
                        "after": "FP16训练: 6GB显存, 70s/epoch"
                    },
                    prerequisites=["支持Tensor Core的GPU", "PyTorch 1.6+"]
                ),
                
                PerformanceTip(
                    title="优化GPU内存使用",
                    description="通过梯度累积和内存清理来优化GPU内存使用。",
                    category="GPU优化",
                    difficulty="高级",
                    impact="高",
                    code_example="""# 梯度累积减少内存使用
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 定期清理GPU缓存
if i % 100 == 0:
    torch.cuda.empty_cache()""",
                    warnings=["频繁清理缓存可能影响性能", "梯度累积会影响批归一化"]
                ),
                
                PerformanceTip(
                    title="使用编译优化",
                    description="使用TorchScript或TensorRT编译模型以获得更好的推理性能。",
                    category="GPU优化",
                    difficulty="高级",
                    impact="高",
                    code_example="""# TorchScript编译
import torch

# 方法1: 脚本化
scripted_model = torch.jit.script(model)

# 方法2: 追踪
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# 保存编译后的模型
torch.jit.save(scripted_model, "model_scripted.pt")

# 加载和使用
loaded_model = torch.jit.load("model_scripted.pt")
loaded_model.eval()""",
                    prerequisites=["模型结构相对固定", "了解TorchScript限制"]
                )
            ]
            
            # 内存优化技巧
            memory_tips = [
                PerformanceTip(
                    title="使用内存映射文件",
                    description="对于大型数据集，使用内存映射可以减少内存占用。",
                    category="内存优化",
                    difficulty="中级",
                    impact="中等",
                    code_example="""# 使用numpy内存映射
import numpy as np

# 创建内存映射数组
data = np.memmap('large_dataset.dat', dtype='float32', mode='r', shape=(1000000, 100))

# 只加载需要的部分
batch_data = data[start_idx:end_idx]

# 自定义Dataset使用内存映射
class MemmapDataset(torch.utils.data.Dataset):
    def __init__(self, filename, shape, dtype='float32'):
        self.data = np.memmap(filename, dtype=dtype, mode='r', shape=shape)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx].copy())""",
                    prerequisites=["数据已预处理为二进制格式"]
                ),
                
                PerformanceTip(
                    title="实现检查点机制",
                    description="定期保存模型检查点，避免训练中断导致的重新开始。",
                    category="内存优化",
                    difficulty="中级",
                    impact="中等",
                    code_example="""# 检查点保存和恢复
def save_checkpoint(model, optimizer, epoch, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

# 训练循环中使用
for epoch in range(start_epoch, num_epochs):
    # 训练代码...
    
    # 每10个epoch保存检查点
    if epoch % 10 == 0:
        save_checkpoint(model, optimizer, epoch, loss, f'checkpoint_epoch_{epoch}.pth')""",
                    warnings=["检查点文件可能很大", "需要定期清理旧检查点"]
                )
            ]
            
            # 分布式优化技巧
            distributed_tips = [
                PerformanceTip(
                    title="使用数据并行训练",
                    description="在多GPU环境中使用数据并行可以显著加速训练。",
                    category="分布式优化",
                    difficulty="高级",
                    impact="高",
                    code_example="""# PyTorch分布式数据并行
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化分布式环境
dist.init_process_group(backend='nccl')

# 设置设备
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# 包装模型
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])

# 使用分布式采样器
from torch.utils.data.distributed import DistributedSampler
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)""",
                    prerequisites=["多GPU环境", "了解分布式训练概念"],
                    warnings=["需要正确设置环境变量", "调试相对困难"]
                )
            ]
            
            # 合并所有技巧
            self.performance_tips.extend(cpu_tips)
            self.performance_tips.extend(gpu_tips)
            self.performance_tips.extend(memory_tips)
            self.performance_tips.extend(distributed_tips)
            
            logger_manager.debug(f"初始化了 {len(self.performance_tips)} 个性能优化技巧")
            
        except Exception as e:
            logger_manager.error(f"初始化性能优化技巧失败: {e}")
    
    def generate_performance_guide(self, output_file: str) -> bool:
        """
        生成性能优化指南
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            是否生成成功
        """
        try:
            content = self._generate_guide_content()
            
            # 保存文件
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger_manager.info(f"性能优化指南生成成功: {output_file}")
            return True
            
        except Exception as e:
            logger_manager.error(f"生成性能优化指南失败: {e}")
            return False
    
    def _generate_guide_content(self) -> str:
        """生成指南内容"""
        try:
            content = """# 深度学习平台性能优化指南

## 概述

本指南提供了深度学习模型训练和推理的性能优化技巧，帮助您充分利用硬件资源，提升模型性能。

## 性能优化原则

### 1. 识别瓶颈
- 使用性能分析工具识别瓶颈
- 监控CPU、GPU、内存和IO使用情况
- 分析数据加载和预处理时间

### 2. 分层优化
- 算法层面：选择合适的模型架构
- 实现层面：优化代码和数据流
- 硬件层面：充分利用GPU和多核CPU

### 3. 平衡权衡
- 精度 vs 速度
- 内存使用 vs 计算速度
- 开发时间 vs 性能提升

## 性能监控工具

```python
# 使用PyTorch Profiler
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, batch_data in enumerate(dataloader):
        # 训练代码
        prof.step()
```

"""
            
            # 按类别组织技巧
            categories = {}
            for tip in self.performance_tips:
                if tip.category not in categories:
                    categories[tip.category] = []
                categories[tip.category].append(tip)
            
            # 生成各类别的内容
            for category, tips in categories.items():
                content += f"## {category}\n\n"
                
                for tip in tips:
                    content += f"### {tip.title}\n\n"
                    content += f"**难度**: {tip.difficulty} | **影响**: {tip.impact}\n\n"
                    content += f"{tip.description}\n\n"
                    
                    if tip.prerequisites:
                        content += "**前置条件**:\n"
                        for prereq in tip.prerequisites:
                            content += f"- {prereq}\n"
                        content += "\n"
                    
                    if tip.code_example:
                        content += "**代码示例**:\n\n"
                        content += f"```python\n{tip.code_example}\n```\n\n"
                    
                    if tip.before_after:
                        content += "**性能对比**:\n"
                        content += f"- 优化前: {tip.before_after.get('before', 'N/A')}\n"
                        content += f"- 优化后: {tip.before_after.get('after', 'N/A')}\n\n"
                    
                    if tip.warnings:
                        content += "**注意事项**:\n"
                        for warning in tip.warnings:
                            content += f"⚠️ {warning}\n"
                        content += "\n"
                    
                    content += "---\n\n"
            
            # 添加最佳实践总结
            content += """## 最佳实践总结

### 训练阶段优化
1. **数据预处理**: 离线预处理数据，使用高效的数据格式
2. **批处理**: 使用合适的批大小，启用数据并行
3. **混合精度**: 在支持的硬件上使用FP16训练
4. **梯度累积**: 在内存受限时使用梯度累积
5. **检查点**: 定期保存模型检查点

### 推理阶段优化
1. **模型编译**: 使用TorchScript或ONNX优化模型
2. **批量推理**: 批量处理多个样本
3. **模型量化**: 使用INT8量化减少模型大小
4. **缓存**: 缓存常用的计算结果
5. **异步处理**: 使用异步IO和计算

### 内存优化
1. **内存映射**: 对大数据集使用内存映射
2. **垃圾回收**: 定期清理不需要的变量
3. **梯度检查点**: 使用梯度检查点减少内存使用
4. **模型并行**: 在内存不足时使用模型并行

### 监控和调试
1. **性能分析**: 使用profiler识别瓶颈
2. **资源监控**: 监控CPU、GPU、内存使用
3. **日志记录**: 记录关键性能指标
4. **A/B测试**: 对比不同优化策略的效果

## 常见问题解决

### Q: 训练速度慢怎么办？
A: 
1. 检查数据加载是否是瓶颈
2. 尝试增加批大小
3. 使用混合精度训练
4. 考虑使用多GPU训练

### Q: 内存不足怎么办？
A:
1. 减小批大小
2. 使用梯度累积
3. 启用梯度检查点
4. 使用内存映射数据集

### Q: GPU利用率低怎么办？
A:
1. 增加批大小
2. 优化数据加载流水线
3. 减少CPU-GPU数据传输
4. 使用异步操作

## 参考资源

- [PyTorch性能调优指南](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [NVIDIA深度学习性能指南](https://docs.nvidia.com/deeplearning/performance/index.html)
- [混合精度训练指南](https://pytorch.org/docs/stable/amp.html)
- [分布式训练最佳实践](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

---

*本指南会持续更新，欢迎提供反馈和建议。*
"""
            
            return content
            
        except Exception as e:
            logger_manager.error(f"生成指南内容失败: {e}")
            return ""
    
    def add_custom_tip(self, tip: PerformanceTip):
        """
        添加自定义优化技巧
        
        Args:
            tip: 性能优化技巧
        """
        try:
            self.performance_tips.append(tip)
            logger_manager.debug(f"添加自定义优化技巧: {tip.title}")
            
        except Exception as e:
            logger_manager.error(f"添加自定义优化技巧失败: {e}")
    
    def get_tips_by_category(self, category: str) -> List[PerformanceTip]:
        """
        按类别获取优化技巧
        
        Args:
            category: 类别名称
            
        Returns:
            该类别的优化技巧列表
        """
        try:
            return [tip for tip in self.performance_tips if tip.category == category]
            
        except Exception as e:
            logger_manager.error(f"按类别获取优化技巧失败: {e}")
            return []


# 全局性能优化指南生成器实例
performance_guide_generator = PerformanceGuideGenerator()


if __name__ == "__main__":
    # 测试性能优化指南生成器功能
    print("⚡ 测试性能优化指南生成器功能...")
    
    try:
        generator = PerformanceGuideGenerator()
        
        # 生成性能优化指南
        if generator.generate_performance_guide("performance_optimization_guide.md"):
            print("✅ 性能优化指南生成成功")
        
        # 测试按类别获取技巧
        cpu_tips = generator.get_tips_by_category("CPU优化")
        print(f"✅ CPU优化技巧: {len(cpu_tips)} 个")
        
        gpu_tips = generator.get_tips_by_category("GPU优化")
        print(f"✅ GPU优化技巧: {len(gpu_tips)} 个")
        
        # 测试添加自定义技巧
        custom_tip = PerformanceTip(
            title="自定义优化技巧",
            description="这是一个自定义的优化技巧示例",
            category="自定义优化",
            difficulty="初级",
            impact="低"
        )
        
        generator.add_custom_tip(custom_tip)
        print("✅ 自定义技巧添加成功")
        
        print("✅ 性能优化指南生成器功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("性能优化指南生成器功能测试完成")
