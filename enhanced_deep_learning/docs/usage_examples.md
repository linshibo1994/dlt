# 大乐透深度学习预测系统使用示例

## 目录

1. [基本使用](#基本使用)
2. [训练模型](#训练模型)
3. [预测号码](#预测号码)
4. [模型集成](#模型集成)
5. [元学习系统](#元学习系统)
6. [性能优化](#性能优化)
7. [高级用法](#高级用法)

## 基本使用

### 命令行使用

```bash
# 显示帮助信息
python3 dlt_main.py dl --help

# 显示系统信息
python3 dlt_main.py dl info --all

# 显示GPU信息
python3 dlt_main.py dl info --gpu
```

### 在Python中使用

```python
from enhanced_deep_learning.system_integration import system_integration

# 获取系统信息
info = system_integration.get_system_info()
print(f"GPU可用性: {'可用' if info['gpu']['available'] else '不可用'}")
print(f"可用模型: {info['models']}")

# 获取预测方法
methods = system_integration.get_prediction_methods()
for method in methods:
    print(f"预测方法: {method['display_name']}")
```

## 训练模型

### 命令行训练

```bash
# 训练Transformer模型
python3 dlt_main.py dl train -m transformer -p 2000 -e 200 -b 32 --gpu --save-model

# 训练GAN模型
python3 dlt_main.py dl train -m gan -p 2000 -e 300 -b 64 --gpu --save-model

# 训练所有模型
python3 dlt_main.py dl train -m all -p 1000 -e 100 --gpu --save-model
```

### 在Python中训练

```python
import pandas as pd
from enhanced_deep_learning.transformer_model import TransformerModel
from enhanced_deep_learning.gan_model import GANModel
from core_modules import data_manager

# 获取数据
df = data_manager.get_data(periods=1000)

# 训练Transformer模型
transformer = TransformerModel()
transformer.train(df, epochs=100, batch_size=32, use_gpu=True)
transformer.save_model()

# 训练GAN模型
gan = GANModel()
gan.train(df, epochs=150, batch_size=64, use_gpu=True)
gan.save_model()
```

## 预测号码

### 命令行预测

```bash
# 使用Transformer模型预测
python3 dlt_main.py dl predict -m transformer -c 5 --confidence

# 使用GAN模型预测
python3 dlt_main.py dl predict -m gan -c 5 --confidence

# 使用集成模型预测
python3 dlt_main.py dl predict -m ensemble -c 10 --confidence --report

# 生成复式投注
python3 dlt_main.py dl predict -m ensemble --compound --front-count 8 --back-count 4
```

### 在Python中预测

```python
from enhanced_deep_learning.transformer_model import TransformerModel
from enhanced_deep_learning.gan_model import GANModel
from enhanced_deep_learning.ensemble_manager import EnsembleManager
from core_modules import data_manager

# 获取数据
df = data_manager.get_data(periods=20)

# 使用Transformer模型预测
transformer = TransformerModel()
transformer.load_model()
predictions, confidences = transformer.predict(df, count=5)

# 显示预测结果
for i, (pred, conf) in enumerate(zip(predictions, confidences)):
    front_balls = pred[0]
    back_balls = pred[1]
    print(f"预测 {i+1}: 前区 {front_balls} 后区 {back_balls} 置信度: {conf:.4f}")

# 使用集成模型预测
ensemble = EnsembleManager()
ensemble.load_models()
predictions, confidences = ensemble.predict(df, count=5)
```

## 模型集成

### 命令行管理集成

```bash
# 列出集成模型
python3 dlt_main.py dl ensemble list

# 添加模型到集成
python3 dlt_main.py dl ensemble add -m transformer -w 0.6

# 更新模型权重
python3 dlt_main.py dl ensemble update -m transformer -w 0.8

# 移除模型
python3 dlt_main.py dl ensemble remove -m gan

# 显示模型权重
python3 dlt_main.py dl ensemble weights

# 设置集成方法
python3 dlt_main.py dl ensemble list --method stacking
```

### 在Python中管理集成

```python
from enhanced_deep_learning.ensemble_manager import EnsembleManager

# 创建集成管理器
ensemble = EnsembleManager()

# 添加模型
ensemble.add_model('transformer', weight=0.6)
ensemble.add_model('gan', weight=0.4)

# 更新模型权重
ensemble.update_model_weight('transformer', weight=0.7)

# 设置集成方法
ensemble.set_ensemble_method('stacking')

# 保存配置
ensemble.save_config()

# 列出模型
models = ensemble.list_models()
print(f"集成模型: {models}")
```

## 元学习系统

### 命令行管理元学习

```bash
# 显示元学习状态
python3 dlt_main.py dl metalearning status

# 启用自动权重调整
python3 dlt_main.py dl metalearning enable --auto-weight

# 启用自动重训练
python3 dlt_main.py dl metalearning enable --auto-retrain

# 设置性能退化阈值
python3 dlt_main.py dl metalearning enable --threshold 0.1

# 禁用自动权重调整
python3 dlt_main.py dl metalearning disable --auto-weight

# 触发重训练
python3 dlt_main.py dl metalearning retrain
```

### 在Python中管理元学习

```python
from enhanced_deep_learning.metalearning_manager import MetaLearningManager

# 创建元学习管理器
meta = MetaLearningManager()

# 启用自动权重调整
meta.enable_auto_weight_adjustment(True)

# 启用自动重训练
meta.enable_auto_retraining(True)

# 设置性能退化阈值
meta.set_performance_threshold(0.05)

# 获取状态
status = meta.get_status()
print(f"元学习状态: {status}")

# 获取性能跟踪
performance = meta.get_performance_tracking()
print(f"性能跟踪: {performance}")

# 触发重训练
meta.trigger_retraining()

# 保存配置
meta.save_config()
```

## 性能优化

### 命令行优化

```bash
# 量化模型
python3 dlt_main.py dl optimize quantize -m transformer --type float16

# 管理缓存
python3 dlt_main.py dl optimize cache
python3 dlt_main.py dl optimize cache --clear-cache

# 监控资源使用
python3 dlt_main.py dl optimize monitor --report
```

### 在Python中优化

```python
from enhanced_deep_learning.model_optimizer import ModelOptimizer, CachedModel
from enhanced_deep_learning.transformer_model import TransformerModel
from enhanced_deep_learning.gpu_accelerator import GPUAccelerator, GPUMemoryManager
from enhanced_deep_learning.batch_processor import BatchProcessor

# 创建组件
accelerator = GPUAccelerator()
memory_manager = GPUMemoryManager(accelerator)
optimizer = ModelOptimizer()
batch_processor = BatchProcessor(memory_manager=memory_manager)

# 加载模型
transformer = TransformerModel()
transformer.load_model()

# 量化模型
quantized_model = optimizer.quantize_model(transformer.model, 'tensorflow', 'float16')
transformer.model = quantized_model
transformer.save_model(suffix='_quantized')

# 创建缓存模型
cached_model = CachedModel(transformer.model, optimizer)

# 监控资源使用
optimizer.start_resource_monitoring()
usage = optimizer.record_resource_usage()
print(f"CPU使用率: {usage['cpu']}%")
print(f"内存使用率: {usage['memory']}%")

# 优化批处理大小
batch_size = batch_processor.optimize_batch_size(500)
print(f"优化后的批处理大小: {batch_size}")
```

## 高级用法

### 数据处理

```python
import pandas as pd
import numpy as np
from enhanced_deep_learning.data_processor import DataProcessor
from enhanced_deep_learning.data_augmentor import DataAugmentor
from enhanced_deep_learning.anomaly_detector import AnomalyDetector
from core_modules import data_manager

# 获取数据
df = data_manager.get_data(periods=500)

# 创建数据处理器
processor = DataProcessor()

# 处理数据
processed_data = processor.process_data(df, augment=True, detect_anomalies=True)

# 获取处理后的特征
normalized_features = processed_data['normalized_features']
print(f"归一化特征形状: {normalized_features.shape}")

# 准备序列数据
X, y = processor.prepare_sequence_data(normalized_features, sequence_length=10)
print(f"序列数据形状: X={X.shape}, y={y.shape}")

# 单独使用数据增强器
augmentor = DataAugmentor()
augmented_features = augmentor.augment_data(normalized_features, factor=1.5)
print(f"增强后的特征形状: {augmented_features.shape}")

# 单独使用异常检测器
detector = AnomalyDetector()
normal_features, anomaly_features = detector.detect_anomalies(normalized_features)
print(f"正常特征数量: {len(normal_features)}, 异常特征数量: {len(anomaly_features)}")
```

### GPU加速和批处理

```python
import numpy as np
from enhanced_deep_learning.gpu_accelerator import GPUAccelerator, GPUMemoryManager
from enhanced_deep_learning.batch_processor import BatchProcessor

# 创建组件
accelerator = GPUAccelerator()
memory_manager = GPUMemoryManager(accelerator)
batch_processor = BatchProcessor(memory_manager=memory_manager)

# 检查GPU可用性
if accelerator.is_gpu_available():
    print(f"可用GPU: {accelerator.get_gpu_devices()}")
    
    # 启用TensorFlow GPU
    accelerator.enable_tensorflow_gpu()
    
    # 启用PyTorch GPU
    accelerator.enable_pytorch_gpu()
    
    # 监控GPU使用情况
    usage = accelerator.monitor_gpu_usage()
    for device, memory in usage['gpu_memory'].items():
        print(f"{device}: 已用 {memory.get('used', 0):.2f}GB / 总共 {memory.get('total', 0):.2f}GB")

# 创建测试数据
data = np.random.rand(1000, 10)

# 定义处理函数
def process_function(batch):
    return batch * 2

# 批量处理
result1 = batch_processor.batch_process(data, process_function)

# 并行处理
result2 = batch_processor.parallel_batch_process(data, process_function, use_processes=True)

# 动态批处理
result3 = batch_processor.dynamic_batch_processing(data, process_function, target_time_per_batch=0.1)
```

### 缓存和资源监控

```python
import numpy as np
import time
from enhanced_deep_learning.model_optimizer import ModelOptimizer, CachedModel

# 创建模型优化器
optimizer = ModelOptimizer()

# 使用缓存装饰器
@optimizer.cache_result
def expensive_calculation(x):
    time.sleep(1)  # 模拟耗时计算
    return x * 2

# 第一次调用（执行计算）
result1 = expensive_calculation(10)
print(f"结果1: {result1}")

# 第二次调用（从缓存加载）
result2 = expensive_calculation(10)
print(f"结果2: {result2}")

# 创建模拟模型
class MockModel:
    def predict(self, inputs):
        time.sleep(0.5)  # 模拟预测时间
        return inputs * 2

# 创建缓存模型
model = MockModel()
cached_model = CachedModel(model, optimizer)

# 预测
inputs = np.array([[1, 2, 3]])
result1 = cached_model.predict(inputs)
result2 = cached_model.predict(inputs)  # 应该从缓存加载

# 获取缓存统计信息
stats = cached_model.get_cache_stats()
print(f"缓存命中率: {stats['hit_rate']:.2%}")

# 资源监控
optimizer.start_resource_monitoring()

for _ in range(5):
    usage = optimizer.record_resource_usage()
    print(f"CPU: {usage['cpu']:.1f}%, 内存: {usage['memory']:.1f}%")
    time.sleep(1)

# 获取资源使用情况摘要
summary = optimizer.get_resource_usage_summary()
print(f"CPU平均使用率: {summary['cpu']['avg']:.1f}%")
print(f"内存平均使用率: {summary['memory']['avg']:.1f}%")

# 保存资源使用情况报告
report_path = optimizer.save_resource_usage_report()
print(f"报告已保存到: {report_path}")
```