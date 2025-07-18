# 大乐透深度学习预测系统 API 参考

## 目录

1. [GPU加速器](#gpu加速器)
2. [批处理系统](#批处理系统)
3. [模型优化器](#模型优化器)
4. [数据处理器](#数据处理器)
5. [CLI命令](#cli命令)
6. [系统集成](#系统集成)

## GPU加速器

### GPUAccelerator

GPU加速器提供GPU加速支持和设备管理功能。

#### 主要方法

```python
# 初始化
accelerator = GPUAccelerator()

# 检查GPU可用性
is_available = accelerator.is_gpu_available()

# 获取GPU设备列表
devices = accelerator.get_gpu_devices()

# 获取GPU内存信息
memory_info = accelerator.get_gpu_memory_info()

# 启用TensorFlow GPU
success = accelerator.enable_tensorflow_gpu(memory_limit=None)

# 启用PyTorch GPU
success = accelerator.enable_pytorch_gpu(device_id=0)

# 将模型迁移到GPU
gpu_model = accelerator.migrate_model_to_gpu(model, framework='tensorflow')

# 清理GPU内存
success = accelerator.clear_gpu_memory(framework='all')

# 监控GPU使用情况
usage_info = accelerator.monitor_gpu_usage()
```

### GPUMemoryManager

GPU内存管理器提供内存限制和自动清理功能。

#### 主要方法

```python
# 初始化
memory_manager = GPUMemoryManager(accelerator)

# 设置内存限制
success = memory_manager.set_memory_limit(limit_mb=1024)

# 启用自动清理
memory_manager.enable_auto_clear(enabled=True)

# 根据需要清理内存
cleared = memory_manager.clear_if_needed(threshold_percent=80.0)

# 优化批处理大小
batch_size = memory_manager.optimize_batch_size(initial_batch_size=32, model_size_mb=500)
```

## 批处理系统

### BatchProcessor

批处理系统提供批量预测和动态批大小调整功能。

#### 主要方法

```python
# 初始化
processor = BatchProcessor(initial_batch_size=32, memory_manager=None, max_workers=None)

# 设置批处理大小
processor.set_batch_size(batch_size=64)

# 启用自适应批处理大小
processor.enable_adaptive_batch_size(enabled=True)

# 优化批处理大小
batch_size = processor.optimize_batch_size(model_size_mb=500)

# 批量处理数据
results = processor.batch_process(data, process_func)

# 并行批量处理数据
results = processor.parallel_batch_process(data, process_func, use_processes=False)

# 动态批处理
results = processor.dynamic_batch_processing(data, process_func, initial_batch_size=None, target_time_per_batch=1.0)
```

## 模型优化器

### ModelOptimizer

模型优化器提供模型量化、中间结果缓存和资源使用监控功能。

#### 主要方法

```python
# 初始化
optimizer = ModelOptimizer(cache_dir=None)

# 量化TensorFlow模型
quantized_model = optimizer.quantize_tensorflow_model(model, quantization_type='float16')

# 量化PyTorch模型
quantized_model = optimizer.quantize_pytorch_model(model, quantization_type='float16')

# 量化模型（通用）
quantized_model = optimizer.quantize_model(model, framework='tensorflow', quantization_type='float16')

# 缓存函数结果（装饰器）
@optimizer.cache_result
def expensive_function(x):
    return x * 2

# 清除结果缓存
optimizer.clear_result_cache()

# 开始资源监控
optimizer.start_resource_monitoring()

# 记录资源使用情况
usage = optimizer.record_resource_usage()

# 获取资源使用情况摘要
summary = optimizer.get_resource_usage_summary()

# 保存资源使用情况报告
report_path = optimizer.save_resource_usage_report(file_path=None)

# 优化模型推理
optimized_model = optimizer.optimize_model_inference(model, framework='tensorflow')
```

### CachedModel

缓存模型提供模型预测结果缓存功能。

#### 主要方法

```python
# 初始化
cached_model = CachedModel(model, optimizer, cache_size=100)

# 预测
results = cached_model.predict(inputs)

# 获取缓存统计信息
stats = cached_model.get_cache_stats()

# 清除缓存
cached_model.clear_cache()
```

## 数据处理器

### DataProcessor

数据处理器整合数据预处理、特征提取、数据增强和异常检测功能。

#### 主要方法

```python
# 初始化
processor = DataProcessor(normalization_method='minmax', anomaly_detection_method='zscore', cache_enabled=True)

# 处理数据
processed_data = processor.process_data(df, augment=False, detect_anomalies=False)

# 准备序列数据
X, y = processor.prepare_sequence_data(features, sequence_length=10)

# 清除缓存
processor.clear_cache()
```

### DataPreprocessor

数据预处理器提供数据标准化和归一化功能。

#### 主要方法

```python
# 初始化
preprocessor = DataPreprocessor(normalization_method='minmax', cache_enabled=True)

# 归一化数据
normalized_data = preprocessor.normalize_data(data, feature_name='default')

# 反归一化数据
original_data = preprocessor.inverse_normalize(data, feature_name='default')
```

### FeatureExtractor

特征提取器提供从原始数据中提取特征的功能。

#### 主要方法

```python
# 初始化
extractor = FeatureExtractor(cache_enabled=True)

# 提取基本特征
basic_features = extractor.extract_basic_features(df)

# 提取统计特征
statistical_features = extractor.extract_statistical_features(df)
```

### DataAugmentor

数据增强器提供数据增强功能。

#### 主要方法

```python
# 初始化
augmentor = DataAugmentor()

# 数据增强
augmented_features = augmentor.augment_data(features, factor=1.5)

# 抖动增强
jittered_features = augmentor.augment_by_jittering(features, noise_level=0.03)

# 混合增强
mixed_features = augmentor.augment_by_mixup(features, alpha=0.2)
```

### AnomalyDetector

异常检测器提供异常数据检测功能。

#### 主要方法

```python
# 初始化
detector = AnomalyDetector(method='isolation_forest')

# 检测异常
normal_features, anomaly_features = detector.detect_anomalies(features, contamination=0.05)
```

## CLI命令

### DeepLearningCommands

深度学习命令集提供训练、预测、集成和元学习命令。

#### 主要方法

```python
# 初始化
commands = DeepLearningCommands()

# 获取命令解析器
parser = commands.get_command_parser()

# 执行命令
result = commands.execute_command(args)

# 训练命令
result = commands.train_command(args)

# 预测命令
result = commands.predict_command(args)

# 集成命令
result = commands.ensemble_command(args)

# 元学习命令
result = commands.metalearning_command(args)

# 优化命令
result = commands.optimize_command(args)

# 信息命令
result = commands.info_command(args)
```

### CLIHandler

CLI处理器提供与主CLI系统的集成。

#### 主要方法

```python
# 初始化
handler = CLIHandler()

# 处理命令
result = handler.handle_command(args)

# 注册到主CLI系统
handler.register_with_main_cli(main_parser)

# 执行主CLI命令
result = handler.execute_main_cli_command(args)
```

## 系统集成

### SystemIntegration

系统集成提供与现有系统的集成功能。

#### 主要方法

```python
# 初始化
integration = SystemIntegration()

# 注册命令到主CLI系统
integration.register_commands(main_parser)

# 处理命令
result = integration.handle_command(args)

# 获取预测方法列表
methods = integration.get_prediction_methods()

# 获取帮助内容
help_content = integration.get_help_content()

# 获取系统信息
info = integration.get_system_info()
```