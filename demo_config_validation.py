#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置验证功能演示

展示backend/app/main.py中的配置验证机制
"""

import sys
import os

# 添加路径
sys.path.insert(0, 'backend/app')
sys.path.insert(0, 'backend/app/core')

print("=" * 70)
print("配置验证功能演示")
print("=" * 70)

# 演示1: 路径配置管理器
print("\n【演示1】路径配置管理器")
print("-" * 70)
try:
    from core.path_config import get_path_manager
    pm = get_path_manager()

    print(f"✅ 项目根目录: {pm.base_dir}")
    print(f"✅ 数据目录: {pm.data_dir}")
    print(f"✅ 数据文件: {pm.data_file}")
    print(f"✅ 缓存目录: {pm.cache_dir}")
    print(f"✅ 日志目录: {pm.logs_dir}")
    print(f"✅ 模型目录: {pm.models_dir}")
    print(f"✅ 报告目录: {pm.reports_dir}")

except Exception as e:
    print(f"❌ 错误: {e}")

# 演示2: DLTPredictorSystem配置验证
print("\n【演示2】DLTPredictorSystem配置验证")
print("-" * 70)
try:
    import core_modules
    from main import DLTPredictorSystem

    print("正在初始化系统...")
    system = DLTPredictorSystem()

    print("\n✅ 配置验证通过！系统已成功初始化")
    print(f"   - 分析器已加载: {system._analyzers_loaded}")
    print(f"   - 预测器已加载: {system._predictors_loaded}")
    print(f"   - 自适应学习已加载: {system._adaptive_loaded}")
    print(f"   - 增强功能可用: {system.enhanced_available}")

except Exception as e:
    print(f"❌ 初始化失败: {e}")

# 演示3: 验证方法详细输出
print("\n【演示3】验证方法详细测试")
print("-" * 70)
try:
    from core.path_config import get_path_manager
    pm = get_path_manager()

    # 测试关键路径
    test_paths = [
        ("数据文件", pm.data_file),
        ("缓存目录", pm.cache_dir),
        ("日志目录", pm.logs_dir),
        ("模型目录", pm.models_dir),
        ("报告目录", pm.reports_dir),
        ("预测配置文件", pm.prediction_config_file),
        ("训练配置文件", pm.training_config_file),
    ]

    print("\n路径检查结果：")
    for name, path in test_paths:
        if path.exists():
            if path.is_file():
                print(f"✅ {name}: {path} (文件)")
            else:
                print(f"✅ {name}: {path} (目录)")
        else:
            print(f"⚠️  {name}: {path} (不存在)")

except Exception as e:
    print(f"❌ 错误: {e}")

# 演示4: 数据文件验证
print("\n【演示4】数据文件详细验证")
print("-" * 70)
try:
    from core.path_config import get_path_manager
    pm = get_path_manager()

    if pm.data_file.exists():
        print(f"✅ 数据文件存在: {pm.data_file}")

        # 检查文件大小
        size = pm.data_file.stat().st_size
        print(f"   文件大小: {size:,} 字节 ({size / 1024 / 1024:.2f} MB)")

        # 尝试读取第一行
        with open(pm.data_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            print(f"   第一行: {first_line[:100]}..." if len(first_line) > 100 else f"   第一行: {first_line}")

            # 统计行数（快速估算）
            lines = sum(1 for _ in f) + 1  # +1 for the first line we already read
            print(f"   总行数: {lines:,}")

        print("\n✅ 数据文件验证通过")
    else:
        print(f"❌ 数据文件不存在: {pm.data_file}")

except Exception as e:
    print(f"❌ 数据文件验证失败: {e}")

# 总结
print("\n" + "=" * 70)
print("配置验证功能演示完成")
print("=" * 70)
print("\n配置验证机制特性：")
print("  1. ✅ 自动验证关键目录和文件")
print("  2. ✅ 自动创建缺失的目录")
print("  3. ✅ 详细的错误信息和日志记录")
print("  4. ✅ 早期失败机制，快速发现配置问题")
print("  5. ✅ 支持多种导入路径，提高兼容性")
print("\n使用建议：")
print("  • 运行任何命令前都会自动进行配置验证")
print("  • 如果配置有问题，系统会立即报错并给出修复建议")
print("  • 日志文件会记录所有验证过程和结果")
print()
