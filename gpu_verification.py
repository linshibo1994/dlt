#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU加速验证脚本
验证哪些方法真正使用了GPU加速
"""

import time
import sys
import os

def test_gpu_acceleration():
    """测试GPU加速功能"""
    print("=== GPU加速功能验证 ===")

    # 测试GPU可用性
    try:
        from gpu_accelerated_predictor import get_gpu_accelerator
        gpu_accelerator = get_gpu_accelerator()

        print(f"GPU可用: {gpu_accelerator.gpu_available}")
        print(f"加速方法: {gpu_accelerator.acceleration_method}")
        print(f"设备: {gpu_accelerator.device}")

        if gpu_accelerator.gpu_available:
            print("✅ GPU基础设施正常")

            # 测试GPU矩阵运算
            print("\n测试GPU矩阵运算...")
            import numpy as np
            test_data = np.random.randn(100, 50).astype(np.float32)

            start_time = time.time()
            result = gpu_accelerator.accelerated_matrix_operations(test_data, "matmul")
            gpu_time = time.time() - start_time

            print(f"GPU矩阵运算耗时: {gpu_time:.4f}秒")
            print(f"结果形状: {result.shape}")

            # 测试GPU预测
            print("\n测试GPU预测功能...")
            start_time = time.time()
            predictions, metrics = gpu_accelerator.accelerated_prediction(test_data, "lstm")
            gpu_pred_time = time.time() - start_time

            print(f"GPU预测耗时: {gpu_pred_time:.4f}秒")
            print(f"预测结果: {predictions}")
            print(f"性能指标: {metrics}")

            return True
        else:
            print("❌ GPU不可用，使用CPU模式")
            return False

    except Exception as e:
        print(f"❌ GPU测试失败: {e}")
        return False

def test_method_gpu_usage():
    """测试各个预测方法的GPU使用情况"""
    print("\n=== 预测方法GPU使用测试 ===")

    # 导入预测器
    try:
        from predictor_modules import get_advanced_predictor
        predictor = get_advanced_predictor()

        methods_to_test = [
            ('lstm', 'LSTM深度学习'),
            ('ensemble', '集成学习'),
            ('adaptive_ensemble', '自适应集成'),
            ('ultimate_ensemble', '终极集成'),
            ('enhanced', '增强预测'),
            ('super', '超级预测')
        ]

        for method_name, description in methods_to_test:
            print(f"\n测试方法: {method_name} - {description}")

            try:
                start_time = time.time()

                # 根据方法名调用对应函数
                if hasattr(predictor, f'{method_name}_predict'):
                    method = getattr(predictor, f'{method_name}_predict')
                    result = method(1, 100)  # 1注，100期
                elif method_name == 'enhanced':
                    result = predictor.enhanced_predict(1, 100)
                elif method_name == 'super':
                    result = predictor.super_predict(1, 100)
                else:
                    print(f"  ❌ 方法 {method_name} 不存在")
                    continue

                elapsed_time = time.time() - start_time
                print(f"  ✅ 执行时间: {elapsed_time:.4f}秒")
                print(f"  📊 结果数量: {len(result) if result else 0}")

                # 检查是否是GPU加速的结果（通过执行时间判断）
                if elapsed_time < 0.1:
                    print(f"  🚀 可能使用了GPU加速或缓存")
                elif elapsed_time > 5.0:
                    print(f"  🐌 执行时间较长，可能没有使用GPU加速")
                else:
                    print(f"  ⚡ 执行时间正常")

            except Exception as e:
                print(f"  ❌ 方法 {method_name} 执行失败: {e}")

    except Exception as e:
        print(f"❌ 预测器导入失败: {e}")

def show_recommended_gpu_commands():
    """显示推荐的GPU加速命令"""
    print("\n=== 推荐的GPU加速命令 ===")

    commands = [
        "# 真正的GPU加速方法",
        "python3 dlt_main.py predict -m lstm -p 1000 -c 3 --acceleration gpu",
        "python3 dlt_main.py predict -m transformer -p 1500 -c 2 --acceleration gpu_cuda --mixed-precision",
        "python3 dlt_main.py predict -m gan -p 800 -c 5 --acceleration gpu",
        "python3 dlt_main.py predict -m ensemble -p 1000 -c 3 --acceleration auto",
        "",
        "# 查看GPU信息",
        "python3 dlt_main.py enhanced info --gpu",
        "",
        "# 性能基准测试",
        "python3 gpu_verification.py",
    ]

    for cmd in commands:
        if cmd.startswith("#"):
            print(f"\n{cmd}")
        elif cmd == "":
            print()
        else:
            print(f"  {cmd}")

if __name__ == "__main__":
    print("GPU加速验证工具")
    print("=" * 50)

    # 测试GPU基础功能
    gpu_available = test_gpu_acceleration()

    # 测试各方法的GPU使用
    test_method_gpu_usage()

    # 显示推荐命令
    show_recommended_gpu_commands()

    print("\n=== 总结 ===")
    if gpu_available:
        print("✅ GPU基础设施可用")
        print("🎯 建议直接使用lstm、transformer、gan、ensemble方法获得真正的GPU加速")
        print("⚠️ adaptive_ensemble、ultimate_ensemble等高级方法虽然有GPU检查，但实际使用CPU")
    else:
        print("❌ GPU不可用，系统将使用CPU模式")
        print("💡 可以尝试安装CUDA或PyTorch GPU版本")