#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
系统全面检查脚本
System Comprehensive Check Script

检查所有预测方法的可用性和功能完整性
"""

import subprocess
import sys
import time
from typing import Dict, List, Tuple

# 定义所有预测方法
PREDICTION_METHODS = {
    "传统方法": [
        "frequency",      # 频率分析
        "hot_cold",       # 冷热分析
        "missing",        # 遗漏分析
        "markov",         # 马尔可夫链
        "bayesian",       # 贝叶斯分析
    ],
    "集成方法": [
        "ensemble",       # 集成预测
        "stacking",       # 堆叠集成
        "adaptive_ensemble",  # 自适应集成
        "ultimate_ensemble",  # 终极集成
    ],
    "智能方法": [
        "super",          # 超级预测
        "adaptive",       # 自适应预测
        "enhanced",       # 增强预测
        "mixed_strategy", # 混合策略
        "highly_integrated",  # 高度集成
        "advanced_integration",  # 高级集成
    ],
    "复式投注": [
        "compound",       # 复式投注
        "duplex",         # 胆拖投注
        "nine_models_compound",  # 九种模型复式
        "markov_compound",       # 马尔可夫复式
    ],
    "深度学习": [
        "lstm",           # LSTM神经网络
        "transformer",    # Transformer
        "gan",            # 生成对抗网络
    ],
    "高级马尔可夫": [
        "markov_custom",  # 自定义马尔可夫
        "markov_2nd",     # 二阶马尔可夫
        "markov_3rd",     # 三阶马尔可夫
        "adaptive_markov", # 自适应马尔可夫
    ],
    "九种模型": [
        "nine_models",    # 九种数学模型
    ]
}

def run_prediction_test(method: str, timeout: int = 60) -> Tuple[bool, str, str]:
    """
    测试单个预测方法
    
    Args:
        method: 预测方法名称
        timeout: 超时时间（秒）
        
    Returns:
        (是否成功, 输出信息, 错误信息)
    """
    try:
        cmd = [
            "python3", "dlt_main.py", "predict", 
            "-m", method, 
            "-p", "50",  # 使用较少的分析期数以加快速度
            "-c", "1"    # 只生成1注
        ]
        
        print(f"  测试命令: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/Users/linshibo/GithubProject/dlt"
        )
        
        success = result.returncode == 0
        output = result.stdout
        error = result.stderr
        
        return success, output, error
        
    except subprocess.TimeoutExpired:
        return False, "", f"测试超时（{timeout}秒）"
    except Exception as e:
        return False, "", f"测试异常: {str(e)}"

def check_prediction_methods():
    """检查所有预测方法"""
    print("🔍 开始检查所有预测方法...")
    print("=" * 60)
    
    results = {}
    total_methods = 0
    successful_methods = 0
    
    for category, methods in PREDICTION_METHODS.items():
        print(f"\n📂 {category} ({len(methods)}个方法)")
        print("-" * 40)
        
        category_results = {}
        
        for method in methods:
            total_methods += 1
            print(f"\n🧪 测试方法: {method}")
            
            # 对深度学习方法使用更长的超时时间
            timeout = 180 if category == "深度学习" else 60
            
            success, output, error = run_prediction_test(method, timeout)
            
            if success:
                successful_methods += 1
                print(f"  ✅ 成功")
                # 提取预测结果
                if "预测结果:" in output:
                    result_lines = output.split("预测结果:")[1].split("\n")[:3]
                    for line in result_lines:
                        if line.strip():
                            print(f"    {line.strip()}")
            else:
                print(f"  ❌ 失败")
                if error:
                    print(f"    错误: {error[:200]}...")
                if "error:" in output.lower():
                    error_lines = [line for line in output.split("\n") if "error" in line.lower()]
                    for line in error_lines[:2]:
                        print(f"    {line.strip()}")
            
            category_results[method] = {
                'success': success,
                'output': output,
                'error': error
            }
            
            # 短暂延迟避免系统过载
            time.sleep(1)
        
        results[category] = category_results
    
    # 输出总结
    print("\n" + "=" * 60)
    print("📊 检查结果总结")
    print("=" * 60)
    print(f"总方法数: {total_methods}")
    print(f"成功方法数: {successful_methods}")
    print(f"失败方法数: {total_methods - successful_methods}")
    print(f"成功率: {successful_methods/total_methods*100:.1f}%")
    
    # 详细失败列表
    failed_methods = []
    for category, category_results in results.items():
        for method, result in category_results.items():
            if not result['success']:
                failed_methods.append(f"{category}.{method}")
    
    if failed_methods:
        print(f"\n❌ 失败的方法 ({len(failed_methods)}个):")
        for method in failed_methods:
            print(f"  - {method}")
    else:
        print(f"\n🎉 所有方法都测试成功！")
    
    return results

def check_gui_interface():
    """检查GUI界面"""
    print("\n🖥️ 检查GUI界面...")
    print("-" * 40)
    
    try:
        # 检查GUI文件是否存在
        import os
        gui_file = "/Users/linshibo/GithubProject/dlt/gui_app.py"
        if os.path.exists(gui_file):
            print("  ✅ GUI文件存在")
            
            # 尝试导入GUI模块检查语法
            sys.path.insert(0, "/Users/linshibo/GithubProject/dlt")
            try:
                import gui_app
                print("  ✅ GUI模块导入成功")
                return True
            except Exception as e:
                print(f"  ❌ GUI模块导入失败: {str(e)[:100]}...")
                return False
        else:
            print("  ❌ GUI文件不存在")
            return False
            
    except Exception as e:
        print(f"  ❌ GUI检查异常: {str(e)}")
        return False

def main():
    """主函数"""
    print("🚀 大乐透预测系统全面检查")
    print("=" * 60)
    
    # 检查预测方法
    prediction_results = check_prediction_methods()
    
    # 检查GUI界面
    gui_success = check_gui_interface()
    
    print("\n" + "=" * 60)
    print("🏁 系统检查完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
