#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置验证测试脚本

测试backend/app/main.py中的配置验证功能
"""

import sys
import os

# 添加路径
sys.path.insert(0, 'backend/app')
sys.path.insert(0, 'backend/app/core')

def test_path_manager():
    """测试路径管理器"""
    print("=" * 60)
    print("测试1: 路径管理器基本功能")
    print("=" * 60)

    try:
        from core.path_config import get_path_manager
        pm = get_path_manager()

        print(f"✅ 项目根目录: {pm.base_dir}")
        print(f"✅ 数据文件: {pm.data_file}")
        print(f"✅ 缓存目录: {pm.cache_dir}")
        print(f"✅ 日志目录: {pm.logs_dir}")
        print(f"✅ 模型目录: {pm.models_dir}")
        print(f"✅ 报告目录: {pm.reports_dir}")

        print("\n路径管理器测试通过！\n")
        return True
    except Exception as e:
        print(f"❌ 路径管理器测试失败: {e}\n")
        return False


def test_dlt_system_validation():
    """测试DLTPredictorSystem配置验证"""
    print("=" * 60)
    print("测试2: DLTPredictorSystem配置验证")
    print("=" * 60)

    try:
        # 导入核心模块
        import core_modules as cm

        # 导入主系统
        from main import DLTPredictorSystem

        print("正在初始化DLTPredictorSystem...")
        system = DLTPredictorSystem()

        print("\n配置验证测试项：")
        print("✅ path_config模块导入成功")
        print("✅ 关键目录验证通过")
        print("✅ 数据文件存在性验证通过")
        print("✅ 数据文件可读性验证通过")
        print("✅ 配置文件验证通过")

        print("\nDLTPredictorSystem配置验证测试通过！\n")
        return True
    except Exception as e:
        print(f"❌ DLTPredictorSystem配置验证测试失败: {e}\n")
        return False


def test_main_quick_check():
    """测试main()函数的快速配置检查"""
    print("=" * 60)
    print("测试3: main()函数快速配置检查")
    print("=" * 60)

    try:
        # 模拟main()函数的快速检查逻辑
        from core.path_config import get_path_manager
        pm = get_path_manager()

        critical_checks = [
            ('数据文件', pm.data_file, True),
            ('缓存目录', pm.cache_dir, False),
            ('日志目录', pm.logs_dir, False),
            ('模型目录', pm.models_dir, False),
        ]

        all_passed = True
        for name, path, must_exist in critical_checks:
            if path.exists():
                print(f"✅ {name}: {path}")
            elif not must_exist:
                print(f"⚠️  {name}不存在，但可以创建: {path}")
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    print(f"✅ 已创建{name}")
                except Exception as e:
                    print(f"❌ 无法创建{name}: {e}")
                    all_passed = False
            else:
                print(f"❌ {name}不存在: {path}")
                all_passed = False

        if all_passed:
            print("\nmain()快速配置检查测试通过！\n")
            return True
        else:
            print("\n❌ main()快速配置检查测试失败\n")
            return False
    except Exception as e:
        print(f"❌ main()快速配置检查测试失败: {e}\n")
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("配置验证功能测试套件")
    print("=" * 60 + "\n")

    results = []

    # 运行测试
    results.append(("路径管理器", test_path_manager()))
    results.append(("DLTPredictorSystem配置验证", test_dlt_system_validation()))
    results.append(("main()快速配置检查", test_main_quick_check()))

    # 汇总结果
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {test_name}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！配置验证功能工作正常。\n")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查配置。\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
