#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型评估命令行工具
提供命令行接口，用于运行模型评估和基准测试
"""

import os
import sys
import argparse
from typing import List, Dict, Tuple, Any
import importlib
import json
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入核心模块
from core_modules import logger_manager, data_manager, cache_manager

# 导入模型评估框架
try:
    from improvements.model_evaluation import get_model_evaluator
    EVALUATOR_AVAILABLE = True
except ImportError:
    EVALUATOR_AVAILABLE = False
    print("⚠️ 模型评估框架未找到，请确保improvements/model_evaluation.py文件存在")

# 导入模型基准测试框架
try:
    from improvements.model_benchmark import get_model_benchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False
    print("⚠️ 模型基准测试框架未找到，请确保improvements/model_benchmark.py文件存在")

# 导入模型优化器
try:
    from improvements.model_optimizer import ModelOptimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    print("⚠️ 模型优化器未找到，请确保improvements/model_optimizer.py文件存在")


def load_predictor_module(module_path, class_name=None):
    """加载预测器模块
    
    Args:
        module_path: 模块路径
        class_name: 类名（可选）
        
    Returns:
        module or class: 加载的模块或类
    """
    try:
        module = importlib.import_module(module_path)
        if class_name:
            return getattr(module, class_name)
        return module
    except ImportError:
        print(f"⚠️ 无法导入模块: {module_path}")
        return None
    except AttributeError:
        print(f"⚠️ 模块 {module_path} 中没有找到类: {class_name}")
        return None


def register_default_predictors(benchmark):
    """注册默认预测器
    
    Args:
        benchmark: 基准测试实例
        
    Returns:
        int: 注册的预测器数量
    """
    registered_count = 0
    
    # 注册随机预测器（基准）
    def mock_random_predict(historical_data):
        """随机预测"""
        import numpy as np
        return [(sorted(np.random.choice(range(1, 36), 5, replace=False)), 
                sorted(np.random.choice(range(1, 13), 2, replace=False))) for _ in range(3)]
    
    benchmark.register_model("随机预测", mock_random_predict, "完全随机的预测方法", "baseline")
    registered_count += 1
    
    # 尝试注册传统预测器
    try:
        from predictor_modules import get_traditional_predictor
        traditional = get_traditional_predictor()
        
        benchmark.register_model(
            "频率预测", 
            lambda data: traditional.frequency_predict(3), 
            "基于历史频率的预测方法", 
            "traditional"
        )
        
        benchmark.register_model(
            "冷热号预测", 
            lambda data: traditional.hot_cold_predict(3), 
            "基于冷热号的预测方法", 
            "traditional"
        )
        
        benchmark.register_model(
            "遗漏值预测", 
            lambda data: traditional.missing_predict(3), 
            "基于遗漏值的预测方法", 
            "traditional"
        )
        
        registered_count += 3
    except ImportError:
        print("⚠️ 传统预测器模块未找到")
    
    # 尝试注册高级预测器
    try:
        from predictor_modules import get_advanced_predictor
        advanced = get_advanced_predictor()
        
        benchmark.register_model(
            "马尔可夫预测", 
            lambda data: advanced.markov_predict(3), 
            "基于马尔可夫链的预测方法", 
            "advanced"
        )
        
        benchmark.register_model(
            "贝叶斯预测", 
            lambda data: advanced.bayesian_predict(3), 
            "基于贝叶斯分析的预测方法", 
            "advanced"
        )
        
        benchmark.register_model(
            "集成预测", 
            lambda data: advanced.ensemble_predict(3), 
            "基于多种算法集成的预测方法", 
            "advanced"
        )
        
        registered_count += 3
    except ImportError:
        print("⚠️ 高级预测器模块未找到")
    
    # 尝试注册增强马尔可夫预测器
    try:
        from improvements.enhanced_markov import get_markov_predictor
        markov_predictor = get_markov_predictor()
        
        benchmark.register_model(
            "二阶马尔可夫", 
            lambda data: markov_predictor.multi_order_markov_predict(3, 300, 2), 
            "基于二阶马尔可夫链的预测方法", 
            "enhanced"
        )
        
        benchmark.register_model(
            "三阶马尔可夫", 
            lambda data: markov_predictor.multi_order_markov_predict(3, 300, 3), 
            "基于三阶马尔可夫链的预测方法", 
            "enhanced"
        )
        
        registered_count += 2
    except ImportError:
        print("⚠️ 增强马尔可夫模块未找到")
    
    # 尝试注册LSTM预测器
    try:
        from advanced_lstm_predictor import AdvancedLSTMPredictor, TENSORFLOW_AVAILABLE
        if TENSORFLOW_AVAILABLE:
            lstm_predictor = AdvancedLSTMPredictor()
            
            benchmark.register_model(
                "LSTM深度学习", 
                lambda data: lstm_predictor.lstm_predict(3), 
                "基于LSTM深度学习的预测方法", 
                "deep_learning"
            )
            
            registered_count += 1
        else:
            print("⚠️ TensorFlow未安装，无法使用LSTM预测器")
    except ImportError:
        print("⚠️ LSTM预测器模块未找到")
    
    return registered_count


def register_custom_predictor(benchmark, module_path, class_name, method_name, model_name, category):
    """注册自定义预测器
    
    Args:
        benchmark: 基准测试实例
        module_path: 模块路径
        class_name: 类名
        method_name: 方法名
        model_name: 模型名称
        category: 模型类别
        
    Returns:
        bool: 是否注册成功
    """
    try:
        # 加载模块和类
        module = importlib.import_module(module_path)
        predictor_class = getattr(module, class_name)
        
        # 创建实例
        if hasattr(module, f"get_{class_name.lower()}"):
            # 如果有获取实例的函数，使用它
            get_instance = getattr(module, f"get_{class_name.lower()}")
            predictor = get_instance()
        else:
            # 否则直接实例化
            predictor = predictor_class()
        
        # 获取预测方法
        predict_method = getattr(predictor, method_name)
        
        # 注册模型
        benchmark.register_model(
            model_name,
            lambda data: predict_method(data) if 'data' in predict_method.__code__.co_varnames else predict_method(3),
            f"自定义预测器: {module_path}.{class_name}.{method_name}",
            category
        )
        
        print(f"✅ 成功注册自定义预测器: {model_name}")
        return True
    except Exception as e:
        print(f"⚠️ 注册自定义预测器失败: {e}")
        return False


def run_benchmark(args):
    """运行基准测试
    
    Args:
        args: 命令行参数
    """
    if not BENCHMARK_AVAILABLE:
        print("❌ 模型基准测试框架未找到，无法运行基准测试")
        return
    
    # 获取基准测试实例
    benchmark = get_model_benchmark()
    
    # 注册预测器
    if args.register_default:
        print("\n📝 注册默认预测器...")
        count = register_default_predictors(benchmark)
        print(f"✅ 成功注册 {count} 个默认预测器")
    
    # 注册自定义预测器
    if args.custom_predictor:
        print("\n📝 注册自定义预测器...")
        for predictor_info in args.custom_predictor:
            parts = predictor_info.split(':')
            if len(parts) != 5:
                print(f"⚠️ 自定义预测器格式错误: {predictor_info}")
                print("正确格式: module_path:class_name:method_name:model_name:category")
                continue
            
            module_path, class_name, method_name, model_name, category = parts
            register_custom_predictor(benchmark, module_path, class_name, method_name, model_name, category)
    
    # 加载预测器配置文件
    if args.predictor_config:
        print(f"\n📝 从配置文件加载预测器: {args.predictor_config}")
        try:
            with open(args.predictor_config, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            for predictor in config.get('predictors', []):
                register_custom_predictor(
                    benchmark,
                    predictor['module_path'],
                    predictor['class_name'],
                    predictor['method_name'],
                    predictor['model_name'],
                    predictor['category']
                )
        except Exception as e:
            print(f"⚠️ 加载预测器配置文件失败: {e}")
    
    # 评估模型
    if args.evaluate_all:
        print(f"\n📊 评估所有模型 (测试期数: {args.test_periods})...")
        benchmark.evaluate_all_models(test_periods=args.test_periods, verbose=True)
    elif args.evaluate:
        for model_name in args.evaluate:
            print(f"\n📊 评估模型: {model_name} (测试期数: {args.test_periods})...")
            benchmark.evaluate_model(model_name, test_periods=args.test_periods, verbose=True)
    
    # 比较模型
    if args.compare:
        print("\n🔄 比较模型...")
        categories = args.categories.split(',') if args.categories else None
        metrics = args.metrics.split(',') if args.metrics else None
        benchmark.compare_models(metrics=metrics, categories=categories, verbose=True)
    
    # 生成报告
    if args.report:
        print("\n📝 生成评估报告...")
        report_path = args.report
        benchmark.generate_report(report_path)
        print(f"✅ 评估报告已保存到: {report_path}")
    
    # 可视化比较
    if args.visualize_comparison:
        print("\n📈 可视化模型比较...")
        metrics = args.metrics.split(',') if args.metrics else None
        benchmark.visualize_comparison(metrics=metrics, output_path=args.visualize_comparison)
        print(f"✅ 比较图表已保存到: {args.visualize_comparison}")
    
    # 可视化模型性能
    if args.visualize_model:
        for model_name in args.visualize_model:
            print(f"\n📊 可视化模型性能: {model_name}...")
            output_path = f"output/{model_name}_performance.png"
            if args.output_dir:
                output_path = os.path.join(args.output_dir, f"{model_name}_performance.png")
            benchmark.visualize_model_performance(model_name, output_path=output_path)
            print(f"✅ 性能图表已保存到: {output_path}")
    
    # 保存结果
    if args.save_results:
        print("\n💾 保存评估结果...")
        benchmark.save_results(args.save_results)
        print(f"✅ 评估结果已保存到: {args.save_results}")
    
    # 加载结果
    if args.load_results:
        print(f"\n📂 加载评估结果: {args.load_results}")
        if benchmark.load_results(args.load_results):
            print("✅ 评估结果加载成功")
        else:
            print("❌ 评估结果加载失败")


def run_optimization(args):
    """运行参数优化
    
    Args:
        args: 命令行参数
    """
    if not OPTIMIZER_AVAILABLE:
        print("❌ 模型优化器未找到，无法运行参数优化")
        return
    
    # 加载模型类
    print(f"\n📝 加载模型类: {args.module_path}.{args.class_name}")
    try:
        module = importlib.import_module(args.module_path)
        model_class = getattr(module, args.class_name)
    except Exception as e:
        print(f"❌ 加载模型类失败: {e}")
        return
    
    # 加载参数空间
    print(f"\n📝 加载参数空间: {args.param_space}")
    try:
        with open(args.param_space, 'r', encoding='utf-8') as f:
            param_space = json.load(f)
    except Exception as e:
        print(f"❌ 加载参数空间失败: {e}")
        return
    
    # 创建模型创建函数
    def create_model(**params):
        return model_class(**params)
    
    # 创建优化器
    optimizer = ModelOptimizer(
        model_creator=create_model,
        param_space=param_space,
        optimization_method=args.method,
        n_iter=args.n_iter
    )
    
    # 执行优化
    print(f"\n🔍 开始参数优化 (方法: {args.method}, 迭代次数: {args.n_iter})...")
    result = optimizer.optimize(
        train_periods=args.train_periods,
        val_periods=args.val_periods,
        metric=args.metric,
        verbose=True
    )
    
    # 打印最佳参数
    print(f"\n🏆 最佳参数: {optimizer.best_params}")
    print(f"🎯 最佳得分: {optimizer.best_score:.4f}")
    
    # 可视化优化过程
    if args.visualize_process:
        print("\n📈 可视化优化过程...")
        optimizer.visualize_optimization_process(args.visualize_process)
        print(f"✅ 优化过程图表已保存到: {args.visualize_process}")
    
    # 可视化参数重要性
    if args.visualize_importance:
        print("\n📊 可视化参数重要性...")
        optimizer.visualize_parameter_importance(args.visualize_importance)
        print(f"✅ 参数重要性图表已保存到: {args.visualize_importance}")
    
    # 保存优化结果
    if args.save_results:
        print("\n💾 保存优化结果...")
        optimizer.save_results(args.save_results)
        print(f"✅ 优化结果已保存到: {args.save_results}")
    
    # 使用基准测试框架评估最佳模型
    if args.benchmark and BENCHMARK_AVAILABLE:
        print("\n🔍 使用基准测试框架评估最佳模型...")
        benchmark_result = optimizer.benchmark_best_model(
            f"优化后的{args.class_name}",
            test_periods=args.test_periods
        )
        
        # 比较基准模型和优化模型
        if args.compare_baseline:
            print("\n🔄 比较基准模型和优化模型...")
            
            # 注册基准模型
            optimizer.benchmark.register_model(
                f"基准{args.class_name}",
                lambda data: model_class().fit(data) or model_class().predict(data),
                f"未优化的{args.class_name}",
                "baseline"
            )
            
            # 评估基准模型
            optimizer.benchmark.evaluate_model(f"基准{args.class_name}", test_periods=args.test_periods)
            
            # 比较模型
            optimizer.benchmark.compare_models(categories=["baseline", "optimized"])


def run_evaluation(args):
    """运行模型评估
    
    Args:
        args: 命令行参数
    """
    if not EVALUATOR_AVAILABLE:
        print("❌ 模型评估框架未找到，无法运行模型评估")
        return
    
    # 获取模型评估器实例
    evaluator = get_model_evaluator()
    
    # 注册预测器
    if args.register_default:
        print("\n📝 注册默认预测器...")
        count = register_default_predictors(evaluator.benchmark)
        print(f"✅ 成功注册 {count} 个默认预测器")
    
    # 注册自定义预测器
    if args.custom_predictor:
        print("\n📝 注册自定义预测器...")
        for predictor_info in args.custom_predictor:
            parts = predictor_info.split(':')
            if len(parts) != 5:
                print(f"⚠️ 自定义预测器格式错误: {predictor_info}")
                print("正确格式: module_path:class_name:method_name:model_name:category")
                continue
            
            module_path, class_name, method_name, model_name, category = parts
            register_custom_predictor(evaluator.benchmark, module_path, class_name, method_name, model_name, category)
    
    # 加载预测器配置文件
    if args.predictor_config:
        print(f"\n📝 从配置文件加载预测器: {args.predictor_config}")
        try:
            with open(args.predictor_config, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            for predictor in config.get('predictors', []):
                register_custom_predictor(
                    evaluator.benchmark,
                    predictor['module_path'],
                    predictor['class_name'],
                    predictor['method_name'],
                    predictor['model_name'],
                    predictor['category']
                )
        except Exception as e:
            print(f"⚠️ 加载预测器配置文件失败: {e}")
    
    # 评估模型
    if args.evaluate_all:
        print(f"\n📊 评估所有模型 (测试期数: {args.test_periods})...")
        evaluator.evaluate_all_models(test_periods=args.test_periods, verbose=True)
    elif args.evaluate:
        for model_name in args.evaluate:
            print(f"\n📊 评估模型: {model_name} (测试期数: {args.test_periods})...")
            evaluator.evaluate_model(model_name, test_periods=args.test_periods, verbose=True)
    
    # 比较模型
    if args.compare:
        print("\n🔄 比较模型...")
        categories = args.categories.split(',') if args.categories else None
        metrics = args.metrics.split(',') if args.metrics else None
        evaluator.compare_models(metrics=metrics, categories=categories, verbose=True)
    
    # 优化模型
    if args.optimize:
        for optimize_info in args.optimize:
            parts = optimize_info.split(':')
            if len(parts) != 3:
                print(f"⚠️ 优化信息格式错误: {optimize_info}")
                print("正确格式: module_path:class_name:param_space_file")
                continue
            
            module_path, class_name, param_space_file = parts
            
            print(f"\n🔍 优化模型: {module_path}.{class_name}")
            
            # 加载模型类
            try:
                module = importlib.import_module(module_path)
                model_class = getattr(module, class_name)
            except Exception as e:
                print(f"❌ 加载模型类失败: {e}")
                continue
            
            # 加载参数空间
            try:
                with open(param_space_file, 'r', encoding='utf-8') as f:
                    param_space = json.load(f)
            except Exception as e:
                print(f"❌ 加载参数空间失败: {e}")
                continue
            
            # 创建模型创建函数
            def create_model(**params):
                return model_class(**params)
            
            # 执行优化
            evaluator.optimize_model(
                model_name=class_name,
                model_creator=create_model,
                param_space=param_space,
                optimization_method=args.method,
                n_iter=args.n_iter,
                train_periods=args.train_periods,
                val_periods=args.val_periods,
                metric=args.metric,
                verbose=True
            )
            
            # 可视化优化过程
            if args.output_dir:
                evaluator.visualize_optimization(class_name, args.output_dir)
    
    # 生成报告
    if args.report:
        print("\n📝 生成评估报告...")
        report_path = args.report
        evaluator.generate_report(report_path)
        print(f"✅ 评估报告已保存到: {report_path}")
    
    # 可视化比较
    if args.visualize_comparison:
        print("\n📈 可视化模型比较...")
        metrics = args.metrics.split(',') if args.metrics else None
        evaluator.visualize_comparison(metrics=metrics, output_path=args.visualize_comparison)
        print(f"✅ 比较图表已保存到: {args.visualize_comparison}")
    
    # 可视化模型性能
    if args.visualize_model:
        for model_name in args.visualize_model:
            print(f"\n📊 可视化模型性能: {model_name}...")
            output_path = f"output/{model_name}_performance.png"
            if args.output_dir:
                output_path = os.path.join(args.output_dir, f"{model_name}_performance.png")
            evaluator.visualize_model_performance(model_name, output_path=output_path)
            print(f"✅ 性能图表已保存到: {output_path}")
    
    # 保存结果
    if args.save_results:
        print("\n💾 保存评估结果...")
        evaluator.save_results(args.save_results if args.save_results != "default" else args.output_dir)
        print(f"✅ 评估结果已保存到: {args.save_results if args.save_results != 'default' else args.output_dir}")


def main():
    """主函数"""
    # 创建命令行解析器
    parser = argparse.ArgumentParser(description="模型评估命令行工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 基准测试子命令
    benchmark_parser = subparsers.add_parser("benchmark", help="运行模型基准测试")
    benchmark_parser.add_argument("--register-default", action="store_true", help="注册默认预测器")
    benchmark_parser.add_argument("--custom-predictor", nargs="+", help="注册自定义预测器 (格式: module_path:class_name:method_name:model_name:category)")
    benchmark_parser.add_argument("--predictor-config", help="预测器配置文件路径")
    benchmark_parser.add_argument("--evaluate-all", action="store_true", help="评估所有注册的模型")
    benchmark_parser.add_argument("--evaluate", nargs="+", help="评估指定的模型")
    benchmark_parser.add_argument("--test-periods", type=int, default=20, help="测试期数")
    benchmark_parser.add_argument("--compare", action="store_true", help="比较模型性能")
    benchmark_parser.add_argument("--categories", help="要比较的模型类别，逗号分隔")
    benchmark_parser.add_argument("--metrics", help="要比较的指标，逗号分隔")
    benchmark_parser.add_argument("--report", help="生成评估报告的输出路径")
    benchmark_parser.add_argument("--visualize-comparison", help="可视化模型比较的输出路径")
    benchmark_parser.add_argument("--visualize-model", nargs="+", help="可视化指定模型的性能")
    benchmark_parser.add_argument("--output-dir", default="output", help="输出目录")
    benchmark_parser.add_argument("--save-results", help="保存评估结果的输出路径")
    benchmark_parser.add_argument("--load-results", help="加载评估结果的输入路径")
    
    # 参数优化子命令
    optimize_parser = subparsers.add_parser("optimize", help="运行参数优化")
    optimize_parser.add_argument("--module-path", required=True, help="模型模块路径")
    optimize_parser.add_argument("--class-name", required=True, help="模型类名")
    optimize_parser.add_argument("--param-space", required=True, help="参数空间文件路径")
    optimize_parser.add_argument("--method", default="grid", choices=["grid", "random", "bayesian"], help="优化方法")
    optimize_parser.add_argument("--n-iter", type=int, default=50, help="随机搜索或贝叶斯优化的迭代次数")
    optimize_parser.add_argument("--train-periods", type=int, default=300, help="训练数据期数")
    optimize_parser.add_argument("--val-periods", type=int, default=50, help="验证数据期数")
    optimize_parser.add_argument("--test-periods", type=int, default=20, help="测试期数")
    optimize_parser.add_argument("--metric", default="accuracy", choices=["accuracy", "hit_rate", "roi"], help="优化指标")
    optimize_parser.add_argument("--visualize-process", help="可视化优化过程的输出路径")
    optimize_parser.add_argument("--visualize-importance", help="可视化参数重要性的输出路径")
    optimize_parser.add_argument("--save-results", help="保存优化结果的输出路径")
    optimize_parser.add_argument("--benchmark", action="store_true", help="使用基准测试框架评估最佳模型")
    optimize_parser.add_argument("--compare-baseline", action="store_true", help="比较基准模型和优化模型")
    
    # 模型评估子命令
    evaluate_parser = subparsers.add_parser("evaluate", help="运行模型评估")
    evaluate_parser.add_argument("--register-default", action="store_true", help="注册默认预测器")
    evaluate_parser.add_argument("--custom-predictor", nargs="+", help="注册自定义预测器 (格式: module_path:class_name:method_name:model_name:category)")
    evaluate_parser.add_argument("--predictor-config", help="预测器配置文件路径")
    evaluate_parser.add_argument("--evaluate-all", action="store_true", help="评估所有注册的模型")
    evaluate_parser.add_argument("--evaluate", nargs="+", help="评估指定的模型")
    evaluate_parser.add_argument("--test-periods", type=int, default=20, help="测试期数")
    evaluate_parser.add_argument("--compare", action="store_true", help="比较模型性能")
    evaluate_parser.add_argument("--categories", help="要比较的模型类别，逗号分隔")
    evaluate_parser.add_argument("--metrics", help="要比较的指标，逗号分隔")
    evaluate_parser.add_argument("--optimize", nargs="+", help="优化模型 (格式: module_path:class_name:param_space_file)")
    evaluate_parser.add_argument("--method", default="grid", choices=["grid", "random", "bayesian"], help="优化方法")
    evaluate_parser.add_argument("--n-iter", type=int, default=50, help="随机搜索或贝叶斯优化的迭代次数")
    evaluate_parser.add_argument("--train-periods", type=int, default=300, help="训练数据期数")
    evaluate_parser.add_argument("--val-periods", type=int, default=50, help="验证数据期数")
    evaluate_parser.add_argument("--metric", default="accuracy", choices=["accuracy", "hit_rate", "roi"], help="优化指标")
    evaluate_parser.add_argument("--report", help="生成评估报告的输出路径")
    evaluate_parser.add_argument("--visualize-comparison", help="可视化模型比较的输出路径")
    evaluate_parser.add_argument("--visualize-model", nargs="+", help="可视化指定模型的性能")
    evaluate_parser.add_argument("--output-dir", default="output", help="输出目录")
    evaluate_parser.add_argument("--save-results", default="default", help="保存评估结果的输出路径")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 确保输出目录存在
    if hasattr(args, 'output_dir') and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 执行相应的命令
    if args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "optimize":
        run_optimization(args)
    elif args.command == "evaluate":
        run_evaluation(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()