#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
深度学习命令集
提供训练、预测、集成和元学习命令
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Tuple, Any, Optional, Union

from core_modules import logger_manager, data_manager, task_manager, with_progress


class CommandDefinition:
    """深度学习命令集"""
    
    def __init__(self):
        """初始化深度学习命令集"""
        self.commands = {
            'train': self.train_command,
            'predict': self.predict_command,
            'ensemble': self.ensemble_command,
            'metalearning': self.metalearning_command,
            'optimize': self.optimize_command,
            'info': self.info_command
        }
        
        logger_manager.info("深度学习命令集初始化完成")
    
    def get_command_parser(self) -> argparse.ArgumentParser:
        """
        获取命令解析器
        
        Returns:
            命令解析器
        """
        parser = argparse.ArgumentParser(description='大乐透深度学习预测系统')
        subparsers = parser.add_subparsers(dest='command', help='可用命令')
        
        # 训练命令
        train_parser = subparsers.add_parser('train', help='训练深度学习模型')
        train_parser.add_argument('-m', '--model', choices=['transformer', 'gan', 'all'], 
                                default='all', help='要训练的模型类型')
        train_parser.add_argument('-p', '--periods', type=int, default=1000, 
                                help='用于训练的历史期数')
        train_parser.add_argument('-e', '--epochs', type=int, default=100, 
                                help='训练轮数')
        train_parser.add_argument('-b', '--batch-size', type=int, default=32, 
                                help='批处理大小')
        train_parser.add_argument('--gpu', action='store_true', 
                                help='是否使用GPU加速')
        train_parser.add_argument('--save-model', action='store_true', 
                                help='是否保存模型')
        
        # 预测命令
        predict_parser = subparsers.add_parser('predict', help='使用深度学习模型进行预测')
        predict_parser.add_argument('-m', '--model', choices=['transformer', 'gan', 'ensemble'], 
                                  default='ensemble', help='要使用的模型类型')
        predict_parser.add_argument('-c', '--count', type=int, default=5, 
                                  help='生成的预测结果数量')
        predict_parser.add_argument('--compound', action='store_true', 
                                  help='是否生成复式投注')
        predict_parser.add_argument('--front-count', type=int, default=7, 
                                  help='前区号码数量（复式投注）')
        predict_parser.add_argument('--back-count', type=int, default=3, 
                                  help='后区号码数量（复式投注）')
        predict_parser.add_argument('--confidence', action='store_true', 
                                  help='是否显示置信度')
        predict_parser.add_argument('--report', action='store_true', 
                                  help='是否生成详细报告')
        
        # 集成命令
        ensemble_parser = subparsers.add_parser('ensemble', help='管理模型集成')
        ensemble_parser.add_argument('action', choices=['list', 'add', 'remove', 'update', 'weights'], 
                                   help='集成操作')
        ensemble_parser.add_argument('-m', '--model', type=str, 
                                   help='模型名称（用于add/remove/update操作）')
        ensemble_parser.add_argument('-w', '--weight', type=float, 
                                   help='模型权重（用于add/update操作）')
        ensemble_parser.add_argument('--method', choices=['weighted', 'stacking'], 
                                   default='weighted', help='集成方法')
        
        # 元学习命令
        meta_parser = subparsers.add_parser('metalearning', help='管理元学习系统')
        meta_parser.add_argument('action', choices=['status', 'enable', 'disable', 'retrain'], 
                               help='元学习操作')
        meta_parser.add_argument('--auto-weight', action='store_true', 
                               help='是否启用自动权重调整')
        meta_parser.add_argument('--auto-retrain', action='store_true', 
                               help='是否启用自动重训练')
        meta_parser.add_argument('--threshold', type=float, default=0.05, 
                               help='性能退化阈值（用于自动重训练）')
        
        # 优化命令
        optimize_parser = subparsers.add_parser('optimize', help='优化深度学习模型')
        optimize_parser.add_argument('action', choices=['quantize', 'cache', 'monitor'], 
                                   help='优化操作')
        optimize_parser.add_argument('-m', '--model', choices=['transformer', 'gan', 'all'], 
                                   default='all', help='要优化的模型类型')
        optimize_parser.add_argument('--type', choices=['float16', 'int8', 'dynamic'], 
                                   default='float16', help='量化类型')
        optimize_parser.add_argument('--clear-cache', action='store_true', 
                                   help='是否清除缓存')
        optimize_parser.add_argument('--report', action='store_true', 
                                   help='是否生成资源使用报告')
        
        # 信息命令
        info_parser = subparsers.add_parser('info', help='显示深度学习系统信息')
        info_parser.add_argument('--gpu', action='store_true', 
                               help='显示GPU信息')
        info_parser.add_argument('--models', action='store_true', 
                               help='显示模型信息')
        info_parser.add_argument('--performance', action='store_true', 
                               help='显示性能信息')
        info_parser.add_argument('--all', action='store_true', 
                               help='显示所有信息')
        
        return parser 
   
    def execute_command(self, args: argparse.Namespace) -> int:
        """
        执行命令
        
        Args:
            args: 命令参数
            
        Returns:
            命令执行结果代码
        """
        command = args.command
        
        if command in self.commands:
            try:
                return self.commands[command](args)
            except Exception as e:
                logger_manager.error(f"执行命令 {command} 失败: {e}")
                return 1
        else:
            logger_manager.error(f"未知命令: {command}")
            return 1
    
    def train_command(self, args: argparse.Namespace) -> int:
        """
        训练命令
        
        Args:
            args: 命令参数
            
        Returns:
            命令执行结果代码
        """
        logger_manager.info(f"执行训练命令: 模型={args.model}, 期数={args.periods}, 轮数={args.epochs}")
        
        # 导入模型模块
        try:
            from .transformer_model import TransformerModel
            from .gan_model import GANModel
        except ImportError as e:
            logger_manager.error(f"导入模型模块失败: {e}")
            return 1
        
        # 获取数据
        df = data_manager.get_data(periods=args.periods)
        
        if df is None or len(df) == 0:
            logger_manager.error("获取训练数据失败")
            return 1
        
        # 训练模型
        if args.model in ['transformer', 'all']:
            try:
                # 创建Transformer模型
                transformer = TransformerModel()
                
                # 训练模型
                transformer.train(df, epochs=args.epochs, batch_size=args.batch_size, 
                                use_gpu=args.gpu)
                
                # 保存模型
                if args.save_model:
                    transformer.save_model()
                
                logger_manager.info("Transformer模型训练完成")
            except Exception as e:
                logger_manager.error(f"训练Transformer模型失败: {e}")
                return 1
        
        if args.model in ['gan', 'all']:
            try:
                # 创建GAN模型
                gan = GANModel()
                
                # 训练模型
                gan.train(df, epochs=args.epochs, batch_size=args.batch_size, 
                        use_gpu=args.gpu)
                
                # 保存模型
                if args.save_model:
                    gan.save_model()
                
                logger_manager.info("GAN模型训练完成")
            except Exception as e:
                logger_manager.error(f"训练GAN模型失败: {e}")
                return 1
        
        logger_manager.info("训练命令执行完成")
        return 0

    def predict_command(self, args: argparse.Namespace) -> int:
        """
        预测命令
        
        Args:
            args: 命令参数
            
        Returns:
            命令执行结果代码
        """
        logger_manager.info(f"执行预测命令: 模型={args.model}, 数量={args.count}")
        
        # 导入模型模块
        try:
            from .transformer_model import TransformerModel
            from .gan_model import GANModel
            from .ensemble_manager import EnsembleManager
        except ImportError as e:
            logger_manager.error(f"导入模型模块失败: {e}")
            return 1
        
        # 获取最新数据
        df = data_manager.get_data(periods=20)
        
        if df is None or len(df) == 0:
            logger_manager.error("获取预测数据失败")
            return 1
        
        # 根据模型类型进行预测
        predictions = []
        confidences = []
        
        if args.model == 'transformer':
            try:
                # 创建Transformer模型
                transformer = TransformerModel()
                
                # 加载模型
                transformer.load_model()
                
                # 预测
                if args.compound:
                    pred, conf = transformer.predict_compound(
                        df, count=args.count, 
                        front_count=args.front_count, 
                        back_count=args.back_count
                    )
                else:
                    pred, conf = transformer.predict(df, count=args.count)
                
                predictions = pred
                confidences = conf
                
                logger_manager.info("Transformer模型预测完成")
            except Exception as e:
                logger_manager.error(f"Transformer模型预测失败: {e}")
                return 1
        
        elif args.model == 'gan':
            try:
                # 创建GAN模型
                gan = GANModel()
                
                # 加载模型
                gan.load_model()
                
                # 预测
                if args.compound:
                    pred, conf = gan.predict_compound(
                        df, count=args.count, 
                        front_count=args.front_count, 
                        back_count=args.back_count
                    )
                else:
                    pred, conf = gan.predict(df, count=args.count)
                
                predictions = pred
                confidences = conf
                
                logger_manager.info("GAN模型预测完成")
            except Exception as e:
                logger_manager.error(f"GAN模型预测失败: {e}")
                return 1
        
        elif args.model == 'ensemble':
            try:
                # 创建集成管理器
                ensemble = EnsembleManager()
                
                # 加载模型
                ensemble.load_models()
                
                # 预测
                if args.compound:
                    pred, conf = ensemble.predict_compound(
                        df, count=args.count, 
                        front_count=args.front_count, 
                        back_count=args.back_count
                    )
                else:
                    pred, conf = ensemble.predict(df, count=args.count)
                
                predictions = pred
                confidences = conf
                
                logger_manager.info("集成模型预测完成")
            except Exception as e:
                logger_manager.error(f"集成模型预测失败: {e}")
                return 1
        
        # 显示预测结果
        print("\n🎯 大乐透深度学习预测结果")
        print("=" * 50)
        
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            front_balls = pred[0]
            back_balls = pred[1]
            
            front_str = " ".join([f"{ball:02d}" for ball in front_balls])
            back_str = " ".join([f"{ball:02d}" for ball in back_balls])
            
            if args.confidence:
                print(f"预测 {i+1}: 前区 [{front_str}] 后区 [{back_str}] 置信度: {conf:.4f}")
            else:
                print(f"预测 {i+1}: 前区 [{front_str}] 后区 [{back_str}]")
        
        print("=" * 50)
        
        # 生成详细报告
        if args.report:
            try:
                report_path = self._generate_prediction_report(
                    args.model, predictions, confidences, args.compound)
                print(f"\n📊 详细报告已保存到: {report_path}")
            except Exception as e:
                logger_manager.error(f"生成预测报告失败: {e}")
        
        logger_manager.info("预测命令执行完成")
        return 0
    
    def _generate_prediction_report(self, model_type: str, predictions: List, 
                                  confidences: List, is_compound: bool) -> str:
        """
        生成预测报告
        
        Args:
            model_type: 模型类型
            predictions: 预测结果
            confidences: 置信度
            is_compound: 是否为复式投注
            
        Returns:
            报告文件路径
        """
        import time
        from pathlib import Path
        
        # 创建报告目录
        report_dir = Path("output/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成报告文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"prediction_report_{model_type}_{timestamp}.json"
        
        # 准备报告数据
        report_data = {
            "model_type": model_type,
            "timestamp": timestamp,
            "is_compound": is_compound,
            "predictions": [],
            "statistics": {
                "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
                "max_confidence": max(confidences) if confidences else 0,
                "min_confidence": min(confidences) if confidences else 0
            }
        }
        
        # 添加预测结果
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            front_balls = pred[0]
            back_balls = pred[1]
            
            report_data["predictions"].append({
                "id": i + 1,
                "front_balls": front_balls,
                "back_balls": back_balls,
                "confidence": conf,
                "front_sum": sum(front_balls),
                "back_sum": sum(back_balls),
                "front_odd_count": sum(1 for x in front_balls if x % 2 == 1),
                "front_even_count": sum(1 for x in front_balls if x % 2 == 0),
                "front_big_count": sum(1 for x in front_balls if x > 17),
                "front_small_count": sum(1 for x in front_balls if x <= 17)
            })
        
        # 保存报告
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return str(report_file)

    def ensemble_command(self, args: argparse.Namespace) -> int:
        """
        集成命令
        
        Args:
            args: 命令参数
            
        Returns:
            命令执行结果代码
        """
        logger_manager.info(f"执行集成命令: 操作={args.action}")
        
        # 导入集成管理器
        try:
            from .ensemble_manager import EnsembleManager
        except ImportError as e:
            logger_manager.error(f"导入集成管理器失败: {e}")
            return 1
        
        # 创建集成管理器
        ensemble = EnsembleManager()
        
        # 执行操作
        if args.action == 'list':
            # 列出模型
            models = ensemble.list_models()
            
            print("\n📋 集成模型列表")
            print("=" * 50)
            
            if not models:
                print("没有可用的模型")
            else:
                for i, (model_name, weight) in enumerate(models.items()):
                    print(f"{i+1}. {model_name}: 权重 = {weight:.4f}")
            
            print("=" * 50)
            print(f"集成方法: {ensemble.get_ensemble_method()}")
        
        elif args.action == 'add':
            # 添加模型
            if not args.model:
                logger_manager.error("添加模型需要指定模型名称")
                return 1
            
            weight = args.weight if args.weight is not None else 1.0
            success = ensemble.add_model(args.model, weight)
            
            if success:
                print(f"✅ 成功添加模型 '{args.model}' 到集成，权重 = {weight:.4f}")
            else:
                print(f"❌ 添加模型 '{args.model}' 失败")
                return 1
        
        elif args.action == 'remove':
            # 移除模型
            if not args.model:
                logger_manager.error("移除模型需要指定模型名称")
                return 1
            
            success = ensemble.remove_model(args.model)
            
            if success:
                print(f"✅ 成功从集成中移除模型 '{args.model}'")
            else:
                print(f"❌ 移除模型 '{args.model}' 失败")
                return 1
        
        elif args.action == 'update':
            # 更新模型权重
            if not args.model or args.weight is None:
                logger_manager.error("更新模型权重需要指定模型名称和权重")
                return 1
            
            success = ensemble.update_model_weight(args.model, args.weight)
            
            if success:
                print(f"✅ 成功更新模型 '{args.model}' 的权重为 {args.weight:.4f}")
            else:
                print(f"❌ 更新模型 '{args.model}' 权重失败")
                return 1
        
        elif args.action == 'weights':
            # 显示权重
            weights = ensemble.get_model_weights()
            
            print("\n⚖️ 模型权重")
            print("=" * 50)
            
            if not weights:
                print("没有可用的模型权重")
            else:
                for model_name, weight in weights.items():
                    print(f"{model_name}: {weight:.4f}")
            
            print("=" * 50)
        
        # 设置集成方法
        if args.method:
            ensemble.set_ensemble_method(args.method)
            print(f"✅ 集成方法已设置为 '{args.method}'")
        
        # 保存配置
        ensemble.save_config()
        
        logger_manager.info("集成命令执行完成")
        return 0
    
    def metalearning_command(self, args: argparse.Namespace) -> int:
        """
        元学习命令
        
        Args:
            args: 命令参数
            
        Returns:
            命令执行结果代码
        """
        logger_manager.info(f"执行元学习命令: 操作={args.action}")
        
        # 导入元学习管理器
        try:
            from .metalearning_manager import MetaLearningManager
        except ImportError as e:
            logger_manager.error(f"导入元学习管理器失败: {e}")
            return 1
        
        # 创建元学习管理器
        meta = MetaLearningManager()
        
        # 执行操作
        if args.action == 'status':
            # 显示状态
            status = meta.get_status()
            
            print("\n🧠 元学习系统状态")
            print("=" * 50)
            print(f"自动权重调整: {'启用' if status.get('auto_weight', False) else '禁用'}")
            print(f"自动重训练: {'启用' if status.get('auto_retrain', False) else '禁用'}")
            print(f"性能退化阈值: {status.get('threshold', 0.05):.4f}")
            print(f"上次权重调整: {status.get('last_weight_update', '从未')}")
            print(f"上次重训练: {status.get('last_retrain', '从未')}")
            print("=" * 50)
            
            # 显示性能跟踪
            performance = meta.get_performance_tracking()
            
            if performance:
                print("\n📈 性能跟踪")
                print("=" * 50)
                
                for model_name, metrics in performance.items():
                    print(f"模型: {model_name}")
                    print(f"  准确率: {metrics.get('accuracy', 0):.4f}")
                    print(f"  一致性: {metrics.get('consistency', 0):.4f}")
                    print(f"  适应性: {metrics.get('adaptability', 0):.4f}")
                    print(f"  趋势: {metrics.get('trend', '稳定')}")
                    print()
                
                print("=" * 50)
        
        elif args.action == 'enable':
            # 启用功能
            if args.auto_weight:
                meta.enable_auto_weight_adjustment(True)
                print("✅ 自动权重调整已启用")
            
            if args.auto_retrain:
                meta.enable_auto_retraining(True)
                print("✅ 自动重训练已启用")
            
            if args.threshold is not None:
                meta.set_performance_threshold(args.threshold)
                print(f"✅ 性能退化阈值已设置为 {args.threshold:.4f}")
        
        elif args.action == 'disable':
            # 禁用功能
            if args.auto_weight:
                meta.enable_auto_weight_adjustment(False)
                print("✅ 自动权重调整已禁用")
            
            if args.auto_retrain:
                meta.enable_auto_retraining(False)
                print("✅ 自动重训练已禁用")
        
        elif args.action == 'retrain':
            # 触发重训练
            print("🔄 触发模型重训练...")
            success = meta.trigger_retraining()
            
            if success:
                print("✅ 模型重训练完成")
            else:
                print("❌ 模型重训练失败")
                return 1
        
        # 保存配置
        meta.save_config()
        
        logger_manager.info("元学习命令执行完成")
        return 0    

    def optimize_command(self, args: argparse.Namespace) -> int:
        """
        优化命令
        
        Args:
            args: 命令参数
            
        Returns:
            命令执行结果代码
        """
        logger_manager.info(f"执行优化命令: 操作={args.action}, 模型={args.model}")
        
        # 导入模型优化器
        try:
            from .model_optimizer import ModelOptimizer
            from .transformer_model import TransformerModel
            from .gan_model import GANModel
        except ImportError as e:
            logger_manager.error(f"导入模型优化器失败: {e}")
            return 1
        
        # 创建模型优化器
        optimizer = ModelOptimizer()
        
        # 执行操作
        if args.action == 'quantize':
            # 量化模型
            print(f"🔄 使用 {args.type} 量化模型...")
            
            if args.model in ['transformer', 'all']:
                try:
                    # 加载Transformer模型
                    transformer = TransformerModel()
                    transformer.load_model()
                    
                    # 量化模型
                    quantized_model = optimizer.quantize_model(
                        transformer.model, 'tensorflow', args.type)
                    
                    # 更新模型
                    transformer.model = quantized_model
                    
                    # 保存模型
                    transformer.save_model(suffix='_quantized')
                    
                    print("✅ Transformer模型量化完成")
                except Exception as e:
                    logger_manager.error(f"量化Transformer模型失败: {e}")
                    return 1
            
            if args.model in ['gan', 'all']:
                try:
                    # 加载GAN模型
                    gan = GANModel()
                    gan.load_model()
                    
                    # 量化生成器
                    quantized_generator = optimizer.quantize_model(
                        gan.generator, 'tensorflow', args.type)
                    
                    # 更新模型
                    gan.generator = quantized_generator
                    
                    # 保存模型
                    gan.save_model(suffix='_quantized')
                    
                    print("✅ GAN模型量化完成")
                except Exception as e:
                    logger_manager.error(f"量化GAN模型失败: {e}")
                    return 1
        
        elif args.action == 'cache':
            # 管理缓存
            if args.clear_cache:
                optimizer.clear_result_cache()
                print("✅ 结果缓存已清除")
            else:
                # 显示缓存统计信息
                cache_dir = os.path.join(optimizer.cache_dir, 'result_cache')
                cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
                
                print("\n💾 缓存统计信息")
                print("=" * 50)
                print(f"缓存文件数量: {len(cache_files)}")
                
                if cache_files:
                    total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in cache_files)
                    print(f"缓存总大小: {total_size / (1024 * 1024):.2f} MB")
                
                print("=" * 50)
        
        elif args.action == 'monitor':
            # 资源监控
            print("📊 开始资源监控...")
            optimizer.start_resource_monitoring()
            
            # 模拟一些操作
            for i in range(5):
                print(f"执行操作 {i+1}/5...")
                import time
                time.sleep(1)
                usage = optimizer.record_resource_usage()
                print(f"  CPU: {usage['cpu']:.1f}%, 内存: {usage['memory']:.1f}%, 磁盘: {usage['disk']:.1f}%")
            
            # 获取资源使用情况摘要
            summary = optimizer.get_resource_usage_summary()
            
            print("\n📊 资源使用情况摘要")
            print("=" * 50)
            
            for resource, metrics in summary.items():
                print(f"{resource.upper()}:")
                print(f"  最小值: {metrics['min']:.1f}%")
                print(f"  最大值: {metrics['max']:.1f}%")
                print(f"  平均值: {metrics['avg']:.1f}%")
                print(f"  当前值: {metrics['current']:.1f}%")
                print()
            
            print("=" * 50)
            
            # 生成报告
            if args.report:
                report_path = optimizer.save_resource_usage_report()
                print(f"\n📊 资源使用情况报告已保存到: {report_path}")
        
        logger_manager.info("优化命令执行完成")
        return 0
    
    def info_command(self, args: argparse.Namespace) -> int:
        """
        信息命令
        
        Args:
            args: 命令参数
            
        Returns:
            命令执行结果代码
        """
        logger_manager.info("执行信息命令")
        
        # 显示所有信息
        if args.all:
            args.gpu = True
            args.models = True
            args.performance = True
        
        # 显示GPU信息
        if args.gpu:
            try:
                from .gpu_accelerator import GPUAccelerator
                
                # 创建GPU加速器
                accelerator = GPUAccelerator()
                
                print("\n🖥️ GPU信息")
                print("=" * 50)
                print(f"GPU可用性: {'可用' if accelerator.is_gpu_available() else '不可用'}")
                
                # 显示GPU设备
                devices = accelerator.get_gpu_devices()
                if devices:
                    print(f"可用GPU设备: {len(devices)}")
                    for i, device in enumerate(devices):
                        print(f"  {i+1}. {device}")
                else:
                    print("没有可用的GPU设备")
                
                # 显示GPU内存信息
                memory_info = accelerator.get_gpu_memory_info()
                if memory_info:
                    print("\nGPU内存信息:")
                    for device, memory in memory_info.items():
                        print(f"  {device}:")
                        for key, value in memory.items():
                            print(f"    {key}: {value:.2f} GB")
                
                print("=" * 50)
            except Exception as e:
                logger_manager.error(f"获取GPU信息失败: {e}")
        
        # 显示模型信息
        if args.models:
            try:
                # 获取模型文件
                model_dir = "models"
                if os.path.exists(model_dir):
                    model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5') or f.endswith('.pt')]
                    
                    print("\n📊 模型信息")
                    print("=" * 50)
                    
                    if model_files:
                        print(f"可用模型文件: {len(model_files)}")
                        for i, model_file in enumerate(model_files):
                            file_path = os.path.join(model_dir, model_file)
                            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                            import time
                            file_date = os.path.getmtime(file_path)
                            file_date_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_date))
                            
                            print(f"  {i+1}. {model_file}")
                            print(f"     大小: {file_size:.2f} MB")
                            print(f"     修改时间: {file_date_str}")
                    else:
                        print("没有可用的模型文件")
                    
                    print("=" * 50)
                else:
                    print("\n⚠️ 模型目录不存在")
            except Exception as e:
                logger_manager.error(f"获取模型信息失败: {e}")
        
        # 显示性能信息
        if args.performance:
            try:
                from .metalearning_manager import MetaLearningManager
                
                # 创建元学习管理器
                meta = MetaLearningManager()
                
                # 获取性能跟踪
                performance = meta.get_performance_tracking()
                
                print("\n📈 性能信息")
                print("=" * 50)
                
                if performance:
                    for model_name, metrics in performance.items():
                        print(f"模型: {model_name}")
                        print(f"  准确率: {metrics.get('accuracy', 0):.4f}")
                        print(f"  一致性: {metrics.get('consistency', 0):.4f}")
                        print(f"  适应性: {metrics.get('adaptability', 0):.4f}")
                        print(f"  趋势: {metrics.get('trend', '稳定')}")
                        print()
                else:
                    print("没有可用的性能跟踪信息")
                
                print("=" * 50)
            except Exception as e:
                logger_manager.error(f"获取性能信息失败: {e}")
        
        logger_manager.info("信息命令执行完成")
        return 0


if __name__ == "__main__":
    # 测试深度学习命令集
    print("🚀 测试深度学习命令集...")
    
    # 创建命令集
    commands = DeepLearningCommands()
    
    # 获取命令解析器
    parser = commands.get_command_parser()
    
    # 解析命令行参数
    args = parser.parse_args(['info', '--all'])
    
    # 执行命令
    result = commands.execute_command(args)
    
    print(f"命令执行结果: {'成功' if result == 0 else '失败'}")
    
    print("深度学习命令集测试完成")