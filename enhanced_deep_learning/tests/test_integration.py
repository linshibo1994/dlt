#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
集成测试
测试各组件之间的交互
"""

import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入测试模块
from enhanced_deep_learning.gpu_accelerator import GPUAccelerator, GPUMemoryManager
from enhanced_deep_learning.batch_processor import BatchProcessor
from enhanced_deep_learning.model_optimizer import ModelOptimizer, CachedModel
from enhanced_deep_learning.cli_commands import DeepLearningCommands
from enhanced_deep_learning.cli_handler import CLIHandler
from enhanced_deep_learning.system_integration import SystemIntegration


class TestComponentIntegration(unittest.TestCase):
    """组件集成测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建组件
        self.accelerator = GPUAccelerator()
        self.memory_manager = GPUMemoryManager(self.accelerator)
        self.batch_processor = BatchProcessor(memory_manager=self.memory_manager)
        self.optimizer = ModelOptimizer()
        self.commands = DeepLearningCommands()
        self.cli_handler = CLIHandler()
        self.system_integration = SystemIntegration()
    
    def test_gpu_batch_integration(self):
        """测试GPU加速器和批处理系统集成"""
        # 模拟GPU内存优化
        with patch.object(self.memory_manager, 'optimize_batch_size', return_value=64):
            # 优化批处理大小
            batch_size = self.batch_processor.optimize_batch_size(500)
            
            # 检查结果
            self.assertEqual(batch_size, 64)
    
    def test_optimizer_cached_model(self):
        """测试模型优化器和缓存模型集成"""
        # 创建模拟模型
        mock_model = MagicMock()
        mock_model.predict.return_value = [1, 2, 3]
        
        # 创建缓存模型
        cached_model = CachedModel(mock_model, self.optimizer)
        
        # 创建测试数据
        import numpy as np
        test_data = np.array([[1, 2, 3]])
        
        # 第一次预测
        result1 = cached_model.predict(test_data)
        
        # 第二次预测（应该使用缓存）
        result2 = cached_model.predict(test_data)
        
        # 检查结果
        self.assertEqual(mock_model.predict.call_count, 1)  # 只应该调用一次
        self.assertEqual(cached_model.cache_hits, 1)
        self.assertEqual(cached_model.cache_misses, 1)
    
    def test_cli_system_integration(self):
        """测试CLI和系统集成"""
        # 模拟CLI处理器
        with patch.object(self.cli_handler, 'handle_command', return_value=0):
            # 处理命令
            result = self.system_integration.handle_command(MagicMock())
            
            # 检查结果
            self.assertEqual(result, 0)
            self.cli_handler.handle_command.assert_called_once()
    
    @patch('enhanced_deep_learning.gpu_accelerator.GPUAccelerator.is_gpu_available')
    @patch('enhanced_deep_learning.model_optimizer.ModelOptimizer.quantize_model')
    def test_gpu_optimizer_integration(self, mock_quantize, mock_is_gpu_available):
        """测试GPU加速器和模型优化器集成"""
        # 设置模拟返回值
        mock_is_gpu_available.return_value = True
        mock_quantize.return_value = MagicMock()
        
        # 创建模拟模型
        mock_model = MagicMock()
        
        # 检查GPU可用性
        gpu_available = self.accelerator.is_gpu_available()
        self.assertTrue(gpu_available)
        
        # 量化模型
        quantized_model = self.optimizer.quantize_model(mock_model, 'tensorflow', 'float16')
        
        # 检查结果
        mock_quantize.assert_called_once_with(mock_model, 'tensorflow', 'float16')
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_commands_cli_integration(self, mock_parse_args):
        """测试命令集和CLI处理器集成"""
        # 设置模拟返回值
        mock_args = MagicMock()
        mock_args.command = 'info'
        mock_args.all = True
        mock_args.gpu = False
        mock_args.models = False
        mock_args.performance = False
        mock_parse_args.return_value = mock_args
        
        # 模拟info_command
        with patch.object(self.commands, 'info_command', return_value=0):
            # 处理命令
            result = self.cli_handler.handle_command(['info', '--all'])
            
            # 检查结果
            self.assertEqual(result, 0)
            self.commands.info_command.assert_called_once()


class TestEndToEndWorkflow(unittest.TestCase):
    """端到端工作流测试类"""
    
    @patch('enhanced_deep_learning.gpu_accelerator.GPUAccelerator.is_gpu_available')
    @patch('enhanced_deep_learning.gpu_accelerator.GPUAccelerator.migrate_model_to_gpu')
    @patch('enhanced_deep_learning.batch_processor.BatchProcessor.batch_process')
    @patch('enhanced_deep_learning.model_optimizer.ModelOptimizer.quantize_model')
    @patch('enhanced_deep_learning.cli_commands.DeepLearningCommands.predict_command')
    def test_prediction_workflow(self, mock_predict, mock_quantize, mock_batch_process, 
                               mock_migrate, mock_is_gpu_available):
        """测试预测工作流"""
        # 设置模拟返回值
        mock_is_gpu_available.return_value = True
        mock_migrate.return_value = MagicMock()
        mock_batch_process.return_value = [1, 2, 3]
        mock_quantize.return_value = MagicMock()
        mock_predict.return_value = 0
        
        # 创建组件
        accelerator = GPUAccelerator()
        memory_manager = GPUMemoryManager(accelerator)
        batch_processor = BatchProcessor(memory_manager=memory_manager)
        optimizer = ModelOptimizer()
        commands = DeepLearningCommands()
        
        # 模拟参数
        args = MagicMock()
        args.command = 'predict'
        args.model = 'ensemble'
        args.count = 5
        args.compound = False
        args.confidence = True
        args.report = False
        
        # 执行预测命令
        result = commands.execute_command(args)
        
        # 检查结果
        self.assertEqual(result, 0)
        mock_predict.assert_called_once_with(args)
    
    @patch('enhanced_deep_learning.gpu_accelerator.GPUAccelerator.is_gpu_available')
    @patch('enhanced_deep_learning.gpu_accelerator.GPUAccelerator.enable_tensorflow_gpu')
    @patch('enhanced_deep_learning.batch_processor.BatchProcessor.optimize_batch_size')
    @patch('enhanced_deep_learning.model_optimizer.ModelOptimizer.start_resource_monitoring')
    @patch('enhanced_deep_learning.model_optimizer.ModelOptimizer.record_resource_usage')
    @patch('enhanced_deep_learning.cli_commands.DeepLearningCommands.train_command')
    def test_training_workflow(self, mock_train, mock_record, mock_start, 
                             mock_optimize, mock_enable, mock_is_gpu_available):
        """测试训练工作流"""
        # 设置模拟返回值
        mock_is_gpu_available.return_value = True
        mock_enable.return_value = True
        mock_optimize.return_value = 64
        mock_start.return_value = None
        mock_record.return_value = {'cpu': 50, 'memory': 60, 'disk': 70, 'time': 123456789}
        mock_train.return_value = 0
        
        # 创建组件
        accelerator = GPUAccelerator()
        memory_manager = GPUMemoryManager(accelerator)
        batch_processor = BatchProcessor(memory_manager=memory_manager)
        optimizer = ModelOptimizer()
        commands = DeepLearningCommands()
        
        # 模拟参数
        args = MagicMock()
        args.command = 'train'
        args.model = 'all'
        args.periods = 1000
        args.epochs = 100
        args.batch_size = 32
        args.gpu = True
        args.save_model = True
        
        # 执行训练命令
        result = commands.execute_command(args)
        
        # 检查结果
        self.assertEqual(result, 0)
        mock_train.assert_called_once_with(args)
        mock_is_gpu_available.assert_called()
        mock_optimize.assert_called()


if __name__ == "__main__":
    unittest.main()