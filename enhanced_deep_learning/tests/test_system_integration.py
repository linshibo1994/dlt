#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
系统集成测试
"""

import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from enhanced_deep_learning.system_integration import SystemIntegration


class TestSystemIntegration(unittest.TestCase):
    """系统集成测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建系统集成
        self.integration = SystemIntegration()
    
    def test_get_prediction_methods(self):
        """测试获取预测方法"""
        # 获取预测方法
        methods = self.integration.get_prediction_methods()
        
        # 检查结果
        self.assertIsInstance(methods, list)
        self.assertEqual(len(methods), 3)  # 应该有3种方法
        
        # 检查方法名称
        method_names = [m['name'] for m in methods]
        self.assertIn('transformer', method_names)
        self.assertIn('gan', method_names)
        self.assertIn('ensemble', method_names)
        
        # 检查方法属性
        for method in methods:
            self.assertIn('name', method)
            self.assertIn('display_name', method)
            self.assertIn('description', method)
            self.assertIn('category', method)
            self.assertIn('command', method)
            self.assertIn('options', method)
            self.assertIsInstance(method['options'], list)
    
    def test_get_help_content(self):
        """测试获取帮助内容"""
        # 获取帮助内容
        help_content = self.integration.get_help_content()
        
        # 检查结果
        self.assertIsInstance(help_content, dict)
        
        # 检查主题
        self.assertIn('deep_learning', help_content)
        self.assertIn('transformer', help_content)
        self.assertIn('gan', help_content)
        self.assertIn('ensemble', help_content)
        
        # 检查内容
        for topic, content in help_content.items():
            self.assertIsInstance(content, str)
            self.assertGreater(len(content), 100)  # 内容应该足够长
    
    @patch('enhanced_deep_learning.system_integration.GPUAccelerator')
    @patch('enhanced_deep_learning.system_integration.MetaLearningManager')
    def test_get_system_info(self, mock_meta, mock_gpu):
        """测试获取系统信息"""
        # 设置模拟对象
        mock_gpu.return_value.is_gpu_available.return_value = True
        mock_gpu.return_value.get_gpu_devices.return_value = ['GPU 0']
        mock_gpu.return_value.get_gpu_memory_info.return_value = {'GPU 0': {'total': 8, 'used': 2}}
        
        mock_meta.return_value.get_status.return_value = {'auto_weight': True, 'auto_retrain': False}
        mock_meta.return_value.get_performance_tracking.return_value = {'transformer': {'accuracy': 0.8}}
        
        # 模拟os.path.exists和os.listdir
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=['transformer_model.h5', 'gan_model.h5', 'ensemble_config.json']):
            
            # 获取系统信息
            info = self.integration.get_system_info()
            
            # 检查结果
            self.assertIsInstance(info, dict)
            self.assertEqual(info['status'], 'ok')
            
            # 检查GPU信息
            self.assertIn('gpu', info)
            self.assertTrue(info['gpu']['available'])
            self.assertEqual(info['gpu']['devices'], ['GPU 0'])
            
            # 检查元学习信息
            self.assertIn('metalearning', info)
            self.assertTrue(info['metalearning']['status']['auto_weight'])
            self.assertFalse(info['metalearning']['status']['auto_retrain'])
            
            # 检查模型信息
            self.assertIn('models', info)
            self.assertTrue(info['models']['transformer']['available'])
            self.assertTrue(info['models']['gan']['available'])
            self.assertTrue(info['models']['ensemble']['available'])
    
    @patch('enhanced_deep_learning.system_integration.cli_handler')
    def test_register_commands(self, mock_cli_handler):
        """测试注册命令"""
        # 创建模拟主解析器
        main_parser = MagicMock()
        
        # 注册命令
        self.integration.register_commands(main_parser)
        
        # 检查是否调用了register_with_main_cli
        mock_cli_handler.register_with_main_cli.assert_called_once_with(main_parser)
    
    @patch('enhanced_deep_learning.system_integration.cli_handler')
    def test_handle_command(self, mock_cli_handler):
        """测试处理命令"""
        # 设置模拟返回值
        mock_cli_handler.execute_main_cli_command.return_value = 0
        
        # 创建模拟参数
        args = MagicMock()
        
        # 处理命令
        result = self.integration.handle_command(args)
        
        # 检查结果
        self.assertEqual(result, 0)
        mock_cli_handler.execute_main_cli_command.assert_called_once_with(args)
    
    def test_check_model_available(self):
        """测试检查模型是否可用"""
        # 模拟os.path.exists和os.listdir
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=['transformer_model.h5', 'gan_model.h5', 'ensemble_config.json']):
            
            # 检查模型可用性
            self.assertTrue(self.integration._check_model_available('transformer'))
            self.assertTrue(self.integration._check_model_available('gan'))
            self.assertTrue(self.integration._check_model_available('ensemble'))
            self.assertFalse(self.integration._check_model_available('unknown'))
        
        # 模拟模型目录不存在
        with patch('os.path.exists', return_value=False):
            self.assertFalse(self.integration._check_model_available('transformer'))


if __name__ == "__main__":
    unittest.main()