#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLI命令测试
"""

import unittest
import os
import sys
import argparse
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from enhanced_deep_learning.cli_commands import DeepLearningCommands
from enhanced_deep_learning.cli_handler import CLIHandler


class TestCLICommands(unittest.TestCase):
    """CLI命令测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建命令集
        self.commands = DeepLearningCommands()
        
        # 创建CLI处理器
        self.handler = CLIHandler()
    
    def test_command_parser(self):
        """测试命令解析器"""
        # 获取命令解析器
        parser = self.commands.get_command_parser()
        
        # 检查解析器类型
        self.assertIsInstance(parser, argparse.ArgumentParser)
        
        # 解析测试命令
        args = parser.parse_args(['info', '--all'])
        
        # 检查解析结果
        self.assertEqual(args.command, 'info')
        self.assertTrue(args.all)
    
    @patch('enhanced_deep_learning.cli_commands.DeepLearningCommands.info_command')
    def test_execute_command(self, mock_info_command):
        """测试执行命令"""
        # 设置模拟返回值
        mock_info_command.return_value = 0
        
        # 创建参数
        args = argparse.Namespace(command='info', all=True, gpu=False, models=False, performance=False)
        
        # 执行命令
        result = self.commands.execute_command(args)
        
        # 检查结果
        self.assertEqual(result, 0)
        mock_info_command.assert_called_once_with(args)
    
    @patch('enhanced_deep_learning.cli_commands.DeepLearningCommands.execute_command')
    def test_cli_handler(self, mock_execute_command):
        """测试CLI处理器"""
        # 设置模拟返回值
        mock_execute_command.return_value = 0
        
        # 处理命令
        result = self.handler.handle_command(['info', '--all'])
        
        # 检查结果
        self.assertEqual(result, 0)
        mock_execute_command.assert_called_once()
    
    @patch('enhanced_deep_learning.cli_handler.CLIHandler.handle_command')
    def test_execute_main_cli_command(self, mock_handle_command):
        """测试执行主CLI命令"""
        # 设置模拟返回值
        mock_handle_command.return_value = 0
        
        # 创建参数
        args = argparse.Namespace(command='dl', dl_command='info', all=True)
        
        # 执行命令
        result = self.handler.execute_main_cli_command(args)
        
        # 检查结果
        self.assertEqual(result, 0)
        mock_handle_command.assert_called_once()
    
    def test_register_with_main_cli(self):
        """测试注册到主CLI系统"""
        # 创建模拟主解析器
        main_parser = MagicMock()
        
        # 模拟add_parser方法
        main_parser.add_parser.return_value.add_subparsers.return_value.add_parser.return_value = MagicMock()
        
        # 注册命令
        self.handler.register_with_main_cli(main_parser)
        
        # 检查是否调用了add_parser
        main_parser.add_parser.assert_called_once_with('dl', help='深度学习预测命令')


if __name__ == "__main__":
    unittest.main()