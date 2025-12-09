#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLI体验增强模块
CLI Experience Enhancer Module

提供命令自动补全、智能提示、历史记录等功能。
"""

import os
import json
import readline
import rlcompleter
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import re

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException


@dataclass
class CommandCompletion:
    """命令补全信息"""
    command: str
    description: str
    options: List[str] = field(default_factory=list)
    subcommands: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


class AutoCompleter:
    """自动补全器"""
    
    def __init__(self):
        """初始化自动补全器"""
        self.commands = {}
        self.history_file = os.path.expanduser("~/.deep_learning_cli_history")
        self.completion_cache = {}
        
        # 设置readline
        self._setup_readline()
        
        logger_manager.debug("自动补全器初始化完成")
    
    def _setup_readline(self):
        """设置readline"""
        try:
            # 启用自动补全
            readline.set_completer(self.complete)
            readline.parse_and_bind("tab: complete")
            
            # 设置历史记录
            if os.path.exists(self.history_file):
                readline.read_history_file(self.history_file)
            
            # 设置历史记录大小
            readline.set_history_length(1000)
            
            logger_manager.debug("Readline设置完成")
            
        except Exception as e:
            logger_manager.warning(f"Readline设置失败: {e}")
    
    def register_command(self, completion: CommandCompletion):
        """
        注册命令补全信息
        
        Args:
            completion: 命令补全信息
        """
        try:
            self.commands[completion.command] = completion
            
            # 清除缓存
            self.completion_cache.clear()
            
            logger_manager.debug(f"命令补全注册成功: {completion.command}")
            
        except Exception as e:
            logger_manager.error(f"注册命令补全失败: {e}")
    
    def complete(self, text: str, state: int) -> Optional[str]:
        """
        自动补全函数
        
        Args:
            text: 当前输入文本
            state: 补全状态
            
        Returns:
            补全建议
        """
        try:
            if state == 0:
                # 第一次调用，生成补全列表
                line = readline.get_line_buffer()
                self.completion_cache[text] = self._generate_completions(line, text)
            
            # 返回补全建议
            completions = self.completion_cache.get(text, [])
            
            if state < len(completions):
                return completions[state]
            else:
                return None
                
        except Exception as e:
            logger_manager.error(f"自动补全失败: {e}")
            return None
    
    def _generate_completions(self, line: str, text: str) -> List[str]:
        """生成补全建议"""
        try:
            completions = []
            
            # 解析命令行
            parts = line.split()
            
            if not parts or (len(parts) == 1 and not line.endswith(' ')):
                # 补全主命令
                for cmd in self.commands.keys():
                    if cmd.startswith(text):
                        completions.append(cmd)
            
            elif len(parts) >= 1:
                # 补全子命令或选项
                main_cmd = parts[0]
                
                if main_cmd in self.commands:
                    completion_info = self.commands[main_cmd]
                    
                    # 补全子命令
                    for subcmd in completion_info.subcommands:
                        if subcmd.startswith(text):
                            completions.append(subcmd)
                    
                    # 补全选项
                    for option in completion_info.options:
                        if option.startswith(text):
                            completions.append(option)
                    
                    # 补全文件路径
                    if text.startswith('./') or text.startswith('/') or text.startswith('~'):
                        file_completions = self._complete_file_path(text)
                        completions.extend(file_completions)
            
            return sorted(completions)
            
        except Exception as e:
            logger_manager.error(f"生成补全建议失败: {e}")
            return []
    
    def _complete_file_path(self, text: str) -> List[str]:
        """补全文件路径"""
        try:
            completions = []
            
            # 展开路径
            expanded_path = os.path.expanduser(text)
            
            if os.path.isdir(expanded_path):
                # 目录补全
                try:
                    for item in os.listdir(expanded_path):
                        item_path = os.path.join(expanded_path, item)
                        if os.path.isdir(item_path):
                            completions.append(text + item + '/')
                        else:
                            completions.append(text + item)
                except PermissionError:
                    pass
            
            else:
                # 文件名补全
                dirname = os.path.dirname(expanded_path)
                basename = os.path.basename(expanded_path)
                
                if os.path.isdir(dirname):
                    try:
                        for item in os.listdir(dirname):
                            if item.startswith(basename):
                                item_path = os.path.join(dirname, item)
                                if text.startswith('~'):
                                    completion = text.replace(os.path.expanduser(text), item_path)
                                else:
                                    completion = os.path.join(os.path.dirname(text), item)
                                
                                if os.path.isdir(item_path):
                                    completion += '/'
                                
                                completions.append(completion)
                    except PermissionError:
                        pass
            
            return completions[:10]  # 限制数量
            
        except Exception as e:
            logger_manager.error(f"文件路径补全失败: {e}")
            return []
    
    def save_history(self):
        """保存历史记录"""
        try:
            readline.write_history_file(self.history_file)
            logger_manager.debug("历史记录保存成功")
            
        except Exception as e:
            logger_manager.error(f"保存历史记录失败: {e}")
    
    def get_command_help(self, command: str) -> str:
        """获取命令帮助"""
        try:
            if command in self.commands:
                completion = self.commands[command]
                
                help_text = f"命令: {completion.command}\n"
                help_text += f"描述: {completion.description}\n"
                
                if completion.subcommands:
                    help_text += f"子命令: {', '.join(completion.subcommands)}\n"
                
                if completion.options:
                    help_text += f"选项: {', '.join(completion.options)}\n"
                
                if completion.examples:
                    help_text += "示例:\n"
                    for example in completion.examples:
                        help_text += f"  {example}\n"
                
                return help_text
            
            return f"未找到命令: {command}"
            
        except Exception as e:
            logger_manager.error(f"获取命令帮助失败: {e}")
            return f"获取帮助失败: {e}"


class SmartPrompt:
    """智能提示器"""
    
    def __init__(self):
        """初始化智能提示器"""
        self.prompt_history = []
        self.context = {}
        
        logger_manager.debug("智能提示器初始化完成")
    
    def generate_prompt(self, context: Dict[str, Any] = None) -> str:
        """
        生成智能提示符
        
        Args:
            context: 上下文信息
            
        Returns:
            提示符字符串
        """
        try:
            if context:
                self.context.update(context)
            
            # 基础提示符
            prompt = "🤖 DL-Platform"
            
            # 添加当前模式
            if 'mode' in self.context:
                prompt += f" [{self.context['mode']}]"
            
            # 添加当前模型
            if 'current_model' in self.context:
                prompt += f" (model: {self.context['current_model']})"
            
            # 添加状态指示
            if 'status' in self.context:
                status = self.context['status']
                if status == 'training':
                    prompt += " 🔄"
                elif status == 'ready':
                    prompt += " ✅"
                elif status == 'error':
                    prompt += " ❌"
            
            prompt += " > "
            
            return prompt
            
        except Exception as e:
            logger_manager.error(f"生成智能提示符失败: {e}")
            return "DL-Platform > "
    
    def add_suggestion(self, command: str, suggestion: str):
        """
        添加命令建议
        
        Args:
            command: 命令
            suggestion: 建议
        """
        try:
            suggestion_text = f"💡 建议: {suggestion}"
            print(suggestion_text)
            
            logger_manager.debug(f"添加命令建议: {command} -> {suggestion}")
            
        except Exception as e:
            logger_manager.error(f"添加命令建议失败: {e}")
    
    def show_tips(self):
        """显示使用技巧"""
        tips = [
            "💡 使用 Tab 键进行命令自动补全",
            "💡 使用 help <command> 查看命令详细帮助",
            "💡 使用 history 查看命令历史",
            "💡 使用 Ctrl+R 搜索历史命令",
            "💡 使用 clear 清屏",
            "💡 使用 exit 或 quit 退出"
        ]
        
        print("\n🎯 使用技巧:")
        for tip in tips:
            print(f"  {tip}")
        print()


class CLIEnhancer:
    """CLI体验增强器"""
    
    def __init__(self):
        """初始化CLI体验增强器"""
        self.auto_completer = AutoCompleter()
        self.smart_prompt = SmartPrompt()
        self.command_aliases = {}
        self.shortcuts = {}
        
        # 注册默认命令
        self._register_default_commands()
        
        logger_manager.info("CLI体验增强器初始化完成")
    
    def _register_default_commands(self):
        """注册默认命令"""
        try:
            # 主要命令
            commands = [
                CommandCompletion(
                    command="train",
                    description="训练深度学习模型",
                    options=["--model", "--data", "--epochs", "--batch-size", "--lr"],
                    subcommands=["lstm", "transformer", "gan"],
                    examples=[
                        "train --model lstm --data data.csv --epochs 100",
                        "train lstm --batch-size 32 --lr 0.001"
                    ]
                ),
                CommandCompletion(
                    command="predict",
                    description="使用模型进行预测",
                    options=["--model", "--input", "--output", "--format"],
                    examples=[
                        "predict --model trained_model.pkl --input test_data.csv",
                        "predict --input data.json --output predictions.json"
                    ]
                ),
                CommandCompletion(
                    command="evaluate",
                    description="评估模型性能",
                    options=["--model", "--test-data", "--metrics"],
                    examples=[
                        "evaluate --model model.pkl --test-data test.csv",
                        "evaluate --metrics accuracy,precision,recall"
                    ]
                ),
                CommandCompletion(
                    command="visualize",
                    description="可视化数据和结果",
                    subcommands=["data", "predictions", "metrics", "dashboard"],
                    options=["--input", "--output", "--type"],
                    examples=[
                        "visualize data --input data.csv",
                        "visualize predictions --input predictions.json"
                    ]
                ),
                CommandCompletion(
                    command="config",
                    description="配置管理",
                    subcommands=["show", "set", "reset"],
                    options=["--key", "--value", "--file"],
                    examples=[
                        "config show",
                        "config set model.batch_size 32"
                    ]
                ),
                CommandCompletion(
                    command="help",
                    description="显示帮助信息",
                    examples=["help", "help train", "help predict"]
                ),
                CommandCompletion(
                    command="history",
                    description="显示命令历史",
                    options=["--limit", "--search"],
                    examples=["history", "history --limit 10"]
                ),
                CommandCompletion(
                    command="clear",
                    description="清屏",
                    examples=["clear"]
                ),
                CommandCompletion(
                    command="exit",
                    description="退出程序",
                    examples=["exit", "quit"]
                )
            ]
            
            for cmd in commands:
                self.auto_completer.register_command(cmd)
            
            # 设置别名
            self.command_aliases = {
                "t": "train",
                "p": "predict",
                "e": "evaluate",
                "v": "visualize",
                "c": "config",
                "h": "help",
                "q": "quit",
                "cls": "clear"
            }
            
            logger_manager.debug("默认命令注册完成")
            
        except Exception as e:
            logger_manager.error(f"注册默认命令失败: {e}")
    
    def enhance_command(self, command: str) -> str:
        """
        增强命令处理
        
        Args:
            command: 原始命令
            
        Returns:
            增强后的命令
        """
        try:
            # 处理别名
            parts = command.split()
            if parts and parts[0] in self.command_aliases:
                parts[0] = self.command_aliases[parts[0]]
                command = ' '.join(parts)
            
            # 处理快捷方式
            if command in self.shortcuts:
                command = self.shortcuts[command]
            
            return command
            
        except Exception as e:
            logger_manager.error(f"增强命令处理失败: {e}")
            return command
    
    def add_alias(self, alias: str, command: str):
        """
        添加命令别名
        
        Args:
            alias: 别名
            command: 实际命令
        """
        try:
            self.command_aliases[alias] = command
            logger_manager.debug(f"添加命令别名: {alias} -> {command}")
            
        except Exception as e:
            logger_manager.error(f"添加命令别名失败: {e}")
    
    def add_shortcut(self, shortcut: str, command: str):
        """
        添加快捷方式
        
        Args:
            shortcut: 快捷方式
            command: 完整命令
        """
        try:
            self.shortcuts[shortcut] = command
            logger_manager.debug(f"添加快捷方式: {shortcut} -> {command}")
            
        except Exception as e:
            logger_manager.error(f"添加快捷方式失败: {e}")
    
    def show_welcome_message(self):
        """显示欢迎信息"""
        welcome_msg = """
🚀 欢迎使用深度学习预测平台 CLI！

✨ 功能特性:
  • 智能命令补全 (Tab键)
  • 命令历史记录 (Ctrl+R搜索)
  • 命令别名和快捷方式
  • 上下文感知提示
  • 详细帮助系统

🎯 快速开始:
  • 输入 'help' 查看所有命令
  • 输入 'help <command>' 查看具体命令帮助
  • 使用 Tab 键自动补全命令和路径
  • 输入 'tips' 查看更多使用技巧

💡 常用命令:
  • train    - 训练模型
  • predict  - 进行预测
  • evaluate - 评估性能
  • visualize - 数据可视化

祝您使用愉快！🎉
"""
        print(welcome_msg)
    
    def cleanup(self):
        """清理资源"""
        try:
            self.auto_completer.save_history()
            logger_manager.debug("CLI增强器清理完成")
            
        except Exception as e:
            logger_manager.error(f"CLI增强器清理失败: {e}")


# 全局CLI增强器实例
cli_enhancer = CLIEnhancer()


if __name__ == "__main__":
    # 测试CLI增强器功能
    print("💻 测试CLI增强器功能...")
    
    try:
        enhancer = CLIEnhancer()
        
        # 显示欢迎信息
        enhancer.show_welcome_message()
        
        # 测试命令增强
        test_commands = ["t --model lstm", "p --input data.csv", "h train"]
        
        for cmd in test_commands:
            enhanced = enhancer.enhance_command(cmd)
            print(f"原始命令: {cmd}")
            print(f"增强命令: {enhanced}")
            print()
        
        # 测试智能提示
        enhancer.smart_prompt.show_tips()
        
        # 测试命令帮助
        help_text = enhancer.auto_completer.get_command_help("train")
        print("命令帮助示例:")
        print(help_text)
        
        print("✅ CLI增强器功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("CLI增强器功能测试完成")
