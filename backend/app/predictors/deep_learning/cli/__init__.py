"""
命令行接口模块
CLI Module

提供命令定义、处理和注册功能。
"""

from .cli_commands import CommandDefinition
from .cli_handler import CommandHandler
from .cli_enhancer import (
    CLIEnhancer, AutoCompleter, SmartPrompt,
    cli_enhancer
)

__all__ = [
    'CommandDefinition',
    'CommandHandler',
    'CLIEnhancer',
    'AutoCompleter',
    'SmartPrompt',
    'cli_enhancer'
]
