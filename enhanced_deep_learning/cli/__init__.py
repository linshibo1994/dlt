"""
命令行接口模块
CLI Module

提供命令定义、处理和注册功能。
"""

from .cli_commands import CommandDefinition
from .cli_handler import CommandHandler

__all__ = [
    'CommandDefinition',
    'CommandHandler'
]
