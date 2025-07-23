"""
系统集成模块
System Integration Module

提供系统集成、命令注册、迁移工具等功能。
"""

from .system_integrator import (
    SystemIntegrator, IntegrationConfig, IntegrationStatus,
    system_integrator
)
from .command_registry import (
    CommandRegistry, CommandDefinition, CommandHandler,
    command_registry
)
from .migration_tools import (
    MigrationTool, MigrationConfig, MigrationStatus,
    migration_tool
)

__all__ = [
    # 系统集成器
    'SystemIntegrator',
    'IntegrationConfig',
    'IntegrationStatus',
    'system_integrator',
    
    # 命令注册表
    'CommandRegistry',
    'CommandDefinition',
    'CommandHandler',
    'command_registry',
    
    # 迁移工具
    'MigrationTool',
    'MigrationConfig',
    'MigrationStatus',
    'migration_tool'
]
