#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置管理器
Configuration Manager

提供配置加载、验证、覆盖和持久化功能。
"""

import os
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
from core_modules import logger_manager

# 尝试导入yaml，如果不可用则跳过
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_dir: str = "config"):
        """
        初始化配置管理器

        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        # 配置存储
        self._configs = {}
        self._default_configs = {}

        # 加载默认配置
        self._load_default_configs()

        logger_manager.info("配置管理器初始化完成")

    def _load_default_configs(self):
        """加载默认配置"""
        self._default_configs = {
            'transformer': DEFAULT_TRANSFORMER_CONFIG,
            'gan': DEFAULT_GAN_CONFIG,
            'ensemble': DEFAULT_ENSEMBLE_CONFIG,
            'meta_learning': DEFAULT_META_LEARNING_CONFIG,
            'system': DEFAULT_SYSTEM_CONFIG
        }

    def load_config(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        从文件加载配置

        Args:
            path: 配置文件路径

        Returns:
            配置字典
        """
        try:
            path = Path(path)

            if not path.exists():
                logger_manager.warning(f"配置文件不存在: {path}")
                return {}

            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix.lower() == '.json':
                    config = json.load(f)
                elif path.suffix.lower() in ['.yml', '.yaml'] and YAML_AVAILABLE:
                    config = yaml.safe_load(f)
                elif path.suffix.lower() in ['.yml', '.yaml'] and not YAML_AVAILABLE:
                    logger_manager.error("YAML支持不可用，请安装PyYAML")
                    return {}
                else:
                    logger_manager.error(f"不支持的配置文件格式: {path.suffix}")
                    return {}

            logger_manager.info(f"配置加载成功: {path}")
            return config

        except Exception as e:
            logger_manager.error(f"配置加载失败: {e}")
            return {}

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键，支持点分隔的嵌套键
            default: 默认值

        Returns:
            配置值
        """
        try:
            keys = key.split('.')
            config = self._configs

            for k in keys:
                if isinstance(config, dict) and k in config:
                    config = config[k]
                else:
                    # 尝试从默认配置获取
                    return self._get_default_config(key, default)

            return config

        except Exception as e:
            logger_manager.error(f"获取配置失败: {key}, {e}")
            return default

    def _get_default_config(self, key: str, default: Any = None) -> Any:
        """从默认配置获取值"""
        try:
            keys = key.split('.')
            config = self._default_configs

            for k in keys:
                if isinstance(config, dict) and k in config:
                    config = config[k]
                else:
                    return default

            return config

        except:
            return default

    def set_config(self, key: str, value: Any) -> None:
        """
        设置配置值

        Args:
            key: 配置键，支持点分隔的嵌套键
            value: 配置值
        """
        try:
            keys = key.split('.')
            config = self._configs

            # 创建嵌套字典结构
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]

            config[keys[-1]] = value
            logger_manager.debug(f"配置设置成功: {key} = {value}")

        except Exception as e:
            logger_manager.error(f"配置设置失败: {key}, {e}")

    def save_config(self, path: Union[str, Path], config: Optional[Dict[str, Any]] = None) -> bool:
        """
        保存配置到文件

        Args:
            path: 保存路径
            config: 要保存的配置，默认保存当前所有配置

        Returns:
            是否保存成功
        """
        try:
            path = Path(path)
            config = config or self._configs

            # 确保目录存在
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                if path.suffix.lower() == '.json':
                    json.dump(config, f, indent=2, ensure_ascii=False)
                elif path.suffix.lower() in ['.yml', '.yaml'] and YAML_AVAILABLE:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                elif path.suffix.lower() in ['.yml', '.yaml'] and not YAML_AVAILABLE:
                    logger_manager.warning("YAML支持不可用，使用JSON格式保存")
                    json.dump(config, f, indent=2, ensure_ascii=False)
                else:
                    # 默认使用JSON格式
                    json.dump(config, f, indent=2, ensure_ascii=False)

            logger_manager.info(f"配置保存成功: {path}")
            return True

        except Exception as e:
            logger_manager.error(f"配置保存失败: {e}")
            return False

    def validate_config(self, schema: Dict[str, Any]) -> bool:
        """
        验证配置

        Args:
            schema: 配置模式

        Returns:
            是否验证通过
        """
        try:
            # 简单的配置验证逻辑
            for key, expected_type in schema.items():
                value = self.get_config(key)
                if value is not None and not isinstance(value, expected_type):
                    logger_manager.error(f"配置验证失败: {key} 应为 {expected_type.__name__}")
                    return False

            logger_manager.info("配置验证通过")
            return True

        except Exception as e:
            logger_manager.error(f"配置验证失败: {e}")
            return False

    def merge_config(self, new_config: Dict[str, Any]) -> None:
        """
        合并配置

        Args:
            new_config: 新配置
        """
        try:
            self._deep_merge(self._configs, new_config)
            logger_manager.info("配置合并完成")

        except Exception as e:
            logger_manager.error(f"配置合并失败: {e}")

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """深度合并字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get_all_configs(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self._configs.copy()

    def clear_config(self) -> None:
        """清空配置"""
        self._configs.clear()
        logger_manager.info("配置已清空")


# Transformer模型默认配置
DEFAULT_TRANSFORMER_CONFIG = {
    "name": "Transformer",
    "sequence_length": 20,
    "d_model": 128,
    "num_heads": 8,
    "num_layers": 4,
    "dff": 512,
    "dropout_rate": 0.1,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 100,
    "early_stopping_patience": 10
}

# GAN模型默认配置
DEFAULT_GAN_CONFIG = {
    "name": "GAN",
    "latent_dim": 100,
    "generator_layers": [128, 256, 128],
    "discriminator_layers": [128, 64, 32],
    "learning_rate": 0.0002,
    "beta1": 0.5,
    "batch_size": 64,
    "epochs": 200,
    "noise_std": 0.1
}

# 集成管理器默认配置
DEFAULT_ENSEMBLE_CONFIG = {
    "weights": {
        "transformer": 0.4,
        "gan": 0.3,
        "lstm": 0.3
    },
    "voting_threshold": 0.5,
    "confidence_threshold": 0.6
}

# 元学习优化器默认配置
DEFAULT_META_LEARNING_CONFIG = {
    "history_size": 50,
    "update_interval": 10,
    "learning_rate": 0.01,
    "performance_threshold": 0.5,
    "retrain_threshold": 0.3
}

# 数据管理器默认配置
DEFAULT_DATA_MANAGER_CONFIG = {
    "window_sizes": [100, 300, 500, 1000],
    "default_window": 500,
    "cache_enabled": True,
    "augmentation_factor": 1.5,
    "normalization": "minmax"
}

# 性能优化器默认配置
DEFAULT_PERFORMANCE_CONFIG = {
    "gpu_enabled": True,
    "batch_size": 32,
    "quantization_enabled": False,
    "cache_intermediate": True,
    "monitor_interval": 5
}

# 系统默认配置
DEFAULT_SYSTEM_CONFIG = {
    "model_dir": "models",
    "cache_dir": "cache",
    "log_level": "INFO",
    "max_workers": 4,
    "memory_limit": "2GB",
    "gpu_memory_fraction": 0.8,
    "enable_mixed_precision": True,
    "enable_xla": False
}

# 全局配置管理器实例
config_manager = ConfigManager()