#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置管理器模块
Configuration Manager Module

提供配置文件管理、环境变量处理、配置验证等功能。
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import configparser

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException


@dataclass
class ConfigSection:
    """配置节"""
    name: str
    data: Dict[str, Any] = field(default_factory=dict)
    required_keys: List[str] = field(default_factory=list)
    optional_keys: List[str] = field(default_factory=list)


class ConfigValidator:
    """配置验证器"""
    
    def __init__(self):
        """初始化配置验证器"""
        logger_manager.debug("配置验证器初始化完成")
    
    def validate_section(self, section: ConfigSection) -> bool:
        """
        验证配置节
        
        Args:
            section: 配置节
            
        Returns:
            是否验证通过
        """
        try:
            # 检查必需键
            for key in section.required_keys:
                if key not in section.data:
                    logger_manager.error(f"配置节 {section.name} 缺少必需键: {key}")
                    return False
            
            return True
            
        except Exception as e:
            logger_manager.error(f"配置验证失败: {e}")
            return False


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "config"):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置目录
        """
        self.config_dir = Path(config_dir)
        self.config_data = {}
        self.validator = ConfigValidator()
        
        # 确保配置目录存在
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载默认配置
        self._load_default_config()
        
        logger_manager.info("配置管理器初始化完成")
    
    def _load_default_config(self):
        """加载默认配置"""
        try:
            default_config = {
                "system": {
                    "log_level": "INFO",
                    "max_workers": 4,
                    "cache_size": 1000,
                    "timeout": 30
                },
                "model": {
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "epochs": 100,
                    "early_stopping": True
                },
                "data": {
                    "train_ratio": 0.8,
                    "validation_ratio": 0.1,
                    "test_ratio": 0.1,
                    "shuffle": True
                },
                "prediction": {
                    "confidence_threshold": 0.8,
                    "max_predictions": 10,
                    "cache_predictions": True
                }
            }
            
            self.config_data.update(default_config)
            
        except Exception as e:
            logger_manager.error(f"加载默认配置失败: {e}")
    
    def load_config(self, config_file: str) -> bool:
        """
        加载配置文件
        
        Args:
            config_file: 配置文件路径
            
        Returns:
            是否加载成功
        """
        try:
            config_path = self.config_dir / config_file
            
            if not config_path.exists():
                logger_manager.warning(f"配置文件不存在: {config_path}")
                return False
            
            # 根据文件扩展名选择加载方式
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            elif config_path.suffix.lower() in ['.ini', '.cfg']:
                config_parser = configparser.ConfigParser()
                config_parser.read(config_path, encoding='utf-8')
                config = {section: dict(config_parser[section]) for section in config_parser.sections()}
            else:
                logger_manager.error(f"不支持的配置文件格式: {config_path.suffix}")
                return False
            
            # 合并配置
            self._merge_config(config)
            
            logger_manager.info(f"配置文件加载成功: {config_file}")
            return True
            
        except Exception as e:
            logger_manager.error(f"加载配置文件失败: {e}")
            return False
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """合并配置"""
        try:
            for key, value in new_config.items():
                if key in self.config_data and isinstance(self.config_data[key], dict) and isinstance(value, dict):
                    self.config_data[key].update(value)
                else:
                    self.config_data[key] = value
                    
        except Exception as e:
            logger_manager.error(f"合并配置失败: {e}")
    
    def save_config(self, config_file: str, format: str = "yaml") -> bool:
        """
        保存配置文件
        
        Args:
            config_file: 配置文件路径
            format: 保存格式 (yaml, json, ini)
            
        Returns:
            是否保存成功
        """
        try:
            config_path = self.config_dir / config_file
            
            if format.lower() in ['yaml', 'yml']:
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config_data, f, default_flow_style=False, allow_unicode=True)
            elif format.lower() == 'json':
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config_data, f, indent=2, ensure_ascii=False)
            elif format.lower() in ['ini', 'cfg']:
                config_parser = configparser.ConfigParser()
                for section, values in self.config_data.items():
                    config_parser[section] = {k: str(v) for k, v in values.items()}
                with open(config_path, 'w', encoding='utf-8') as f:
                    config_parser.write(f)
            else:
                logger_manager.error(f"不支持的保存格式: {format}")
                return False
            
            logger_manager.info(f"配置文件保存成功: {config_file}")
            return True
            
        except Exception as e:
            logger_manager.error(f"保存配置文件失败: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键 (支持点分隔的嵌套键)
            default: 默认值
            
        Returns:
            配置值
        """
        try:
            keys = key.split('.')
            value = self.config_data
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception as e:
            logger_manager.error(f"获取配置值失败: {e}")
            return default
    
    def set(self, key: str, value: Any):
        """
        设置配置值
        
        Args:
            key: 配置键 (支持点分隔的嵌套键)
            value: 配置值
        """
        try:
            keys = key.split('.')
            config = self.config_data
            
            # 创建嵌套结构
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # 设置值
            config[keys[-1]] = value
            
            logger_manager.debug(f"配置值设置成功: {key} = {value}")
            
        except Exception as e:
            logger_manager.error(f"设置配置值失败: {e}")
    
    def has(self, key: str) -> bool:
        """
        检查配置键是否存在
        
        Args:
            key: 配置键
            
        Returns:
            是否存在
        """
        try:
            keys = key.split('.')
            value = self.config_data
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return False
            
            return True
            
        except Exception as e:
            logger_manager.error(f"检查配置键失败: {e}")
            return False
    
    def remove(self, key: str) -> bool:
        """
        删除配置键
        
        Args:
            key: 配置键
            
        Returns:
            是否删除成功
        """
        try:
            keys = key.split('.')
            config = self.config_data
            
            # 导航到父级
            for k in keys[:-1]:
                if isinstance(config, dict) and k in config:
                    config = config[k]
                else:
                    return False
            
            # 删除键
            if isinstance(config, dict) and keys[-1] in config:
                del config[keys[-1]]
                logger_manager.debug(f"配置键删除成功: {key}")
                return True
            
            return False
            
        except Exception as e:
            logger_manager.error(f"删除配置键失败: {e}")
            return False
    
    def get_section(self, section_name: str) -> Dict[str, Any]:
        """
        获取配置节
        
        Args:
            section_name: 节名称
            
        Returns:
            配置节数据
        """
        return self.config_data.get(section_name, {})
    
    def set_section(self, section_name: str, section_data: Dict[str, Any]):
        """
        设置配置节
        
        Args:
            section_name: 节名称
            section_data: 节数据
        """
        self.config_data[section_name] = section_data
        logger_manager.debug(f"配置节设置成功: {section_name}")
    
    def load_from_env(self, prefix: str = "DL_"):
        """
        从环境变量加载配置
        
        Args:
            prefix: 环境变量前缀
        """
        try:
            for key, value in os.environ.items():
                if key.startswith(prefix):
                    config_key = key[len(prefix):].lower().replace('_', '.')
                    
                    # 尝试转换数据类型
                    try:
                        # 尝试转换为数字
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        # 尝试转换为布尔值
                        if value.lower() in ['true', 'false']:
                            value = value.lower() == 'true'
                        # 否则保持字符串
                    
                    self.set(config_key, value)
            
            logger_manager.info(f"从环境变量加载配置完成，前缀: {prefix}")
            
        except Exception as e:
            logger_manager.error(f"从环境变量加载配置失败: {e}")
    
    def get_all_config(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self.config_data.copy()
    
    def clear(self):
        """清空所有配置"""
        self.config_data.clear()
        self._load_default_config()
        logger_manager.info("配置已清空并重置为默认值")


# 全局配置管理器实例
config_manager = ConfigManager()


if __name__ == "__main__":
    # 测试配置管理器功能
    print("⚙️ 测试配置管理器功能...")
    
    try:
        manager = ConfigManager()
        
        # 测试基本操作
        manager.set("test.key", "test_value")
        value = manager.get("test.key")
        print(f"✅ 配置设置和获取: {value}")
        
        # 测试环境变量加载
        os.environ["DL_TEST_ENV"] = "env_value"
        manager.load_from_env("DL_")
        env_value = manager.get("test.env")
        print(f"✅ 环境变量加载: {env_value}")
        
        # 测试配置节
        section_data = {"key1": "value1", "key2": "value2"}
        manager.set_section("test_section", section_data)
        section = manager.get_section("test_section")
        print(f"✅ 配置节操作: {len(section)} 个键")
        
        print("✅ 配置管理器功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("配置管理器功能测试完成")
