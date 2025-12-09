# -*- coding: utf-8 -*-
"""
大乐透智能预测系统 - 统一路径配置模块

该模块提供统一的路径管理，将所有硬编码路径集中管理。
所有其他模块应该从这里导入路径配置。
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# 获取项目根目录
# path_config.py 位于 backend/app/core/，需要向上4级到达项目根目录
# backend/app/core/path_config.py -> backend/app/core -> backend/app -> backend -> dlt(项目根)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()

# 路径配置缓存
_path_config: Optional[Dict[str, Any]] = None


def get_project_root() -> Path:
    """获取项目根目录"""
    return PROJECT_ROOT


def load_paths_config() -> Dict[str, Any]:
    """加载 paths.yaml 配置文件"""
    global _path_config
    if _path_config is not None:
        return _path_config

    config_path = PROJECT_ROOT / "config" / "paths.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            _path_config = yaml.safe_load(f)
    else:
        # 使用默认配置
        _path_config = get_default_paths()

    return _path_config


def get_default_paths() -> Dict[str, Any]:
    """获取默认路径配置"""
    return {
        'data': {
            'root': 'data',
            'historical': 'data/dlt_data_all.csv'
        },
        'artifacts': {
            'root': 'artifacts',
            'cache': {
                'root': 'artifacts/cache',
                'analysis': 'artifacts/cache/analysis',
                'data': 'artifacts/cache/data',
                'models': 'artifacts/cache/models'
            },
            'logs': {
                'root': 'artifacts/logs',
                'app': 'artifacts/logs/app.log',
                'deep_learning': 'artifacts/logs/deep_learning.log',
                'errors': 'artifacts/logs/errors.log'
            },
            'models': {
                'root': 'artifacts/models',
                'lstm': 'artifacts/models/lstm',
                'transformer': 'artifacts/models/transformer',
                'gan': 'artifacts/models/gan'
            },
            'reports': {
                'root': 'artifacts/reports',
                'predictions': 'artifacts/reports/predictions',
                'analysis': 'artifacts/reports/analysis',
                'backtest': 'artifacts/reports/backtest',
                'visualization': 'artifacts/reports/visualization'
            }
        },
        'config': {
            'root': 'config',
            'main': 'config/config.json',
            'prediction': 'config/prediction.yaml',
            'training': 'config/training.yaml',
            'acceleration': 'config/acceleration.yaml',
            'gui': 'config/gui_config.json'
        }
    }


class PathManager:
    """路径管理器 - 提供便捷的路径访问接口"""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or PROJECT_ROOT
        self.config = load_paths_config()
        self._ensure_directories()

    def _ensure_directories(self):
        """确保必要的目录存在"""
        dirs_to_create = [
            self.cache_dir,
            self.logs_dir,
            self.models_dir,
            self.reports_dir,
            self.cache_analysis_dir,
            self.cache_data_dir,
            self.cache_models_dir,
            self.reports_predictions_dir,
            self.reports_analysis_dir,
            self.reports_backtest_dir,
            self.reports_visualization_dir,
            self.models_lstm_dir,
            self.models_transformer_dir,
            self.models_gan_dir,
        ]
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)

    # ===== 数据目录 =====
    @property
    def data_dir(self) -> Path:
        """数据目录"""
        return self.base_dir / self.config['data']['root']

    @property
    def data_file(self) -> Path:
        """历史数据文件"""
        return self.base_dir / self.config['data']['historical']

    # ===== 缓存目录 =====
    @property
    def cache_dir(self) -> Path:
        """缓存根目录"""
        return self.base_dir / self.config['artifacts']['cache']['root']

    @property
    def cache_analysis_dir(self) -> Path:
        """分析缓存目录"""
        return self.base_dir / self.config['artifacts']['cache']['analysis']

    @property
    def cache_data_dir(self) -> Path:
        """数据缓存目录"""
        return self.base_dir / self.config['artifacts']['cache']['data']

    @property
    def cache_models_dir(self) -> Path:
        """模型缓存目录"""
        return self.base_dir / self.config['artifacts']['cache']['models']

    # ===== 日志目录 =====
    @property
    def logs_dir(self) -> Path:
        """日志根目录"""
        return self.base_dir / self.config['artifacts']['logs']['root']

    @property
    def app_log_file(self) -> Path:
        """应用日志文件"""
        return self.base_dir / self.config['artifacts']['logs']['app']

    @property
    def deep_learning_log_file(self) -> Path:
        """深度学习日志文件"""
        return self.base_dir / self.config['artifacts']['logs']['deep_learning']

    @property
    def errors_log_file(self) -> Path:
        """错误日志文件"""
        return self.base_dir / self.config['artifacts']['logs']['errors']

    # ===== 模型目录 =====
    @property
    def models_dir(self) -> Path:
        """模型根目录"""
        return self.base_dir / self.config['artifacts']['models']['root']

    @property
    def models_lstm_dir(self) -> Path:
        """LSTM模型目录"""
        return self.base_dir / self.config['artifacts']['models']['lstm']

    @property
    def models_transformer_dir(self) -> Path:
        """Transformer模型目录"""
        return self.base_dir / self.config['artifacts']['models']['transformer']

    @property
    def models_gan_dir(self) -> Path:
        """GAN模型目录"""
        return self.base_dir / self.config['artifacts']['models']['gan']

    # ===== 报告目录 =====
    @property
    def reports_dir(self) -> Path:
        """报告根目录"""
        return self.base_dir / self.config['artifacts']['reports']['root']

    @property
    def reports_predictions_dir(self) -> Path:
        """预测结果目录"""
        return self.base_dir / self.config['artifacts']['reports']['predictions']

    @property
    def reports_analysis_dir(self) -> Path:
        """分析报告目录"""
        return self.base_dir / self.config['artifacts']['reports']['analysis']

    @property
    def reports_backtest_dir(self) -> Path:
        """回测结果目录"""
        return self.base_dir / self.config['artifacts']['reports']['backtest']

    @property
    def reports_visualization_dir(self) -> Path:
        """可视化输出目录"""
        return self.base_dir / self.config['artifacts']['reports']['visualization']

    # ===== 配置目录 =====
    @property
    def config_dir(self) -> Path:
        """配置目录"""
        return self.base_dir / self.config['config']['root']

    @property
    def main_config_file(self) -> Path:
        """主配置文件"""
        return self.base_dir / self.config['config']['main']

    @property
    def prediction_config_file(self) -> Path:
        """预测配置文件"""
        return self.base_dir / self.config['config']['prediction']

    @property
    def training_config_file(self) -> Path:
        """训练配置文件"""
        return self.base_dir / self.config['config']['training']

    @property
    def acceleration_config_file(self) -> Path:
        """加速配置文件"""
        return self.base_dir / self.config['config']['acceleration']

    @property
    def gui_config_file(self) -> Path:
        """GUI配置文件"""
        return self.base_dir / self.config['config']['gui']

    # ===== 兼容性方法 - 旧路径到新路径的映射 =====
    def get_legacy_path(self, legacy_path: str) -> Path:
        """
        将旧路径映射到新路径
        用于兼容旧代码中的硬编码路径
        """
        mappings = {
            'cache/': 'artifacts/cache/',
            'cache\\': 'artifacts/cache/',
            'logs/': 'artifacts/logs/',
            'logs\\': 'artifacts/logs/',
            'models/': 'artifacts/models/',
            'models\\': 'artifacts/models/',
            'output/': 'artifacts/reports/',
            'output\\': 'artifacts/reports/',
        }

        for old, new in mappings.items():
            if legacy_path.startswith(old):
                return self.base_dir / legacy_path.replace(old, new, 1)

        return self.base_dir / legacy_path


# 全局路径管理器实例
_path_manager: Optional[PathManager] = None


def get_path_manager() -> PathManager:
    """获取全局路径管理器实例"""
    global _path_manager
    if _path_manager is None:
        _path_manager = PathManager()
    return _path_manager


# ===== 便捷函数 - 直接获取常用路径 =====

def get_data_dir() -> Path:
    """获取数据目录"""
    return get_path_manager().data_dir


def get_data_file() -> Path:
    """获取历史数据文件路径"""
    return get_path_manager().data_file


def get_cache_dir() -> Path:
    """获取缓存目录"""
    return get_path_manager().cache_dir


def get_logs_dir() -> Path:
    """获取日志目录"""
    return get_path_manager().logs_dir


def get_models_dir() -> Path:
    """获取模型目录"""
    return get_path_manager().models_dir


def get_reports_dir() -> Path:
    """获取报告目录"""
    return get_path_manager().reports_dir


def get_config_dir() -> Path:
    """获取配置目录"""
    return get_path_manager().config_dir


# ===== 兼容性常量 - 供旧代码使用 =====

# 初始化路径管理器以获取路径
_pm = get_path_manager()

# 缓存路径
CACHE_DIR = str(_pm.cache_dir)
CACHE_ANALYSIS_DIR = str(_pm.cache_analysis_dir)
CACHE_DATA_DIR = str(_pm.cache_data_dir)
CACHE_MODELS_DIR = str(_pm.cache_models_dir)

# 日志路径
LOGS_DIR = str(_pm.logs_dir)
APP_LOG_FILE = str(_pm.app_log_file)
DEEP_LEARNING_LOG_FILE = str(_pm.deep_learning_log_file)
ERRORS_LOG_FILE = str(_pm.errors_log_file)

# 模型路径
MODELS_DIR = str(_pm.models_dir)
MODELS_LSTM_DIR = str(_pm.models_lstm_dir)
MODELS_TRANSFORMER_DIR = str(_pm.models_transformer_dir)
MODELS_GAN_DIR = str(_pm.models_gan_dir)

# 报告路径
REPORTS_DIR = str(_pm.reports_dir)
REPORTS_PREDICTIONS_DIR = str(_pm.reports_predictions_dir)
REPORTS_ANALYSIS_DIR = str(_pm.reports_analysis_dir)
REPORTS_BACKTEST_DIR = str(_pm.reports_backtest_dir)
REPORTS_VISUALIZATION_DIR = str(_pm.reports_visualization_dir)

# 数据路径
DATA_DIR = str(_pm.data_dir)
DATA_FILE = str(_pm.data_file)

# 配置路径
CONFIG_DIR = str(_pm.config_dir)
MAIN_CONFIG_FILE = str(_pm.main_config_file)
PREDICTION_CONFIG_FILE = str(_pm.prediction_config_file)
TRAINING_CONFIG_FILE = str(_pm.training_config_file)
