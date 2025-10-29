#!/usr/bin/env python3
"""
配置管理模块
负责管理测试配置和参数
"""

import json
import os
from copy import deepcopy
from typing import Dict, List, Any, Optional


PREDICTION_METHOD_CATEGORIES = {
    "basic": ["frequency", "hot_cold", "missing"],
    "markov": [
        "markov",
        "markov_2nd",
        "markov_3rd",
        "adaptive_markov",
        "markov_custom",
        "markov_compound"
    ],
    "probabilistic": ["bayesian"],
    "ensemble": ["ensemble", "stacking", "adaptive_ensemble", "ultimate_ensemble"],
    "intelligent": [
        "super",
        "adaptive",
        "mixed_strategy",
        "highly_integrated",
        "advanced_integration",
        "enhanced"
    ],
    "compound": ["compound", "duplex", "nine_models", "nine_models_compound", "markov_compound"],
    "deep_learning": ["lstm", "transformer", "gan"],
}

ALL_SUPPORTED_METHODS = sorted({method for methods in PREDICTION_METHOD_CATEGORIES.values() for method in methods})


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = None):
        if config_dir is None:
            self.config_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'config'
            )
        else:
            self.config_dir = config_dir
        
        # 确保配置目录存在
        os.makedirs(self.config_dir, exist_ok=True)
        
        self.default_config = self._get_default_config()
        self.config = self.load_config()
        self._ensure_supported_methods()

    def _get_default_config(self) -> Dict:
        """获取包含所有预测方法的默认配置"""
        return {
            "test_settings": {
                "timeout_seconds": 0,
                "max_retries": 2,
                "parallel_workers": 4,
                "stop_on_major_prize": True,
                "major_prize_levels": [1, 2]
            },
            "prediction_methods": deepcopy(PREDICTION_METHOD_CATEGORIES),
            "parameter_ranges": {
                "periods": {
                    "min": 10,
                    "max": 2000,
                    "basic_test": [50, 100, 200, 500, 1000],
                    "comprehensive_test": [50, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000],
                    "progressive_step": 50
                },
                "count": {
                    "min": 1,
                    "max": 10,
                    "basic_test": [1, 2, 3, 5],
                    "optimization_test": [1, 2, 3, 4, 5, 8, 10]
                }
            },
            "test_strategies": {
                "quick": {
                    "description": "快速测试模式",
                    "methods": ["frequency", "markov", "markov_2nd", "bayesian", "ensemble", "lstm"],
                    "periods_list": [100, 500],
                    "count_list": [1, 2],
                    "max_tests_per_method": 4
                },
                "comprehensive": {
                    "description": "全面测试模式 - 测试所有预测方法",
                    "methods": "all",
                    "periods_range": [50, 1000],
                    "count_range": [1, 5],
                    "progressive_testing": True,
                    "priority_methods": [
                        "frequency", "markov", "markov_2nd", "markov_3rd",
                        "adaptive_markov", "markov_custom", "bayesian", "ensemble",
                        "stacking", "adaptive_ensemble", "ultimate_ensemble",
                        "lstm", "transformer", "gan", "compound", "markov_compound",
                        "nine_models", "nine_models_compound", "super", "adaptive", "enhanced"
                    ]
                },
                "optimization": {
                    "description": "优化测试模式",
                    "methods": [
                        "markov", "markov_2nd", "markov_3rd", "adaptive_markov", "markov_custom",
                        "bayesian", "ensemble", "stacking", "adaptive_ensemble", "ultimate_ensemble",
                        "lstm", "transformer", "gan", "super", "adaptive"
                    ],
                    "periods_range": [10, 2000],
                    "count_range": [1, 10],
                    "random_testing": True,
                    "max_tests": 1000
                }
            },
            "output_settings": {
                "log_level": "INFO",
                "save_all_results": True,
                "result_formats": ["json", "csv", "html"],
                "real_time_display": True,
                "progress_update_interval": 5,
                "detailed_winner_report": True
            },
            "performance_settings": {
                "memory_limit_mb": 1024,
                "disk_space_limit_mb": 2048,
                "max_execution_time_hours": 24
            }
        }

    def load_config(self, config_file: str = "config.json") -> Dict:
        """加载配置文件"""
        config_path = os.path.join(self.config_dir, config_file)
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                
                # 合并用户配置和默认配置
                config = self._merge_config(self.default_config, user_config)
                print(f"已加载配置文件: {config_path}")
                return config
                
            except Exception as e:
                print(f"加载配置文件失败: {e}，使用默认配置")
                return self.default_config.copy()
        else:
            print("配置文件不存在，使用默认配置")
            return self.default_config.copy()

    def save_config(self, config_file: str = "config.json") -> bool:
        """保存配置文件"""
        config_path = os.path.join(self.config_dir, config_file)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"配置已保存: {config_path}")
            return True
        except Exception as e:
            print(f"保存配置文件失败: {e}")
            return False

    def _merge_config(self, default: Dict, user: Dict) -> Dict:
        """合并配置"""
        merged = default.copy()
        
        for key, value in user.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_config(merged[key], value)
            else:
                merged[key] = value
        
        return merged

    def _ensure_supported_methods(self) -> None:
        """确保配置中的预测方法与系统支持的方法一致"""
        methods_section = self.config.setdefault('prediction_methods', {})

        for category, default_methods in PREDICTION_METHOD_CATEGORIES.items():
            existing = methods_section.get(category, [])
            combined = {method for method in existing if method in ALL_SUPPORTED_METHODS}
            combined.update(default_methods)
            methods_section[category] = sorted(combined)

        # 移除配置中可能存在的无效方法
        for category, method_list in list(methods_section.items()):
            filtered = [method for method in method_list if method in ALL_SUPPORTED_METHODS]
            methods_section[category] = sorted(set(filtered))

    def get(self, key_path: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any) -> None:
        """设置配置值"""
        keys = key_path.split('.')
        target = self.config
        
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        target[keys[-1]] = value

    def get_prediction_methods(self, category: str = None) -> List[str]:
        """获取预测方法列表"""
        methods = self.config.get('prediction_methods', {})

        if category in (None, 'all'):
            aggregated = {method for method_list in methods.values() for method in method_list}
            return sorted(method for method in aggregated if method in ALL_SUPPORTED_METHODS)

        return methods.get(category, [])

    def get_test_strategy(self, strategy_name: str) -> Optional[Dict]:
        """获取测试策略配置"""
        strategies = self.get('test_strategies', {})
        return strategies.get(strategy_name)

    def get_parameter_range(self, param_name: str) -> Optional[Dict]:
        """获取参数范围配置"""
        ranges = self.get('parameter_ranges', {})
        return ranges.get(param_name)

    def create_default_configs(self) -> None:
        """创建默认配置文件"""
        # 主配置文件
        self.save_config('config.json')
        
        # 测试策略配置
        strategies_config = {
            "strategies": self.get('test_strategies', {})
        }
        with open(os.path.join(self.config_dir, 'test_strategies.json'), 'w', encoding='utf-8') as f:
            json.dump(strategies_config, f, indent=2, ensure_ascii=False)
        
        # 方法配置
        methods_config = {
            "methods": self.get('prediction_methods', {}),
            "parameter_ranges": self.get('parameter_ranges', {})
        }
        with open(os.path.join(self.config_dir, 'methods_config.json'), 'w', encoding='utf-8') as f:
            json.dump(methods_config, f, indent=2, ensure_ascii=False)
        
        print("默认配置文件已创建")

    def validate_config(self) -> List[str]:
        """验证配置的有效性"""
        errors = []
        
        # 检查必需的配置项
        required_paths = [
            'test_settings.timeout_seconds',
            'prediction_methods',
            'parameter_ranges.periods',
            'parameter_ranges.count'
        ]
        
        for path in required_paths:
            if self.get(path) is None:
                errors.append(f"缺少必需的配置项: {path}")
        
        # 检查数值范围
        timeout = self.get('test_settings.timeout_seconds', 0)
        if timeout is not None:
            if not isinstance(timeout, int):
                errors.append("timeout_seconds 必须是整数或留空")
            elif timeout < 0:
                errors.append("timeout_seconds 不能为负数")
        
        # 检查预测方法
        methods = self.get_prediction_methods()
        if not methods:
            errors.append("至少需要配置一个预测方法")

        configured_methods = {
            method
            for method_list in self.config.get('prediction_methods', {}).values()
            for method in method_list
        }
        unsupported_methods = sorted(set(configured_methods) - set(ALL_SUPPORTED_METHODS))
        if unsupported_methods:
            errors.append(
                "检测到不受支持的预测方法: " + ", ".join(unsupported_methods)
            )
        
        # 检查参数范围
        periods_range = self.get_parameter_range('periods')
        if periods_range:
            min_periods = periods_range.get('min', 0)
            max_periods = periods_range.get('max', 0)
            if min_periods >= max_periods:
                errors.append("periods 的最小值必须小于最大值")
        
        return errors

    def get_runtime_config(self, strategy: str = None) -> Dict:
        """获取运行时配置"""
        runtime_config = {
            'timeout': None,
            'max_retries': self.get('test_settings.max_retries', 2),
            'parallel_workers': self.get('test_settings.parallel_workers', 4),
            'stop_on_major_prize': self.get('test_settings.stop_on_major_prize', True),
            'major_prize_levels': self.get('test_settings.major_prize_levels', [1, 2])
        }

        timeout_config = self.get('test_settings.timeout_seconds', 0)
        if isinstance(timeout_config, int) and timeout_config > 0:
            runtime_config['timeout'] = timeout_config
        
        if strategy:
            strategy_config = self.get_test_strategy(strategy)
            if strategy_config:
                runtime_config['strategy'] = strategy_config
        
        return runtime_config


def test_config_manager():
    """测试函数"""
    config_dir = "/tmp/test_predictor_config"
    config_manager = ConfigManager(config_dir)
    
    print("=== 配置管理测试 ===")
    
    # 测试默认配置
    print("\n1. 默认配置测试:")
    print(f"超时时间: {config_manager.get('test_settings.timeout_seconds')}")
    print(f"重试次数: {config_manager.get('test_settings.max_retries')}")
    
    # 测试获取预测方法
    print("\n2. 预测方法:")
    basic_methods = config_manager.get_prediction_methods('basic')
    print(f"基础方法: {basic_methods}")
    
    # 测试配置验证
    print("\n3. 配置验证:")
    errors = config_manager.validate_config()
    if errors:
        print(f"配置错误: {errors}")
    else:
        print("配置验证通过")
    
    # 测试创建配置文件
    print("\n4. 创建配置文件:")
    config_manager.create_default_configs()
    
    # 清理测试目录
    import shutil
    shutil.rmtree(config_dir, ignore_errors=True)


if __name__ == "__main__":
    test_config_manager()