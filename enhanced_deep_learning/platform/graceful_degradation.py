#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
优雅降级系统模块
Graceful Degradation System Module

提供功能降级、回退策略、兼容性处理等功能。
"""

import os
import sys
import warnings
from typing import Dict, List, Any, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
import functools
import importlib

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException
from .platform_detector import platform_detector


class FeatureLevel(Enum):
    """功能级别枚举"""
    ESSENTIAL = "essential"      # 必需功能
    STANDARD = "standard"        # 标准功能
    ENHANCED = "enhanced"        # 增强功能
    EXPERIMENTAL = "experimental" # 实验功能


class FallbackStrategy(Enum):
    """回退策略枚举"""
    DISABLE = "disable"          # 禁用功能
    FALLBACK = "fallback"        # 使用备用实现
    WARN_AND_CONTINUE = "warn_and_continue"  # 警告并继续
    FAIL_GRACEFULLY = "fail_gracefully"     # 优雅失败


@dataclass
class FeatureAvailability:
    """功能可用性"""
    name: str
    level: FeatureLevel
    available: bool
    reason: str = ""
    fallback_available: bool = False
    fallback_implementation: Optional[Callable] = None
    requirements: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)


@dataclass
class DegradationRule:
    """降级规则"""
    feature_name: str
    condition: Callable[[], bool]
    strategy: FallbackStrategy
    fallback_implementation: Optional[Callable] = None
    warning_message: str = ""
    error_message: str = ""


class FeatureRegistry:
    """功能注册表"""
    
    def __init__(self):
        """初始化功能注册表"""
        self.features = {}
        self.degradation_rules = {}
        
        logger_manager.debug("功能注册表初始化完成")
    
    def register_feature(self, feature: FeatureAvailability):
        """
        注册功能
        
        Args:
            feature: 功能可用性信息
        """
        try:
            self.features[feature.name] = feature
            logger_manager.debug(f"功能注册成功: {feature.name}")
            
        except Exception as e:
            logger_manager.error(f"注册功能失败: {e}")
    
    def register_degradation_rule(self, rule: DegradationRule):
        """
        注册降级规则
        
        Args:
            rule: 降级规则
        """
        try:
            self.degradation_rules[rule.feature_name] = rule
            logger_manager.debug(f"降级规则注册成功: {rule.feature_name}")
            
        except Exception as e:
            logger_manager.error(f"注册降级规则失败: {e}")
    
    def get_feature(self, name: str) -> Optional[FeatureAvailability]:
        """获取功能信息"""
        return self.features.get(name)
    
    def get_degradation_rule(self, name: str) -> Optional[DegradationRule]:
        """获取降级规则"""
        return self.degradation_rules.get(name)
    
    def list_features(self, level: FeatureLevel = None) -> List[FeatureAvailability]:
        """
        列出功能
        
        Args:
            level: 功能级别过滤
            
        Returns:
            功能列表
        """
        features = list(self.features.values())
        
        if level:
            features = [f for f in features if f.level == level]
        
        return features


class GracefulDegradation:
    """优雅降级系统"""
    
    def __init__(self):
        """初始化优雅降级系统"""
        self.feature_registry = FeatureRegistry()
        self.platform_info = platform_detector.detect_platform()
        
        # 初始化核心功能检测
        self._initialize_core_features()
        
        logger_manager.info("优雅降级系统初始化完成")
    
    def _initialize_core_features(self):
        """初始化核心功能检测"""
        try:
            # GPU加速功能
            gpu_feature = FeatureAvailability(
                name="gpu_acceleration",
                level=FeatureLevel.ENHANCED,
                available=self._check_gpu_availability(),
                reason="GPU不可用或CUDA未安装" if not self._check_gpu_availability() else "",
                fallback_available=True,
                requirements=["CUDA", "cuDNN", "GPU驱动"],
                alternatives=["CPU计算", "OpenMP并行"]
            )
            self.feature_registry.register_feature(gpu_feature)
            
            # 分布式训练功能
            distributed_feature = FeatureAvailability(
                name="distributed_training",
                level=FeatureLevel.ENHANCED,
                available=self._check_distributed_availability(),
                reason="分布式库不可用" if not self._check_distributed_availability() else "",
                fallback_available=True,
                requirements=["torch.distributed", "多GPU或多节点"],
                alternatives=["单机训练", "数据并行"]
            )
            self.feature_registry.register_feature(distributed_feature)
            
            # 混合精度训练功能
            mixed_precision_feature = FeatureAvailability(
                name="mixed_precision",
                level=FeatureLevel.STANDARD,
                available=self._check_mixed_precision_availability(),
                reason="不支持混合精度" if not self._check_mixed_precision_availability() else "",
                fallback_available=True,
                requirements=["Tensor Core GPU", "PyTorch 1.6+"],
                alternatives=["FP32训练"]
            )
            self.feature_registry.register_feature(mixed_precision_feature)
            
            # 可视化功能
            visualization_feature = FeatureAvailability(
                name="interactive_visualization",
                level=FeatureLevel.STANDARD,
                available=self._check_visualization_availability(),
                reason="可视化库不可用" if not self._check_visualization_availability() else "",
                fallback_available=True,
                requirements=["plotly", "dash", "matplotlib"],
                alternatives=["基础图表", "文本输出"]
            )
            self.feature_registry.register_feature(visualization_feature)
            
            # 注册降级规则
            self._register_degradation_rules()
            
        except Exception as e:
            logger_manager.error(f"初始化核心功能检测失败: {e}")
    
    def _check_gpu_availability(self) -> bool:
        """检查GPU可用性"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
        except Exception:
            return False
    
    def _check_distributed_availability(self) -> bool:
        """检查分布式训练可用性"""
        try:
            import torch.distributed
            return True
        except ImportError:
            return False
        except Exception:
            return False
    
    def _check_mixed_precision_availability(self) -> bool:
        """检查混合精度可用性"""
        try:
            import torch
            from torch.cuda.amp import autocast, GradScaler
            return torch.cuda.is_available()
        except ImportError:
            return False
        except Exception:
            return False
    
    def _check_visualization_availability(self) -> bool:
        """检查可视化可用性"""
        try:
            import plotly
            import dash
            return True
        except ImportError:
            try:
                import matplotlib
                return True
            except ImportError:
                return False
        except Exception:
            return False
    
    def _register_degradation_rules(self):
        """注册降级规则"""
        try:
            # GPU加速降级规则
            gpu_rule = DegradationRule(
                feature_name="gpu_acceleration",
                condition=lambda: not self._check_gpu_availability(),
                strategy=FallbackStrategy.FALLBACK,
                fallback_implementation=self._cpu_fallback,
                warning_message="GPU不可用，使用CPU计算"
            )
            self.feature_registry.register_degradation_rule(gpu_rule)
            
            # 分布式训练降级规则
            distributed_rule = DegradationRule(
                feature_name="distributed_training",
                condition=lambda: not self._check_distributed_availability(),
                strategy=FallbackStrategy.FALLBACK,
                fallback_implementation=self._single_node_fallback,
                warning_message="分布式训练不可用，使用单机训练"
            )
            self.feature_registry.register_degradation_rule(distributed_rule)
            
            # 混合精度降级规则
            mixed_precision_rule = DegradationRule(
                feature_name="mixed_precision",
                condition=lambda: not self._check_mixed_precision_availability(),
                strategy=FallbackStrategy.FALLBACK,
                fallback_implementation=self._fp32_fallback,
                warning_message="混合精度不可用，使用FP32训练"
            )
            self.feature_registry.register_degradation_rule(mixed_precision_rule)
            
            # 可视化降级规则
            visualization_rule = DegradationRule(
                feature_name="interactive_visualization",
                condition=lambda: not self._check_visualization_availability(),
                strategy=FallbackStrategy.FALLBACK,
                fallback_implementation=self._text_output_fallback,
                warning_message="交互式可视化不可用，使用文本输出"
            )
            self.feature_registry.register_degradation_rule(visualization_rule)
            
        except Exception as e:
            logger_manager.error(f"注册降级规则失败: {e}")
    
    def _cpu_fallback(self, *args, **kwargs):
        """CPU回退实现"""
        logger_manager.info("使用CPU计算回退")
        # 这里可以实现CPU特定的优化
        return "cpu_device"
    
    def _single_node_fallback(self, *args, **kwargs):
        """单节点回退实现"""
        logger_manager.info("使用单节点训练回退")
        return "single_node_training"
    
    def _fp32_fallback(self, *args, **kwargs):
        """FP32回退实现"""
        logger_manager.info("使用FP32精度回退")
        return "fp32_precision"
    
    def _text_output_fallback(self, *args, **kwargs):
        """文本输出回退实现"""
        logger_manager.info("使用文本输出回退")
        return "text_output"
    
    def check_feature_availability(self, feature_name: str) -> bool:
        """
        检查功能可用性
        
        Args:
            feature_name: 功能名称
            
        Returns:
            是否可用
        """
        try:
            feature = self.feature_registry.get_feature(feature_name)
            if not feature:
                logger_manager.warning(f"未知功能: {feature_name}")
                return False
            
            # 检查降级条件
            rule = self.feature_registry.get_degradation_rule(feature_name)
            if rule and rule.condition():
                return False
            
            return feature.available
            
        except Exception as e:
            logger_manager.error(f"检查功能可用性失败: {e}")
            return False
    
    def get_feature_implementation(self, feature_name: str) -> Optional[Callable]:
        """
        获取功能实现
        
        Args:
            feature_name: 功能名称
            
        Returns:
            功能实现或回退实现
        """
        try:
            feature = self.feature_registry.get_feature(feature_name)
            if not feature:
                return None
            
            # 检查是否需要降级
            rule = self.feature_registry.get_degradation_rule(feature_name)
            if rule and rule.condition():
                # 应用降级策略
                return self._apply_degradation_strategy(rule)
            
            return None  # 返回None表示使用默认实现
            
        except Exception as e:
            logger_manager.error(f"获取功能实现失败: {e}")
            return None
    
    def _apply_degradation_strategy(self, rule: DegradationRule) -> Optional[Callable]:
        """应用降级策略"""
        try:
            if rule.strategy == FallbackStrategy.DISABLE:
                logger_manager.warning(f"功能已禁用: {rule.feature_name}")
                return None
            
            elif rule.strategy == FallbackStrategy.FALLBACK:
                if rule.warning_message:
                    logger_manager.warning(rule.warning_message)
                return rule.fallback_implementation
            
            elif rule.strategy == FallbackStrategy.WARN_AND_CONTINUE:
                if rule.warning_message:
                    warnings.warn(rule.warning_message, UserWarning)
                return None
            
            elif rule.strategy == FallbackStrategy.FAIL_GRACEFULLY:
                if rule.error_message:
                    logger_manager.error(rule.error_message)
                raise DeepLearningException(rule.error_message or f"功能不可用: {rule.feature_name}")
            
            return None
            
        except Exception as e:
            logger_manager.error(f"应用降级策略失败: {e}")
            return None
    
    def with_graceful_degradation(self, feature_name: str):
        """
        装饰器：为函数添加优雅降级支持
        
        Args:
            feature_name: 功能名称
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    # 检查功能可用性
                    if not self.check_feature_availability(feature_name):
                        # 获取回退实现
                        fallback_impl = self.get_feature_implementation(feature_name)
                        if fallback_impl:
                            return fallback_impl(*args, **kwargs)
                        else:
                            logger_manager.warning(f"功能不可用且无回退实现: {feature_name}")
                            return None
                    
                    # 执行原始函数
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    logger_manager.error(f"功能执行失败: {feature_name}, 错误: {e}")
                    
                    # 尝试回退实现
                    fallback_impl = self.get_feature_implementation(feature_name)
                    if fallback_impl:
                        logger_manager.info(f"使用回退实现: {feature_name}")
                        return fallback_impl(*args, **kwargs)
                    
                    raise
            
            return wrapper
        return decorator
    
    def get_system_capabilities(self) -> Dict[str, Any]:
        """获取系统能力报告"""
        try:
            capabilities = {
                'platform': {
                    'os': self.platform_info.os_name,
                    'version': self.platform_info.os_version,
                    'architecture': self.platform_info.architecture.value
                },
                'features': {},
                'degradations': []
            }
            
            # 检查所有功能
            for feature_name, feature in self.feature_registry.features.items():
                capabilities['features'][feature_name] = {
                    'available': feature.available,
                    'level': feature.level.value,
                    'reason': feature.reason,
                    'fallback_available': feature.fallback_available,
                    'requirements': feature.requirements,
                    'alternatives': feature.alternatives
                }
                
                # 检查是否需要降级
                rule = self.feature_registry.get_degradation_rule(feature_name)
                if rule and rule.condition():
                    capabilities['degradations'].append({
                        'feature': feature_name,
                        'strategy': rule.strategy.value,
                        'reason': rule.warning_message or rule.error_message
                    })
            
            return capabilities
            
        except Exception as e:
            logger_manager.error(f"获取系统能力报告失败: {e}")
            return {}


# 全局优雅降级系统实例
graceful_degradation = GracefulDegradation()


if __name__ == "__main__":
    # 测试优雅降级系统功能
    print("🛡️ 测试优雅降级系统功能...")
    
    try:
        degradation = GracefulDegradation()
        
        # 测试功能可用性检查
        features_to_test = [
            "gpu_acceleration",
            "distributed_training",
            "mixed_precision",
            "interactive_visualization"
        ]
        
        for feature in features_to_test:
            available = degradation.check_feature_availability(feature)
            status = "✅" if available else "❌"
            print(f"{status} {feature}: {'可用' if available else '不可用'}")
        
        # 测试装饰器
        @degradation.with_graceful_degradation("gpu_acceleration")
        def test_gpu_function():
            return "GPU计算完成"
        
        result = test_gpu_function()
        print(f"✅ 装饰器测试结果: {result}")
        
        # 获取系统能力报告
        capabilities = degradation.get_system_capabilities()
        print(f"✅ 系统能力报告生成成功，包含 {len(capabilities.get('features', {}))} 个功能")
        
        if capabilities.get('degradations'):
            print(f"⚠️ 检测到 {len(capabilities['degradations'])} 个功能降级")
        
        print("✅ 优雅降级系统功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("优雅降级系统功能测试完成")
