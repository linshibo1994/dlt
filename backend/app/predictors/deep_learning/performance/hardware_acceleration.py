#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
硬件加速模块
Hardware Acceleration Module

提供GPU加速、硬件检测、加速器配置等功能。
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException


class AcceleratorType(Enum):
    """加速器类型枚举"""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"


@dataclass
class AcceleratorConfig:
    """加速器配置"""
    accelerator_type: AcceleratorType
    device_id: int = 0
    memory_limit: Optional[int] = None
    mixed_precision: bool = False


class HardwareAccelerator:
    """硬件加速器"""
    
    def __init__(self):
        """初始化硬件加速器"""
        self.available_devices = self._detect_devices()
        
        logger_manager.info("硬件加速器初始化完成")
    
    def _detect_devices(self) -> Dict[str, Any]:
        """检测可用设备"""
        devices = {"cpu": True}
        
        try:
            import torch
            if torch.cuda.is_available():
                devices["gpu"] = {
                    "count": torch.cuda.device_count(),
                    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                }
        except ImportError:
            pass
        
        return devices
    
    def get_optimal_device(self) -> str:
        """获取最优设备"""
        if "gpu" in self.available_devices:
            return "cuda"
        return "cpu"
    
    def configure_device(self, config: AcceleratorConfig) -> bool:
        """配置设备"""
        try:
            if config.accelerator_type == AcceleratorType.GPU:
                if "gpu" not in self.available_devices:
                    logger_manager.warning("GPU不可用，回退到CPU")
                    return False
            
            logger_manager.info(f"设备配置成功: {config.accelerator_type.value}")
            return True
            
        except Exception as e:
            logger_manager.error(f"设备配置失败: {e}")
            return False


# 全局硬件加速器实例
hardware_accelerator = HardwareAccelerator()


if __name__ == "__main__":
    print("⚡ 测试硬件加速器功能...")
    
    try:
        accelerator = HardwareAccelerator()
        
        # 检测设备
        devices = accelerator.available_devices
        print(f"✅ 可用设备: {list(devices.keys())}")
        
        # 获取最优设备
        optimal_device = accelerator.get_optimal_device()
        print(f"✅ 最优设备: {optimal_device}")
        
        print("✅ 硬件加速器功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    print("硬件加速器功能测试完成")
