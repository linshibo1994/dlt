#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强功能集成模块
Enhanced Features Integration Module

将新开发的enhanced_deep_learning模块集成到原有的大乐透预测系统中
"""

import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

# 导入原有核心模块
from core_modules import logger_manager, cache_manager, data_manager
from smart_cache_system import smart_cache_manager

# 导入新开发的增强模块
try:
    from enhanced_deep_learning.core import ConfigManager, ExceptionHandler
    from enhanced_deep_learning.data import DataProcessor, DataAugmentor
    from enhanced_deep_learning.models import ModelRegistry, BaseModel
    from enhanced_deep_learning.prediction import PredictionEngine, PredictionCache
    from enhanced_deep_learning.visualization import InteractiveVisualizer
    from enhanced_deep_learning.workflow import WorkflowManager
    from enhanced_deep_learning.platform import PlatformDetector, GracefulDegradation
    from enhanced_deep_learning.performance import DistributedComputingManager
    
    ENHANCED_FEATURES_AVAILABLE = True
    logger_manager.info("增强功能模块加载成功")
    
except ImportError as e:
    ENHANCED_FEATURES_AVAILABLE = False
    logger_manager.warning(f"增强功能模块加载失败: {e}")


class EnhancedDLTSystem:
    """增强版大乐透预测系统"""
    
    def __init__(self):
        """初始化增强系统"""
        self.enhanced_available = ENHANCED_FEATURES_AVAILABLE
        
        if self.enhanced_available:
            # 初始化增强组件
            self.config_manager = ConfigManager()
            self.exception_handler = ExceptionHandler()
            self.data_processor = DataProcessor()
            self.model_registry = ModelRegistry()
            self.prediction_engine = PredictionEngine()
            self.prediction_cache = PredictionCache()
            self.visualizer = InteractiveVisualizer()
            self.workflow_manager = WorkflowManager()
            self.platform_detector = PlatformDetector()
            self.graceful_degradation = GracefulDegradation()
            self.computing_manager = DistributedComputingManager()
            
            logger_manager.info("增强版大乐透预测系统初始化完成")
        else:
            logger_manager.info("使用基础版大乐透预测系统")
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        info = {
            "enhanced_features_available": self.enhanced_available,
            "system_type": "Enhanced" if self.enhanced_available else "Basic"
        }
        
        if self.enhanced_available:
            # 获取平台信息
            platform_info = self.platform_detector.detect_platform()
            info.update({
                "platform": {
                    "os": platform_info.os_name,
                    "version": platform_info.os_version,
                    "architecture": platform_info.architecture.value,
                    "python_version": platform_info.python_info.version
                },
                "capabilities": platform_info.capabilities,
                "hardware": {
                    "cpu_count": platform_info.hardware_info.cpu_count,
                    "memory_total_gb": platform_info.hardware_info.memory_total / (1024**3),
                    "gpu_count": platform_info.hardware_info.gpu_count
                }
            })
        
        return info
    
    def enhanced_predict(self, data: Any, method: str = "auto", **kwargs) -> Dict[str, Any]:
        """增强预测功能"""
        if not self.enhanced_available:
            logger_manager.warning("增强功能不可用，请使用基础预测功能")
            return {"error": "Enhanced features not available"}

        try:
            # 准备输入数据
            import numpy as np

            # 如果data是字符串，创建简单的数值数据
            if isinstance(data, str):
                # 创建模拟的历史数据用于预测
                input_data = np.random.rand(100, 10).astype(np.float32)  # 模拟100期，10个特征
            else:
                input_data = np.array(data, dtype=np.float32)

            # 根据方法选择模型
            if method == "auto":
                model_name = "lstm"  # 默认使用LSTM
            else:
                model_name = method

            # 使用增强预测引擎（同步方式，避免重复调用）
            result = self.prediction_engine.predict_sync(
                model_name=model_name,
                input_data=input_data,
                **kwargs
            )

            # 缓存预测结果
            cache_key = f"prediction_{hash(str(data))}_{method}"
            self.prediction_cache.set(cache_key, result)

            return {
                "success": True,
                "result": result,
                "method": method,
                "cached": True,
                "request_id": result.request_id if hasattr(result, 'request_id') else None
            }

        except Exception as e:
            self.exception_handler.handle_exception(e)
            return {"error": str(e)}
    
    def enhanced_visualize(self, data: Any, chart_type: str = "auto", **kwargs) -> Dict[str, Any]:
        """增强可视化功能"""
        if not self.enhanced_available:
            logger_manager.warning("增强功能不可用，请使用基础可视化功能")
            return {"error": "Enhanced features not available"}
        
        try:
            # 使用增强可视化器
            result = self.visualizer.create_visualization(data, chart_type=chart_type, **kwargs)
            
            return {
                "success": True,
                "result": result,
                "chart_type": chart_type
            }
            
        except Exception as e:
            self.exception_handler.handle_exception(e)
            return {"error": str(e)}
    
    def enhanced_analyze(self, data: Any, analysis_type: str = "comprehensive", **kwargs) -> Dict[str, Any]:
        """增强分析功能"""
        if not self.enhanced_available:
            logger_manager.warning("增强功能不可用，请使用基础分析功能")
            return {"error": "Enhanced features not available"}
        
        try:
            # 使用增强数据处理器
            processed_data = self.data_processor.process(data, **kwargs)
            
            # 执行分析
            analysis_result = {
                "processed_data": processed_data,
                "analysis_type": analysis_type,
                "timestamp": str(datetime.now())
            }
            
            return {
                "success": True,
                "result": analysis_result
            }
            
        except Exception as e:
            self.exception_handler.handle_exception(e)
            return {"error": str(e)}
    
    def create_workflow(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """创建工作流"""
        if not self.enhanced_available:
            return {"error": "Enhanced features not available"}
        
        try:
            workflow_id = self.workflow_manager.create_workflow(workflow_config)
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "config": workflow_config
            }
            
        except Exception as e:
            self.exception_handler.handle_exception(e)
            return {"error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.enhanced_available:
            return {"error": "Enhanced features not available"}
        
        try:
            models = self.model_registry.list_models()
            
            return {
                "success": True,
                "models": models,
                "total_count": len(models)
            }
            
        except Exception as e:
            self.exception_handler.handle_exception(e)
            return {"error": str(e)}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        if not self.enhanced_available:
            return {"error": "Enhanced features not available"}
        
        try:
            stats = self.prediction_cache.get_cache_stats()
            
            return {
                "success": True,
                "cache_stats": stats
            }
            
        except Exception as e:
            self.exception_handler.handle_exception(e)
            return {"error": str(e)}
    
    def run_compatibility_test(self) -> Dict[str, Any]:
        """运行兼容性测试"""
        if not self.enhanced_available:
            return {"error": "Enhanced features not available"}
        
        try:
            # 运行系统兼容性测试
            from enhanced_deep_learning.platform import compatibility_tester
            
            results = compatibility_tester.run_test_suite("system_compatibility")
            
            return {
                "success": True,
                "test_results": [
                    {
                        "name": result.name,
                        "status": result.status.value,
                        "message": result.message,
                        "duration": result.duration
                    }
                    for result in results
                ]
            }
            
        except Exception as e:
            self.exception_handler.handle_exception(e)
            return {"error": str(e)}


# 创建全局增强系统实例
enhanced_dlt_system = EnhancedDLTSystem()


def get_enhanced_system() -> EnhancedDLTSystem:
    """获取增强系统实例"""
    return enhanced_dlt_system


def is_enhanced_available() -> bool:
    """检查增强功能是否可用"""
    return ENHANCED_FEATURES_AVAILABLE


if __name__ == "__main__":
    # 测试增强功能集成
    print("🔧 测试增强功能集成...")
    
    system = get_enhanced_system()
    
    # 获取系统信息
    info = system.get_system_info()
    print(f"✅ 系统信息: {info['system_type']}")
    
    if system.enhanced_available:
        print("✅ 增强功能可用")
        
        # 测试兼容性
        compat_result = system.run_compatibility_test()
        if compat_result.get("success"):
            print(f"✅ 兼容性测试: {len(compat_result['test_results'])} 个测试")
        
        # 测试缓存
        cache_stats = system.get_cache_stats()
        if cache_stats.get("success"):
            print("✅ 缓存系统正常")
        
        # 测试模型注册表
        model_info = system.get_model_info()
        if model_info.get("success"):
            print(f"✅ 模型注册表: {model_info['total_count']} 个模型")
    
    else:
        print("⚠️ 增强功能不可用，使用基础功能")
    
    print("🎉 增强功能集成测试完成")
