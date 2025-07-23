#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
深度学习模块异常类
定义各种异常类型和错误处理机制
"""


class DeepLearningException(Exception):
    """深度学习模块基础异常类"""

    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


class ModelException(DeepLearningException):
    """模型相关异常"""

    def __init__(self, message: str, model_name: str = None, error_code: str = None):
        self.model_name = model_name
        super().__init__(message, error_code)


class DataException(DeepLearningException):
    """数据相关异常"""

    def __init__(self, message: str, data_type: str = None, error_code: str = None):
        self.data_type = data_type
        super().__init__(message, error_code)


class PredictionException(DeepLearningException):
    """预测相关异常"""

    def __init__(self, message: str, prediction_id: str = None, error_code: str = None):
        self.prediction_id = prediction_id
        super().__init__(message, error_code)

class ModelInitializationError(Exception):
    """模型初始化错误"""
    
    def __init__(self, model_name, message):
        self.model_name = model_name
        self.message = message
        super().__init__(f"模型 {model_name} 初始化失败: {message}")


class TrainingDataError(Exception):
    """训练数据错误"""
    
    def __init__(self, message):
        self.message = message
        super().__init__(f"训练数据错误: {message}")


class PredictionError(Exception):
    """预测过程错误"""
    
    def __init__(self, model_name, message):
        self.model_name = model_name
        self.message = message
        super().__init__(f"模型 {model_name} 预测失败: {message}")


class ResourceAllocationError(Exception):
    """资源分配错误"""
    
    def __init__(self, resource_type, message):
        self.resource_type = resource_type
        self.message = message
        super().__init__(f"{resource_type} 资源分配失败: {message}")


class ModelCompatibilityError(Exception):
    """模型兼容性错误"""
    
    def __init__(self, model_name, message):
        self.model_name = model_name
        self.message = message
        super().__init__(f"模型 {model_name} 兼容性错误: {message}")


class ModelNotFoundError(Exception):
    """模型文件不存在错误"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        super().__init__(f"模型文件不存在: {model_path}")


class InvalidConfigurationError(Exception):
    """配置参数无效错误"""
    
    def __init__(self, config_name, message):
        self.config_name = config_name
        self.message = message
        super().__init__(f"配置 {config_name} 无效: {message}")


def handle_model_error(func):
    """模型错误处理装饰器"""
    from functools import wraps
    from core_modules import logger_manager
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ModelInitializationError as e:
            logger_manager.error(f"模型初始化错误: {e}")
            # 尝试使用备用配置
            if hasattr(args[0], 'use_fallback_config'):
                logger_manager.info("尝试使用备用配置...")
                args[0].use_fallback_config()
                return func(*args, **kwargs)
            return None
        except TrainingDataError as e:
            logger_manager.error(f"训练数据错误: {e}")
            # 尝试使用缓存的上一版本数据
            if hasattr(args[0], 'use_cached_data'):
                logger_manager.info("尝试使用缓存数据...")
                args[0].use_cached_data()
                return func(*args, **kwargs)
            return None
        except PredictionError as e:
            logger_manager.error(f"预测错误: {e}")
            # 回退到更简单的模型
            if hasattr(args[0], 'use_simple_model'):
                logger_manager.info("回退到简单模型...")
                args[0].use_simple_model()
                return func(*args, **kwargs)
            return None
        except ResourceAllocationError as e:
            logger_manager.error(f"资源分配错误: {e}")
            # 降低资源需求
            if hasattr(args[0], 'reduce_resource_usage'):
                logger_manager.info("降低资源需求...")
                args[0].reduce_resource_usage()
                return func(*args, **kwargs)
            return None
        except ModelCompatibilityError as e:
            logger_manager.error(f"模型兼容性错误: {e}")
            # 尝试转换数据格式
            if hasattr(args[0], 'convert_data_format'):
                logger_manager.info("尝试转换数据格式...")
                args[0].convert_data_format()
                return func(*args, **kwargs)
            return None
        except Exception as e:
            logger_manager.error(f"未知错误: {e}")
            return None
    
    return wrapper