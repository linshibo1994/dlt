#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
异常处理器模块
Exception Handler Module

提供统一的异常处理、错误记录、错误恢复等功能。
"""

import sys
import traceback
from typing import Dict, Any, Optional, Callable, Type, List
from dataclasses import dataclass, field
from enum import Enum
import functools

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException


class ErrorLevel(Enum):
    """错误级别枚举"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """错误上下文"""
    error_type: str
    error_message: str
    error_level: ErrorLevel
    module_name: str = ""
    function_name: str = ""
    line_number: int = 0
    stack_trace: str = ""
    additional_info: Dict[str, Any] = field(default_factory=dict)


class ExceptionHandler:
    """异常处理器"""
    
    def __init__(self):
        """初始化异常处理器"""
        self.error_handlers = {}
        self.error_history = []
        self.max_history_size = 1000
        
        # 注册默认错误处理器
        self._register_default_handlers()
        
        logger_manager.info("异常处理器初始化完成")
    
    def _register_default_handlers(self):
        """注册默认错误处理器"""
        try:
            # 注册常见异常的处理器
            self.register_handler(ValueError, self._handle_value_error)
            self.register_handler(TypeError, self._handle_type_error)
            self.register_handler(FileNotFoundError, self._handle_file_not_found_error)
            self.register_handler(ImportError, self._handle_import_error)
            self.register_handler(DeepLearningException, self._handle_deep_learning_error)
            
        except Exception as e:
            logger_manager.error(f"注册默认错误处理器失败: {e}")
    
    def register_handler(self, exception_type: Type[Exception], handler: Callable):
        """
        注册异常处理器
        
        Args:
            exception_type: 异常类型
            handler: 处理函数
        """
        try:
            self.error_handlers[exception_type] = handler
            logger_manager.debug(f"异常处理器注册成功: {exception_type.__name__}")
            
        except Exception as e:
            logger_manager.error(f"注册异常处理器失败: {e}")
    
    def handle_exception(self, exception: Exception, context: Optional[ErrorContext] = None) -> bool:
        """
        处理异常
        
        Args:
            exception: 异常对象
            context: 错误上下文
            
        Returns:
            是否处理成功
        """
        try:
            # 创建错误上下文
            if context is None:
                context = self._create_error_context(exception)
            
            # 记录错误历史
            self._record_error(context)
            
            # 查找合适的处理器
            handler = self._find_handler(type(exception))
            
            if handler:
                # 执行处理器
                return handler(exception, context)
            else:
                # 使用默认处理
                return self._default_handler(exception, context)
                
        except Exception as e:
            logger_manager.error(f"处理异常失败: {e}")
            return False
    
    def _create_error_context(self, exception: Exception) -> ErrorContext:
        """创建错误上下文"""
        try:
            # 获取调用栈信息
            tb = traceback.extract_tb(exception.__traceback__)
            
            if tb:
                frame = tb[-1]
                module_name = frame.filename
                function_name = frame.name
                line_number = frame.lineno
            else:
                module_name = ""
                function_name = ""
                line_number = 0
            
            # 获取堆栈跟踪
            stack_trace = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
            
            # 确定错误级别
            error_level = self._determine_error_level(exception)
            
            context = ErrorContext(
                error_type=type(exception).__name__,
                error_message=str(exception),
                error_level=error_level,
                module_name=module_name,
                function_name=function_name,
                line_number=line_number,
                stack_trace=stack_trace
            )
            
            return context
            
        except Exception as e:
            logger_manager.error(f"创建错误上下文失败: {e}")
            return ErrorContext(
                error_type="UnknownError",
                error_message="创建错误上下文失败",
                error_level=ErrorLevel.ERROR
            )
    
    def _determine_error_level(self, exception: Exception) -> ErrorLevel:
        """确定错误级别"""
        if isinstance(exception, (SystemExit, KeyboardInterrupt)):
            return ErrorLevel.CRITICAL
        elif isinstance(exception, (ImportError, ModuleNotFoundError)):
            return ErrorLevel.ERROR
        elif isinstance(exception, (ValueError, TypeError)):
            return ErrorLevel.WARNING
        elif isinstance(exception, DeepLearningException):
            return ErrorLevel.ERROR
        else:
            return ErrorLevel.ERROR
    
    def _find_handler(self, exception_type: Type[Exception]) -> Optional[Callable]:
        """查找异常处理器"""
        # 直接匹配
        if exception_type in self.error_handlers:
            return self.error_handlers[exception_type]
        
        # 查找父类匹配
        for registered_type, handler in self.error_handlers.items():
            if issubclass(exception_type, registered_type):
                return handler
        
        return None
    
    def _record_error(self, context: ErrorContext):
        """记录错误历史"""
        try:
            self.error_history.append(context)
            
            # 限制历史记录大小
            if len(self.error_history) > self.max_history_size:
                self.error_history = self.error_history[-self.max_history_size:]
            
            # 记录日志
            log_message = f"{context.error_type}: {context.error_message}"
            if context.function_name:
                log_message += f" (在 {context.function_name}:{context.line_number})"
            
            if context.error_level == ErrorLevel.CRITICAL:
                logger_manager.critical(log_message)
            elif context.error_level == ErrorLevel.ERROR:
                logger_manager.error(log_message)
            elif context.error_level == ErrorLevel.WARNING:
                logger_manager.warning(log_message)
            else:
                logger_manager.info(log_message)
                
        except Exception as e:
            logger_manager.error(f"记录错误历史失败: {e}")
    
    def _default_handler(self, exception: Exception, context: ErrorContext) -> bool:
        """默认异常处理器"""
        try:
            logger_manager.error(f"未处理的异常: {context.error_type} - {context.error_message}")
            
            # 在调试模式下打印堆栈跟踪
            if logger_manager.get_level() == "DEBUG":
                logger_manager.debug(f"堆栈跟踪:\n{context.stack_trace}")
            
            return True
            
        except Exception as e:
            logger_manager.error(f"默认异常处理失败: {e}")
            return False
    
    # 具体异常处理器
    def _handle_value_error(self, exception: ValueError, context: ErrorContext) -> bool:
        """处理值错误"""
        logger_manager.warning(f"值错误: {context.error_message}")
        return True
    
    def _handle_type_error(self, exception: TypeError, context: ErrorContext) -> bool:
        """处理类型错误"""
        logger_manager.warning(f"类型错误: {context.error_message}")
        return True
    
    def _handle_file_not_found_error(self, exception: FileNotFoundError, context: ErrorContext) -> bool:
        """处理文件未找到错误"""
        logger_manager.error(f"文件未找到: {context.error_message}")
        return True
    
    def _handle_import_error(self, exception: ImportError, context: ErrorContext) -> bool:
        """处理导入错误"""
        logger_manager.error(f"导入错误: {context.error_message}")
        return True
    
    def _handle_deep_learning_error(self, exception: DeepLearningException, context: ErrorContext) -> bool:
        """处理深度学习异常"""
        logger_manager.error(f"深度学习异常: {context.error_message}")
        return True
    
    def with_exception_handling(self, reraise: bool = False):
        """
        异常处理装饰器
        
        Args:
            reraise: 是否重新抛出异常
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # 处理异常
                    handled = self.handle_exception(e)
                    
                    if not handled or reraise:
                        raise
                    
                    return None
            
            return wrapper
        return decorator
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        try:
            if not self.error_history:
                return {"total_errors": 0}
            
            # 按错误类型统计
            error_types = {}
            error_levels = {}
            
            for error in self.error_history:
                # 错误类型统计
                if error.error_type in error_types:
                    error_types[error.error_type] += 1
                else:
                    error_types[error.error_type] = 1
                
                # 错误级别统计
                level = error.error_level.value
                if level in error_levels:
                    error_levels[level] += 1
                else:
                    error_levels[level] = 1
            
            return {
                "total_errors": len(self.error_history),
                "error_types": error_types,
                "error_levels": error_levels,
                "most_common_error": max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
            }
            
        except Exception as e:
            logger_manager.error(f"获取错误统计信息失败: {e}")
            return {"error": str(e)}
    
    def clear_error_history(self):
        """清空错误历史"""
        self.error_history.clear()
        logger_manager.info("错误历史已清空")
    
    def get_recent_errors(self, count: int = 10) -> List[ErrorContext]:
        """
        获取最近的错误
        
        Args:
            count: 错误数量
            
        Returns:
            最近的错误列表
        """
        return self.error_history[-count:] if self.error_history else []


# 全局异常处理器实例
exception_handler = ExceptionHandler()


if __name__ == "__main__":
    # 测试异常处理器功能
    print("🛡️ 测试异常处理器功能...")
    
    try:
        handler = ExceptionHandler()
        
        # 测试异常处理
        try:
            raise ValueError("测试值错误")
        except ValueError as e:
            handled = handler.handle_exception(e)
            print(f"✅ 异常处理: {handled}")
        
        # 测试装饰器
        @handler.with_exception_handling()
        def test_function():
            raise TypeError("测试类型错误")
        
        result = test_function()
        print(f"✅ 装饰器处理: {result is None}")
        
        # 测试错误统计
        stats = handler.get_error_statistics()
        print(f"✅ 错误统计: {stats['total_errors']} 个错误")
        
        print("✅ 异常处理器功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("异常处理器功能测试完成")
