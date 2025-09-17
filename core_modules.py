#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
核心模块集成
整合数据管理、缓存管理、日志管理、进度管理等核心功能
"""

import os
import sys
import json
import time
import threading
import logging
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from logging.handlers import RotatingFileHandler


# ==================== 缓存管理器 ====================
class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        self.model_cache_dir = os.path.join(cache_dir, "models")
        self.analysis_cache_dir = os.path.join(cache_dir, "analysis")
        self.data_cache_dir = os.path.join(cache_dir, "data")
        
        self._ensure_cache_dirs()
    
    def _ensure_cache_dirs(self):
        """确保缓存目录存在"""
        for dir_path in [self.cache_dir, self.model_cache_dir, self.analysis_cache_dir, self.data_cache_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
    
    def get_cache_path(self, cache_type: str, key: str) -> str:
        """获取缓存文件路径"""
        if cache_type == "models":
            return os.path.join(self.model_cache_dir, f"{key}.pkl")
        elif cache_type == "analysis":
            return os.path.join(self.analysis_cache_dir, f"{key}.json")
        elif cache_type == "data":
            return os.path.join(self.data_cache_dir, f"{key}.csv")
        else:
            return os.path.join(self.cache_dir, f"{key}.cache")
    
    def save_cache(self, cache_type: str, key: str, data: Any) -> bool:
        """保存缓存"""
        try:
            cache_path = self.get_cache_path(cache_type, key)
            
            if cache_type == "models":
                import joblib
                joblib.dump(data, cache_path)
            elif cache_type == "analysis":
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            elif cache_type == "data":
                if isinstance(data, pd.DataFrame):
                    data.to_csv(cache_path, index=False)
                else:
                    pd.DataFrame(data).to_csv(cache_path, index=False)
            else:
                import pickle
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            
            return True
        except Exception as e:
            print(f"❌ 保存缓存失败: {e}")
            return False
    
    def load_cache(self, cache_type: str, key: str) -> Any:
        """加载缓存"""
        try:
            cache_path = self.get_cache_path(cache_type, key)
            
            if not os.path.exists(cache_path):
                return None
            
            if cache_type == "models":
                import joblib
                return joblib.load(cache_path)
            elif cache_type == "analysis":
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif cache_type == "data":
                return pd.read_csv(cache_path)
            else:
                import pickle
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        
        except Exception as e:
            print(f"❌ 加载缓存失败: {e}")
            return None
    
    def clear_cache(self, cache_type: str = "all") -> int:
        """清理缓存"""
        cleared_count = 0
        
        if cache_type == "all":
            dirs_to_clear = [self.model_cache_dir, self.analysis_cache_dir, self.data_cache_dir]
        elif cache_type == "models":
            dirs_to_clear = [self.model_cache_dir]
        elif cache_type == "analysis":
            dirs_to_clear = [self.analysis_cache_dir]
        elif cache_type == "data":
            dirs_to_clear = [self.data_cache_dir]
        else:
            return 0
        
        for dir_path in dirs_to_clear:
            if os.path.exists(dir_path):
                for filename in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, filename)
                    try:
                        # 如果是目录，尝试递归删除
                        if os.path.isdir(file_path):
                            import shutil
                            try:
                                shutil.rmtree(file_path)
                                cleared_count += 1
                            except PermissionError:
                                # 权限不足时，尝试修改权限后删除
                                try:
                                    import stat
                                    os.chmod(file_path, stat.S_IWRITE)
                                    shutil.rmtree(file_path)
                                    cleared_count += 1
                                except Exception:
                                    # 如果仍然无法删除，跳过并记录警告
                                    logger_manager.warning(f"跳过无法删除的目录: {filename}")
                        else:
                            # 普通文件
                            os.remove(file_path)
                            cleared_count += 1
                    except PermissionError:
                        # 权限不足时，尝试修改权限后删除
                        try:
                            import stat
                            os.chmod(file_path, stat.S_IWRITE)
                            os.remove(file_path)
                            cleared_count += 1
                        except Exception:
                            # 如果仍然无法删除，跳过并记录警告
                            logger_manager.warning(f"跳过无法删除的文件: {filename}")
                    except Exception as e:
                        # 其他错误，记录警告但不中断程序
                        logger_manager.warning(f"删除缓存文件失败 {filename}: {e}")
        
        return cleared_count
    
    def get_cache_info(self) -> Dict:
        """获取缓存信息"""
        info = {
            'cache_dir': self.cache_dir,
            'models': {'files': 0, 'size_mb': 0.0},
            'analysis': {'files': 0, 'size_mb': 0.0},
            'data': {'files': 0, 'size_mb': 0.0},
            'total': {'files': 0, 'size_mb': 0.0}
        }
        
        for cache_type, dir_path in [
            ('models', self.model_cache_dir),
            ('analysis', self.analysis_cache_dir),
            ('data', self.data_cache_dir)
        ]:
            if os.path.exists(dir_path):
                for filename in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, filename)
                    if os.path.isfile(file_path):
                        size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        info[cache_type]['files'] += 1
                        info[cache_type]['size_mb'] += size_mb
                        info['total']['files'] += 1
                        info['total']['size_mb'] += size_mb
        
        # 四舍五入
        for key in info:
            if isinstance(info[key], dict) and 'size_mb' in info[key]:
                info[key]['size_mb'] = round(info[key]['size_mb'], 2)
        
        return info


# ==================== 进度管理器 ====================
class ProgressBar:
    """进度条类"""
    
    def __init__(self, total: int, description: str = "", width: int = 50):
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        self.start_time = time.time()
        self.interrupted = False
    
    def update(self, step: int = 1, message: str = ""):
        """更新进度"""
        if self.interrupted:
            return
        
        self.current = min(self.current + step, self.total)
        current_time = time.time()
        
        percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        elapsed_time = current_time - self.start_time
        
        if self.current > 0:
            estimated_total_time = elapsed_time * self.total / self.current
            remaining_time = max(0, estimated_total_time - elapsed_time)
        else:
            remaining_time = 0
        
        filled_width = int(self.width * self.current / self.total) if self.total > 0 else 0
        bar = "█" * filled_width + "░" * (self.width - filled_width)
        
        elapsed_str = self._format_time(elapsed_time)
        remaining_str = self._format_time(remaining_time)
        
        progress_str = f"\r{self.description} |{bar}| {self.current}/{self.total} "
        progress_str += f"({percentage:.1f}%) "
        progress_str += f"[已用: {elapsed_str}, 剩余: {remaining_str}]"
        
        if message:
            progress_str += f" - {message}"
        
        sys.stdout.write(progress_str)
        sys.stdout.flush()
    
    def finish(self, message: str = "完成"):
        """完成进度条"""
        if not self.interrupted:
            self.current = self.total
            self.update(0, message)
        print()
    
    def interrupt(self, message: str = "已中断"):
        """中断进度条"""
        self.interrupted = True
        print(f"\n{message}")
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes:.0f}m{secs:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h{minutes:.0f}m"


class TaskManager:
    """任务管理器"""
    
    def __init__(self):
        self.current_task = None
        self.interrupt_flag = threading.Event()
        self.interrupt_callback = None
    
    def create_progress_bar(self, total: int, description: str = "") -> ProgressBar:
        """创建进度条"""
        progress_bar = ProgressBar(total, description)
        self.current_task = progress_bar
        return progress_bar
    
    def set_interrupt_callback(self, callback):
        """设置中断回调函数"""
        self.interrupt_callback = callback
    
    def interrupt_current_task(self):
        """中断当前任务"""
        self.interrupt_flag.set()
        if self.current_task:
            self.current_task.interrupt()
        if self.interrupt_callback:
            self.interrupt_callback()
    
    def is_interrupted(self) -> bool:
        """检查是否被中断"""
        return self.interrupt_flag.is_set()
    
    def reset_interrupt(self):
        """重置中断标志"""
        self.interrupt_flag.clear()


# ==================== 日志管理器 ====================
class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


class LoggerManager:
    """日志管理器"""
    
    def __init__(self, log_dir="logs", max_file_size=10*1024*1024, backup_count=5):
        self.log_dir = log_dir
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.logger = logging.getLogger('dlt_predictor')
        self.logger.setLevel(logging.DEBUG)
        
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """设置日志处理器"""
        # 文件处理器
        file_handler = RotatingFileHandler(
            os.path.join(self.log_dir, 'dlt_predictor.log'),
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # 错误文件处理器
        error_handler = RotatingFileHandler(
            os.path.join(self.log_dir, 'errors.log'),
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(console_handler)
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        if exception:
            self.logger.error(f"{message}: {str(exception)}", extra=kwargs)
            self.logger.debug(traceback.format_exc())
        else:
            self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        if exception:
            self.logger.critical(f"{message}: {str(exception)}", extra=kwargs)
            self.logger.debug(traceback.format_exc())
        else:
            self.logger.critical(message, extra=kwargs)


# ==================== 数据管理器 ====================
class DataManager:
    """数据管理器"""
    
    def __init__(self, data_file="data/dlt_data_all.csv", cache_manager=None):
        self.data_file = data_file
        self.cache_manager = cache_manager or CacheManager()
        self.df = None
        self.data_stats = {}
        
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        try:
            # 尝试从缓存加载
            cached_data = self.cache_manager.load_cache("data", "dlt_data_all")

            if cached_data is not None and not cached_data.empty:
                self.df = cached_data
                print(f"✅ 从缓存加载数据: {len(self.df)} 期")
            else:
                # 从文件加载
                self.df = pd.read_csv(self.data_file)

                # 数据文件已经按期号降序排列（最新的在前面），无需重新排序
                print(f"[OK] 从文件加载数据: {len(self.df)} 期")

                # 保存到缓存
                self.cache_manager.save_cache("data", "dlt_data_all", self.df)

            self._calculate_stats()

        except Exception as e:
            print(f"[ERROR] 加载数据失败: {e}")
    
    def _calculate_stats(self):
        """计算数据统计信息"""
        if self.df is None or len(self.df) == 0:
            return

        # 数据已按期号降序排列，第一行是最新数据，最后一行是最早数据
        self.data_stats = {
            'total_periods': len(self.df),
            'date_range': {
                'start': self.df.iloc[-1]['date'],  # 最早日期
                'end': self.df.iloc[0]['date']      # 最新日期
            },
            'latest_issue': self.df.iloc[0]['issue']  # 最新期号
        }
    
    def get_data(self) -> pd.DataFrame:
        """获取数据"""
        return self.df
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.data_stats
    
    def get_period_range(self, start_period: int, end_period: int = None) -> pd.DataFrame:
        """获取指定期数范围的数据"""
        if self.df is None:
            return pd.DataFrame()
        
        if end_period is None:
            return self.df.iloc[start_period:]
        else:
            return self.df.iloc[start_period:end_period]
    
    def get_latest_periods(self, count: int) -> pd.DataFrame:
        """获取最新的几期数据"""
        if self.df is None:
            return pd.DataFrame()
        
        return self.df.tail(count)
    
    def parse_balls(self, row) -> Tuple[List[int], List[int]]:
        """解析号码"""
        try:
            front_balls = [int(x.strip()) for x in str(row['front_balls']).split(',')]
            back_balls = [int(x.strip()) for x in str(row['back_balls']).split(',')]
            return front_balls, back_balls
        except Exception as e:
            print(f"❌ 解析号码失败: {e}")
            return [], []


# ==================== 全局实例 ====================
cache_manager = CacheManager()
task_manager = TaskManager()
logger_manager = LoggerManager()
data_manager = DataManager(cache_manager=cache_manager)


# ==================== 装饰器 ====================
def with_progress(total: int, description: str = ""):
    """进度条装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            progress_bar = task_manager.create_progress_bar(total, description)
            try:
                result = func(progress_bar, *args, **kwargs)
                progress_bar.finish()
                return result
            except KeyboardInterrupt:
                progress_bar.interrupt("用户中断")
                raise
            except Exception as e:
                progress_bar.interrupt(f"错误: {e}")
                raise
        return wrapper
    return decorator


def log_function(func):
    """函数调用日志装饰器"""
    def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}"
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger_manager.debug(f"函数调用成功: {func_name} (耗时: {duration:.3f}s)")
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger_manager.error(f"函数调用失败: {func_name} (耗时: {duration:.3f}s)", e)
            raise
    
    return wrapper


if __name__ == "__main__":
    # 测试核心模块
    print("🔧 测试核心模块...")

    # 测试缓存管理器
    print("📦 测试缓存管理器...")
    cache_info = cache_manager.get_cache_info()
    print(f"缓存信息: {cache_info}")

    # 测试数据管理器
    print("📊 测试数据管理器...")
    stats = data_manager.get_stats()
    print(f"数据统计: {stats}")

    # 测试日志管理器
    print("📝 测试日志管理器...")
    logger_manager.info("核心模块测试完成")

    print("✅ 核心模块测试完成")
