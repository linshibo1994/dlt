#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
智能缓存系统
Smart Cache System

解决缓存数据不准确和不实时的问题
"""

import os
import json
import time
import hashlib
import threading
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
import pandas as pd

from core_modules import logger_manager, data_manager


class SmartCacheManager:
    """智能缓存管理器 - 支持数据版本控制和自动失效"""
    
    def __init__(self, cache_dir="cache", max_memory_cache_size=1000):
        self.cache_dir = cache_dir
        self.analysis_cache_dir = os.path.join(cache_dir, "analysis")
        self.models_cache_dir = os.path.join(cache_dir, "models")
        self.data_cache_dir = os.path.join(cache_dir, "data")
        
        # 内存缓存 (LRU)
        self.memory_cache = OrderedDict()
        self.max_memory_cache_size = max_memory_cache_size
        self.cache_lock = threading.RLock()
        
        # 数据版本跟踪
        self.data_version_cache = {}
        self.last_data_check = 0
        
        self._ensure_cache_dirs()
        self._load_data_version_info()
    
    def _ensure_cache_dirs(self):
        """确保缓存目录存在"""
        for dir_path in [self.cache_dir, self.analysis_cache_dir, 
                        self.models_cache_dir, self.data_cache_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
    
    def _load_data_version_info(self):
        """加载数据版本信息"""
        try:
            version_file = os.path.join(self.cache_dir, "data_version.json")
            if os.path.exists(version_file):
                with open(version_file, 'r', encoding='utf-8') as f:
                    self.data_version_cache = json.load(f)
        except Exception as e:
            logger_manager.warning(f"加载数据版本信息失败: {e}")
            self.data_version_cache = {}
    
    def _save_data_version_info(self):
        """保存数据版本信息"""
        try:
            version_file = os.path.join(self.cache_dir, "data_version.json")
            with open(version_file, 'w', encoding='utf-8') as f:
                json.dump(self.data_version_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger_manager.error(f"保存数据版本信息失败: {e}")
    
    def _get_data_signature(self) -> str:
        """获取当前数据的签名（用于版本控制）"""
        try:
            df = data_manager.get_data()
            if df is None or df.empty:
                return "empty_data"
            
            # 使用数据的行数、最新期号、最新开奖日期生成签名
            latest_row = df.iloc[0]  # 数据按期号降序排列，第一行是最新数据
            signature_data = {
                'total_rows': len(df),
                'latest_period': str(latest_row.get('期号', '')),
                'latest_date': str(latest_row.get('开奖日期', '')),
                'data_hash': hashlib.md5(str(df.head(10).values).encode()).hexdigest()[:8]
            }
            
            signature = hashlib.md5(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
            return signature
            
        except Exception as e:
            logger_manager.error(f"获取数据签名失败: {e}")
            return f"error_{int(time.time())}"
    
    def _generate_cache_key(self, method_name: str, periods: Optional[int] = None, 
                          **kwargs) -> str:
        """生成智能缓存键"""
        # 获取数据签名
        data_signature = self._get_data_signature()
        
        # 构建缓存键组件
        key_components = [
            method_name,
            f"periods_{periods or 'all'}",
            f"data_{data_signature}"
        ]
        
        # 添加其他参数
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            params_str = "_".join([f"{k}_{v}" for k, v in sorted_kwargs])
            key_components.append(f"params_{params_str}")
        
        cache_key = "_".join(key_components)
        
        # 限制键长度
        if len(cache_key) > 200:
            cache_key = f"{method_name}_{hashlib.md5(cache_key.encode()).hexdigest()}"
        
        return cache_key
    
    def _is_cache_valid(self, cache_data: Dict) -> bool:
        """检查缓存是否有效"""
        try:
            # 检查缓存数据结构
            if not isinstance(cache_data, dict):
                return False
            
            if 'data' not in cache_data or 'metadata' not in cache_data:
                return False
            
            metadata = cache_data['metadata']
            
            # 检查数据版本
            cached_data_signature = metadata.get('data_signature')
            current_data_signature = self._get_data_signature()
            
            if cached_data_signature != current_data_signature:
                logger_manager.debug(f"缓存失效: 数据版本不匹配 {cached_data_signature} != {current_data_signature}")
                return False
            
            # 检查时间戳（可选的额外验证）
            cache_time = metadata.get('timestamp')
            if cache_time:
                cache_datetime = datetime.fromisoformat(cache_time)
                # 缓存超过24小时自动失效
                if datetime.now() - cache_datetime > timedelta(hours=24):
                    logger_manager.debug("缓存失效: 超过24小时")
                    return False
            
            return True
            
        except Exception as e:
            logger_manager.error(f"检查缓存有效性失败: {e}")
            return False
    
    def get_from_memory_cache(self, cache_key: str) -> Optional[Any]:
        """从内存缓存获取数据"""
        with self.cache_lock:
            if cache_key in self.memory_cache:
                # 移动到末尾（LRU）
                value = self.memory_cache.pop(cache_key)
                self.memory_cache[cache_key] = value
                
                if self._is_cache_valid(value):
                    return value['data']
                else:
                    # 删除无效缓存
                    del self.memory_cache[cache_key]
            
            return None
    
    def set_to_memory_cache(self, cache_key: str, data: Any):
        """设置内存缓存"""
        with self.cache_lock:
            # 构建缓存数据
            cache_data = {
                'data': data,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'data_signature': self._get_data_signature()
                }
            }
            
            # LRU淘汰
            if len(self.memory_cache) >= self.max_memory_cache_size:
                # 删除最旧的项
                oldest_key = next(iter(self.memory_cache))
                del self.memory_cache[oldest_key]
            
            self.memory_cache[cache_key] = cache_data
    
    def load_cache(self, cache_type: str, method_name: str, periods: Optional[int] = None, 
                   **kwargs) -> Optional[Any]:
        """智能加载缓存"""
        try:
            cache_key = self._generate_cache_key(method_name, periods, **kwargs)
            
            # 1. 先尝试内存缓存
            memory_result = self.get_from_memory_cache(cache_key)
            if memory_result is not None:
                logger_manager.debug(f"从内存缓存加载: {cache_key}")
                return memory_result
            
            # 2. 尝试文件缓存
            cache_path = os.path.join(self.analysis_cache_dir, f"{cache_key}.json")
            
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                if self._is_cache_valid(cache_data):
                    # 加载到内存缓存
                    self.set_to_memory_cache(cache_key, cache_data['data'])
                    logger_manager.debug(f"从文件缓存加载: {cache_key}")
                    return cache_data['data']
                else:
                    # 删除无效的文件缓存
                    os.remove(cache_path)
                    logger_manager.debug(f"删除无效文件缓存: {cache_key}")
            
            return None
            
        except Exception as e:
            logger_manager.error(f"加载缓存失败: {e}")
            return None
    
    def save_cache(self, cache_type: str, method_name: str, data: Any, 
                   periods: Optional[int] = None, **kwargs) -> bool:
        """智能保存缓存"""
        try:
            cache_key = self._generate_cache_key(method_name, periods, **kwargs)
            
            # 1. 保存到内存缓存
            self.set_to_memory_cache(cache_key, data)
            
            # 2. 保存到文件缓存
            cache_data = {
                'data': data,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'data_signature': self._get_data_signature(),
                    'method_name': method_name,
                    'periods': periods,
                    'kwargs': kwargs
                }
            }
            
            cache_path = os.path.join(self.analysis_cache_dir, f"{cache_key}.json")
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2, default=str)
            
            logger_manager.debug(f"保存缓存: {cache_key}")
            return True
            
        except Exception as e:
            logger_manager.error(f"保存缓存失败: {e}")
            return False
    
    def clear_cache(self, cache_type: str = "all", method_pattern: str = None) -> int:
        """清理缓存"""
        cleared_count = 0
        
        try:
            # 清理内存缓存
            with self.cache_lock:
                if method_pattern:
                    keys_to_remove = [k for k in self.memory_cache.keys() 
                                    if method_pattern in k]
                    for key in keys_to_remove:
                        del self.memory_cache[key]
                        cleared_count += 1
                else:
                    cleared_count += len(self.memory_cache)
                    self.memory_cache.clear()
            
            # 清理文件缓存
            if cache_type in ["all", "analysis"]:
                if os.path.exists(self.analysis_cache_dir):
                    for filename in os.listdir(self.analysis_cache_dir):
                        if filename.endswith('.json'):
                            if method_pattern is None or method_pattern in filename:
                                file_path = os.path.join(self.analysis_cache_dir, filename)
                                os.remove(file_path)
                                cleared_count += 1
            
            logger_manager.info(f"清理缓存完成，删除 {cleared_count} 个缓存项")
            return cleared_count
            
        except Exception as e:
            logger_manager.error(f"清理缓存失败: {e}")
            return cleared_count
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        stats = {
            'memory_cache': {
                'size': len(self.memory_cache),
                'max_size': self.max_memory_cache_size
            },
            'file_cache': {
                'analysis_files': 0,
                'total_size_mb': 0.0
            },
            'data_signature': self._get_data_signature()
        }
        
        try:
            if os.path.exists(self.analysis_cache_dir):
                total_size = 0
                file_count = 0
                for filename in os.listdir(self.analysis_cache_dir):
                    if filename.endswith('.json'):
                        file_path = os.path.join(self.analysis_cache_dir, filename)
                        total_size += os.path.getsize(file_path)
                        file_count += 1
                
                stats['file_cache']['analysis_files'] = file_count
                stats['file_cache']['total_size_mb'] = total_size / (1024 * 1024)
        
        except Exception as e:
            logger_manager.error(f"获取缓存统计失败: {e}")
        
        return stats


# 创建全局智能缓存管理器实例
smart_cache_manager = SmartCacheManager()


class CacheMigrationTool:
    """缓存迁移工具 - 将旧缓存系统迁移到智能缓存系统"""

    def __init__(self):
        self.old_cache_manager = None
        self.smart_cache_manager = smart_cache_manager

    def migrate_from_old_system(self):
        """从旧缓存系统迁移"""
        try:
            from core_modules import cache_manager as old_cache_manager
            self.old_cache_manager = old_cache_manager

            logger_manager.info("开始缓存系统迁移...")

            # 清理所有旧缓存（因为它们可能包含过期数据）
            cleared_count = old_cache_manager.clear_cache("all")
            logger_manager.info(f"清理旧缓存: {cleared_count} 个文件")

            # 清理智能缓存系统
            self.smart_cache_manager.clear_cache("all")
            logger_manager.info("智能缓存系统已准备就绪")

            return True

        except Exception as e:
            logger_manager.error(f"缓存迁移失败: {e}")
            return False

    def create_cache_compatibility_layer(self):
        """创建缓存兼容层，让旧代码可以使用新缓存系统"""
        try:
            # 在core_modules中替换cache_manager
            import core_modules

            # 创建兼容性包装器
            class CompatibilityCacheManager:
                def __init__(self):
                    self.smart_cache = smart_cache_manager

                def load_cache(self, cache_type: str, key: str):
                    """兼容旧的load_cache接口"""
                    # 解析旧的缓存键，提取方法名和期数
                    method_name, periods = self._parse_old_cache_key(key)
                    return self.smart_cache.load_cache(cache_type, method_name, periods)

                def save_cache(self, cache_type: str, key: str, data):
                    """兼容旧的save_cache接口"""
                    method_name, periods = self._parse_old_cache_key(key)
                    return self.smart_cache.save_cache(cache_type, method_name, data, periods)

                def clear_cache(self, cache_type: str = "all"):
                    """兼容旧的clear_cache接口"""
                    return self.smart_cache.clear_cache(cache_type)

                def get_cache_info(self):
                    """兼容旧的get_cache_info接口"""
                    stats = self.smart_cache.get_cache_stats()
                    # 转换为旧格式
                    return {
                        'cache_dir': self.smart_cache.cache_dir,
                        'analysis': {
                            'files': stats['file_cache']['analysis_files'],
                            'size_mb': stats['file_cache']['total_size_mb']
                        },
                        'total': {
                            'files': stats['file_cache']['analysis_files'],
                            'size_mb': stats['file_cache']['total_size_mb']
                        }
                    }

                def _parse_old_cache_key(self, key: str) -> Tuple[str, Optional[int]]:
                    """解析旧的缓存键格式"""
                    # 旧格式例如: "enhanced_frequency_analysis_100"
                    parts = key.split('_')

                    # 尝试提取期数
                    periods = None
                    method_parts = []

                    for part in reversed(parts):
                        if part.isdigit() and periods is None:
                            periods = int(part)
                        elif part == "all" and periods is None:
                            periods = None
                            break
                        else:
                            method_parts.insert(0, part)

                    method_name = "_".join(method_parts) if method_parts else key
                    return method_name, periods

            # 替换全局cache_manager
            core_modules.cache_manager = CompatibilityCacheManager()

            logger_manager.info("缓存兼容层创建成功")
            return True

        except Exception as e:
            logger_manager.error(f"创建缓存兼容层失败: {e}")
            return False


def migrate_cache_system():
    """执行缓存系统迁移"""
    migration_tool = CacheMigrationTool()

    # 执行迁移
    if migration_tool.migrate_from_old_system():
        # 创建兼容层
        if migration_tool.create_cache_compatibility_layer():
            logger_manager.info("缓存系统迁移完成")
            return True

    logger_manager.error("缓存系统迁移失败")
    return False
