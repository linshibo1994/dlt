#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
预测缓存模块
Prediction Cache Module

提供预测结果缓存、缓存策略、缓存管理等功能。
"""

import os
import time
import json
import pickle
import hashlib
import threading
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import OrderedDict
import numpy as np

from core_modules import logger_manager
from ..utils.exceptions import PredictionException
from .prediction_engine import PredictionResult, PredictionRequest


class CacheStrategy(Enum):
    """缓存策略枚举"""
    LRU = "lru"  # 最近最少使用
    LFU = "lfu"  # 最少使用频率
    TTL = "ttl"  # 生存时间
    FIFO = "fifo"  # 先进先出
    ADAPTIVE = "adaptive"  # 自适应


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: PredictionResult
    created_time: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[int] = None  # 生存时间（秒）
    size: int = 0  # 条目大小（字节）
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        
        return (datetime.now() - self.created_time).total_seconds() > self.ttl
    
    def update_access(self):
        """更新访问信息"""
        self.last_accessed = datetime.now()
        self.access_count += 1


class CacheKeyGenerator:
    """缓存键生成器"""
    
    @staticmethod
    def generate_key(request: PredictionRequest) -> str:
        """
        生成缓存键
        
        Args:
            request: 预测请求
            
        Returns:
            缓存键
        """
        try:
            # 创建键的组成部分
            key_parts = [
                request.model_name,
                request.model_version or "latest",
                request.mode.value
            ]
            
            # 添加输入数据的哈希
            input_hash = hashlib.md5(request.input_data.tobytes()).hexdigest()[:16]
            key_parts.append(input_hash)
            
            # 添加参数的哈希
            if request.parameters:
                params_str = json.dumps(request.parameters, sort_keys=True)
                params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
                key_parts.append(params_hash)
            
            # 组合键
            cache_key = ":".join(key_parts)
            
            return cache_key
            
        except Exception as e:
            logger_manager.error(f"生成缓存键失败: {e}")
            return f"fallback:{request.request_id}"
    
    @staticmethod
    def generate_key_from_data(model_name: str, input_data: np.ndarray,
                              model_version: str = None, **kwargs) -> str:
        """
        从数据生成缓存键
        
        Args:
            model_name: 模型名称
            input_data: 输入数据
            model_version: 模型版本
            **kwargs: 其他参数
            
        Returns:
            缓存键
        """
        try:
            key_parts = [
                model_name,
                model_version or "latest"
            ]
            
            # 添加输入数据的哈希
            input_hash = hashlib.md5(input_data.tobytes()).hexdigest()[:16]
            key_parts.append(input_hash)
            
            # 添加参数的哈希
            if kwargs:
                params_str = json.dumps(kwargs, sort_keys=True)
                params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
                key_parts.append(params_hash)
            
            return ":".join(key_parts)
            
        except Exception as e:
            logger_manager.error(f"从数据生成缓存键失败: {e}")
            return f"fallback:{int(time.time())}"


class LRUCache:
    """LRU缓存实现"""
    
    def __init__(self, max_size: int = 1000):
        """
        初始化LRU缓存
        
        Args:
            max_size: 最大缓存大小
        """
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        
        logger_manager.debug(f"LRU缓存初始化，最大大小: {max_size}")
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """获取缓存项"""
        with self.lock:
            if key in self.cache:
                entry = self.cache.pop(key)
                entry.update_access()
                self.cache[key] = entry  # 移动到末尾
                return entry
            return None
    
    def put(self, key: str, entry: CacheEntry):
        """添加缓存项"""
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # 移除最久未使用的项
                self.cache.popitem(last=False)
            
            self.cache[key] = entry
    
    def remove(self, key: str) -> bool:
        """移除缓存项"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """获取缓存大小"""
        with self.lock:
            return len(self.cache)
    
    def keys(self) -> List[str]:
        """获取所有键"""
        with self.lock:
            return list(self.cache.keys())


class TTLCache:
    """TTL缓存实现"""
    
    def __init__(self, default_ttl: int = 3600):
        """
        初始化TTL缓存
        
        Args:
            default_ttl: 默认生存时间（秒）
        """
        self.default_ttl = default_ttl
        self.cache = {}
        self.lock = threading.RLock()
        
        logger_manager.debug(f"TTL缓存初始化，默认TTL: {default_ttl}秒")
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """获取缓存项"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry.is_expired():
                    del self.cache[key]
                    return None
                
                entry.update_access()
                return entry
            return None
    
    def put(self, key: str, entry: CacheEntry):
        """添加缓存项"""
        with self.lock:
            if entry.ttl is None:
                entry.ttl = self.default_ttl
            
            self.cache[key] = entry
    
    def remove(self, key: str) -> bool:
        """移除缓存项"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
    
    def cleanup_expired(self) -> int:
        """清理过期项"""
        with self.lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
            
            return len(expired_keys)
    
    def size(self) -> int:
        """获取缓存大小"""
        with self.lock:
            return len(self.cache)


class PredictionCache:
    """预测缓存主类"""
    
    def __init__(self, strategy: CacheStrategy = CacheStrategy.LRU,
                 max_size: int = 1000, default_ttl: int = 3600,
                 enable_persistence: bool = False, cache_dir: str = "cache"):
        """
        初始化预测缓存
        
        Args:
            strategy: 缓存策略
            max_size: 最大缓存大小
            default_ttl: 默认生存时间
            enable_persistence: 是否启用持久化
            cache_dir: 缓存目录
        """
        self.strategy = strategy
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.enable_persistence = enable_persistence
        self.cache_dir = cache_dir
        
        # 创建缓存实现
        if strategy == CacheStrategy.LRU:
            self.cache_impl = LRUCache(max_size)
        elif strategy == CacheStrategy.TTL:
            self.cache_impl = TTLCache(default_ttl)
        else:
            # 默认使用LRU
            self.cache_impl = LRUCache(max_size)
        
        self.key_generator = CacheKeyGenerator()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'puts': 0,
            'removes': 0,
            'evictions': 0
        }
        self.lock = threading.RLock()
        
        # 创建缓存目录
        if enable_persistence:
            os.makedirs(cache_dir, exist_ok=True)
        
        # 启动清理线程
        self.cleanup_thread = None
        self.running = False
        if strategy == CacheStrategy.TTL:
            self.start_cleanup_thread()
        
        logger_manager.info(f"预测缓存初始化完成，策略: {strategy.value}")
    
    def get(self, request: PredictionRequest) -> Optional[PredictionResult]:
        """
        获取缓存的预测结果
        
        Args:
            request: 预测请求
            
        Returns:
            缓存的预测结果
        """
        try:
            cache_key = self.key_generator.generate_key(request)
            
            with self.lock:
                entry = self.cache_impl.get(cache_key)
                
                if entry:
                    self.stats['hits'] += 1
                    logger_manager.debug(f"缓存命中: {cache_key}")
                    return entry.value
                else:
                    self.stats['misses'] += 1
                    logger_manager.debug(f"缓存未命中: {cache_key}")
                    return None
                    
        except Exception as e:
            logger_manager.error(f"获取缓存失败: {e}")
            return None
    
    def put(self, request: PredictionRequest, result: PredictionResult,
            ttl: Optional[int] = None):
        """
        缓存预测结果
        
        Args:
            request: 预测请求
            result: 预测结果
            ttl: 生存时间
        """
        try:
            cache_key = self.key_generator.generate_key(request)
            
            # 计算条目大小
            entry_size = self._calculate_size(result)
            
            # 创建缓存条目
            entry = CacheEntry(
                key=cache_key,
                value=result,
                created_time=datetime.now(),
                last_accessed=datetime.now(),
                ttl=ttl or self.default_ttl,
                size=entry_size
            )
            
            with self.lock:
                self.cache_impl.put(cache_key, entry)
                self.stats['puts'] += 1
            
            # 持久化
            if self.enable_persistence:
                self._persist_entry(cache_key, entry)
            
            logger_manager.debug(f"缓存已添加: {cache_key}")
            
        except Exception as e:
            logger_manager.error(f"添加缓存失败: {e}")

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        设置缓存值（简化接口）

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间（秒）
        """
        try:
            # 使用原始的CacheEntry类
            from datetime import datetime

            # 创建一个简单的预测结果包装器
            class SimpleResult:
                def __init__(self, value):
                    self.value = value

            entry = CacheEntry(
                key=key,
                value=SimpleResult(value),
                created_time=datetime.now(),
                last_accessed=datetime.now(),
                ttl=ttl
            )

            with self.lock:
                self.cache_impl.put(key, entry)
                self.stats['puts'] += 1

            logger_manager.debug(f"缓存值已设置: {key}")

        except Exception as e:
            logger_manager.error(f"设置缓存值失败: {e}")

    def get_simple(self, key: str) -> Optional[Any]:
        """
        获取缓存值（简化接口）

        Args:
            key: 缓存键

        Returns:
            缓存值
        """
        try:
            with self.lock:
                entry = self.cache_impl.get(key)

                if entry:
                    # 检查TTL（使用原始的is_expired方法）
                    if hasattr(entry, 'is_expired') and entry.is_expired():
                        self.cache_impl.remove(key)
                        self.stats['misses'] += 1
                        return None

                    self.stats['hits'] += 1

                    # 返回包装的值
                    if hasattr(entry.value, 'value'):
                        return entry.value.value
                    else:
                        return entry.value
                else:
                    self.stats['misses'] += 1
                    return None

        except Exception as e:
            logger_manager.error(f"获取缓存值失败: {e}")
            return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        try:
            with self.lock:
                total_requests = self.stats['hits'] + self.stats['misses']
                hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0.0

                return {
                    'hits': self.stats['hits'],
                    'misses': self.stats['misses'],
                    'puts': self.stats['puts'],
                    'removes': self.stats['removes'],
                    'hit_rate': hit_rate,
                    'total_requests': total_requests,
                    'cache_size': len(self.cache_impl.cache) if hasattr(self.cache_impl, 'cache') else 0
                }
        except Exception as e:
            logger_manager.error(f"获取缓存统计失败: {e}")
            return {}

    def remove(self, request: PredictionRequest) -> bool:
        """
        移除缓存项
        
        Args:
            request: 预测请求
            
        Returns:
            是否移除成功
        """
        try:
            cache_key = self.key_generator.generate_key(request)
            
            with self.lock:
                success = self.cache_impl.remove(cache_key)
            
            # 删除持久化文件
            if self.enable_persistence and success:
                self._remove_persisted_entry(cache_key)
            
            if success:
                logger_manager.debug(f"缓存已移除: {cache_key}")
            
            return success
            
        except Exception as e:
            logger_manager.error(f"移除缓存失败: {e}")
            return False
    
    def clear(self):
        """清空缓存"""
        try:
            with self.lock:
                self.cache_impl.clear()
                self.stats = {
                    'hits': 0,
                    'misses': 0,
                    'puts': 0,
                    'evictions': 0
                }
            
            # 清空持久化文件
            if self.enable_persistence:
                self._clear_persisted_cache()
            
            logger_manager.info("缓存已清空")
            
        except Exception as e:
            logger_manager.error(f"清空缓存失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'strategy': self.strategy.value,
                'size': self.cache_impl.size(),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'stats': self.stats.copy(),
                'memory_usage': self._calculate_memory_usage()
            }
    
    def start_cleanup_thread(self):
        """启动清理线程"""
        if self.running:
            return
        
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop)
        self.cleanup_thread.daemon = True
        self.cleanup_thread.start()
        
        logger_manager.debug("缓存清理线程已启动")
    
    def stop_cleanup_thread(self):
        """停止清理线程"""
        try:
            self.running = False
            if self.cleanup_thread:
                self.cleanup_thread.join(timeout=5)

            logger_manager.debug("缓存清理线程已停止")
        except Exception:
            # 在Python关闭时，某些模块可能已经被清理，忽略错误
            pass
    
    def _cleanup_loop(self):
        """清理循环"""
        while self.running:
            try:
                if hasattr(self.cache_impl, 'cleanup_expired'):
                    expired_count = self.cache_impl.cleanup_expired()
                    if expired_count > 0:
                        logger_manager.debug(f"清理过期缓存项: {expired_count} 个")
                
                time.sleep(300)  # 每5分钟清理一次
                
            except Exception as e:
                logger_manager.error(f"缓存清理失败: {e}")
                time.sleep(60)
    
    def _calculate_size(self, result: PredictionResult) -> int:
        """计算结果大小"""
        try:
            size = 0
            
            if result.predictions is not None:
                size += result.predictions.nbytes
            
            if result.confidence_scores is not None:
                size += result.confidence_scores.nbytes
            
            # 估算其他字段大小
            size += len(str(result.request_id)) * 2
            size += len(str(result.error_message)) * 2
            size += len(json.dumps(result.model_info)) * 2
            size += len(json.dumps(result.metadata)) * 2
            
            return size
            
        except Exception:
            return 1024  # 默认1KB
    
    def _calculate_memory_usage(self) -> int:
        """计算内存使用量"""
        try:
            total_size = 0
            
            if hasattr(self.cache_impl, 'cache'):
                for entry in self.cache_impl.cache.values():
                    total_size += entry.size
            
            return total_size
            
        except Exception:
            return 0
    
    def _persist_entry(self, key: str, entry: CacheEntry):
        """持久化缓存条目"""
        try:
            if not self.enable_persistence:
                return
            
            file_path = os.path.join(self.cache_dir, f"{key}.pkl")
            
            with open(file_path, 'wb') as f:
                pickle.dump(entry, f)
                
        except Exception as e:
            logger_manager.error(f"持久化缓存条目失败: {e}")
    
    def _remove_persisted_entry(self, key: str):
        """删除持久化的缓存条目"""
        try:
            file_path = os.path.join(self.cache_dir, f"{key}.pkl")
            
            if os.path.exists(file_path):
                os.remove(file_path)
                
        except Exception as e:
            logger_manager.error(f"删除持久化缓存条目失败: {e}")
    
    def _clear_persisted_cache(self):
        """清空持久化缓存"""
        try:
            if not os.path.exists(self.cache_dir):
                return
            
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(self.cache_dir, filename)
                    os.remove(file_path)
                    
        except Exception as e:
            logger_manager.error(f"清空持久化缓存失败: {e}")
    
    def __del__(self):
        """析构函数"""
        try:
            self.stop_cleanup_thread()
        except Exception:
            # 在Python关闭时，某些模块可能已经被清理，忽略错误
            pass


# 全局预测缓存实例
prediction_cache = PredictionCache()


if __name__ == "__main__":
    # 测试预测缓存功能
    print("💾 测试预测缓存功能...")
    
    try:
        from .prediction_engine import PredictionRequest, PredictionResult, PredictionMode, PredictionStatus
        import numpy as np
        
        # 创建测试缓存
        cache = PredictionCache(
            strategy=CacheStrategy.LRU,
            max_size=100,
            enable_persistence=False
        )
        
        # 创建测试请求和结果
        test_request = PredictionRequest(
            request_id="test_001",
            model_name="test_model",
            input_data=np.random.random((10, 5)),
            mode=PredictionMode.SINGLE
        )
        
        test_result = PredictionResult(
            request_id="test_001",
            status=PredictionStatus.COMPLETED,
            predictions=np.random.random((10, 2)),
            confidence_scores=np.random.random(10),
            execution_time=1.5
        )
        
        # 测试缓存添加
        cache.put(test_request, test_result)
        print("✅ 缓存添加成功")
        
        # 测试缓存获取
        cached_result = cache.get(test_request)
        if cached_result:
            print("✅ 缓存获取成功")
        else:
            print("❌ 缓存获取失败")
        
        # 测试缓存统计
        stats = cache.get_stats()
        print(f"✅ 缓存统计获取成功，命中率: {stats['hit_rate']:.2f}")
        
        # 测试缓存移除
        if cache.remove(test_request):
            print("✅ 缓存移除成功")
        
        # 测试缓存清空
        cache.clear()
        print("✅ 缓存清空成功")
        
        print("✅ 预测缓存功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("预测缓存功能测试完成")
