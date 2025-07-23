#!/usr/bin/env python3
"""
资源监控器
监控系统资源使用情况，包括CPU、内存、GPU等
"""

import psutil
import time
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import os
from collections import deque

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from core_modules import logger_manager


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化资源监控器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 监控参数
        self.monitor_interval = self.config.get('monitor_interval', 1.0)  # 监控间隔（秒）
        self.history_size = self.config.get('history_size', 1000)  # 历史记录大小
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'cpu': 80.0,      # CPU使用率阈值
            'memory': 85.0,   # 内存使用率阈值
            'disk': 90.0,     # 磁盘使用率阈值
            'gpu': 90.0       # GPU使用率阈值
        })
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None
        
        # 历史数据
        self.cpu_history = deque(maxlen=self.history_size)
        self.memory_history = deque(maxlen=self.history_size)
        self.disk_history = deque(maxlen=self.history_size)
        self.gpu_history = deque(maxlen=self.history_size)
        self.network_history = deque(maxlen=self.history_size)
        
        # 警报记录
        self.alerts = []
        
        logger_manager.info("资源监控器初始化完成")
    
    def start_monitoring(self) -> bool:
        """开始监控"""
        try:
            if self.is_monitoring:
                logger_manager.warning("资源监控已在运行")
                return True
            
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            
            logger_manager.info("资源监控已启动")
            return True
            
        except Exception as e:
            logger_manager.error(f"启动资源监控失败: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """停止监控"""
        try:
            if not self.is_monitoring:
                logger_manager.warning("资源监控未在运行")
                return True
            
            self.is_monitoring = False
            
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
            
            logger_manager.info("资源监控已停止")
            return True
            
        except Exception as e:
            logger_manager.error(f"停止资源监控失败: {e}")
            return False
    
    def _monitor_loop(self):
        """监控循环"""
        logger_manager.info("资源监控循环开始")
        
        while self.is_monitoring:
            try:
                # 收集资源使用情况
                timestamp = datetime.now()
                
                # CPU监控
                cpu_usage = self._get_cpu_usage()
                self.cpu_history.append((timestamp, cpu_usage))
                
                # 内存监控
                memory_usage = self._get_memory_usage()
                self.memory_history.append((timestamp, memory_usage))
                
                # 磁盘监控
                disk_usage = self._get_disk_usage()
                self.disk_history.append((timestamp, disk_usage))
                
                # GPU监控
                if GPU_AVAILABLE:
                    gpu_usage = self._get_gpu_usage()
                    self.gpu_history.append((timestamp, gpu_usage))
                
                # 网络监控
                network_usage = self._get_network_usage()
                self.network_history.append((timestamp, network_usage))
                
                # 检查警报
                self._check_alerts(timestamp, cpu_usage, memory_usage, disk_usage)
                
                # 等待下次监控
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger_manager.error(f"资源监控循环错误: {e}")
                time.sleep(self.monitor_interval)
        
        logger_manager.info("资源监控循环结束")
    
    def _get_cpu_usage(self) -> Dict[str, float]:
        """获取CPU使用情况"""
        try:
            # 总体CPU使用率
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # 每个核心的使用率
            cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
            
            # CPU频率
            cpu_freq = psutil.cpu_freq()
            
            # 负载平均值
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
            
            return {
                'total': cpu_percent,
                'per_core': cpu_per_core,
                'frequency': cpu_freq.current if cpu_freq else 0,
                'load_avg_1m': load_avg[0],
                'load_avg_5m': load_avg[1],
                'load_avg_15m': load_avg[2]
            }
            
        except Exception as e:
            logger_manager.error(f"获取CPU使用情况失败: {e}")
            return {}
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        try:
            # 虚拟内存
            virtual_memory = psutil.virtual_memory()
            
            # 交换内存
            swap_memory = psutil.swap_memory()
            
            return {
                'virtual_total': virtual_memory.total / (1024**3),  # GB
                'virtual_used': virtual_memory.used / (1024**3),
                'virtual_available': virtual_memory.available / (1024**3),
                'virtual_percent': virtual_memory.percent,
                'swap_total': swap_memory.total / (1024**3),
                'swap_used': swap_memory.used / (1024**3),
                'swap_percent': swap_memory.percent
            }
            
        except Exception as e:
            logger_manager.error(f"获取内存使用情况失败: {e}")
            return {}
    
    def _get_disk_usage(self) -> Dict[str, Any]:
        """获取磁盘使用情况"""
        try:
            disk_usage = {}
            
            # 获取所有磁盘分区
            partitions = psutil.disk_partitions()
            
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    
                    disk_usage[partition.device] = {
                        'mountpoint': partition.mountpoint,
                        'fstype': partition.fstype,
                        'total': usage.total / (1024**3),  # GB
                        'used': usage.used / (1024**3),
                        'free': usage.free / (1024**3),
                        'percent': (usage.used / usage.total) * 100 if usage.total > 0 else 0
                    }
                    
                except PermissionError:
                    # 某些分区可能没有访问权限
                    continue
            
            # 磁盘I/O统计
            disk_io = psutil.disk_io_counters()
            if disk_io:
                disk_usage['io'] = {
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes,
                    'read_count': disk_io.read_count,
                    'write_count': disk_io.write_count
                }
            
            return disk_usage
            
        except Exception as e:
            logger_manager.error(f"获取磁盘使用情况失败: {e}")
            return {}
    
    def _get_gpu_usage(self) -> List[Dict[str, Any]]:
        """获取GPU使用情况"""
        try:
            if not GPU_AVAILABLE:
                return []
            
            gpus = GPUtil.getGPUs()
            gpu_info = []
            
            for gpu in gpus:
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,  # 转换为百分比
                    'memory_used': gpu.memoryUsed,  # MB
                    'memory_total': gpu.memoryTotal,  # MB
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100 if gpu.memoryTotal > 0 else 0,
                    'temperature': gpu.temperature
                })
            
            return gpu_info
            
        except Exception as e:
            logger_manager.error(f"获取GPU使用情况失败: {e}")
            return []
    
    def _get_network_usage(self) -> Dict[str, Any]:
        """获取网络使用情况"""
        try:
            # 网络I/O统计
            network_io = psutil.net_io_counters()
            
            # 网络连接
            connections = psutil.net_connections()
            
            return {
                'bytes_sent': network_io.bytes_sent if network_io else 0,
                'bytes_recv': network_io.bytes_recv if network_io else 0,
                'packets_sent': network_io.packets_sent if network_io else 0,
                'packets_recv': network_io.packets_recv if network_io else 0,
                'connections_count': len(connections)
            }
            
        except Exception as e:
            logger_manager.error(f"获取网络使用情况失败: {e}")
            return {}
    
    def _check_alerts(self, timestamp: datetime, cpu_usage: Dict, memory_usage: Dict, disk_usage: Dict):
        """检查警报条件"""
        try:
            # CPU警报
            if cpu_usage.get('total', 0) > self.alert_thresholds['cpu']:
                self._add_alert(timestamp, 'cpu', cpu_usage['total'], self.alert_thresholds['cpu'])
            
            # 内存警报
            if memory_usage.get('virtual_percent', 0) > self.alert_thresholds['memory']:
                self._add_alert(timestamp, 'memory', memory_usage['virtual_percent'], self.alert_thresholds['memory'])
            
            # 磁盘警报
            for device, usage in disk_usage.items():
                if isinstance(usage, dict) and usage.get('percent', 0) > self.alert_thresholds['disk']:
                    self._add_alert(timestamp, f'disk_{device}', usage['percent'], self.alert_thresholds['disk'])
            
            # GPU警报
            if GPU_AVAILABLE and self.gpu_history:
                _, gpu_usage_data = self.gpu_history[-1]
                for gpu in gpu_usage_data:
                    if gpu.get('load', 0) > self.alert_thresholds['gpu']:
                        self._add_alert(timestamp, f'gpu_{gpu["id"]}', gpu['load'], self.alert_thresholds['gpu'])
            
        except Exception as e:
            logger_manager.error(f"检查警报失败: {e}")
    
    def _add_alert(self, timestamp: datetime, resource: str, value: float, threshold: float):
        """添加警报"""
        alert = {
            'timestamp': timestamp,
            'resource': resource,
            'value': value,
            'threshold': threshold,
            'message': f"{resource} 使用率 {value:.1f}% 超过阈值 {threshold:.1f}%"
        }
        
        self.alerts.append(alert)
        logger_manager.warning(alert['message'])
        
        # 限制警报历史大小
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-500:]
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前资源状态"""
        try:
            status = {
                'timestamp': datetime.now(),
                'cpu': self._get_cpu_usage(),
                'memory': self._get_memory_usage(),
                'disk': self._get_disk_usage(),
                'network': self._get_network_usage(),
                'monitoring': self.is_monitoring
            }
            
            if GPU_AVAILABLE:
                status['gpu'] = self._get_gpu_usage()
            
            return status
            
        except Exception as e:
            logger_manager.error(f"获取当前状态失败: {e}")
            return {}
    
    def get_history_summary(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """获取历史数据摘要"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
            
            summary = {
                'duration_minutes': duration_minutes,
                'cpu': self._summarize_history(self.cpu_history, cutoff_time, 'total'),
                'memory': self._summarize_history(self.memory_history, cutoff_time, 'virtual_percent'),
                'alerts_count': len([a for a in self.alerts if a['timestamp'] > cutoff_time])
            }
            
            return summary
            
        except Exception as e:
            logger_manager.error(f"获取历史摘要失败: {e}")
            return {}
    
    def _summarize_history(self, history: deque, cutoff_time: datetime, key: str) -> Dict[str, float]:
        """汇总历史数据"""
        try:
            values = []
            
            for timestamp, data in history:
                if timestamp > cutoff_time and isinstance(data, dict) and key in data:
                    values.append(data[key])
            
            if not values:
                return {}
            
            return {
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'count': len(values)
            }
            
        except Exception as e:
            logger_manager.error(f"汇总历史数据失败: {e}")
            return {}
    
    def save_report(self, file_path: str = None) -> str:
        """保存监控报告"""
        try:
            if file_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"resource_report_{timestamp}.json"
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'current_status': self.get_current_status(),
                'history_summary': self.get_history_summary(),
                'recent_alerts': self.alerts[-50:] if self.alerts else [],
                'configuration': {
                    'monitor_interval': self.monitor_interval,
                    'alert_thresholds': self.alert_thresholds
                }
            }
            
            # 序列化datetime对象
            def json_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=json_serializer, ensure_ascii=False)
            
            logger_manager.info(f"监控报告已保存到 {file_path}")
            return file_path
            
        except Exception as e:
            logger_manager.error(f"保存监控报告失败: {e}")
            return ""
    
    def clear_history(self):
        """清除历史数据"""
        try:
            self.cpu_history.clear()
            self.memory_history.clear()
            self.disk_history.clear()
            self.gpu_history.clear()
            self.network_history.clear()
            self.alerts.clear()
            
            logger_manager.info("历史数据已清除")
            
        except Exception as e:
            logger_manager.error(f"清除历史数据失败: {e}")


# 全局资源监控器实例
resource_monitor = ResourceMonitor()
