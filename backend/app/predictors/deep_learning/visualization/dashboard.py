#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
仪表板模块
Dashboard Module

提供实时监控面板、指标展示、告警系统等功能。
"""

import os
import time
import json
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import queue
import uuid

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException
from .interactive_visualizer import InteractiveVisualizer, ChartData, VisualizationConfig, ChartType


class WidgetType(Enum):
    """组件类型枚举"""
    METRIC = "metric"
    CHART = "chart"
    TABLE = "table"
    TEXT = "text"
    GAUGE = "gauge"
    PROGRESS = "progress"
    ALERT = "alert"


class AlertLevel(Enum):
    """告警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """告警信息"""
    id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    source: str
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Metric:
    """指标数据"""
    name: str
    value: Union[int, float, str]
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    trend: Optional[str] = None  # "up", "down", "stable"
    target: Optional[Union[int, float]] = None
    threshold_warning: Optional[Union[int, float]] = None
    threshold_critical: Optional[Union[int, float]] = None


class DashboardWidget:
    """仪表板组件基类"""
    
    def __init__(self, widget_id: str, widget_type: WidgetType, title: str = ""):
        """
        初始化仪表板组件
        
        Args:
            widget_id: 组件ID
            widget_type: 组件类型
            title: 组件标题
        """
        self.widget_id = widget_id
        self.widget_type = widget_type
        self.title = title
        self.created_time = datetime.now()
        self.last_updated = datetime.now()
        self.data = {}
        self.config = {}
        
        logger_manager.debug(f"创建仪表板组件: {widget_id}, 类型: {widget_type.value}")
    
    def update_data(self, data: Any):
        """更新组件数据"""
        self.data = data
        self.last_updated = datetime.now()
        logger_manager.debug(f"更新组件数据: {self.widget_id}")
    
    def get_data(self) -> Any:
        """获取组件数据"""
        return self.data
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'widget_id': self.widget_id,
            'widget_type': self.widget_type.value,
            'title': self.title,
            'created_time': self.created_time.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'data': self.data,
            'config': self.config
        }


class MetricWidget(DashboardWidget):
    """指标组件"""
    
    def __init__(self, widget_id: str, title: str = ""):
        """初始化指标组件"""
        super().__init__(widget_id, WidgetType.METRIC, title)
        self.metrics = {}
    
    def add_metric(self, metric: Metric):
        """添加指标"""
        self.metrics[metric.name] = metric
        self.last_updated = datetime.now()
        logger_manager.debug(f"添加指标: {metric.name} = {metric.value}")
    
    def update_metric(self, name: str, value: Union[int, float, str], 
                     unit: str = "", trend: str = None):
        """更新指标"""
        if name in self.metrics:
            metric = self.metrics[name]
            metric.value = value
            metric.unit = unit
            metric.trend = trend
            metric.timestamp = datetime.now()
        else:
            metric = Metric(name=name, value=value, unit=unit, trend=trend)
            self.metrics[name] = metric
        
        self.last_updated = datetime.now()
        logger_manager.debug(f"更新指标: {name} = {value}")
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """获取指标"""
        return self.metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, Metric]:
        """获取所有指标"""
        return self.metrics.copy()
    
    def check_thresholds(self) -> List[Alert]:
        """检查阈值告警"""
        alerts = []
        
        for metric in self.metrics.values():
            if isinstance(metric.value, (int, float)):
                # 检查临界阈值
                if (metric.threshold_critical is not None and 
                    metric.value >= metric.threshold_critical):
                    alert = Alert(
                        id=str(uuid.uuid4()),
                        level=AlertLevel.CRITICAL,
                        title=f"指标临界告警: {metric.name}",
                        message=f"{metric.name} 值 {metric.value} 超过临界阈值 {metric.threshold_critical}",
                        timestamp=datetime.now(),
                        source=self.widget_id
                    )
                    alerts.append(alert)
                
                # 检查警告阈值
                elif (metric.threshold_warning is not None and 
                      metric.value >= metric.threshold_warning):
                    alert = Alert(
                        id=str(uuid.uuid4()),
                        level=AlertLevel.WARNING,
                        title=f"指标警告: {metric.name}",
                        message=f"{metric.name} 值 {metric.value} 超过警告阈值 {metric.threshold_warning}",
                        timestamp=datetime.now(),
                        source=self.widget_id
                    )
                    alerts.append(alert)
        
        return alerts


class ChartWidget(DashboardWidget):
    """图表组件"""
    
    def __init__(self, widget_id: str, title: str = "", chart_type: ChartType = ChartType.LINE):
        """初始化图表组件"""
        super().__init__(widget_id, WidgetType.CHART, title)
        self.chart_type = chart_type
        self.visualizer = InteractiveVisualizer()
        self.chart_data = None
        self.chart_config = None
    
    def update_chart(self, data: ChartData, config: VisualizationConfig = None):
        """更新图表"""
        self.chart_data = data
        
        if config is None:
            config = VisualizationConfig(
                chart_type=self.chart_type,
                title=self.title,
                width=400,
                height=300
            )
        
        self.chart_config = config
        
        # 创建或更新图表
        fig = self.visualizer.create_chart(self.widget_id, data, config)
        self.data = fig
        self.last_updated = datetime.now()
        
        logger_manager.debug(f"更新图表组件: {self.widget_id}")
    
    def get_chart_html(self) -> str:
        """获取图表HTML"""
        try:
            if self.data and hasattr(self.data, 'to_html'):
                return self.data.to_html(include_plotlyjs='cdn')
            return "<div>图表数据不可用</div>"
        except Exception as e:
            logger_manager.error(f"获取图表HTML失败: {e}")
            return f"<div>图表错误: {e}</div>"


class TableWidget(DashboardWidget):
    """表格组件"""
    
    def __init__(self, widget_id: str, title: str = ""):
        """初始化表格组件"""
        super().__init__(widget_id, WidgetType.TABLE, title)
        self.columns = []
        self.rows = []
    
    def update_table(self, data: Union[pd.DataFrame, List[Dict], List[List]]):
        """更新表格数据"""
        if isinstance(data, pd.DataFrame):
            self.columns = data.columns.tolist()
            self.rows = data.values.tolist()
        elif isinstance(data, list) and data:
            if isinstance(data[0], dict):
                # 字典列表
                self.columns = list(data[0].keys())
                self.rows = [[row[col] for col in self.columns] for row in data]
            elif isinstance(data[0], list):
                # 二维列表
                self.rows = data
                if not self.columns:
                    self.columns = [f"Column {i+1}" for i in range(len(data[0]))]
        
        self.data = {'columns': self.columns, 'rows': self.rows}
        self.last_updated = datetime.now()
        
        logger_manager.debug(f"更新表格组件: {self.widget_id}, 行数: {len(self.rows)}")
    
    def get_table_html(self) -> str:
        """获取表格HTML"""
        try:
            if not self.columns or not self.rows:
                return "<div>表格数据不可用</div>"
            
            html = "<table class='table table-striped'>"
            
            # 表头
            html += "<thead><tr>"
            for col in self.columns:
                html += f"<th>{col}</th>"
            html += "</tr></thead>"
            
            # 表体
            html += "<tbody>"
            for row in self.rows:
                html += "<tr>"
                for cell in row:
                    html += f"<td>{cell}</td>"
                html += "</tr>"
            html += "</tbody>"
            
            html += "</table>"
            return html
            
        except Exception as e:
            logger_manager.error(f"获取表格HTML失败: {e}")
            return f"<div>表格错误: {e}</div>"


class AlertManager:
    """告警管理器"""
    
    def __init__(self, max_alerts: int = 1000):
        """
        初始化告警管理器
        
        Args:
            max_alerts: 最大告警数量
        """
        self.max_alerts = max_alerts
        self.alerts = []
        self.alert_handlers = {}
        self.lock = threading.RLock()
        
        logger_manager.info("告警管理器初始化完成")
    
    def add_alert(self, alert: Alert):
        """添加告警"""
        with self.lock:
            self.alerts.append(alert)
            
            # 限制告警数量
            if len(self.alerts) > self.max_alerts:
                self.alerts = self.alerts[-self.max_alerts:]
            
            # 触发告警处理器
            self._trigger_handlers(alert)
            
            logger_manager.info(f"添加告警: {alert.level.value} - {alert.title}")
    
    def resolve_alert(self, alert_id: str):
        """解决告警"""
        with self.lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    logger_manager.info(f"解决告警: {alert_id}")
                    break
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        with self.lock:
            return [alert for alert in self.alerts if not alert.resolved]
    
    def get_alerts_by_level(self, level: AlertLevel) -> List[Alert]:
        """按级别获取告警"""
        with self.lock:
            return [alert for alert in self.alerts if alert.level == level and not alert.resolved]
    
    def register_handler(self, level: AlertLevel, handler: Callable[[Alert], None]):
        """注册告警处理器"""
        if level not in self.alert_handlers:
            self.alert_handlers[level] = []
        self.alert_handlers[level].append(handler)
        
        logger_manager.info(f"注册告警处理器: {level.value}")
    
    def _trigger_handlers(self, alert: Alert):
        """触发告警处理器"""
        try:
            handlers = self.alert_handlers.get(alert.level, [])
            for handler in handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger_manager.error(f"告警处理器执行失败: {e}")
        except Exception as e:
            logger_manager.error(f"触发告警处理器失败: {e}")
    
    def clear_resolved_alerts(self):
        """清除已解决的告警"""
        with self.lock:
            original_count = len(self.alerts)
            self.alerts = [alert for alert in self.alerts if not alert.resolved]
            cleared_count = original_count - len(self.alerts)
            
            if cleared_count > 0:
                logger_manager.info(f"清除已解决告警: {cleared_count} 条")


class Dashboard:
    """仪表板"""
    
    def __init__(self, dashboard_id: str, title: str = ""):
        """
        初始化仪表板
        
        Args:
            dashboard_id: 仪表板ID
            title: 仪表板标题
        """
        self.dashboard_id = dashboard_id
        self.title = title
        self.widgets = {}
        self.layout = []
        self.alert_manager = AlertManager()
        self.auto_refresh = False
        self.refresh_interval = 5  # 秒
        self.refresh_thread = None
        self.created_time = datetime.now()
        
        logger_manager.info(f"创建仪表板: {dashboard_id}")
    
    def add_widget(self, widget: DashboardWidget, position: Tuple[int, int] = (0, 0)):
        """
        添加组件
        
        Args:
            widget: 仪表板组件
            position: 位置 (row, col)
        """
        self.widgets[widget.widget_id] = widget
        
        # 更新布局
        self.layout.append({
            'widget_id': widget.widget_id,
            'position': position,
            'size': (1, 1)  # (rows, cols)
        })
        
        logger_manager.info(f"添加组件到仪表板: {widget.widget_id}")
    
    def remove_widget(self, widget_id: str):
        """移除组件"""
        if widget_id in self.widgets:
            del self.widgets[widget_id]
            self.layout = [item for item in self.layout if item['widget_id'] != widget_id]
            logger_manager.info(f"移除组件: {widget_id}")
    
    def get_widget(self, widget_id: str) -> Optional[DashboardWidget]:
        """获取组件"""
        return self.widgets.get(widget_id)
    
    def update_widget_data(self, widget_id: str, data: Any):
        """更新组件数据"""
        widget = self.get_widget(widget_id)
        if widget:
            widget.update_data(data)
            
            # 检查指标告警
            if isinstance(widget, MetricWidget):
                alerts = widget.check_thresholds()
                for alert in alerts:
                    self.alert_manager.add_alert(alert)
    
    def start_auto_refresh(self, interval: int = 5):
        """启动自动刷新"""
        self.refresh_interval = interval
        self.auto_refresh = True
        
        if self.refresh_thread is None or not self.refresh_thread.is_alive():
            self.refresh_thread = threading.Thread(target=self._refresh_loop)
            self.refresh_thread.daemon = True
            self.refresh_thread.start()
        
        logger_manager.info(f"启动仪表板自动刷新，间隔: {interval}秒")
    
    def stop_auto_refresh(self):
        """停止自动刷新"""
        self.auto_refresh = False
        logger_manager.info("停止仪表板自动刷新")
    
    def _refresh_loop(self):
        """刷新循环"""
        while self.auto_refresh:
            try:
                # 这里可以添加自动数据更新逻辑
                time.sleep(self.refresh_interval)
            except Exception as e:
                logger_manager.error(f"仪表板刷新失败: {e}")
                time.sleep(self.refresh_interval)
    
    def generate_html(self) -> str:
        """生成仪表板HTML"""
        try:
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{self.title}</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    .widget {{ margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                    .metric-value {{ font-size: 2em; font-weight: bold; }}
                    .metric-unit {{ font-size: 0.8em; color: #666; }}
                    .alert-critical {{ background-color: #f8d7da; border-color: #f5c6cb; }}
                    .alert-warning {{ background-color: #fff3cd; border-color: #ffeaa7; }}
                </style>
            </head>
            <body>
                <div class="container-fluid">
                    <h1 class="mb-4">{self.title}</h1>
                    <div class="row">
            """
            
            # 添加组件
            for widget_id, widget in self.widgets.items():
                html += f'<div class="col-md-6 col-lg-4"><div class="widget">'
                html += f'<h5>{widget.title}</h5>'
                
                if isinstance(widget, MetricWidget):
                    html += self._generate_metric_html(widget)
                elif isinstance(widget, ChartWidget):
                    html += widget.get_chart_html()
                elif isinstance(widget, TableWidget):
                    html += widget.get_table_html()
                
                html += '</div></div>'
            
            # 添加告警区域
            active_alerts = self.alert_manager.get_active_alerts()
            if active_alerts:
                html += '<div class="col-12"><div class="widget">'
                html += '<h5>活跃告警</h5>'
                for alert in active_alerts[:5]:  # 只显示前5个
                    alert_class = f"alert-{alert.level.value}"
                    html += f'<div class="alert {alert_class}">'
                    html += f'<strong>{alert.title}</strong><br>{alert.message}'
                    html += f'<small class="text-muted"> - {alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</small>'
                    html += '</div>'
                html += '</div></div>'
            
            html += """
                    </div>
                </div>
                <script>
                    // 自动刷新页面
                    setTimeout(function() {
                        location.reload();
                    }, 30000); // 30秒刷新一次
                </script>
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            logger_manager.error(f"生成仪表板HTML失败: {e}")
            return f"<html><body><h1>仪表板错误: {e}</h1></body></html>"
    
    def _generate_metric_html(self, widget: MetricWidget) -> str:
        """生成指标HTML"""
        html = ""
        for metric in widget.get_all_metrics().values():
            trend_icon = ""
            if metric.trend == "up":
                trend_icon = "↗️"
            elif metric.trend == "down":
                trend_icon = "↘️"
            elif metric.trend == "stable":
                trend_icon = "➡️"
            
            html += f"""
            <div class="metric mb-2">
                <div class="metric-name">{metric.name} {trend_icon}</div>
                <div class="metric-value">{metric.value} <span class="metric-unit">{metric.unit}</span></div>
            </div>
            """
        
        return html
    
    def export_config(self, file_path: str):
        """导出仪表板配置"""
        try:
            config = {
                'dashboard_id': self.dashboard_id,
                'title': self.title,
                'created_time': self.created_time.isoformat(),
                'widgets': [widget.to_dict() for widget in self.widgets.values()],
                'layout': self.layout
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger_manager.info(f"仪表板配置已导出: {file_path}")
            
        except Exception as e:
            logger_manager.error(f"导出仪表板配置失败: {e}")


class DashboardManager:
    """仪表板管理器"""
    
    def __init__(self):
        """初始化仪表板管理器"""
        self.dashboards = {}
        logger_manager.info("仪表板管理器初始化完成")
    
    def create_dashboard(self, dashboard_id: str, title: str = "") -> Dashboard:
        """创建仪表板"""
        dashboard = Dashboard(dashboard_id, title)
        self.dashboards[dashboard_id] = dashboard
        
        logger_manager.info(f"创建仪表板: {dashboard_id}")
        return dashboard
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """获取仪表板"""
        return self.dashboards.get(dashboard_id)
    
    def remove_dashboard(self, dashboard_id: str):
        """移除仪表板"""
        if dashboard_id in self.dashboards:
            dashboard = self.dashboards[dashboard_id]
            dashboard.stop_auto_refresh()
            del self.dashboards[dashboard_id]
            logger_manager.info(f"移除仪表板: {dashboard_id}")
    
    def list_dashboards(self) -> List[str]:
        """列出所有仪表板"""
        return list(self.dashboards.keys())


# 全局仪表板管理器实例
dashboard_manager = DashboardManager()


if __name__ == "__main__":
    # 测试仪表板功能
    print("📊 测试仪表板功能...")
    
    try:
        # 创建仪表板
        dashboard = dashboard_manager.create_dashboard("test_dashboard", "测试仪表板")
        
        # 创建指标组件
        metric_widget = MetricWidget("metrics", "系统指标")
        metric_widget.update_metric("CPU使用率", 75.5, "%", "up")
        metric_widget.update_metric("内存使用率", 60.2, "%", "stable")
        metric_widget.update_metric("磁盘使用率", 45.8, "%", "down")
        
        dashboard.add_widget(metric_widget, (0, 0))
        
        # 创建图表组件
        chart_widget = ChartWidget("performance_chart", "性能趋势", ChartType.LINE)
        
        from .interactive_visualizer import ChartData, VisualizationConfig
        chart_data = ChartData(
            x=list(range(10)),
            y=[i*2 + np.random.random() for i in range(10)],
            labels=["性能指标"]
        )
        chart_config = VisualizationConfig(
            chart_type=ChartType.LINE,
            title="性能趋势",
            width=400,
            height=300
        )
        chart_widget.update_chart(chart_data, chart_config)
        
        dashboard.add_widget(chart_widget, (0, 1))
        
        # 生成HTML
        html = dashboard.generate_html()
        
        if html and len(html) > 100:
            print("✅ 仪表板HTML生成成功")
        
        # 测试告警
        alert = Alert(
            id="test_alert",
            level=AlertLevel.WARNING,
            title="测试告警",
            message="这是一个测试告警",
            timestamp=datetime.now(),
            source="test"
        )
        
        dashboard.alert_manager.add_alert(alert)
        active_alerts = dashboard.alert_manager.get_active_alerts()
        
        if active_alerts:
            print(f"✅ 告警系统工作正常，活跃告警: {len(active_alerts)}")
        
        print("✅ 仪表板功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("仪表板功能测试完成")
