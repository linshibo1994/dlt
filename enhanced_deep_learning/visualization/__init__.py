"""
可视化模块
Visualization Module

提供交互式可视化、实时监控面板、图表生成等功能。
"""

from .interactive_visualizer import (
    InteractiveVisualizer, ChartType, VisualizationConfig,
    interactive_visualizer
)
from .dashboard import (
    Dashboard, DashboardWidget, WidgetType, MetricWidget,
    ChartWidget, dashboard_manager
)
from .report_generator import (
    ReportGenerator, ReportTemplate, ReportFormat,
    report_generator
)

__all__ = [
    # 交互式可视化
    'InteractiveVisualizer',
    'ChartType',
    'VisualizationConfig',
    'interactive_visualizer',
    
    # 仪表板
    'Dashboard',
    'DashboardWidget',
    'WidgetType',
    'MetricWidget',
    'ChartWidget',
    'dashboard_manager',
    
    # 报告生成
    'ReportGenerator',
    'ReportTemplate',
    'ReportFormat',
    'report_generator'
]
