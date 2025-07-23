#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
交互式可视化模块
Interactive Visualization Module

提供交互式图表、实时数据可视化、多维数据展示等功能。
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
import base64
from io import BytesIO

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException


class ChartType(Enum):
    """图表类型枚举"""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BOX = "box"
    VIOLIN = "violin"
    PIE = "pie"
    AREA = "area"
    CANDLESTICK = "candlestick"
    SURFACE = "surface"
    TREEMAP = "treemap"


@dataclass
class VisualizationConfig:
    """可视化配置"""
    chart_type: ChartType
    title: str = ""
    width: int = 800
    height: int = 600
    theme: str = "plotly"  # "plotly", "seaborn", "matplotlib"
    interactive: bool = True
    animation: bool = False
    export_format: str = "html"  # "html", "png", "svg", "pdf"
    color_scheme: str = "viridis"
    show_legend: bool = True
    show_grid: bool = True


@dataclass
class ChartData:
    """图表数据"""
    x: Union[List, np.ndarray, pd.Series]
    y: Union[List, np.ndarray, pd.Series]
    z: Optional[Union[List, np.ndarray, pd.Series]] = None
    labels: Optional[List[str]] = None
    colors: Optional[List[str]] = None
    sizes: Optional[Union[List, np.ndarray]] = None
    text: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PlotlyVisualizer:
    """Plotly可视化器"""
    
    def __init__(self):
        """初始化Plotly可视化器"""
        if not PLOTLY_AVAILABLE:
            raise DeepLearningException("Plotly未安装，无法使用交互式可视化")
        
        self.figures = {}
        logger_manager.info("Plotly可视化器初始化完成")
    
    def create_line_chart(self, data: ChartData, config: VisualizationConfig):
        """创建折线图"""
        try:
            if not PLOTLY_AVAILABLE:
                return None
            fig = go.Figure()
            
            if isinstance(data.y[0], (list, np.ndarray)):
                # 多条线
                for i, y_series in enumerate(data.y):
                    name = data.labels[i] if data.labels and i < len(data.labels) else f"Series {i+1}"
                    fig.add_trace(go.Scatter(
                        x=data.x,
                        y=y_series,
                        mode='lines+markers',
                        name=name,
                        line=dict(width=2),
                        marker=dict(size=6)
                    ))
            else:
                # 单条线
                fig.add_trace(go.Scatter(
                    x=data.x,
                    y=data.y,
                    mode='lines+markers',
                    name=data.labels[0] if data.labels else "Data",
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
            
            self._apply_layout(fig, config)
            return fig
            
        except Exception as e:
            logger_manager.error(f"创建折线图失败: {e}")
            return None
    
    def create_bar_chart(self, data: ChartData, config: VisualizationConfig):
        """创建柱状图"""
        try:
            if not PLOTLY_AVAILABLE:
                return None
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=data.x,
                y=data.y,
                name=data.labels[0] if data.labels else "Data",
                marker=dict(
                    color=data.colors if data.colors else px.colors.qualitative.Set1,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                text=data.text,
                textposition='auto'
            ))
            
            self._apply_layout(fig, config)
            return fig
            
        except Exception as e:
            logger_manager.error(f"创建柱状图失败: {e}")
            return None
    
    def create_scatter_plot(self, data: ChartData, config: VisualizationConfig):
        """创建散点图"""
        try:
            if not PLOTLY_AVAILABLE:
                return None
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data.x,
                y=data.y,
                mode='markers',
                name=data.labels[0] if data.labels else "Data",
                marker=dict(
                    size=data.sizes if data.sizes is not None else 8,
                    color=data.colors if data.colors else data.y,
                    colorscale=config.color_scheme,
                    showscale=True,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                text=data.text,
                hovertemplate='<b>%{text}</b><br>X: %{x}<br>Y: %{y}<extra></extra>'
            ))
            
            self._apply_layout(fig, config)
            return fig
            
        except Exception as e:
            logger_manager.error(f"创建散点图失败: {e}")
            return None
    
    def create_heatmap(self, data: ChartData, config: VisualizationConfig):
        """创建热力图"""
        try:
            if not PLOTLY_AVAILABLE:
                return None
            fig = go.Figure()
            
            # 假设z是二维数组
            fig.add_trace(go.Heatmap(
                z=data.z,
                x=data.x if data.x is not None else None,
                y=data.y if data.y is not None else None,
                colorscale=config.color_scheme,
                showscale=True,
                hoverongaps=False
            ))
            
            self._apply_layout(fig, config)
            return fig
            
        except Exception as e:
            logger_manager.error(f"创建热力图失败: {e}")
            return None

    def create_3d_surface(self, data: ChartData, config: VisualizationConfig):
        """创建3D表面图"""
        try:
            if not PLOTLY_AVAILABLE:
                return None
            fig = go.Figure()
            
            fig.add_trace(go.Surface(
                z=data.z,
                x=data.x if data.x is not None else None,
                y=data.y if data.y is not None else None,
                colorscale=config.color_scheme,
                showscale=True
            ))
            
            fig.update_layout(
                title=config.title,
                scene=dict(
                    xaxis_title='X Axis',
                    yaxis_title='Y Axis',
                    zaxis_title='Z Axis'
                ),
                width=config.width,
                height=config.height
            )
            
            return fig
            
        except Exception as e:
            logger_manager.error(f"创建3D表面图失败: {e}")
            return None

    def create_pie_chart(self, data: ChartData, config: VisualizationConfig):
        """创建饼图"""
        try:
            if not PLOTLY_AVAILABLE:
                return None
            fig = go.Figure()
            
            fig.add_trace(go.Pie(
                labels=data.labels if data.labels else data.x,
                values=data.y,
                hole=0.3,  # 甜甜圈图
                textinfo='label+percent',
                textposition='auto',
                marker=dict(
                    colors=data.colors if data.colors else px.colors.qualitative.Set1,
                    line=dict(color='#000000', width=2)
                )
            ))
            
            self._apply_layout(fig, config)
            return fig
            
        except Exception as e:
            logger_manager.error(f"创建饼图失败: {e}")
            return None

    def _apply_layout(self, fig, config: VisualizationConfig):
        """应用布局配置"""
        try:
            fig.update_layout(
                title=dict(
                    text=config.title,
                    x=0.5,
                    font=dict(size=16, family="Arial, sans-serif")
                ),
                width=config.width,
                height=config.height,
                showlegend=config.show_legend,
                template=config.theme,
                hovermode='closest',
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            if config.show_grid:
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            
        except Exception as e:
            logger_manager.error(f"应用布局配置失败: {e}")


class MatplotlibVisualizer:
    """Matplotlib可视化器"""
    
    def __init__(self):
        """初始化Matplotlib可视化器"""
        if not MATPLOTLIB_AVAILABLE:
            raise DeepLearningException("Matplotlib未安装，无法使用静态可视化")
        
        plt.style.use('seaborn-v0_8' if SEABORN_AVAILABLE else 'default')
        self.figures = {}
        logger_manager.info("Matplotlib可视化器初始化完成")
    
    def create_line_chart(self, data: ChartData, config: VisualizationConfig) -> plt.Figure:
        """创建折线图"""
        try:
            fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))
            
            if isinstance(data.y[0], (list, np.ndarray)):
                # 多条线
                for i, y_series in enumerate(data.y):
                    label = data.labels[i] if data.labels and i < len(data.labels) else f"Series {i+1}"
                    ax.plot(data.x, y_series, label=label, marker='o', linewidth=2, markersize=4)
            else:
                # 单条线
                label = data.labels[0] if data.labels else "Data"
                ax.plot(data.x, data.y, label=label, marker='o', linewidth=2, markersize=4)
            
            self._apply_matplotlib_layout(ax, config)
            return fig
            
        except Exception as e:
            logger_manager.error(f"创建Matplotlib折线图失败: {e}")
            return plt.figure()
    
    def create_bar_chart(self, data: ChartData, config: VisualizationConfig) -> plt.Figure:
        """创建柱状图"""
        try:
            fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))
            
            bars = ax.bar(data.x, data.y, color=data.colors if data.colors else 'skyblue', 
                         edgecolor='black', linewidth=1)
            
            # 添加数值标签
            if data.text:
                for bar, text in zip(bars, data.text):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           text, ha='center', va='bottom')
            
            self._apply_matplotlib_layout(ax, config)
            return fig
            
        except Exception as e:
            logger_manager.error(f"创建Matplotlib柱状图失败: {e}")
            return plt.figure()
    
    def _apply_matplotlib_layout(self, ax, config: VisualizationConfig):
        """应用Matplotlib布局配置"""
        try:
            ax.set_title(config.title, fontsize=14, fontweight='bold')
            
            if config.show_legend:
                ax.legend()
            
            if config.show_grid:
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
        except Exception as e:
            logger_manager.error(f"应用Matplotlib布局配置失败: {e}")


class InteractiveVisualizer:
    """交互式可视化器"""
    
    def __init__(self, preferred_backend: str = "plotly"):
        """
        初始化交互式可视化器
        
        Args:
            preferred_backend: 首选后端 ("plotly", "matplotlib")
        """
        self.preferred_backend = preferred_backend
        self.plotly_viz = None
        self.matplotlib_viz = None
        
        # 初始化可用的可视化器
        if PLOTLY_AVAILABLE and preferred_backend == "plotly":
            self.plotly_viz = PlotlyVisualizer()
            self.active_backend = "plotly"
        elif MATPLOTLIB_AVAILABLE:
            self.matplotlib_viz = MatplotlibVisualizer()
            self.active_backend = "matplotlib"
        else:
            raise DeepLearningException("没有可用的可视化后端")
        
        self.charts = {}
        self.animations = {}
        
        logger_manager.info(f"交互式可视化器初始化完成，使用后端: {self.active_backend}")
    
    def create_chart(self, chart_id: str, data: ChartData, 
                    config: VisualizationConfig) -> Any:
        """
        创建图表
        
        Args:
            chart_id: 图表ID
            data: 图表数据
            config: 可视化配置
            
        Returns:
            图表对象
        """
        try:
            if self.active_backend == "plotly" and self.plotly_viz:
                fig = self._create_plotly_chart(data, config)
            else:
                fig = self._create_matplotlib_chart(data, config)
            
            self.charts[chart_id] = {
                'figure': fig,
                'config': config,
                'data': data,
                'created_time': time.time()
            }
            
            logger_manager.info(f"创建图表: {chart_id}, 类型: {config.chart_type.value}")
            return fig
            
        except Exception as e:
            logger_manager.error(f"创建图表失败: {e}")
            return None
    
    def _create_plotly_chart(self, data: ChartData, config: VisualizationConfig):
        """创建Plotly图表"""
        if config.chart_type == ChartType.LINE:
            return self.plotly_viz.create_line_chart(data, config)
        elif config.chart_type == ChartType.BAR:
            return self.plotly_viz.create_bar_chart(data, config)
        elif config.chart_type == ChartType.SCATTER:
            return self.plotly_viz.create_scatter_plot(data, config)
        elif config.chart_type == ChartType.HEATMAP:
            return self.plotly_viz.create_heatmap(data, config)
        elif config.chart_type == ChartType.PIE:
            return self.plotly_viz.create_pie_chart(data, config)
        elif config.chart_type == ChartType.SURFACE:
            return self.plotly_viz.create_3d_surface(data, config)
        else:
            logger_manager.warning(f"不支持的Plotly图表类型: {config.chart_type}")
            return self.plotly_viz.create_line_chart(data, config)
    
    def _create_matplotlib_chart(self, data: ChartData, config: VisualizationConfig):
        """创建Matplotlib图表"""
        if config.chart_type == ChartType.LINE:
            return self.matplotlib_viz.create_line_chart(data, config)
        elif config.chart_type == ChartType.BAR:
            return self.matplotlib_viz.create_bar_chart(data, config)
        else:
            logger_manager.warning(f"不支持的Matplotlib图表类型: {config.chart_type}")
            return self.matplotlib_viz.create_line_chart(data, config)
    
    def update_chart(self, chart_id: str, new_data: ChartData):
        """更新图表数据"""
        try:
            if chart_id not in self.charts:
                logger_manager.warning(f"图表不存在: {chart_id}")
                return
            
            chart_info = self.charts[chart_id]
            config = chart_info['config']
            
            # 重新创建图表
            new_fig = self.create_chart(chart_id, new_data, config)
            
            logger_manager.debug(f"更新图表: {chart_id}")
            return new_fig
            
        except Exception as e:
            logger_manager.error(f"更新图表失败: {e}")
    
    def export_chart(self, chart_id: str, file_path: str, format: str = "html"):
        """导出图表"""
        try:
            if chart_id not in self.charts:
                logger_manager.warning(f"图表不存在: {chart_id}")
                return False
            
            fig = self.charts[chart_id]['figure']
            
            if self.active_backend == "plotly":
                if format == "html":
                    fig.write_html(file_path)
                elif format == "png":
                    fig.write_image(file_path)
                elif format == "svg":
                    fig.write_image(file_path)
                else:
                    logger_manager.warning(f"不支持的导出格式: {format}")
                    return False
            else:
                # Matplotlib导出
                if format in ["png", "svg", "pdf"]:
                    fig.savefig(file_path, format=format, dpi=300, bbox_inches='tight')
                else:
                    logger_manager.warning(f"Matplotlib不支持的导出格式: {format}")
                    return False
            
            logger_manager.info(f"图表已导出: {file_path}")
            return True
            
        except Exception as e:
            logger_manager.error(f"导出图表失败: {e}")
            return False
    
    def create_animation(self, chart_id: str, data_frames: List[ChartData],
                        config: VisualizationConfig, interval: int = 500):
        """创建动画图表"""
        try:
            if self.active_backend != "plotly":
                logger_manager.warning("动画功能仅支持Plotly后端")
                return None
            
            # 创建动画帧
            frames = []
            for i, frame_data in enumerate(data_frames):
                frame_fig = self._create_plotly_chart(frame_data, config)
                frames.append(go.Frame(data=frame_fig.data, name=str(i)))
            
            # 创建基础图表
            base_fig = self._create_plotly_chart(data_frames[0], config)
            base_fig.frames = frames
            
            # 添加播放控件
            base_fig.update_layout(
                updatemenus=[{
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": interval, "redraw": True},
                                          "fromcurrent": True, "transition": {"duration": 300}}],
                            "label": "Play",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                            "mode": "immediate", "transition": {"duration": 0}}],
                            "label": "Pause",
                            "method": "animate"
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }]
            )
            
            self.animations[chart_id] = base_fig
            
            logger_manager.info(f"创建动画图表: {chart_id}, 帧数: {len(data_frames)}")
            return base_fig
            
        except Exception as e:
            logger_manager.error(f"创建动画图表失败: {e}")
            return None
    
    def get_chart_info(self, chart_id: str) -> Dict[str, Any]:
        """获取图表信息"""
        if chart_id in self.charts:
            chart_info = self.charts[chart_id]
            return {
                'chart_id': chart_id,
                'chart_type': chart_info['config'].chart_type.value,
                'created_time': chart_info['created_time'],
                'backend': self.active_backend,
                'interactive': chart_info['config'].interactive
            }
        return {}
    
    def list_charts(self) -> List[str]:
        """列出所有图表ID"""
        return list(self.charts.keys())
    
    def clear_charts(self):
        """清除所有图表"""
        self.charts.clear()
        self.animations.clear()
        logger_manager.info("已清除所有图表")


# 全局交互式可视化器实例
interactive_visualizer = InteractiveVisualizer()


if __name__ == "__main__":
    # 测试交互式可视化功能
    print("📊 测试交互式可视化功能...")
    
    try:
        # 创建测试数据
        x_data = list(range(10))
        y_data = [i**2 for i in x_data]
        
        test_data = ChartData(
            x=x_data,
            y=y_data,
            labels=["平方函数"]
        )
        
        test_config = VisualizationConfig(
            chart_type=ChartType.LINE,
            title="测试折线图",
            width=800,
            height=600
        )
        
        # 创建图表
        visualizer = InteractiveVisualizer()
        fig = visualizer.create_chart("test_chart", test_data, test_config)
        
        if fig:
            print("✅ 折线图创建成功")
            
            # 测试导出
            if visualizer.export_chart("test_chart", "test_chart.html", "html"):
                print("✅ 图表导出成功")
        
        # 测试柱状图
        bar_config = VisualizationConfig(
            chart_type=ChartType.BAR,
            title="测试柱状图"
        )
        
        bar_fig = visualizer.create_chart("test_bar", test_data, bar_config)
        if bar_fig:
            print("✅ 柱状图创建成功")
        
        # 列出所有图表
        charts = visualizer.list_charts()
        print(f"✅ 创建的图表: {charts}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    print("交互式可视化功能测试完成")
