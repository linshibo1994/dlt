#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可视化工具模块
Visualization Utils Module

提供数据可视化、模型性能可视化和预测结果可视化功能。
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from core_modules import logger_manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')


class VisualizationUtils:
    """可视化工具类"""
    
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        logger_manager.info("可视化工具初始化完成")
    
    def plot_prediction_results(self, predictions: List[Dict[str, Any]], 
                               save_path: Optional[str] = None) -> None:
        """可视化预测结果"""
        try:
            if not predictions:
                logger_manager.warning("没有预测结果可视化")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('彩票预测结果分析', fontsize=16, fontweight='bold')
            
            # 提取数据
            methods = [p.get('method', 'Unknown') for p in predictions]
            confidences = [p.get('confidence', 0.5) for p in predictions]
            front_balls = []
            back_balls = []
            
            for p in predictions:
                if 'front_balls' in p:
                    front_balls.extend(p['front_balls'])
                if 'back_balls' in p:
                    back_balls.extend(p['back_balls'])
            
            # 1. 置信度分布
            axes[0, 0].bar(methods, confidences, color=self.colors[:len(methods)])
            axes[0, 0].set_title('预测方法置信度分布')
            axes[0, 0].set_ylabel('置信度')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. 前区号码分布
            if front_balls:
                axes[0, 1].hist(front_balls, bins=35, range=(1, 36), alpha=0.7, color='skyblue')
                axes[0, 1].set_title('前区号码分布')
                axes[0, 1].set_xlabel('号码')
                axes[0, 1].set_ylabel('频次')
            
            # 3. 后区号码分布
            if back_balls:
                axes[1, 0].hist(back_balls, bins=12, range=(1, 13), alpha=0.7, color='lightcoral')
                axes[1, 0].set_title('后区号码分布')
                axes[1, 0].set_xlabel('号码')
                axes[1, 0].set_ylabel('频次')
            
            # 4. 预测方法统计
            method_counts = pd.Series(methods).value_counts()
            axes[1, 1].pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%')
            axes[1, 1].set_title('预测方法使用分布')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger_manager.info(f"预测结果图表已保存: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger_manager.error("预测结果可视化失败", e)
    
    def plot_performance_history(self, performance_data: Dict[str, List[Dict[str, float]]], 
                                save_path: Optional[str] = None) -> None:
        """可视化性能历史"""
        try:
            if not performance_data:
                logger_manager.warning("没有性能数据可视化")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('模型性能历史分析', fontsize=16, fontweight='bold')
            
            # 准备数据
            models = list(performance_data.keys())
            metrics = ['accuracy', 'consistency', 'adaptability', 'overall_score']
            
            for i, metric in enumerate(metrics):
                ax = axes[i // 2, i % 2]
                
                for j, model in enumerate(models):
                    history = performance_data[model]
                    if history:
                        values = [h.get(metric, 0) for h in history]
                        epochs = range(1, len(values) + 1)
                        ax.plot(epochs, values, marker='o', label=model, 
                               color=self.colors[j % len(self.colors)])
                
                ax.set_title(f'{metric.replace("_", " ").title()} 趋势')
                ax.set_xlabel('时期')
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger_manager.info(f"性能历史图表已保存: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger_manager.error("性能历史可视化失败", e)
    
    def plot_resource_usage(self, resource_data: Dict[str, Any], 
                           save_path: Optional[str] = None) -> None:
        """可视化资源使用情况"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('系统资源使用情况', fontsize=16, fontweight='bold')
            
            # CPU使用率
            if 'cpu' in resource_data:
                cpu_data = resource_data['cpu']
                if isinstance(cpu_data, dict) and 'total' in cpu_data:
                    axes[0, 0].bar(['CPU使用率'], [cpu_data['total']], color='lightblue')
                    axes[0, 0].set_title('CPU使用率')
                    axes[0, 0].set_ylabel('使用率 (%)')
                    axes[0, 0].set_ylim(0, 100)
            
            # 内存使用情况
            if 'memory' in resource_data:
                memory_data = resource_data['memory']
                if isinstance(memory_data, dict):
                    memory_labels = []
                    memory_values = []
                    for key, value in memory_data.items():
                        if isinstance(value, (int, float)):
                            memory_labels.append(key.replace('_', ' ').title())
                            memory_values.append(value)
                    
                    if memory_labels:
                        axes[0, 1].bar(memory_labels, memory_values, color='lightgreen')
                        axes[0, 1].set_title('内存使用情况')
                        axes[0, 1].set_ylabel('使用量')
                        axes[0, 1].tick_params(axis='x', rotation=45)
            
            # GPU使用情况
            if 'gpu' in resource_data:
                gpu_data = resource_data['gpu']
                if isinstance(gpu_data, dict):
                    gpu_names = list(gpu_data.keys())
                    gpu_usage = [gpu_data[name].get('utilization', 0) for name in gpu_names]
                    
                    if gpu_names:
                        axes[1, 0].bar(gpu_names, gpu_usage, color='orange')
                        axes[1, 0].set_title('GPU使用率')
                        axes[1, 0].set_ylabel('使用率 (%)')
                        axes[1, 0].set_ylim(0, 100)
                        axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 磁盘IO
            if 'disk' in resource_data:
                disk_data = resource_data['disk']
                if isinstance(disk_data, dict):
                    io_labels = ['读取', '写入']
                    io_values = [disk_data.get('read_bytes', 0), disk_data.get('write_bytes', 0)]
                    
                    axes[1, 1].bar(io_labels, io_values, color='purple')
                    axes[1, 1].set_title('磁盘IO')
                    axes[1, 1].set_ylabel('字节数')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger_manager.info(f"资源使用图表已保存: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger_manager.error("资源使用可视化失败", e)
    
    def plot_interactive_predictions(self, predictions: List[Dict[str, Any]]) -> None:
        """创建交互式预测结果图表"""
        try:
            if not predictions:
                logger_manager.warning("没有预测结果可视化")
                return
            
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('置信度分布', '前区号码热力图', '后区号码分布', '预测方法对比'),
                specs=[[{"type": "bar"}, {"type": "heatmap"}],
                       [{"type": "histogram"}, {"type": "pie"}]]
            )
            
            # 提取数据
            methods = [p.get('method', 'Unknown') for p in predictions]
            confidences = [p.get('confidence', 0.5) for p in predictions]
            
            # 1. 置信度分布
            fig.add_trace(
                go.Bar(x=methods, y=confidences, name='置信度'),
                row=1, col=1
            )
            
            # 2. 前区号码热力图
            front_balls = []
            for p in predictions:
                if 'front_balls' in p:
                    front_balls.extend(p['front_balls'])
            
            if front_balls:
                front_counts = pd.Series(front_balls).value_counts().sort_index()
                # 创建5x7的热力图矩阵
                heatmap_data = np.zeros((5, 7))
                for i in range(1, 36):
                    row = (i - 1) // 7
                    col = (i - 1) % 7
                    heatmap_data[row, col] = front_counts.get(i, 0)
                
                fig.add_trace(
                    go.Heatmap(z=heatmap_data, colorscale='Blues'),
                    row=1, col=2
                )
            
            # 3. 后区号码分布
            back_balls = []
            for p in predictions:
                if 'back_balls' in p:
                    back_balls.extend(p['back_balls'])
            
            if back_balls:
                fig.add_trace(
                    go.Histogram(x=back_balls, nbinsx=12, name='后区分布'),
                    row=2, col=1
                )
            
            # 4. 预测方法饼图
            method_counts = pd.Series(methods).value_counts()
            fig.add_trace(
                go.Pie(labels=method_counts.index, values=method_counts.values),
                row=2, col=2
            )
            
            # 更新布局
            fig.update_layout(
                title_text="彩票预测结果交互式分析",
                showlegend=False,
                height=800
            )
            
            fig.show()
            
        except Exception as e:
            logger_manager.error("交互式预测结果可视化失败", e)
    
    def create_dashboard(self, data: Dict[str, Any]) -> None:
        """创建综合仪表板"""
        try:
            # 这里可以集成更复杂的仪表板功能
            # 比如使用Dash或Streamlit
            logger_manager.info("仪表板功能开发中...")
            
        except Exception as e:
            logger_manager.error("仪表板创建失败", e)


# 全局实例
visualization_utils = VisualizationUtils()


if __name__ == "__main__":
    # 测试代码
    test_predictions = [
        {
            'method': 'LSTM',
            'confidence': 0.85,
            'front_balls': [1, 5, 12, 23, 35],
            'back_balls': [3, 8]
        },
        {
            'method': 'Transformer',
            'confidence': 0.78,
            'front_balls': [2, 8, 15, 28, 33],
            'back_balls': [5, 11]
        }
    ]
    
    visualization_utils.plot_prediction_results(test_predictions)
