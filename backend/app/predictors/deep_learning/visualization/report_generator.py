#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
报告生成器模块
Report Generator Module

提供自动报告生成、多格式导出、定时报告等功能。
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
import uuid
import base64
from io import BytesIO
# 尝试导入schedule，如果不可用则跳过
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException
from .interactive_visualizer import InteractiveVisualizer, ChartData, VisualizationConfig, ChartType
from .dashboard import Dashboard, MetricWidget, ChartWidget, Alert


class ReportFormat(Enum):
    """报告格式枚举"""
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    JSON = "json"
    EXCEL = "excel"


class ReportType(Enum):
    """报告类型枚举"""
    PERFORMANCE = "performance"
    TRAINING = "training"
    PREDICTION = "prediction"
    SYSTEM = "system"
    CUSTOM = "custom"


@dataclass
class ReportSection:
    """报告章节"""
    title: str
    content: str
    section_type: str = "text"  # "text", "chart", "table", "metric"
    data: Any = None
    order: int = 0


@dataclass
class ReportTemplate:
    """报告模板"""
    template_id: str
    name: str
    description: str
    sections: List[ReportSection] = field(default_factory=list)
    format: ReportFormat = ReportFormat.HTML
    auto_generate: bool = False
    schedule_cron: str = ""  # cron表达式
    created_time: datetime = field(default_factory=datetime.now)


class HTMLReportBuilder:
    """HTML报告构建器"""
    
    def __init__(self):
        """初始化HTML报告构建器"""
        self.template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }}
                .section {{ margin-bottom: 30px; }}
                .section-title {{ color: #333; border-bottom: 1px solid #ccc; padding-bottom: 10px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; text-align: center; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
                .metric-label {{ font-size: 0.9em; color: #666; }}
                .chart {{ text-align: center; margin: 20px 0; }}
                .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .table th {{ background-color: #f2f2f2; }}
                .footer {{ margin-top: 50px; text-align: center; color: #666; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>生成时间: {timestamp}</p>
            </div>
            {content}
            <div class="footer">
                <p>此报告由大乐透智能预测系统自动生成</p>
            </div>
        </body>
        </html>
        """
        
        logger_manager.info("HTML报告构建器初始化完成")
    
    def build_section(self, section: ReportSection) -> str:
        """构建报告章节"""
        try:
            html = f'<div class="section">'
            html += f'<h2 class="section-title">{section.title}</h2>'
            
            if section.section_type == "text":
                html += f'<p>{section.content}</p>'
            
            elif section.section_type == "metric" and section.data:
                html += '<div class="metrics">'
                if isinstance(section.data, dict):
                    for name, value in section.data.items():
                        html += f'''
                        <div class="metric">
                            <div class="metric-value">{value}</div>
                            <div class="metric-label">{name}</div>
                        </div>
                        '''
                html += '</div>'
            
            elif section.section_type == "table" and section.data:
                html += self._build_table(section.data)
            
            elif section.section_type == "chart" and section.data:
                html += f'<div class="chart">{section.data}</div>'
            
            html += '</div>'
            return html
            
        except Exception as e:
            logger_manager.error(f"构建报告章节失败: {e}")
            return f'<div class="section"><h2>{section.title}</h2><p>章节生成失败: {e}</p></div>'
    
    def _build_table(self, data: Any) -> str:
        """构建表格"""
        try:
            if isinstance(data, pd.DataFrame):
                return data.to_html(classes='table', table_id='data-table')
            elif isinstance(data, list) and data:
                if isinstance(data[0], dict):
                    # 字典列表
                    columns = list(data[0].keys())
                    html = '<table class="table"><thead><tr>'
                    for col in columns:
                        html += f'<th>{col}</th>'
                    html += '</tr></thead><tbody>'
                    
                    for row in data:
                        html += '<tr>'
                        for col in columns:
                            html += f'<td>{row.get(col, "")}</td>'
                        html += '</tr>'
                    
                    html += '</tbody></table>'
                    return html
            
            return '<p>表格数据格式不支持</p>'
            
        except Exception as e:
            logger_manager.error(f"构建表格失败: {e}")
            return f'<p>表格生成失败: {e}</p>'
    
    def build_report(self, title: str, sections: List[ReportSection]) -> str:
        """构建完整报告"""
        try:
            # 按顺序排序章节
            sorted_sections = sorted(sections, key=lambda x: x.order)
            
            # 构建内容
            content = ""
            for section in sorted_sections:
                content += self.build_section(section)
            
            # 生成完整HTML
            html = self.template.format(
                title=title,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                content=content
            )
            
            return html
            
        except Exception as e:
            logger_manager.error(f"构建HTML报告失败: {e}")
            return f"<html><body><h1>报告生成失败: {e}</h1></body></html>"


class MarkdownReportBuilder:
    """Markdown报告构建器"""
    
    def __init__(self):
        """初始化Markdown报告构建器"""
        logger_manager.info("Markdown报告构建器初始化完成")
    
    def build_section(self, section: ReportSection) -> str:
        """构建报告章节"""
        try:
            md = f"## {section.title}\n\n"
            
            if section.section_type == "text":
                md += f"{section.content}\n\n"
            
            elif section.section_type == "metric" and section.data:
                if isinstance(section.data, dict):
                    md += "| 指标 | 值 |\n|------|----|\n"
                    for name, value in section.data.items():
                        md += f"| {name} | {value} |\n"
                    md += "\n"
            
            elif section.section_type == "table" and section.data:
                md += self._build_table(section.data)
            
            return md
            
        except Exception as e:
            logger_manager.error(f"构建Markdown章节失败: {e}")
            return f"## {section.title}\n\n章节生成失败: {e}\n\n"
    
    def _build_table(self, data: Any) -> str:
        """构建Markdown表格"""
        try:
            if isinstance(data, pd.DataFrame):
                return data.to_markdown() + "\n\n"
            elif isinstance(data, list) and data:
                if isinstance(data[0], dict):
                    columns = list(data[0].keys())
                    md = "| " + " | ".join(columns) + " |\n"
                    md += "|" + "---|" * len(columns) + "\n"
                    
                    for row in data:
                        values = [str(row.get(col, "")) for col in columns]
                        md += "| " + " | ".join(values) + " |\n"
                    
                    md += "\n"
                    return md
            
            return "表格数据格式不支持\n\n"
            
        except Exception as e:
            logger_manager.error(f"构建Markdown表格失败: {e}")
            return f"表格生成失败: {e}\n\n"
    
    def build_report(self, title: str, sections: List[ReportSection]) -> str:
        """构建完整报告"""
        try:
            # 按顺序排序章节
            sorted_sections = sorted(sections, key=lambda x: x.order)
            
            # 构建内容
            md = f"# {title}\n\n"
            md += f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            md += "---\n\n"
            
            for section in sorted_sections:
                md += self.build_section(section)
            
            md += "---\n\n"
            md += "*此报告由大乐透智能预测系统自动生成*\n"
            
            return md
            
        except Exception as e:
            logger_manager.error(f"构建Markdown报告失败: {e}")
            return f"# 报告生成失败\n\n{e}\n"


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self):
        """初始化报告生成器"""
        self.templates = {}
        self.html_builder = HTMLReportBuilder()
        self.markdown_builder = MarkdownReportBuilder()
        self.visualizer = InteractiveVisualizer()
        self.scheduled_reports = {}
        self.scheduler_running = False
        self.scheduler_thread = None
        
        logger_manager.info("报告生成器初始化完成")
    
    def create_template(self, template: ReportTemplate):
        """创建报告模板"""
        self.templates[template.template_id] = template
        
        # 如果是自动生成报告，添加到调度器
        if template.auto_generate and template.schedule_cron:
            self._schedule_report(template)
        
        logger_manager.info(f"创建报告模板: {template.template_id}")
    
    def generate_report(self, template_id: str, data: Dict[str, Any] = None,
                       output_path: str = None) -> str:
        """
        生成报告
        
        Args:
            template_id: 模板ID
            data: 报告数据
            output_path: 输出路径
            
        Returns:
            报告内容
        """
        try:
            if template_id not in self.templates:
                raise DeepLearningException(f"报告模板不存在: {template_id}")
            
            template = self.templates[template_id]
            
            # 处理动态数据
            sections = self._process_sections(template.sections, data or {})
            
            # 根据格式生成报告
            if template.format == ReportFormat.HTML:
                content = self.html_builder.build_report(template.name, sections)
            elif template.format == ReportFormat.MARKDOWN:
                content = self.markdown_builder.build_report(template.name, sections)
            elif template.format == ReportFormat.JSON:
                content = self._generate_json_report(template.name, sections)
            else:
                raise DeepLearningException(f"不支持的报告格式: {template.format}")
            
            # 保存到文件
            if output_path:
                self._save_report(content, output_path, template.format)
            
            logger_manager.info(f"生成报告完成: {template_id}")
            return content
            
        except Exception as e:
            logger_manager.error(f"生成报告失败: {e}")
            return f"报告生成失败: {e}"
    
    def _process_sections(self, sections: List[ReportSection], 
                         data: Dict[str, Any]) -> List[ReportSection]:
        """处理报告章节"""
        processed_sections = []
        
        for section in sections:
            processed_section = ReportSection(
                title=section.title,
                content=section.content,
                section_type=section.section_type,
                data=section.data,
                order=section.order
            )
            
            # 处理动态数据
            if section.section_type == "metric" and section.title in data:
                processed_section.data = data[section.title]
            elif section.section_type == "table" and section.title in data:
                processed_section.data = data[section.title]
            elif section.section_type == "chart" and section.title in data:
                # 生成图表
                chart_data = data[section.title]
                if isinstance(chart_data, dict) and 'x' in chart_data and 'y' in chart_data:
                    chart = self._generate_chart(chart_data)
                    processed_section.data = chart
            
            processed_sections.append(processed_section)
        
        return processed_sections
    
    def _generate_chart(self, chart_data: Dict[str, Any]) -> str:
        """生成图表"""
        try:
            data = ChartData(
                x=chart_data['x'],
                y=chart_data['y'],
                labels=chart_data.get('labels', ['Data'])
            )
            
            config = VisualizationConfig(
                chart_type=ChartType(chart_data.get('type', 'line')),
                title=chart_data.get('title', ''),
                width=600,
                height=400
            )
            
            chart_id = str(uuid.uuid4())
            fig = self.visualizer.create_chart(chart_id, data, config)
            
            if hasattr(fig, 'to_html'):
                return fig.to_html(include_plotlyjs='cdn')
            else:
                return "<p>图表生成失败</p>"
                
        except Exception as e:
            logger_manager.error(f"生成图表失败: {e}")
            return f"<p>图表生成失败: {e}</p>"
    
    def _generate_json_report(self, title: str, sections: List[ReportSection]) -> str:
        """生成JSON格式报告"""
        try:
            report_data = {
                'title': title,
                'generated_time': datetime.now().isoformat(),
                'sections': []
            }
            
            for section in sections:
                section_data = {
                    'title': section.title,
                    'content': section.content,
                    'type': section.section_type,
                    'order': section.order
                }
                
                if section.data is not None:
                    if isinstance(section.data, pd.DataFrame):
                        section_data['data'] = section.data.to_dict('records')
                    else:
                        section_data['data'] = section.data
                
                report_data['sections'].append(section_data)
            
            return json.dumps(report_data, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger_manager.error(f"生成JSON报告失败: {e}")
            return json.dumps({'error': str(e)})
    
    def _save_report(self, content: str, output_path: str, format: ReportFormat):
        """保存报告到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger_manager.info(f"报告已保存: {output_path}")
            
        except Exception as e:
            logger_manager.error(f"保存报告失败: {e}")
    
    def _schedule_report(self, template: ReportTemplate):
        """调度报告生成"""
        try:
            if not SCHEDULE_AVAILABLE:
                logger_manager.warning("schedule库不可用，无法调度报告")
                return

            # 简化的调度实现
            if template.schedule_cron == "daily":
                schedule.every().day.at("09:00").do(
                    self._generate_scheduled_report, template.template_id
                )
            elif template.schedule_cron == "weekly":
                schedule.every().monday.at("09:00").do(
                    self._generate_scheduled_report, template.template_id
                )
            elif template.schedule_cron == "monthly":
                schedule.every().month.do(
                    self._generate_scheduled_report, template.template_id
                )
            
            self.scheduled_reports[template.template_id] = template
            
            # 启动调度器
            if not self.scheduler_running:
                self._start_scheduler()
            
            logger_manager.info(f"调度报告: {template.template_id}, 计划: {template.schedule_cron}")
            
        except Exception as e:
            logger_manager.error(f"调度报告失败: {e}")
    
    def _generate_scheduled_report(self, template_id: str):
        """生成调度报告"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/{template_id}_{timestamp}.html"
            
            self.generate_report(template_id, output_path=output_path)
            
            logger_manager.info(f"调度报告生成完成: {output_path}")
            
        except Exception as e:
            logger_manager.error(f"调度报告生成失败: {e}")
    
    def _start_scheduler(self):
        """启动调度器"""
        if self.scheduler_running:
            return
        
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger_manager.info("报告调度器已启动")
    
    def _scheduler_loop(self):
        """调度器循环"""
        while self.scheduler_running:
            try:
                if SCHEDULE_AVAILABLE:
                    schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
            except Exception as e:
                logger_manager.error(f"调度器运行失败: {e}")
                time.sleep(60)
    
    def stop_scheduler(self):
        """停止调度器"""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger_manager.info("报告调度器已停止")
    
    def create_performance_report(self, data: Dict[str, Any]) -> str:
        """创建性能报告"""
        try:
            sections = [
                ReportSection(
                    title="系统概览",
                    content="系统性能指标概览",
                    section_type="metric",
                    data=data.get('metrics', {}),
                    order=1
                ),
                ReportSection(
                    title="性能趋势",
                    content="",
                    section_type="chart",
                    data=data.get('performance_chart', {}),
                    order=2
                ),
                ReportSection(
                    title="详细数据",
                    content="",
                    section_type="table",
                    data=data.get('performance_table', []),
                    order=3
                )
            ]
            
            return self.html_builder.build_report("性能报告", sections)
            
        except Exception as e:
            logger_manager.error(f"创建性能报告失败: {e}")
            return f"性能报告生成失败: {e}"
    
    def get_template(self, template_id: str) -> Optional[ReportTemplate]:
        """获取报告模板"""
        return self.templates.get(template_id)
    
    def list_templates(self) -> List[str]:
        """列出所有模板"""
        return list(self.templates.keys())
    
    def remove_template(self, template_id: str):
        """移除报告模板"""
        if template_id in self.templates:
            del self.templates[template_id]
            
            if template_id in self.scheduled_reports:
                del self.scheduled_reports[template_id]
            
            logger_manager.info(f"移除报告模板: {template_id}")


# 全局报告生成器实例
report_generator = ReportGenerator()


if __name__ == "__main__":
    # 测试报告生成功能
    print("📄 测试报告生成功能...")
    
    try:
        # 创建测试模板
        sections = [
            ReportSection(
                title="系统指标",
                content="当前系统运行指标",
                section_type="metric",
                data={"CPU使用率": "75%", "内存使用率": "60%", "磁盘使用率": "45%"},
                order=1
            ),
            ReportSection(
                title="说明",
                content="这是一个测试报告，展示了系统的基本运行状态。",
                section_type="text",
                order=2
            )
        ]
        
        template = ReportTemplate(
            template_id="test_template",
            name="测试报告",
            description="用于测试的报告模板",
            sections=sections,
            format=ReportFormat.HTML
        )
        
        generator = ReportGenerator()
        generator.create_template(template)
        
        # 生成报告
        report_content = generator.generate_report("test_template")
        
        if report_content and len(report_content) > 100:
            print("✅ HTML报告生成成功")
        
        # 测试Markdown格式
        template.format = ReportFormat.MARKDOWN
        generator.create_template(template)
        
        md_content = generator.generate_report("test_template")
        if md_content and "# 测试报告" in md_content:
            print("✅ Markdown报告生成成功")
        
        # 测试JSON格式
        template.format = ReportFormat.JSON
        generator.create_template(template)
        
        json_content = generator.generate_report("test_template")
        if json_content and '"title"' in json_content:
            print("✅ JSON报告生成成功")
        
        print("✅ 报告生成功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("报告生成功能测试完成")
