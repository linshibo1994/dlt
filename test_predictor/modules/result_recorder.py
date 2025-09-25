#!/usr/bin/env python3
"""
结果记录模块
负责记录测试结果和生成报告
"""

import json
import csv
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import html


class ResultRecorder:
    """结果记录器"""
    
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            self.output_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'results'
            )
        else:
            self.output_dir = output_dir
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建时间戳用于文件命名
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_test_results(self, results: List[Dict], filename: str = None) -> str:
        """保存测试结果到JSON文件"""
        if filename is None:
            filename = f"test_results_{self.timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"测试结果已保存: {filepath}")
            return filepath
        except Exception as e:
            print(f"保存测试结果失败: {e}")
            return ""

    def save_summary_report(self, summary: Dict, filename: str = None) -> str:
        """保存汇总报告"""
        if filename is None:
            filename = f"summary_report_{self.timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"汇总报告已保存: {filepath}")
            return filepath
        except Exception as e:
            print(f"保存汇总报告失败: {e}")
            return ""

    def export_to_csv(self, results: List[Dict], filename: str = None) -> str:
        """导出结果到CSV文件"""
        if filename is None:
            filename = f"test_results_{self.timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        if not results:
            return ""
        
        try:
            # 准备CSV数据
            csv_rows = []
            for result in results:
                test_case = result.get('test_case', {})
                
                base_row = {
                    'timestamp': result.get('timestamp', ''),
                    'method': test_case.get('method', ''),
                    'periods': test_case.get('periods', 0),
                    'count': test_case.get('count', 0),
                    'strategy': test_case.get('strategy', ''),
                    'success': result.get('success', False),
                    'execution_time': result.get('execution_time', 0),
                    'total_predictions': result.get('total_predictions', 0),
                    'winning_predictions': result.get('winning_predictions', 0),
                    'winning_rate': result.get('winning_rate', 0),
                    'best_prize_level': result.get('best_prize_level', 0),
                    'is_winning': result.get('is_winning', False),
                    'is_major_prize': result.get('is_major_prize', False)
                }
                
                # 添加预测详情
                predictions_analysis = result.get('predictions_analysis', [])
                if predictions_analysis:
                    for i, analysis in enumerate(predictions_analysis):
                        row = base_row.copy()
                        prediction = analysis.get('prediction', {})
                        winning_result = analysis.get('winning_result', {})
                        
                        row.update({
                            'prediction_index': i + 1,
                            'predicted_front': prediction.get('front_balls', ''),
                            'predicted_back': prediction.get('back_balls', ''),
                            'winning_front': winning_result.get('winning_front', []),
                            'winning_back': winning_result.get('winning_back', []),
                            'front_matches': winning_result.get('front_matches', 0),
                            'back_matches': winning_result.get('back_matches', 0),
                            'match_combination': winning_result.get('match_combination', ''),
                            'prize_level': winning_result.get('prize_level', 0),
                            'prize_name': winning_result.get('prize_name', ''),
                            'prediction_is_winning': winning_result.get('is_winning', False)
                        })
                        
                        csv_rows.append(row)
                else:
                    csv_rows.append(base_row)
            
            # 写入CSV文件
            if csv_rows:
                fieldnames = csv_rows[0].keys()
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(csv_rows)
                
                print(f"CSV结果已保存: {filepath}")
                return filepath
            
        except Exception as e:
            print(f"导出CSV失败: {e}")
            return ""

    def generate_html_report(self, summary: Dict, results: List[Dict], 
                           filename: str = None) -> str:
        """生成HTML报告"""
        if filename is None:
            filename = f"test_report_{self.timestamp}.html"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            html_content = self._build_html_report(summary, results)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"HTML报告已生成: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"生成HTML报告失败: {e}")
            return ""

    def _build_html_report(self, summary: Dict, results: List[Dict]) -> str:
        """构建HTML报告内容"""
        # 获取统计数据
        total_tests = summary.get('total_tests', 0)
        winning_tests = summary.get('winning_tests', 0)
        major_prizes = summary.get('major_prizes', 0)
        execution_time = summary.get('execution_time', 0)
        
        # 构建HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>大乐透预测方法测试报告</title>
    <style>
        body {{ 
            font-family: 'Microsoft YaHei', Arial, sans-serif; 
            margin: 20px; 
            background-color: #f5f5f5;
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            padding: 20px; 
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{ 
            text-align: center; 
            color: #333; 
            border-bottom: 2px solid #007acc;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .summary {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 20px; 
            margin-bottom: 30px;
        }}
        .summary-card {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            padding: 20px; 
            border-radius: 10px;
            text-align: center;
        }}
        .summary-card h3 {{ margin: 0 0 10px 0; }}
        .summary-card .value {{ font-size: 2em; font-weight: bold; }}
        .section {{ 
            margin-bottom: 30px; 
            background: #f9f9f9; 
            padding: 20px; 
            border-radius: 8px;
        }}
        .section h2 {{ 
            color: #333; 
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }}
        table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin-top: 20px;
        }}
        th, td {{ 
            padding: 12px; 
            text-align: left; 
            border-bottom: 1px solid #ddd;
        }}
        th {{ 
            background-color: #007acc; 
            color: white;
        }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .winning {{ color: #28a745; font-weight: bold; }}
        .major-prize {{ color: #dc3545; font-weight: bold; }}
        .method-stats {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 20px;
        }}
        .method-card {{ 
            background: white; 
            padding: 15px; 
            border-radius: 8px;
            border-left: 4px solid #007acc;
        }}
        .prize-badge {{ 
            display: inline-block; 
            padding: 4px 8px; 
            background: #28a745; 
            color: white; 
            border-radius: 4px; 
            font-size: 0.8em;
        }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 大乐透预测方法测试报告</h1>
            <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>📊 测试概览</h2>
            <div class="summary">
                <div class="summary-card">
                    <h3>总测试数</h3>
                    <div class="value">{total_tests}</div>
                </div>
                <div class="summary-card">
                    <h3>中奖测试</h3>
                    <div class="value">{winning_tests}</div>
                </div>
                <div class="summary-card">
                    <h3>重大奖项</h3>
                    <div class="value">{major_prizes}</div>
                </div>
                <div class="summary-card">
                    <h3>执行时间</h3>
                    <div class="value">{execution_time:.1f}s</div>
                </div>
            </div>
        </div>
        
        {self._build_prize_statistics_section(summary)}
        
        {self._build_method_statistics_section(summary)}
        
        {self._build_winning_results_section(results)}
        
        {self._build_detailed_results_section(results)}
    </div>
</body>
</html>
        """
        
        return html_content

    def _build_prize_statistics_section(self, summary: Dict) -> str:
        """构建奖级统计部分"""
        prize_stats = summary.get('prize_statistics', {})
        
        if not prize_stats:
            return ""
        
        html = """
        <div class="section">
            <h2>🏆 中奖等级统计</h2>
            <table>
                <thead>
                    <tr>
                        <th>奖级</th>
                        <th>奖项名称</th>
                        <th>中奖次数</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for level in sorted(prize_stats.keys()):
            stat = prize_stats[level]
            html += f"""
                    <tr>
                        <td>{level}等奖</td>
                        <td>{stat['name']}</td>
                        <td><span class="prize-badge">{stat['count']}</span></td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        return html

    def _build_method_statistics_section(self, summary: Dict) -> str:
        """构建方法统计部分"""
        method_stats = summary.get('method_statistics', {})
        
        if not method_stats:
            return ""
        
        html = """
        <div class="section">
            <h2>🔍 预测方法统计</h2>
            <div class="method-stats">
        """
        
        # 按最佳奖级排序
        sorted_methods = sorted(
            method_stats.items(), 
            key=lambda x: x[1].get('best_prize_level', 0), 
            reverse=True
        )
        
        for method, stats in sorted_methods:
            best_prize = stats.get('best_prize_level', 0)
            test_rate = stats.get('test_winning_rate', 0) * 100
            pred_rate = stats.get('prediction_winning_rate', 0) * 100
            
            prize_class = "major-prize" if best_prize <= 2 else "winning" if best_prize > 0 else ""
            
            html += f"""
            <div class="method-card">
                <h3>{method}</h3>
                <p><strong>测试数量:</strong> {stats.get('total_tests', 0)}</p>
                <p><strong>预测数量:</strong> {stats.get('total_predictions', 0)}</p>
                <p><strong>测试中奖率:</strong> {test_rate:.1f}%</p>
                <p><strong>预测中奖率:</strong> {pred_rate:.1f}%</p>
                <p><strong>最佳成绩:</strong> 
                    <span class="{prize_class}">
                        {best_prize}等奖
                    </span>
                </p>
            </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        return html

    def _build_winning_results_section(self, results: List[Dict]) -> str:
        """构建中奖结果部分"""
        winning_results = [r for r in results if r.get('is_winning')]
        
        if not winning_results:
            return ""
        
        html = """
        <div class="section">
            <h2>🎯 中奖结果详情</h2>
            <table>
                <thead>
                    <tr>
                        <th>时间</th>
                        <th>预测方法</th>
                        <th>参数</th>
                        <th>奖级</th>
                        <th>中奖号码</th>
                        <th>匹配情况</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for result in winning_results:
            test_case = result.get('test_case', {})
            method = test_case.get('method', '')
            periods = test_case.get('periods', 0)
            count = test_case.get('count', 0)
            prize_level = result.get('best_prize_level', 0)
            timestamp = result.get('timestamp', '')
            
            # 获取第一个中奖预测的详情
            winning_prediction = None
            for analysis in result.get('predictions_analysis', []):
                if analysis.get('winning_result', {}).get('is_winning'):
                    winning_prediction = analysis
                    break
            
            if winning_prediction:
                prediction = winning_prediction['prediction']
                winning_result = winning_prediction['winning_result']
                
                prize_class = "major-prize" if prize_level <= 2 else "winning"
                
                html += f"""
                <tr class="{prize_class}">
                    <td>{timestamp[:19]}</td>
                    <td>{method}</td>
                    <td>p={periods}, c={count}</td>
                    <td>{prize_level}等奖</td>
                    <td>{prediction.get('front_balls', '')} + {prediction.get('back_balls', '')}</td>
                    <td>{winning_result.get('match_combination', '')}</td>
                </tr>
                """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        return html

    def _build_detailed_results_section(self, results: List[Dict]) -> str:
        """构建详细结果部分（只显示部分结果）"""
        html = """
        <div class="section">
            <h2>📋 详细测试结果 (前20项)</h2>
            <table>
                <thead>
                    <tr>
                        <th>方法</th>
                        <th>参数</th>
                        <th>状态</th>
                        <th>预测数</th>
                        <th>中奖数</th>
                        <th>最佳奖级</th>
                        <th>执行时间</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for result in results[:20]:  # 只显示前20个结果
            test_case = result.get('test_case', {})
            method = test_case.get('method', '')
            periods = test_case.get('periods', 0)
            count = test_case.get('count', 0)
            success = result.get('success', False)
            total_pred = result.get('total_predictions', 0)
            winning_pred = result.get('winning_predictions', 0)
            prize_level = result.get('best_prize_level', 0)
            exec_time = result.get('execution_time', 0)
            
            status_class = "winning" if result.get('is_winning') else ""
            status_text = "成功" if success else "失败"
            prize_text = f"{prize_level}等奖" if prize_level > 0 else "-"
            
            html += f"""
            <tr class="{status_class}">
                <td>{method}</td>
                <td>p={periods}, c={count}</td>
                <td>{status_text}</td>
                <td>{total_pred}</td>
                <td>{winning_pred}</td>
                <td>{prize_text}</td>
                <td>{exec_time:.1f}s</td>
            </tr>
            """
        
        if len(results) > 20:
            html += f"""
            <tr>
                <td colspan="7" style="text-align: center; color: #666;">
                    ... 还有 {len(results) - 20} 个结果未显示
                </td>
            </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        return html

    def generate_winner_report(self, results: List[Dict], filename: str = None) -> str:
        """生成详细的中奖者报告，包含所有奖级信息"""
        winning_results = [r for r in results if r.get('is_winning')]
        
        if not winning_results:
            print("没有中奖结果，无法生成中奖者报告")
            return ""
        
        if filename is None:
            filename = f"winner_report_{self.timestamp}.txt"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("🎉 恭喜！大乐透预测测试中奖详细报告 🎉\n")
                f.write("=" * 60 + "\n\n")
                
                # 统计各奖级总数
                prize_summary = {}
                total_winnings = 0
                
                # 先统计所有中奖情况
                for result in winning_results:
                    winnings = result.get('winnings', [])
                    if winnings:
                        # 使用新格式
                        for winning in winnings:
                            level = winning.get('prize_level', 0)
                            total_winnings += 1
                            prize_summary[level] = prize_summary.get(level, 0) + 1
                    else:
                        # 向后兼容旧格式
                        level = result.get('best_prize_level', 0)
                        if level > 0:
                            total_winnings += 1
                            prize_summary[level] = prize_summary.get(level, 0) + 1
                
                # 写入汇总信息
                f.write("🏆 中奖汇总统计\n")
                f.write("-" * 30 + "\n")
                f.write(f"总中奖次数: {total_winnings} 次\n")
                f.write(f"中奖测试数: {len(winning_results)} 个\n\n")
                
                if prize_summary:
                    f.write("各奖级中奖统计:\n")
                    for level in sorted(prize_summary.keys()):
                        count = prize_summary[level]
                        prize_names = {1: "一等奖", 2: "二等奖", 3: "三等奖", 4: "四等奖", 
                                     5: "五等奖", 6: "六等奖", 7: "七等奖", 8: "八等奖", 9: "九等奖"}
                        name = prize_names.get(level, f"{level}等奖")
                        f.write(f"  {name}: {count} 次\n")
                    f.write("\n")
                
                f.write("🎯 详细中奖记录\n")
                f.write("=" * 60 + "\n\n")
                
                # 按最高奖级排序
                sorted_results = sorted(winning_results, key=lambda x: x.get('best_prize_level', 0))
                
                for i, result in enumerate(sorted_results, 1):
                    test_case = result.get('test_case', {})
                    
                    f.write(f"中奖记录 #{i}\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"预测方法: {test_case.get('method', '')}\n")
                    f.write(f"分析期数: {test_case.get('periods', 0)} 期\n")
                    f.write(f"生成注数: {test_case.get('count', 0)} 注\n")
                    f.write(f"测试时间: {result.get('timestamp', '')}\n")
                    f.write(f"最高奖级: {result.get('best_prize_level', 0)}等奖\n")
                    
                    # 显示详细的中奖信息
                    winnings = result.get('winnings', [])
                    if winnings:
                        f.write(f"中奖详情:\n")
                        for j, winning in enumerate(winnings, 1):
                            level = winning.get('prize_level', 0)
                            name = {1: "一等奖", 2: "二等奖", 3: "三等奖", 4: "四等奖", 
                                   5: "五等奖", 6: "六等奖", 7: "七等奖", 8: "八等奖", 9: "九等奖"}.get(level, f"{level}等奖")
                            
                            f.write(f"  第{j}注中奖: {name} [{winning.get('match_combination', '')}]\n")
                            f.write(f"    预测号码: 前区{winning.get('predicted_front', [])} 后区{winning.get('predicted_back', [])}\n")
                            f.write(f"    中奖号码: 前区{winning.get('matched_front_numbers', [])} 后区{winning.get('matched_back_numbers', [])}\n")
                    else:
                        # 向后兼容处理
                        for analysis in result.get('predictions_analysis', []):
                            winning_result = analysis.get('winning_result', {})
                            if winning_result.get('is_winning'):
                                prediction = analysis.get('prediction', {})
                                f.write(f"中奖号码: {prediction.get('front_balls', '')} + {prediction.get('back_balls', '')}\n")
                                f.write(f"匹配情况: {winning_result.get('match_combination', '')}\n")
                                break
                    
                    f.write("\n")
                
                # 找出最佳方法
                method_stats = {}
                for result in winning_results:
                    method = result['test_case']['method']
                    if method not in method_stats:
                        method_stats[method] = {'count': 0, 'best_level': 0, 'periods': [], 'details': []}
                    
                    method_stats[method]['count'] += 1
                    method_stats[method]['best_level'] = max(method_stats[method]['best_level'], result.get('best_prize_level', 0))
                    method_stats[method]['periods'].append(result['test_case']['periods'])
                    method_stats[method]['details'].append({
                        'level': result.get('best_prize_level', 0),
                        'periods': result['test_case']['periods'],
                        'count': result['test_case']['count']
                    })
                
                f.write("🎯 最佳预测方法分析\n")
                f.write("-" * 30 + "\n")
                
                # 按最佳奖级和中奖次数排序
                sorted_methods = sorted(method_stats.items(), key=lambda x: (x[1]['best_level'], x[1]['count']), reverse=True)
                
                for method, stats in sorted_methods[:5]:
                    f.write(f"\n方法: {method}\n")
                    f.write(f"  中奖次数: {stats['count']} 次\n")
                    f.write(f"  最高奖级: {stats['best_level']}等奖\n")
                    f.write(f"  使用期数范围: {min(stats['periods'])}-{max(stats['periods'])} 期\n")
                    
                    # 显示详细的中奖记录
                    if stats['best_level'] <= 3:  # 对于重要奖级，显示详细信息
                        f.write("  详细记录:\n")
                        for detail in stats['details']:
                            f.write(f"    {detail['level']}等奖 - 分析{detail['periods']}期 生成{detail['count']}注\n")
                
                f.write(f"\n\n📊 报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("🎉 祝贺所有中奖的预测方法！🎉\n")
            
            print(f"详细中奖者报告已生成: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"生成中奖者报告失败: {e}")
            return ""


def test_result_recorder():
    """测试函数"""
    recorder = ResultRecorder("/tmp/test_results")
    
    # 模拟测试数据
    test_results = [
        {
            'success': True,
            'test_case': {'method': 'frequency', 'periods': 100, 'count': 1},
            'total_predictions': 1,
            'winning_predictions': 1,
            'winning_rate': 1.0,
            'best_prize_level': 3,
            'is_winning': True,
            'is_major_prize': False,
            'timestamp': datetime.now().isoformat(),
            'execution_time': 2.5,
            'predictions_analysis': [
                {
                    'prediction': {'front_balls': '01,02,03,04,05', 'back_balls': '01,02'},
                    'winning_result': {
                        'is_winning': True,
                        'prize_level': 3,
                        'prize_name': '三等奖',
                        'front_matches': 5,
                        'back_matches': 0,
                        'match_combination': '5+0'
                    }
                }
            ]
        }
    ]
    
    summary = {
        'strategy': 'test',
        'total_tests': 1,
        'winning_tests': 1,
        'major_prizes': 0,
        'execution_time': 2.5,
        'prize_statistics': {3: {'count': 1, 'name': '三等奖'}},
        'method_statistics': {
            'frequency': {
                'total_tests': 1,
                'winning_tests': 1,
                'best_prize_level': 3,
                'test_winning_rate': 1.0
            }
        }
    }
    
    print("=== 结果记录测试 ===")
    
    # 测试保存结果
    print("\n1. 保存JSON结果:")
    json_file = recorder.save_test_results(test_results)
    print(f"JSON文件: {json_file}")
    
    # 测试生成HTML报告
    print("\n2. 生成HTML报告:")
    html_file = recorder.generate_html_report(summary, test_results)
    print(f"HTML文件: {html_file}")
    
    # 测试CSV导出
    print("\n3. 导出CSV:")
    csv_file = recorder.export_to_csv(test_results)
    print(f"CSV文件: {csv_file}")
    
    # 测试中奖者报告
    print("\n4. 生成中奖者报告:")
    winner_file = recorder.generate_winner_report(test_results)
    print(f"中奖者报告: {winner_file}")


if __name__ == "__main__":
    test_result_recorder()