#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可视化工具
提供大乐透数据的各种可视化功能
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class VisualizationTool:
    """可视化工具类"""
    
    def __init__(self, data_file):
        self.data_file = data_file
        self.df = None
        self.load_data()
    
    def load_data(self):
        """加载数据"""
        try:
            self.df = pd.read_csv(self.data_file)
            self.df = self.df.sort_values('issue', ascending=True)
            print(f"可视化工具加载数据: {len(self.df)} 条记录")
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def parse_balls(self, balls_str):
        """解析号码字符串"""
        return [int(ball.strip()) for ball in str(balls_str).split(",")]
    
    def generate_all_charts(self, output_dir="output/visualization"):
        """生成所有可视化图表"""
        if self.df is None:
            print("数据未加载")
            return False
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("🎨 开始生成可视化图表...")
        
        try:
            # 1. 频率分布图
            self.generate_frequency_charts(output_dir)
            
            # 2. 热力图
            self.generate_heatmaps(output_dir)
            
            # 3. 走势图
            self.generate_trend_charts(output_dir)
            
            # 4. 分布分析图
            self.generate_distribution_charts(output_dir)
            
            # 5. 相关性分析图
            self.generate_correlation_charts(output_dir)
            
            # 6. 生成总结报告
            self.generate_summary_report(output_dir)
            
            print(f"✅ 所有图表已保存到: {output_dir}")
            return True
            
        except Exception as e:
            print(f"生成图表失败: {e}")
            return False
    
    def generate_frequency_charts(self, output_dir):
        """生成频率分布图"""
        print("  📊 生成频率分布图...")
        
        # 计算频率
        front_frequency = defaultdict(int)
        back_frequency = defaultdict(int)
        
        for _, row in self.df.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])
            
            for ball in front_balls:
                front_frequency[ball] += 1
            for ball in back_balls:
                back_frequency[ball] += 1
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 前区频率分布
        front_balls = list(range(1, 36))
        front_counts = [front_frequency[ball] for ball in front_balls]
        
        bars1 = ax1.bar(front_balls, front_counts, color='skyblue', alpha=0.7)
        ax1.set_title('前区号码频率分布', fontsize=16, fontweight='bold')
        ax1.set_xlabel('号码', fontsize=12)
        ax1.set_ylabel('出现次数', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 标注最高频率
        max_idx = front_counts.index(max(front_counts))
        ax1.annotate(f'最高: {front_balls[max_idx]}号\n{max(front_counts)}次',
                    xy=(front_balls[max_idx], max(front_counts)),
                    xytext=(front_balls[max_idx]+3, max(front_counts)+10),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
        
        # 后区频率分布
        back_balls = list(range(1, 13))
        back_counts = [back_frequency[ball] for ball in back_balls]
        
        bars2 = ax2.bar(back_balls, back_counts, color='lightcoral', alpha=0.7)
        ax2.set_title('后区号码频率分布', fontsize=16, fontweight='bold')
        ax2.set_xlabel('号码', fontsize=12)
        ax2.set_ylabel('出现次数', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 标注最高频率
        max_idx = back_counts.index(max(back_counts))
        ax2.annotate(f'最高: {back_balls[max_idx]}号\n{max(back_counts)}次',
                    xy=(back_balls[max_idx], max(back_counts)),
                    xytext=(back_balls[max_idx]+1, max(back_counts)+5),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/frequency_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_heatmaps(self, output_dir):
        """生成热力图"""
        print("  🔥 生成热力图...")
        
        # 计算最近50期的遗漏值变化
        recent_data = self.df.tail(50)
        
        front_missing_history = []
        back_missing_history = []
        
        for i in range(len(recent_data)):
            # 计算到第i期为止的遗漏值
            subset_data = recent_data.iloc[i:i+1]
            front_missing = self._calculate_missing_values(subset_data, 'front')
            back_missing = self._calculate_missing_values(subset_data, 'back')
            
            front_missing_history.append([front_missing.get(j, 0) for j in range(1, 36)])
            back_missing_history.append([back_missing.get(j, 0) for j in range(1, 13)])
        
        # 创建热力图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 前区遗漏值热力图
        front_matrix = np.array(front_missing_history).T
        sns.heatmap(front_matrix, cmap='YlOrRd', ax=ax1,
                   xticklabels=range(1, 51), yticklabels=range(1, 36),
                   cbar_kws={'label': '遗漏期数'})
        ax1.set_title('前区号码遗漏值热力图 (最近50期)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('期数', fontsize=12)
        ax1.set_ylabel('号码', fontsize=12)
        
        # 后区遗漏值热力图
        back_matrix = np.array(back_missing_history).T
        sns.heatmap(back_matrix, cmap='YlOrRd', ax=ax2,
                   xticklabels=range(1, 51), yticklabels=range(1, 13),
                   cbar_kws={'label': '遗漏期数'})
        ax2.set_title('后区号码遗漏值热力图 (最近50期)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('期数', fontsize=12)
        ax2.set_ylabel('号码', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/missing_value_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_missing_values(self, data, ball_type):
        """计算遗漏值"""
        missing = {}
        max_ball = 35 if ball_type == 'front' else 12
        
        for ball in range(1, max_ball + 1):
            missing[ball] = 0
        
        # 计算每个号码的遗漏期数
        for ball in range(1, max_ball + 1):
            found = False
            for i, (_, row) in enumerate(data.iterrows()):
                balls = self.parse_balls(row[f'{ball_type}_balls'])
                if ball in balls:
                    missing[ball] = i
                    found = True
                    break
            if not found:
                missing[ball] = len(data)
        
        return missing
    
    def generate_trend_charts(self, output_dir):
        """生成走势图"""
        print("  📈 生成走势图...")
        
        recent_data = self.df.tail(100)  # 最近100期
        
        # 计算和值走势
        front_sums = []
        back_sums = []
        issues = []
        
        for _, row in recent_data.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])
            
            front_sums.append(sum(front_balls))
            back_sums.append(sum(back_balls))
            issues.append(row['issue'])
        
        # 创建走势图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 和值走势图
        ax1.plot(range(len(front_sums)), front_sums, 'b-o', markersize=3, linewidth=1, label='前区和值')
        ax1.set_title('前区和值走势图 (最近100期)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('和值', fontsize=12)
        ax1.set_xlabel('期数', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(range(len(back_sums)), back_sums, 'r-o', markersize=3, linewidth=1, label='后区和值')
        ax2.set_title('后区和值走势图 (最近100期)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('和值', fontsize=12)
        ax2.set_xlabel('期数', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 奇偶比例走势
        front_odd_ratios = []
        back_odd_ratios = []
        
        for _, row in recent_data.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])
            
            front_odd_count = sum(1 for ball in front_balls if ball % 2 == 1)
            back_odd_count = sum(1 for ball in back_balls if ball % 2 == 1)
            
            front_odd_ratios.append(front_odd_count / len(front_balls))
            back_odd_ratios.append(back_odd_count / len(back_balls))
        
        ax3.plot(range(len(front_odd_ratios)), front_odd_ratios, 'g-o', markersize=3, linewidth=1)
        ax3.set_title('前区奇数比例走势图 (最近100期)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('奇数比例', fontsize=12)
        ax3.set_xlabel('期数', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='理论平均: 0.5')
        ax3.legend()
        
        ax4.plot(range(len(back_odd_ratios)), back_odd_ratios, 'm-o', markersize=3, linewidth=1)
        ax4.set_title('后区奇数比例走势图 (最近100期)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('奇数比例', fontsize=12)
        ax4.set_xlabel('期数', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='理论平均: 0.5')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/trend_charts.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_distribution_charts(self, output_dir):
        """生成分布分析图"""
        print("  📊 生成分布分析图...")
        
        # 计算各种分布
        front_sums = []
        back_sums = []
        front_spans = []  # 极差
        back_spans = []
        
        for _, row in self.df.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])
            
            front_sums.append(sum(front_balls))
            back_sums.append(sum(back_balls))
            front_spans.append(max(front_balls) - min(front_balls))
            back_spans.append(max(back_balls) - min(back_balls))
        
        # 创建分布图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 和值分布
        ax1.hist(front_sums, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('前区和值分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel('和值', fontsize=12)
        ax1.set_ylabel('频次', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axvline(np.mean(front_sums), color='red', linestyle='--', label=f'平均值: {np.mean(front_sums):.1f}')
        ax1.legend()
        
        ax2.hist(back_sums, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_title('后区和值分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel('和值', fontsize=12)
        ax2.set_ylabel('频次', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axvline(np.mean(back_sums), color='red', linestyle='--', label=f'平均值: {np.mean(back_sums):.1f}')
        ax2.legend()
        
        # 极差分布
        ax3.hist(front_spans, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_title('前区极差分布', fontsize=14, fontweight='bold')
        ax3.set_xlabel('极差', fontsize=12)
        ax3.set_ylabel('频次', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.axvline(np.mean(front_spans), color='red', linestyle='--', label=f'平均值: {np.mean(front_spans):.1f}')
        ax3.legend()
        
        ax4.hist(back_spans, bins=10, alpha=0.7, color='gold', edgecolor='black')
        ax4.set_title('后区极差分布', fontsize=14, fontweight='bold')
        ax4.set_xlabel('极差', fontsize=12)
        ax4.set_ylabel('频次', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.axvline(np.mean(back_spans), color='red', linestyle='--', label=f'平均值: {np.mean(back_spans):.1f}')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/distribution_charts.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_correlation_charts(self, output_dir):
        """生成相关性分析图"""
        print("  🔗 生成相关性分析图...")
        
        # 构建号码出现矩阵
        front_matrix = []
        back_matrix = []
        
        for _, row in self.df.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])
            
            # 前区one-hot编码
            front_vector = [0] * 35
            for ball in front_balls:
                front_vector[ball-1] = 1
            front_matrix.append(front_vector)
            
            # 后区one-hot编码
            back_vector = [0] * 12
            for ball in back_balls:
                back_vector[ball-1] = 1
            back_matrix.append(back_vector)
        
        # 计算相关性矩阵
        front_df = pd.DataFrame(front_matrix, columns=[f'前{i}' for i in range(1, 36)])
        back_df = pd.DataFrame(back_matrix, columns=[f'后{i}' for i in range(1, 13)])
        
        front_corr = front_df.corr()
        back_corr = back_df.corr()
        
        # 创建相关性热力图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 前区相关性热力图（只显示部分，避免过于密集）
        front_corr_sample = front_corr.iloc[:15, :15]  # 只显示前15个号码
        sns.heatmap(front_corr_sample, annot=False, cmap='coolwarm', center=0, ax=ax1,
                   cbar_kws={'label': '相关系数'})
        ax1.set_title('前区号码相关性热力图 (1-15号)', fontsize=14, fontweight='bold')
        
        # 后区相关性热力图
        sns.heatmap(back_corr, annot=True, cmap='coolwarm', center=0, ax=ax2,
                   cbar_kws={'label': '相关系数'}, fmt='.2f')
        ax2.set_title('后区号码相关性热力图', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_charts.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self, output_dir):
        """生成可视化总结报告"""
        print("  📋 生成总结报告...")
        
        # 计算基本统计信息
        front_frequency = defaultdict(int)
        back_frequency = defaultdict(int)
        
        for _, row in self.df.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])
            
            for ball in front_balls:
                front_frequency[ball] += 1
            for ball in back_balls:
                back_frequency[ball] += 1
        
        # 生成报告
        with open(f"{output_dir}/visualization_report.txt", 'w', encoding='utf-8') as f:
            f.write("大乐透数据可视化分析报告\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"数据概况:\n")
            f.write(f"  总期数: {len(self.df)}\n")
            f.write(f"  期号范围: {self.df['issue'].min()} - {self.df['issue'].max()}\n")
            f.write(f"  日期范围: {self.df['date'].min()} - {self.df['date'].max()}\n\n")
            
            f.write(f"前区号码统计:\n")
            front_sorted = sorted(front_frequency.items(), key=lambda x: x[1], reverse=True)
            f.write(f"  最热号码: {front_sorted[0][0]}号 ({front_sorted[0][1]}次)\n")
            f.write(f"  最冷号码: {front_sorted[-1][0]}号 ({front_sorted[-1][1]}次)\n")
            f.write(f"  平均出现: {sum(front_frequency.values()) / len(front_frequency):.1f}次\n\n")
            
            f.write(f"后区号码统计:\n")
            back_sorted = sorted(back_frequency.items(), key=lambda x: x[1], reverse=True)
            f.write(f"  最热号码: {back_sorted[0][0]}号 ({back_sorted[0][1]}次)\n")
            f.write(f"  最冷号码: {back_sorted[-1][0]}号 ({back_sorted[-1][1]}次)\n")
            f.write(f"  平均出现: {sum(back_frequency.values()) / len(back_frequency):.1f}次\n\n")
            
            f.write(f"生成的图表:\n")
            f.write(f"  - frequency_distribution.png: 号码频率分布图\n")
            f.write(f"  - missing_value_heatmap.png: 遗漏值热力图\n")
            f.write(f"  - trend_charts.png: 走势图\n")
            f.write(f"  - distribution_charts.png: 分布分析图\n")
            f.write(f"  - correlation_charts.png: 相关性分析图\n")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="大乐透可视化工具")
    parser.add_argument("data_file", help="数据文件路径")
    parser.add_argument("-o", "--output", default="output/visualization", help="输出目录")
    parser.add_argument("-a", "--all", action="store_true", help="生成所有图表")
    parser.add_argument("-f", "--frequency", action="store_true", help="生成频率分布图")
    parser.add_argument("--heatmap", action="store_true", help="生成热力图")
    parser.add_argument("-t", "--trend", action="store_true", help="生成走势图")
    parser.add_argument("-d", "--distribution", action="store_true", help="生成分布分析图")
    parser.add_argument("-c", "--correlation", action="store_true", help="生成相关性分析图")
    parser.add_argument("-r", "--report", action="store_true", help="生成总结报告")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_file):
        print(f"数据文件不存在: {args.data_file}")
        return
    
    # 创建可视化工具
    viz = VisualizationTool(args.data_file)
    
    # 根据参数生成相应图表
    if args.all:
        viz.generate_all_charts(args.output)
    else:
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        
        if args.frequency:
            viz.generate_frequency_charts(args.output)
        if args.heatmap:
            viz.generate_heatmaps(args.output)
        if args.trend:
            viz.generate_trend_charts(args.output)
        if args.distribution:
            viz.generate_distribution_charts(args.output)
        if args.correlation:
            viz.generate_correlation_charts(args.output)
        if args.report:
            viz.generate_summary_report(args.output)
        
        # 如果没有指定任何选项，生成所有图表
        if not any([args.frequency, args.heatmap, args.trend, args.distribution, args.correlation, args.report]):
            viz.generate_all_charts(args.output)


if __name__ == "__main__":
    main()
