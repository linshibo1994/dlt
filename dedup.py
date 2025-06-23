#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据去重工具
用于去除大乐透数据中的重复记录
"""

import os
import pandas as pd
import argparse
from collections import Counter


def remove_duplicates(data_file, output_file=None, keep='first'):
    """去除数据文件中的重复记录
    
    Args:
        data_file: 输入数据文件路径
        output_file: 输出数据文件路径，如果为None则覆盖原文件
        keep: 保留策略，'first'保留第一个，'last'保留最后一个
    
    Returns:
        去重后的记录数量
    """
    try:
        # 读取数据
        df = pd.read_csv(data_file)
        original_count = len(df)
        
        print(f"原始数据记录数: {original_count}")
        
        # 检查重复的期号
        duplicate_issues = df[df.duplicated(subset=['issue'], keep=False)]
        if len(duplicate_issues) > 0:
            print(f"发现重复期号: {len(duplicate_issues)}条记录")
            print("重复期号列表:")
            for issue in duplicate_issues['issue'].unique():
                count = len(duplicate_issues[duplicate_issues['issue'] == issue])
                print(f"  期号 {issue}: {count}条记录")
        
        # 去除重复记录（基于期号）
        df_dedup = df.drop_duplicates(subset=['issue'], keep=keep)
        dedup_count = len(df_dedup)
        
        print(f"去重后记录数: {dedup_count}")
        print(f"删除重复记录: {original_count - dedup_count}条")
        
        # 按期号排序（降序）
        df_dedup = df_dedup.sort_values('issue', ascending=False).reset_index(drop=True)
        
        # 保存结果
        if output_file is None:
            output_file = data_file
        
        df_dedup.to_csv(output_file, index=False)
        print(f"去重后的数据已保存到: {output_file}")
        
        return dedup_count
        
    except Exception as e:
        print(f"数据去重失败: {e}")
        return 0


def check_data_integrity(data_file):
    """检查数据完整性
    
    Args:
        data_file: 数据文件路径
    
    Returns:
        检查结果字典
    """
    try:
        df = pd.read_csv(data_file)
        
        results = {
            'total_records': len(df),
            'missing_values': {},
            'invalid_formats': [],
            'duplicate_issues': [],
            'issue_gaps': []
        }
        
        # 检查缺失值
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                results['missing_values'][col] = missing_count
        
        # 检查数据格式
        for idx, row in df.iterrows():
            try:
                # 检查前区号码格式
                front_balls = row['front_balls'].split(',')
                if len(front_balls) != 5:
                    results['invalid_formats'].append(f"期号 {row['issue']}: 前区号码数量不正确")
                
                for ball in front_balls:
                    ball_num = int(ball)
                    if not (1 <= ball_num <= 35):
                        results['invalid_formats'].append(f"期号 {row['issue']}: 前区号码 {ball} 超出范围")
                
                # 检查后区号码格式
                back_balls = row['back_balls'].split(',')
                if len(back_balls) != 2:
                    results['invalid_formats'].append(f"期号 {row['issue']}: 后区号码数量不正确")
                
                for ball in back_balls:
                    ball_num = int(ball)
                    if not (1 <= ball_num <= 12):
                        results['invalid_formats'].append(f"期号 {row['issue']}: 后区号码 {ball} 超出范围")
                        
            except Exception as e:
                results['invalid_formats'].append(f"期号 {row['issue']}: 数据格式错误 - {e}")
        
        # 检查重复期号
        issue_counts = Counter(df['issue'])
        for issue, count in issue_counts.items():
            if count > 1:
                results['duplicate_issues'].append(f"期号 {issue}: {count}条记录")
        
        # 检查期号连续性（假设期号是连续的）
        issues = sorted([int(issue) for issue in df['issue']])
        for i in range(len(issues) - 1):
            if issues[i+1] - issues[i] != 1:
                results['issue_gaps'].append(f"期号 {issues[i]} 到 {issues[i+1]} 之间有间隔")
        
        return results
        
    except Exception as e:
        print(f"数据完整性检查失败: {e}")
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="大乐透数据去重工具")
    parser.add_argument("data_file", help="数据文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径")
    parser.add_argument("-k", "--keep", choices=['first', 'last'], default='first', 
                       help="保留策略，first保留第一个，last保留最后一个")
    parser.add_argument("-c", "--check", action="store_true", help="检查数据完整性")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_file):
        print(f"数据文件不存在: {args.data_file}")
        return
    
    if args.check:
        print("检查数据完整性...")
        results = check_data_integrity(args.data_file)
        if results:
            print(f"\n数据完整性检查结果:")
            print(f"总记录数: {results['total_records']}")
            
            if results['missing_values']:
                print(f"缺失值:")
                for col, count in results['missing_values'].items():
                    print(f"  {col}: {count}个缺失值")
            else:
                print("无缺失值")
            
            if results['invalid_formats']:
                print(f"格式错误:")
                for error in results['invalid_formats'][:10]:  # 只显示前10个错误
                    print(f"  {error}")
                if len(results['invalid_formats']) > 10:
                    print(f"  ... 还有 {len(results['invalid_formats']) - 10} 个错误")
            else:
                print("无格式错误")
            
            if results['duplicate_issues']:
                print(f"重复期号:")
                for dup in results['duplicate_issues']:
                    print(f"  {dup}")
            else:
                print("无重复期号")
            
            if results['issue_gaps']:
                print(f"期号间隔:")
                for gap in results['issue_gaps'][:5]:  # 只显示前5个间隔
                    print(f"  {gap}")
                if len(results['issue_gaps']) > 5:
                    print(f"  ... 还有 {len(results['issue_gaps']) - 5} 个间隔")
            else:
                print("期号连续")
    
    # 执行去重
    print("\n开始数据去重...")
    count = remove_duplicates(args.data_file, args.output, args.keep)
    if count > 0:
        print("数据去重完成")
    else:
        print("数据去重失败")


if __name__ == "__main__":
    main()
