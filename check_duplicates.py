#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
重复检查工具
用于检查大乐透数据中的重复记录和数据质量问题
"""

import os
import pandas as pd
import argparse
from collections import Counter, defaultdict


def check_duplicates(data_file):
    """检查数据文件中的重复记录
    
    Args:
        data_file: 数据文件路径
    
    Returns:
        检查结果字典
    """
    try:
        df = pd.read_csv(data_file)
        
        results = {
            'total_records': len(df),
            'duplicate_issues': {},
            'duplicate_combinations': {},
            'statistics': {}
        }
        
        print(f"检查数据文件: {data_file}")
        print(f"总记录数: {len(df)}")
        
        # 检查重复期号
        issue_counts = Counter(df['issue'])
        duplicate_issues = {issue: count for issue, count in issue_counts.items() if count > 1}
        
        if duplicate_issues:
            print(f"\n发现重复期号: {len(duplicate_issues)}个")
            results['duplicate_issues'] = duplicate_issues
            for issue, count in duplicate_issues.items():
                print(f"  期号 {issue}: {count}条记录")
                # 显示重复记录的详细信息
                dup_records = df[df['issue'] == issue]
                for idx, row in dup_records.iterrows():
                    print(f"    [{idx}] {row['date']} - 前区: {row['front_balls']}, 后区: {row['back_balls']}")
        else:
            print("\n未发现重复期号")
        
        # 检查重复号码组合
        combination_counts = Counter()
        for _, row in df.iterrows():
            combination = f"{row['front_balls']}|{row['back_balls']}"
            combination_counts[combination] += 1
        
        duplicate_combinations = {combo: count for combo, count in combination_counts.items() if count > 1}
        
        if duplicate_combinations:
            print(f"\n发现重复号码组合: {len(duplicate_combinations)}个")
            results['duplicate_combinations'] = duplicate_combinations
            for combo, count in duplicate_combinations.items():
                front_balls, back_balls = combo.split('|')
                print(f"  前区: {front_balls}, 后区: {back_balls} - 出现 {count} 次")
                # 显示这些重复组合对应的期号
                matching_records = df[(df['front_balls'] == front_balls) & (df['back_balls'] == back_balls)]
                issues = matching_records['issue'].tolist()
                print(f"    期号: {', '.join(map(str, issues))}")
        else:
            print("\n未发现重复号码组合")
        
        # 统计信息
        results['statistics'] = {
            'unique_issues': len(set(df['issue'])),
            'unique_combinations': len(set(combination_counts.keys())),
            'duplicate_issue_count': len(duplicate_issues),
            'duplicate_combination_count': len(duplicate_combinations)
        }
        
        print(f"\n统计信息:")
        print(f"  唯一期号数: {results['statistics']['unique_issues']}")
        print(f"  唯一号码组合数: {results['statistics']['unique_combinations']}")
        print(f"  重复期号数: {results['statistics']['duplicate_issue_count']}")
        print(f"  重复号码组合数: {results['statistics']['duplicate_combination_count']}")
        
        return results
        
    except Exception as e:
        print(f"重复检查失败: {e}")
        return None


def check_data_quality(data_file):
    """检查数据质量
    
    Args:
        data_file: 数据文件路径
    
    Returns:
        数据质量检查结果
    """
    try:
        df = pd.read_csv(data_file)
        
        quality_issues = {
            'format_errors': [],
            'range_errors': [],
            'missing_data': [],
            'inconsistent_data': []
        }
        
        print(f"\n检查数据质量...")
        
        for idx, row in df.iterrows():
            try:
                issue = row['issue']
                
                # 检查前区号码
                front_balls_str = str(row['front_balls'])
                if pd.isna(row['front_balls']) or front_balls_str == 'nan':
                    quality_issues['missing_data'].append(f"期号 {issue}: 前区号码缺失")
                    continue
                
                front_balls = front_balls_str.split(',')
                if len(front_balls) != 5:
                    quality_issues['format_errors'].append(f"期号 {issue}: 前区号码数量不正确 ({len(front_balls)}个)")
                    continue
                
                front_nums = []
                for ball in front_balls:
                    try:
                        ball_num = int(ball.strip())
                        if not (1 <= ball_num <= 35):
                            quality_issues['range_errors'].append(f"期号 {issue}: 前区号码 {ball_num} 超出范围 (1-35)")
                        front_nums.append(ball_num)
                    except ValueError:
                        quality_issues['format_errors'].append(f"期号 {issue}: 前区号码 '{ball}' 不是有效数字")
                
                # 检查前区号码是否有重复
                if len(set(front_nums)) != len(front_nums):
                    quality_issues['inconsistent_data'].append(f"期号 {issue}: 前区号码有重复")
                
                # 检查后区号码
                back_balls_str = str(row['back_balls'])
                if pd.isna(row['back_balls']) or back_balls_str == 'nan':
                    quality_issues['missing_data'].append(f"期号 {issue}: 后区号码缺失")
                    continue
                
                back_balls = back_balls_str.split(',')
                if len(back_balls) != 2:
                    quality_issues['format_errors'].append(f"期号 {issue}: 后区号码数量不正确 ({len(back_balls)}个)")
                    continue
                
                back_nums = []
                for ball in back_balls:
                    try:
                        ball_num = int(ball.strip())
                        if not (1 <= ball_num <= 12):
                            quality_issues['range_errors'].append(f"期号 {issue}: 后区号码 {ball_num} 超出范围 (1-12)")
                        back_nums.append(ball_num)
                    except ValueError:
                        quality_issues['format_errors'].append(f"期号 {issue}: 后区号码 '{ball}' 不是有效数字")
                
                # 检查后区号码是否有重复
                if len(set(back_nums)) != len(back_nums):
                    quality_issues['inconsistent_data'].append(f"期号 {issue}: 后区号码有重复")
                
            except Exception as e:
                quality_issues['format_errors'].append(f"期号 {issue}: 数据解析错误 - {e}")
        
        # 输出质量检查结果
        total_issues = sum(len(issues) for issues in quality_issues.values())
        
        if total_issues == 0:
            print("数据质量检查通过，未发现问题")
        else:
            print(f"发现 {total_issues} 个数据质量问题:")
            
            if quality_issues['format_errors']:
                print(f"\n格式错误 ({len(quality_issues['format_errors'])}个):")
                for error in quality_issues['format_errors'][:10]:
                    print(f"  {error}")
                if len(quality_issues['format_errors']) > 10:
                    print(f"  ... 还有 {len(quality_issues['format_errors']) - 10} 个格式错误")
            
            if quality_issues['range_errors']:
                print(f"\n范围错误 ({len(quality_issues['range_errors'])}个):")
                for error in quality_issues['range_errors'][:10]:
                    print(f"  {error}")
                if len(quality_issues['range_errors']) > 10:
                    print(f"  ... 还有 {len(quality_issues['range_errors']) - 10} 个范围错误")
            
            if quality_issues['missing_data']:
                print(f"\n缺失数据 ({len(quality_issues['missing_data'])}个):")
                for error in quality_issues['missing_data'][:10]:
                    print(f"  {error}")
                if len(quality_issues['missing_data']) > 10:
                    print(f"  ... 还有 {len(quality_issues['missing_data']) - 10} 个缺失数据")
            
            if quality_issues['inconsistent_data']:
                print(f"\n数据不一致 ({len(quality_issues['inconsistent_data'])}个):")
                for error in quality_issues['inconsistent_data'][:10]:
                    print(f"  {error}")
                if len(quality_issues['inconsistent_data']) > 10:
                    print(f"  ... 还有 {len(quality_issues['inconsistent_data']) - 10} 个不一致问题")
        
        return quality_issues
        
    except Exception as e:
        print(f"数据质量检查失败: {e}")
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="大乐透数据重复检查工具")
    parser.add_argument("data_file", help="数据文件路径")
    parser.add_argument("-q", "--quality", action="store_true", help="同时进行数据质量检查")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_file):
        print(f"数据文件不存在: {args.data_file}")
        return
    
    # 检查重复记录
    results = check_duplicates(args.data_file)
    
    # 如果需要，进行数据质量检查
    if args.quality:
        quality_results = check_data_quality(args.data_file)


if __name__ == "__main__":
    main()
