#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据管理器
整合数据爬虫、数据处理、工具函数等功能
"""

import os
import csv
import time
import random
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataCrawler:
    """数据爬虫"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        
        # 确保数据目录存在
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # 设置请求头
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        
        # 500彩票网大乐透历史数据API URL
        self.api_url = "https://datachart.500.com/dlt/history/newinc/history.php"
    
    def get_history_data(self, count=50, get_all=False):
        """获取历史开奖数据"""
        results = []
        
        try:
            if get_all:
                print("开始从500彩票网获取所有历史大乐透数据...")
                # 分批获取所有数据，每次最多500期
                batch_size = 500
                total_fetched = 0
                
                while True:
                    batch_results = self._fetch_batch_data(batch_size, total_fetched)
                    if not batch_results:
                        break
                    
                    results.extend(batch_results)
                    total_fetched += len(batch_results)
                    print(f"已获取 {total_fetched} 期数据...")
                    
                    # 如果这批数据少于batch_size，说明已经获取完所有数据
                    if len(batch_results) < batch_size:
                        break
                    
                    # 添加延迟避免请求过于频繁
                    time.sleep(2)
                
                print(f"全量获取完成，共获取 {len(results)} 期大乐透开奖数据")
            else:
                print(f"开始从500彩票网获取最近{count}期大乐透数据...")
                results = self._fetch_batch_data(count, 0)
                print(f"成功获取 {len(results)} 期大乐透开奖数据")
        
        except Exception as e:
            print(f"获取数据失败: {e}")
        
        return results
    
    def _fetch_batch_data(self, limit, offset):
        """获取一批数据"""
        results = []
        
        try:
            # 设置请求参数
            params = {
                'limit': limit,
                'sort': 0
            }
            
            # 如果有偏移量，需要调整请求方式
            if offset > 0:
                params['start'] = offset
            
            # 发送请求获取页面
            response = requests.get(self.api_url, headers=self.headers, params=params, timeout=30)
            response.encoding = 'gb2312'  # 500彩票网使用gb2312编码
            
            if response.status_code != 200:
                print(f"请求失败，状态码: {response.status_code}")
                return results
            
            # 解析HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找数据表格
            table = soup.find('div', {'class': 'chart'})
            if not table:
                print("未找到开奖数据表格")
                return results
            
            # 解析表格行
            rows = table.find_all('tr')
            
            for i, row in enumerate(rows):
                try:
                    cells = row.find_all('td')
                    if len(cells) != 15:  # 根据实际情况调整
                        continue
                    
                    # 提取期号 (第0列)
                    issue = cells[0].get_text().strip()
                    
                    # 提取开奖日期 (第14列)
                    date = cells[14].get_text().strip()
                    
                    # 提取前区号码 (第1-5列)
                    front_balls = []
                    for j in range(1, 6):
                        ball = cells[j].get_text().strip()
                        if ball.isdigit():
                            front_balls.append(ball.zfill(2))
                    
                    # 提取后区号码 (第6-7列)
                    back_balls = []
                    for j in range(6, 8):
                        ball = cells[j].get_text().strip()
                        if ball.isdigit():
                            back_balls.append(ball.zfill(2))
                    
                    # 验证数据完整性
                    if len(front_balls) == 5 and len(back_balls) == 2 and issue.isdigit():
                        result = {
                            "issue": issue,
                            "date": date,
                            "front_balls": ",".join(front_balls),
                            "back_balls": ",".join(back_balls)
                        }
                        results.append(result)
                        if not offset:  # 只在第一批数据时显示详细信息
                            print(f"获取第{issue}期数据: 前区 {','.join(front_balls)}, 后区 {','.join(back_balls)}")
                
                except Exception as e:
                    print(f"解析第{i+1}行数据失败: {e}")
                    continue
        
        except Exception as e:
            print(f"获取批次数据失败: {e}")
        
        return results
    
    def save_to_csv(self, results, filename):
        """保存数据到CSV文件"""
        if not results:
            print("没有数据需要保存")
            return
        
        try:
            filepath = os.path.join(self.data_dir, filename)
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['issue', 'date', 'front_balls', 'back_balls']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for result in results:
                    writer.writerow(result)
            
            print(f"数据已保存到: {filepath}")
            print(f"共保存 {len(results)} 条记录")
            
        except Exception as e:
            print(f"保存数据失败: {e}")
    
    def update_data(self, existing_file, new_count=10):
        """更新现有数据文件"""
        try:
            # 读取现有数据
            existing_data = []
            existing_issues = set()
            
            if os.path.exists(existing_file):
                with open(existing_file, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        existing_data.append(row)
                        existing_issues.add(row['issue'])
                print(f"现有数据文件包含 {len(existing_data)} 条记录")
            else:
                print("现有数据文件不存在，将创建新文件")
            
            # 获取最新数据
            print(f"获取最新 {new_count} 期数据...")
            new_results = self.get_history_data(new_count)
            
            if not new_results:
                print("未获取到新数据")
                return len(existing_data)
            
            # 过滤出真正的新数据
            truly_new_results = []
            for result in new_results:
                if result['issue'] not in existing_issues:
                    truly_new_results.append(result)
            
            if not truly_new_results:
                print("没有新的数据需要更新")
                return len(existing_data)
            
            print(f"发现 {len(truly_new_results)} 条新数据")
            
            # 合并数据（新数据在前）
            all_results = truly_new_results + existing_data
            
            # 按期号排序（降序）
            all_results.sort(key=lambda x: int(x["issue"]), reverse=True)
            
            # 保存更新后的数据
            self.save_to_csv(all_results, os.path.basename(existing_file))
            
            print(f"数据更新完成，总记录数: {len(all_results)}")
            return len(all_results)
            
        except Exception as e:
            print(f"数据更新失败: {e}")
            return 0


class DataProcessor:
    """数据处理器"""
    
    def __init__(self, data_file):
        self.data_file = data_file
        self.df = None
        self.load_data()
    
    def load_data(self):
        """加载数据"""
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"数据加载成功: {len(self.df)} 条记录")
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def remove_duplicates(self, output_file=None, keep='first'):
        """去除重复记录"""
        if self.df is None:
            print("数据未加载")
            return 0
        
        try:
            original_count = len(self.df)
            print(f"原始记录数: {original_count}")
            
            # 去除重复记录（基于期号）
            df_dedup = self.df.drop_duplicates(subset=['issue'], keep=keep)
            dedup_count = len(df_dedup)
            
            print(f"去重后记录数: {dedup_count}")
            print(f"删除重复记录: {original_count - dedup_count}条")
            
            # 按期号排序（降序）
            df_dedup = df_dedup.sort_values('issue', ascending=False).reset_index(drop=True)
            
            # 保存结果
            if output_file is None:
                output_file = self.data_file
            
            df_dedup.to_csv(output_file, index=False)
            print(f"去重后的数据已保存到: {output_file}")
            
            # 更新内存中的数据
            self.df = df_dedup
            
            return dedup_count
            
        except Exception as e:
            print(f"数据去重失败: {e}")
            return 0
    
    def check_duplicates(self):
        """检查重复记录"""
        if self.df is None:
            print("数据未加载")
            return {}
        
        try:
            results = {
                'total_records': len(self.df),
                'duplicate_issues': {},
                'duplicate_combinations': {},
                'statistics': {}
            }
            
            print(f"检查数据文件: {self.data_file}")
            print(f"总记录数: {len(self.df)}")
            
            # 检查重复期号
            issue_counts = Counter(self.df['issue'])
            duplicate_issues = {issue: count for issue, count in issue_counts.items() if count > 1}
            
            if duplicate_issues:
                print(f"\n发现重复期号: {len(duplicate_issues)}个")
                results['duplicate_issues'] = duplicate_issues
                for issue, count in duplicate_issues.items():
                    print(f"  期号 {issue}: {count}条记录")
            else:
                print("\n未发现重复期号")
            
            # 检查重复号码组合
            self.df['combination'] = self.df['front_balls'] + '|' + self.df['back_balls']
            combo_counts = Counter(self.df['combination'])
            duplicate_combos = {combo: count for combo, count in combo_counts.items() if count > 1}
            
            if duplicate_combos:
                print(f"\n发现重复号码组合: {len(duplicate_combos)}个")
                results['duplicate_combinations'] = duplicate_combos
                for combo, count in list(duplicate_combos.items())[:5]:  # 只显示前5个
                    print(f"  组合 {combo}: {count}次")
            else:
                print("\n未发现重复号码组合")
            
            # 统计信息
            results['statistics'] = {
                'unique_issues': len(issue_counts),
                'unique_combinations': len(combo_counts),
                'duplicate_issue_count': len(duplicate_issues),
                'duplicate_combination_count': len(duplicate_combos)
            }
            
            return results
            
        except Exception as e:
            print(f"重复检查失败: {e}")
            return {}
    
    def validate_data(self):
        """验证数据质量"""
        if self.df is None:
            print("数据未加载")
            return False
        
        try:
            quality_issues = []
            
            print(f"\n检查数据质量...")
            
            for idx, row in self.df.iterrows():
                try:
                    issue = row['issue']
                    
                    # 检查前区号码
                    front_balls_str = str(row['front_balls'])
                    if pd.isna(row['front_balls']) or front_balls_str == 'nan':
                        quality_issues.append(f"期号 {issue}: 前区号码缺失")
                        continue
                    
                    front_balls = front_balls_str.split(',')
                    if len(front_balls) != 5:
                        quality_issues.append(f"期号 {issue}: 前区号码数量不正确 ({len(front_balls)}个)")
                        continue
                    
                    for ball in front_balls:
                        try:
                            ball_num = int(ball.strip())
                            if not (1 <= ball_num <= 35):
                                quality_issues.append(f"期号 {issue}: 前区号码 {ball_num} 超出范围 (1-35)")
                        except ValueError:
                            quality_issues.append(f"期号 {issue}: 前区号码 '{ball}' 不是有效数字")
                    
                    # 检查后区号码
                    back_balls_str = str(row['back_balls'])
                    if pd.isna(row['back_balls']) or back_balls_str == 'nan':
                        quality_issues.append(f"期号 {issue}: 后区号码缺失")
                        continue
                    
                    back_balls = back_balls_str.split(',')
                    if len(back_balls) != 2:
                        quality_issues.append(f"期号 {issue}: 后区号码数量不正确 ({len(back_balls)}个)")
                        continue
                    
                    for ball in back_balls:
                        try:
                            ball_num = int(ball.strip())
                            if not (1 <= ball_num <= 12):
                                quality_issues.append(f"期号 {issue}: 后区号码 {ball_num} 超出范围 (1-12)")
                        except ValueError:
                            quality_issues.append(f"期号 {issue}: 后区号码 '{ball}' 不是有效数字")
                
                except Exception as e:
                    quality_issues.append(f"期号 {issue}: 数据解析错误 - {e}")
            
            # 输出质量检查结果
            if len(quality_issues) == 0:
                print("✅ 数据质量检查通过，未发现问题")
                return True
            else:
                print(f"⚠️ 发现 {len(quality_issues)} 个数据质量问题:")
                for issue in quality_issues[:10]:  # 只显示前10个
                    print(f"  - {issue}")
                if len(quality_issues) > 10:
                    print(f"  ... 还有 {len(quality_issues) - 10} 个问题")
                return False
            
        except Exception as e:
            print(f"数据质量检查失败: {e}")
            return False


class DataUtils:
    """数据工具函数"""
    
    @staticmethod
    def generate_random_numbers():
        """生成随机大乐透号码"""
        # 生成5个不重复的前区号码（1-35）
        front_balls = sorted(np.random.choice(range(1, 36), 5, replace=False))
        # 生成2个不重复的后区号码（1-12）
        back_balls = sorted(np.random.choice(range(1, 13), 2, replace=False))
        
        return front_balls, back_balls
    
    @staticmethod
    def format_numbers(front_balls, back_balls):
        """格式化大乐透号码"""
        front_str = " ".join([f"{ball:02d}" for ball in front_balls])
        back_str = " ".join([f"{ball:02d}" for ball in back_balls])
        
        return f"前区: {front_str} | 后区: {back_str}"
    
    @staticmethod
    def calculate_prize(my_fronts, my_backs, winning_fronts, winning_backs):
        """计算中奖等级"""
        # 计算前区匹配数
        front_matches = len(set(my_fronts) & set(winning_fronts))
        # 计算后区匹配数
        back_matches = len(set(my_backs) & set(winning_backs))
        
        # 根据匹配数判断中奖等级
        if front_matches == 5 and back_matches == 2:
            return 1  # 一等奖
        elif front_matches == 5 and back_matches == 1:
            return 2  # 二等奖
        elif front_matches == 5 and back_matches == 0:
            return 3  # 三等奖
        elif front_matches == 4 and back_matches == 2:
            return 4  # 四等奖
        elif front_matches == 4 and back_matches == 1:
            return 5  # 五等奖
        elif front_matches == 3 and back_matches == 2:
            return 6  # 六等奖
        elif front_matches == 4 and back_matches == 0:
            return 7  # 七等奖
        elif (front_matches == 3 and back_matches == 1) or (front_matches == 2 and back_matches == 2):
            return 8  # 八等奖
        elif (front_matches == 3 and back_matches == 0) or (front_matches == 1 and back_matches == 2) or \
             (front_matches == 2 and back_matches == 1) or (front_matches == 0 and back_matches == 2):
            return 9  # 九等奖
        else:
            return 0  # 未中奖
    
    @staticmethod
    def get_prize_name(prize_level):
        """获取奖项名称"""
        prize_names = {
            0: "未中奖",
            1: "一等奖",
            2: "二等奖",
            3: "三等奖",
            4: "四等奖",
            5: "五等奖",
            6: "六等奖",
            7: "七等奖",
            8: "八等奖",
            9: "九等奖"
        }
        return prize_names.get(prize_level, "未知")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="大乐透数据管理器")
    parser.add_argument("action", choices=['crawl', 'process', 'check', 'validate'], help="操作类型")
    parser.add_argument("-d", "--data", default="data/dlt_data_all.csv", help="数据文件路径")
    parser.add_argument("-c", "--count", type=int, default=50, help="爬取期数")
    parser.add_argument("-a", "--all", action="store_true", help="爬取所有历史数据")
    parser.add_argument("-u", "--update", action="store_true", help="更新现有数据")
    parser.add_argument("--dedup", action="store_true", help="去除重复记录")
    
    args = parser.parse_args()
    
    if args.action == 'crawl':
        crawler = DataCrawler()
        
        if args.update:
            crawler.update_data(args.data, args.count)
        else:
            if args.all:
                results = crawler.get_history_data(get_all=True)
            else:
                results = crawler.get_history_data(args.count)
            
            if results:
                crawler.save_to_csv(results, os.path.basename(args.data))
    
    elif args.action == 'process':
        if not os.path.exists(args.data):
            print(f"数据文件不存在: {args.data}")
            return
        
        processor = DataProcessor(args.data)
        
        if args.dedup:
            processor.remove_duplicates()
    
    elif args.action == 'check':
        if not os.path.exists(args.data):
            print(f"数据文件不存在: {args.data}")
            return
        
        processor = DataProcessor(args.data)
        processor.check_duplicates()
    
    elif args.action == 'validate':
        if not os.path.exists(args.data):
            print(f"数据文件不存在: {args.data}")
            return
        
        processor = DataProcessor(args.data)
        processor.validate_data()


if __name__ == "__main__":
    main()
