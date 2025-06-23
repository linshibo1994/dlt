#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
大乐透数据爬虫模块 - 500彩票网数据源
从500彩票网获取大乐透历史开奖数据
"""

import os
import csv
import time
import random
import requests
from bs4 import BeautifulSoup
import re


class DLT500Crawler:
    """大乐透500彩票网爬虫
    用于从500彩票网获取大乐透开奖数据
    """

    def __init__(self, data_dir="data"):
        """初始化爬虫
        
        Args:
            data_dir: 数据保存目录，默认为data
        """
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
        """从500彩票网获取大乐透历史开奖数据

        Args:
            count: 获取的期数，默认50期
            get_all: 是否获取所有历史数据，如果为True则忽略count参数

        Returns:
            开奖结果列表
        """
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
        """获取一批数据

        Args:
            limit: 获取数量
            offset: 偏移量

        Returns:
            开奖结果列表
        """
        results = []

        try:
            # 设置请求参数
            params = {
                'limit': limit,
                'sort': 0
            }

            # 如果有偏移量，需要调整请求方式
            if offset > 0:
                # 500彩票网可能不支持offset，这里用分页方式
                params['start'] = offset

            # 发送请求获取页面
            response = requests.get(self.api_url, headers=self.headers, params=params, timeout=30)
            response.encoding = 'gb2312'  # 500彩票网使用gb2312编码

            if response.status_code != 200:
                print(f"请求失败，状态码: {response.status_code}")
                return results

            # 解析HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # 查找数据表格 - 根据博客中的信息，应该查找class="chart"的div
            table = soup.find('div', {'class': 'chart'})
            if not table:
                print("未找到开奖数据表格")
                return results

            # 解析表格行
            rows = table.find_all('tr')

            for i, row in enumerate(rows):
                try:
                    cells = row.find_all('td')
                    if len(cells) != 15:  # 根据博客，应该有15个td
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

    def save_to_csv(self, results, filename="dlt_data.csv"):
        """保存数据到CSV文件
        
        Args:
            results: 开奖结果列表
            filename: 文件名
        
        Returns:
            保存的文件路径
        """
        if not results:
            print("没有数据需要保存")
            return None
        
        try:
            # 如果filename已经包含路径，直接使用；否则加上data_dir
            if os.path.dirname(filename):
                file_path = filename
            else:
                file_path = os.path.join(self.data_dir, filename)
            
            # 写入CSV文件
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['issue', 'date', 'front_balls', 'back_balls']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # 写入表头
                writer.writeheader()
                
                # 写入数据
                for result in results:
                    writer.writerow(result)
            
            print(f"数据已保存到: {file_path}")
            print(f"共保存 {len(results)} 条记录")
            
            return file_path
            
        except Exception as e:
            print(f"保存数据失败: {e}")
            return None

    def update_data(self, existing_file, new_count=10):
        """更新现有数据文件
        
        Args:
            existing_file: 现有数据文件路径
            new_count: 获取最新数据的期数
        
        Returns:
            更新后的记录总数
        """
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


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="大乐透500彩票网数据爬虫")
    parser.add_argument("-c", "--count", type=int, default=50, help="获取的期数")
    parser.add_argument("-o", "--output", default="dlt_data.csv", help="输出文件名")
    parser.add_argument("-u", "--update", help="更新现有数据文件")
    parser.add_argument("-n", "--new-count", type=int, default=10, help="更新时获取的最新期数")
    parser.add_argument("-a", "--all", action="store_true", help="获取所有历史数据")

    args = parser.parse_args()

    crawler = DLT500Crawler()

    if args.update:
        # 更新现有数据
        crawler.update_data(args.update, args.new_count)
    else:
        # 获取新数据
        if args.all:
            results = crawler.get_history_data(get_all=True)
        else:
            results = crawler.get_history_data(args.count)

        if results:
            crawler.save_to_csv(results, args.output)
        else:
            print("未获取到数据")


if __name__ == "__main__":
    main()
