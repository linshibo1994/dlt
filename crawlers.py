#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据爬虫模块
支持从中彩网、500彩票网等数据源爬取大乐透历史开奖数据
"""

import os
import re
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from typing import List, Dict, Optional

from core_modules import logger_manager, data_manager


class BaseCrawler:
    """爬虫基类"""
    
    def __init__(self, data_file="data/dlt_data_all.csv"):
        self.data_file = data_file
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # 确保数据目录存在
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
    
    def save_data(self, data: List[Dict]):
        """保存数据到CSV文件（按期号倒序排列，最新期在前）"""
        if not data:
            logger_manager.warning("没有数据需要保存")
            return

        df = pd.DataFrame(data)

        # 确保期号是整数类型用于正确排序
        df['issue'] = df['issue'].astype(int)

        # 如果文件已存在，合并数据
        if os.path.exists(self.data_file):
            existing_df = pd.read_csv(self.data_file)
            existing_df['issue'] = existing_df['issue'].astype(int)

            # 合并数据并去重，保留最新数据
            combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=['issue'], keep='last')
        else:
            combined_df = df

        # 按期号倒序排列（最新期在前）
        combined_df = combined_df.sort_values('issue', ascending=False).reset_index(drop=True)

        combined_df.to_csv(self.data_file, index=False, encoding='utf-8')
        logger_manager.info(f"数据已保存到 {self.data_file}，共 {len(combined_df)} 期")
    
    def get_latest_issue(self) -> Optional[int]:
        """获取本地数据中的最新期号"""
        if os.path.exists(self.data_file):
            try:
                df = pd.read_csv(self.data_file)
                if not df.empty:
                    # 数据按期号倒序排列，第一行就是最新期
                    return int(df.iloc[0]['issue'])
            except Exception as e:
                logger_manager.error("读取本地数据失败", e)
        return None

    def crawl_recent_data(self, max_pages=3):
        """爬取最近的数据（增量更新）"""
        logger_manager.info(f"开始增量更新，最大页数: {max_pages}")

        all_data = []
        latest_local_issue = self.get_latest_issue()

        # 获取最新的远程数据来确定需要更新的期数
        first_page_data = self.crawl_page(1)
        if not first_page_data:
            logger_manager.warning("无法获取远程数据")
            return 0

        latest_remote_issue = int(first_page_data[0]['issue'])

        if latest_local_issue:
            if latest_remote_issue <= latest_local_issue:
                logger_manager.info(f"本地数据已是最新（本地: {latest_local_issue}, 远程: {latest_remote_issue}）")
                return 0

            missing_count = latest_remote_issue - latest_local_issue
            logger_manager.info(f"发现 {missing_count} 期新数据需要更新")
        else:
            logger_manager.info("本地无数据，开始初始化")

        # 爬取数据直到获取所有缺失的期数
        for page in range(1, max_pages + 1):
            page_data = self.crawl_page(page)

            if not page_data:
                break

            # 如果有本地最新期号，只保留比它新的数据
            if latest_local_issue:
                new_data = []
                for item in page_data:
                    current_issue = int(item['issue'])
                    if current_issue > latest_local_issue:
                        new_data.append(item)
                    else:
                        # 遇到已存在的期号，停止爬取
                        logger_manager.info(f"遇到已存在期号 {current_issue}，停止爬取")
                        break

                all_data.extend(new_data)

                # 如果这一页没有新数据，停止爬取
                if not new_data:
                    break
            else:
                all_data.extend(page_data)

        if all_data:
            self.save_data(all_data)
            logger_manager.info(f"增量更新完成，新增 {len(all_data)} 期数据")
        else:
            logger_manager.info("没有新数据需要更新")

        return len(all_data)

    def crawl_all_data(self, max_pages=100):
        """爬取所有数据"""
        logger_manager.info(f"开始爬取所有数据，最大页数: {max_pages}")

        all_data = []

        for page in range(1, max_pages + 1):
            page_data = self.crawl_page(page)

            if not page_data:
                logger_manager.info(f"第 {page} 页没有数据，停止爬取")
                break

            all_data.extend(page_data)
            logger_manager.info(f"已爬取 {len(all_data)} 期数据")

            # 添加延迟避免请求过快
            time.sleep(0.5)

        if all_data:
            self.save_data(all_data)
            logger_manager.info(f"爬取完成，共获取 {len(all_data)} 期数据")

        return len(all_data)


class ZhcwCrawler(BaseCrawler):
    """中彩网爬虫（使用API接口）"""

    def __init__(self, data_file="data/dlt_data_all.csv"):
        super().__init__(data_file)
        self.api_url = "https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry"
        self.session.headers.update({
            'Referer': 'https://www.zhcw.com/'
        })

    def crawl_page(self, page: int = 1) -> List[Dict]:
        """爬取指定页面的数据"""
        try:
            # 构建API请求参数
            params = {
                'gameNo': '85',  # 大乐透游戏编号
                'provinceId': '0',
                'pageSize': '30',
                'isVerify': '1',
                'pageNo': str(page)
            }

            # 发送API请求
            response = self.session.get(self.api_url, params=params, timeout=15)
            response.raise_for_status()

            # 解析JSON响应
            data_json = response.json()

            if not data_json.get('success'):
                logger_manager.error(f"API请求失败: {data_json.get('errorMessage', '未知错误')}")
                return []

            # 提取数据列表
            value = data_json.get('value', {})
            lottery_list = value.get('list', [])

            if not lottery_list:
                logger_manager.warning(f"第 {page} 页没有数据")
                return []

            data = []
            for item in lottery_list:
                try:
                    # 提取期号
                    issue = item.get('lotteryDrawNum', '').strip()

                    # 提取开奖日期
                    date = item.get('lotteryDrawTime', '').strip()

                    # 提取开奖号码
                    numbers_text = item.get('lotteryDrawResult', '').strip()

                    if issue and date and numbers_text:
                        # 分割号码
                        numbers = numbers_text.split()

                        if len(numbers) >= 7:
                            front_balls = [num.zfill(2) for num in numbers[:5]]
                            back_balls = [num.zfill(2) for num in numbers[5:7]]

                            data.append({
                                'issue': issue,
                                'date': date,
                                'front_balls': ','.join(front_balls),
                                'back_balls': ','.join(back_balls)
                            })

                except Exception as e:
                    logger_manager.warning(f"解析数据项失败: {e}")
                    continue

            logger_manager.info(f"页面 {page} 爬取到 {len(data)} 期数据")
            return data

        except Exception as e:
            logger_manager.error(f"爬取页面 {page} 失败: {e}")
            return []
    
    def crawl_recent_data(self, max_pages_or_periods) -> int:
        """爬取最近的数据（支持按页数或期数）"""
        if isinstance(max_pages_or_periods, int) and max_pages_or_periods <= 10:
            # 如果是小数字，认为是页数（增量更新）
            return self._crawl_by_pages(max_pages_or_periods)
        else:
            # 否则认为是期数
            return self._crawl_by_periods(int(max_pages_or_periods))

    def _crawl_by_pages(self, max_pages: int) -> int:
        """按页数爬取（增量更新）"""
        logger_manager.info(f"开始增量更新，最大页数: {max_pages}")

        all_data = []
        latest_local_issue = self.get_latest_issue()

        # 获取最新的远程数据来确定需要更新的期数
        first_page_data = self.crawl_page(1)
        if not first_page_data:
            logger_manager.warning("无法获取远程数据")
            return 0

        latest_remote_issue = int(first_page_data[0]['issue'])

        if latest_local_issue:
            if latest_remote_issue <= latest_local_issue:
                logger_manager.info(f"本地数据已是最新（本地: {latest_local_issue}, 远程: {latest_remote_issue}）")
                return 0

            missing_count = latest_remote_issue - latest_local_issue
            logger_manager.info(f"发现 {missing_count} 期新数据需要更新")
        else:
            logger_manager.info("本地无数据，开始初始化")

        # 爬取数据直到获取所有缺失的期数
        for page in range(1, max_pages + 1):
            page_data = self.crawl_page(page)

            if not page_data:
                break

            # 如果有本地最新期号，只保留比它新的数据
            if latest_local_issue:
                new_data = []
                for item in page_data:
                    current_issue = int(item['issue'])

                    if current_issue > latest_local_issue:
                        new_data.append(item)
                    else:
                        # 遇到已存在的期号，停止爬取
                        logger_manager.info(f"遇到已存在期号 {current_issue}，停止爬取")
                        break

                all_data.extend(new_data)

                # 如果这一页没有新数据，停止爬取
                if not new_data:
                    break
            else:
                all_data.extend(page_data)

            time.sleep(0.5)

        if all_data:
            self.save_data(all_data)
            logger_manager.info(f"增量更新完成，新增 {len(all_data)} 期数据")
        else:
            logger_manager.info("没有新数据需要更新")

        return len(all_data)

    def _crawl_by_periods(self, periods: int) -> int:
        """按期数爬取"""
        logger_manager.info(f"开始爬取最近 {periods} 期数据...")

        all_data = []
        pages_needed = (periods // 30) + 1  # 每页大约30期数据

        for page in range(1, pages_needed + 1):
            logger_manager.info(f"正在爬取第 {page} 页...")

            page_data = self.crawl_page(page)
            if not page_data:
                logger_manager.warning(f"第 {page} 页无数据，停止爬取")
                break

            all_data.extend(page_data)

            # 如果已经获取足够的数据，停止爬取
            if len(all_data) >= periods:
                all_data = all_data[:periods]
                break

            # 延时避免被封
            time.sleep(0.5)

        if all_data:
            self.save_data(all_data)
            logger_manager.info(f"爬取完成，共获取 {len(all_data)} 期数据")
        else:
            logger_manager.info("没有获取到数据")

        return len(all_data)

    def crawl_all_data(self, max_pages: int = 100) -> int:
        """爬取所有数据"""
        logger_manager.info("开始从中彩网爬取大乐透数据...")

        all_data = []
        latest_local_issue = self.get_latest_issue()

        for page in range(1, max_pages + 1):
            logger_manager.info(f"正在爬取第 {page} 页...")

            page_data = self.crawl_page(page)
            if not page_data:
                logger_manager.warning(f"第 {page} 页无数据，停止爬取")
                break

            # 检查是否已经爬取到本地最新数据
            if latest_local_issue:
                page_issues = [item['issue'] for item in page_data]
                if latest_local_issue in page_issues:
                    # 只保留比本地最新期号更新的数据
                    new_data = [item for item in page_data if item['issue'] > latest_local_issue]
                    all_data.extend(new_data)
                    logger_manager.info(f"已爬取到本地最新数据 {latest_local_issue}，停止爬取")
                    break

            all_data.extend(page_data)

            # 延时避免被封
            time.sleep(1)

        if all_data:
            self.save_data(all_data)
            logger_manager.info(f"爬取完成，共获取 {len(all_data)} 期新数据")
        else:
            logger_manager.info("没有新数据需要更新")

        return len(all_data)


class Crawler500(BaseCrawler):
    """500彩票网爬虫"""
    
    def __init__(self, data_file="data/dlt_data_all.csv"):
        super().__init__(data_file)
        self.base_url = "https://www.500.com"
        self.dlt_url = "https://www.500.com/static/public/dlt/xml/dltdata.xml"
    
    def crawl_xml_data(self) -> List[Dict]:
        """从XML接口爬取数据"""
        try:
            response = self.session.get(self.dlt_url, timeout=10)
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            # 解析XML数据
            from xml.etree import ElementTree as ET
            root = ET.fromstring(response.text)
            
            data = []
            for row in root.findall('.//row'):
                try:
                    # 解析属性
                    issue = row.get('expect')
                    date = row.get('opentime')
                    opencode = row.get('opencode')
                    
                    if opencode:
                        # 解析号码 (格式: 01,02,03,04,05+06,07)
                        parts = opencode.split('+')
                        if len(parts) == 2:
                            front_balls = parts[0]
                            back_balls = parts[1]
                            
                            data.append({
                                'issue': issue,
                                'date': date,
                                'front_balls': front_balls,
                                'back_balls': back_balls
                            })
                
                except Exception as e:
                    logger_manager.error(f"解析XML行数据失败: {e}")
                    continue
            
            logger_manager.info(f"从500彩票网爬取到 {len(data)} 期数据")
            return data
        
        except Exception as e:
            logger_manager.error("从500彩票网爬取数据失败", e)
            return []
    
    def crawl_all_data(self) -> int:
        """爬取所有数据"""
        logger_manager.info("开始从500彩票网爬取大乐透数据...")
        
        all_data = self.crawl_xml_data()
        
        if all_data:
            # 过滤新数据
            latest_local_issue = self.get_latest_issue()
            if latest_local_issue:
                new_data = [item for item in all_data if int(item['issue']) > latest_local_issue]
                if new_data:
                    self.save_data(new_data)
                    logger_manager.info(f"爬取完成，共获取 {len(new_data)} 期新数据")
                    return len(new_data)
                else:
                    logger_manager.info("没有新数据需要更新")
                    return 0
            else:
                self.save_data(all_data)
                logger_manager.info(f"爬取完成，共获取 {len(all_data)} 期数据")
                return len(all_data)
        else:
            logger_manager.warning("未获取到任何数据")
            return 0

    def crawl_recent_data(self, max_pages_or_periods) -> int:
        """爬取最近的数据（500彩票网只支持全量获取）"""
        logger_manager.info("500彩票网爬虫执行增量更新...")
        return self.crawl_all_data()


def update_data(source: str = "zhcw") -> int:
    """更新数据的便捷函数"""
    if source == "zhcw":
        crawler = ZhcwCrawler()
    elif source == "500":
        crawler = Crawler500()
    else:
        logger_manager.error(f"不支持的数据源: {source}")
        return 0
    
    return crawler.crawl_all_data()


if __name__ == "__main__":
    # 测试爬虫
    print("🕷️ 测试数据爬虫...")
    
    # 测试中彩网爬虫
    print("📊 测试中彩网爬虫...")
    zhcw_crawler = ZhcwCrawler()
    zhcw_count = zhcw_crawler.crawl_all_data(max_pages=2)
    print(f"中彩网爬取结果: {zhcw_count} 期")
    
    # 测试500彩票网爬虫
    print("📊 测试500彩票网爬虫...")
    crawler500 = Crawler500()
    count500 = crawler500.crawl_all_data()
    print(f"500彩票网爬取结果: {count500} 期")
    
    print("✅ 爬虫模块测试完成")
