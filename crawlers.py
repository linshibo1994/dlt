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
        """保存数据到CSV文件"""
        if not data:
            logger_manager.warning("没有数据需要保存")
            return
        
        df = pd.DataFrame(data)
        
        # 如果文件已存在，合并数据
        if os.path.exists(self.data_file):
            existing_df = pd.read_csv(self.data_file)
            # 去重，保留最新数据
            combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=['issue'], keep='last')
            combined_df = combined_df.sort_values('issue').reset_index(drop=True)
        else:
            combined_df = df.sort_values('issue').reset_index(drop=True)
        
        combined_df.to_csv(self.data_file, index=False, encoding='utf-8')
        logger_manager.info(f"数据已保存到 {self.data_file}，共 {len(combined_df)} 期")
    
    def get_latest_issue(self) -> Optional[str]:
        """获取本地数据中的最新期号"""
        if os.path.exists(self.data_file):
            try:
                df = pd.read_csv(self.data_file)
                if not df.empty:
                    return df.iloc[-1]['issue']
            except Exception as e:
                logger_manager.error("读取本地数据失败", e)
        return None


class ZhcwCrawler(BaseCrawler):
    """中彩网爬虫"""
    
    def __init__(self, data_file="data/dlt_data_all.csv"):
        super().__init__(data_file)
        self.base_url = "https://www.zhcw.com"
        self.dlt_url = "https://www.zhcw.com/kjxx/dlt/"
    
    def crawl_page(self, page: int = 1) -> List[Dict]:
        """爬取指定页面的数据"""
        url = f"{self.dlt_url}?page={page}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找开奖数据表格
            table = soup.find('table', class_='kjjg_table')
            if not table:
                logger_manager.warning(f"页面 {page} 未找到数据表格")
                return []
            
            data = []
            rows = table.find('tbody').find_all('tr')
            
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 4:
                    try:
                        # 解析期号
                        issue = cols[0].text.strip()
                        
                        # 解析开奖日期
                        date = cols[1].text.strip()
                        
                        # 解析开奖号码
                        balls_td = cols[2]
                        ball_spans = balls_td.find_all('span')
                        
                        if len(ball_spans) >= 7:
                            front_balls = [span.text.strip() for span in ball_spans[:5]]
                            back_balls = [span.text.strip() for span in ball_spans[5:7]]
                            
                            data.append({
                                'issue': issue,
                                'date': date,
                                'front_balls': ','.join(front_balls),
                                'back_balls': ','.join(back_balls)
                            })
                    
                    except Exception as e:
                        logger_manager.error(f"解析行数据失败: {e}")
                        continue
            
            logger_manager.info(f"页面 {page} 爬取到 {len(data)} 期数据")
            return data
        
        except Exception as e:
            logger_manager.error(f"爬取页面 {page} 失败", e)
            return []
    
    def crawl_recent_data(self, periods: int) -> int:
        """爬取最近指定期数的数据"""
        logger_manager.info(f"开始从中彩网爬取最近 {periods} 期数据...")

        all_data = []
        pages_needed = (periods // 20) + 1  # 每页大约20期数据

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
            time.sleep(1)

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
                new_data = [item for item in all_data if item['issue'] > latest_local_issue]
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
