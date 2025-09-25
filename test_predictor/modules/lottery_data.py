#!/usr/bin/env python3
"""
大乐透开奖数据管理模块
负责读取、解析和管理历史开奖数据
"""

import csv
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple


class LotteryData:
    """大乐透开奖数据管理器"""
    
    def __init__(self, data_file_path: str = None):
        if data_file_path is None:
            # 默认数据文件路径
            self.data_file_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'data', 'dlt_data_all.csv'
            )
        else:
            self.data_file_path = data_file_path
        
        self._data_cache = None
        self._last_modified = None

    def _check_file_exists(self) -> bool:
        """检查数据文件是否存在"""
        return os.path.exists(self.data_file_path)

    def _get_file_modified_time(self) -> float:
        """获取文件修改时间"""
        if not self._check_file_exists():
            return 0
        return os.path.getmtime(self.data_file_path)

    def _should_reload_data(self) -> bool:
        """检查是否需要重新加载数据"""
        current_modified = self._get_file_modified_time()
        return (self._data_cache is None or 
                self._last_modified is None or 
                current_modified > self._last_modified)

    def load_data(self, force_reload: bool = False) -> List[Dict]:
        """加载开奖数据"""
        if not force_reload and not self._should_reload_data():
            return self._data_cache

        if not self._check_file_exists():
            raise FileNotFoundError(f"开奖数据文件不存在: {self.data_file_path}")

        data = []
        try:
            with open(self.data_file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # 验证数据格式
                    if self._validate_row(row):
                        processed_row = self._process_row(row)
                        data.append(processed_row)
                    else:
                        print(f"警告: 跳过无效数据行 - {row}")
            
            # 按期号排序（最新的在前）
            data.sort(key=lambda x: x['issue'], reverse=True)
            
            self._data_cache = data
            self._last_modified = self._get_file_modified_time()
            
            print(f"成功加载 {len(data)} 期开奖数据")
            return data
            
        except Exception as e:
            raise Exception(f"加载开奖数据失败: {str(e)}")

    def _validate_row(self, row: Dict) -> bool:
        """验证数据行格式"""
        required_fields = ['issue', 'date', 'front_balls', 'back_balls']
        
        # 检查必需字段
        for field in required_fields:
            if field not in row or not row[field].strip():
                return False
        
        try:
            # 验证期号格式
            issue = row['issue'].strip()
            if not issue.isdigit() or len(issue) != 5:
                return False
            
            # 验证日期格式
            datetime.strptime(row['date'], '%Y-%m-%d')
            
            # 验证号码格式
            front_balls = row['front_balls'].strip('"').strip("'")
            back_balls = row['back_balls'].strip('"').strip("'")
            
            front_nums = [int(x.strip()) for x in front_balls.split(',')]
            back_nums = [int(x.strip()) for x in back_balls.split(',')]
            
            # 检查数量
            if len(front_nums) != 5 or len(back_nums) != 2:
                return False
            
            # 检查范围
            for num in front_nums:
                if not (1 <= num <= 35):
                    return False
            
            for num in back_nums:
                if not (1 <= num <= 12):
                    return False
            
            return True
            
        except (ValueError, TypeError):
            return False

    def _process_row(self, row: Dict) -> Dict:
        """处理数据行"""
        processed = {
            'issue': int(row['issue']),
            'date': row['date'],
            'front_balls': row['front_balls'].strip('"').strip("'"),
            'back_balls': row['back_balls'].strip('"').strip("'")
        }
        
        # 解析号码列表
        processed['front_balls_list'] = sorted([int(x.strip()) for x in processed['front_balls'].split(',')])
        processed['back_balls_list'] = sorted([int(x.strip()) for x in processed['back_balls'].split(',')])
        
        # 添加日期对象
        processed['date_obj'] = datetime.strptime(row['date'], '%Y-%m-%d')
        
        return processed

    def get_latest_result(self) -> Optional[Dict]:
        """获取最新一期开奖结果"""
        data = self.load_data()
        if not data:
            return None
        return data[0]  # 数据已按期号降序排列

    def get_result_by_issue(self, issue: int) -> Optional[Dict]:
        """根据期号获取开奖结果"""
        data = self.load_data()
        for result in data:
            if result['issue'] == issue:
                return result
        return None

    def get_results_range(self, start_issue: int = None, end_issue: int = None, 
                         limit: int = None) -> List[Dict]:
        """获取指定范围的开奖结果"""
        data = self.load_data()
        
        filtered_data = data
        
        # 过滤期号范围
        if start_issue is not None:
            filtered_data = [r for r in filtered_data if r['issue'] >= start_issue]
        
        if end_issue is not None:
            filtered_data = [r for r in filtered_data if r['issue'] <= end_issue]
        
        # 限制数量
        if limit is not None:
            filtered_data = filtered_data[:limit]
        
        return filtered_data

    def get_recent_results(self, count: int = 10) -> List[Dict]:
        """获取最近几期的开奖结果"""
        data = self.load_data()
        return data[:count]

    def get_statistics(self) -> Dict:
        """获取数据统计信息"""
        data = self.load_data()
        if not data:
            return {'total_count': 0}
        
        return {
            'total_count': len(data),
            'latest_issue': data[0]['issue'],
            'latest_date': data[0]['date'],
            'earliest_issue': data[-1]['issue'],
            'earliest_date': data[-1]['date'],
            'date_range': f"{data[-1]['date']} ~ {data[0]['date']}"
        }

    def search_by_numbers(self, front_numbers: List[int] = None, 
                         back_numbers: List[int] = None, 
                         match_type: str = 'any') -> List[Dict]:
        """根据号码搜索历史开奖结果"""
        data = self.load_data()
        results = []
        
        for result in data:
            front_match = True
            back_match = True
            
            if front_numbers:
                if match_type == 'all':
                    front_match = all(num in result['front_balls_list'] for num in front_numbers)
                elif match_type == 'any':
                    front_match = any(num in result['front_balls_list'] for num in front_numbers)
                else:  # exact
                    front_match = set(front_numbers) == set(result['front_balls_list'])
            
            if back_numbers:
                if match_type == 'all':
                    back_match = all(num in result['back_balls_list'] for num in back_numbers)
                elif match_type == 'any':
                    back_match = any(num in result['back_balls_list'] for num in back_numbers)
                else:  # exact
                    back_match = set(back_numbers) == set(result['back_balls_list'])
            
            if front_match and back_match:
                results.append(result)
        
        return results

    def export_data(self, output_file: str, format_type: str = 'csv') -> bool:
        """导出数据"""
        data = self.load_data()
        
        try:
            if format_type.lower() == 'csv':
                with open(output_file, 'w', newline='', encoding='utf-8') as file:
                    if data:
                        writer = csv.DictWriter(file, fieldnames=['issue', 'date', 'front_balls', 'back_balls'])
                        writer.writeheader()
                        for row in data:
                            writer.writerow({
                                'issue': row['issue'],
                                'date': row['date'],
                                'front_balls': row['front_balls'],
                                'back_balls': row['back_balls']
                            })
            return True
            
        except Exception as e:
            print(f"导出数据失败: {str(e)}")
            return False


def test_lottery_data():
    """测试函数"""
    data_manager = LotteryData()
    
    print("=== 大乐透数据管理测试 ===")
    
    # 测试加载数据
    print("\n1. 加载数据测试:")
    try:
        stats = data_manager.get_statistics()
        print(f"数据统计: {stats}")
    except Exception as e:
        print(f"加载失败: {e}")
        return
    
    # 测试获取最新结果
    print("\n2. 最新开奖结果:")
    latest = data_manager.get_latest_result()
    if latest:
        print(f"期号: {latest['issue']}")
        print(f"日期: {latest['date']}")
        print(f"前区: {latest['front_balls']}")
        print(f"后区: {latest['back_balls']}")
    
    # 测试获取最近几期
    print("\n3. 最近5期结果:")
    recent = data_manager.get_recent_results(5)
    for result in recent:
        print(f"{result['issue']} ({result['date']}): {result['front_balls']} + {result['back_balls']}")


if __name__ == "__main__":
    test_lottery_data()