#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增量训练数据更新器
实现增量更新训练数据的功能
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
from datetime import datetime
import hashlib
import json

from core_modules import logger_manager, data_manager, cache_manager


class IncrementalDataUpdater:
    """增量数据更新器"""
    
    def __init__(self, data_dir: str = "data", cache_dir: str = "artifacts/cache/incremental"):
        """
        初始化增量数据更新器
        
        Args:
            data_dir: 数据目录
            cache_dir: 缓存目录
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.df = data_manager.get_data()
        self.cache_manager = cache_manager
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 加载数据指纹
        self.data_fingerprints = self._load_fingerprints()
        
        if self.df is None:
            logger_manager.error("数据未加载")
        else:
            logger_manager.info(f"增量数据更新器初始化完成，数据量: {len(self.df)}")
    
    def update_data(self, new_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        更新数据
        
        Args:
            new_data: 新数据
            
        Returns:
            更新后的数据和更新信息
        """
        if self.df is None:
            logger_manager.error("原始数据未加载，无法更新")
            return new_data, {'status': 'error', 'message': '原始数据未加载'}
        
        if new_data is None or len(new_data) == 0:
            logger_manager.warning("没有新数据可更新")
            return self.df, {'status': 'no_update', 'message': '没有新数据'}
        
        # 检查数据一致性
        if not self._check_data_consistency(new_data):
            logger_manager.error("新数据格式与现有数据不一致")
            return self.df, {'status': 'error', 'message': '数据格式不一致'}
        
        # 备份原始数据
        self._backup_data()
        
        # 计算新数据指纹
        new_fingerprints = self._calculate_fingerprints(new_data)
        
        # 找出新增数据
        added_data = []
        for _, row in new_data.iterrows():
            issue = str(row['issue'])
            fingerprint = new_fingerprints.get(issue)
            
            if issue not in self.data_fingerprints or self.data_fingerprints[issue] != fingerprint:
                added_data.append(row)
        
        if not added_data:
            logger_manager.info("没有新增数据")
            return self.df, {'status': 'no_update', 'message': '没有新增数据'}
        
        # 合并数据
        added_df = pd.DataFrame(added_data)
        updated_df = pd.concat([added_df, self.df])
        
        # 去重并排序
        updated_df = updated_df.drop_duplicates(subset=['issue'])
        updated_df = updated_df.sort_values(by='issue', ascending=False).reset_index(drop=True)
        
        # 更新数据指纹
        self.data_fingerprints.update(new_fingerprints)
        self._save_fingerprints()
        
        # 保存更新后的数据
        self._save_updated_data(updated_df)
        
        # 更新内存中的数据
        self.df = updated_df
        
        logger_manager.info(f"数据更新完成，新增 {len(added_data)} 条数据，总数据量: {len(updated_df)}")
        
        return updated_df, {
            'status': 'success',
            'added_count': len(added_data),
            'total_count': len(updated_df),
            'added_issues': [str(row['issue']) for row in added_data]
        }
    
    def _check_data_consistency(self, new_data: pd.DataFrame) -> bool:
        """
        检查数据一致性
        
        Args:
            new_data: 新数据
            
        Returns:
            数据是否一致
        """
        # 检查必要列是否存在
        required_columns = ['issue', 'date', 'front_balls', 'back_balls']
        for col in required_columns:
            if col not in new_data.columns:
                logger_manager.error(f"新数据缺少必要列: {col}")
                return False
            
            if col not in self.df.columns:
                logger_manager.error(f"原始数据缺少必要列: {col}")
                return False
        
        # 检查数据类型
        try:
            # 尝试解析一行数据
            if len(new_data) > 0:
                row = new_data.iloc[0]
                front_balls, back_balls = data_manager.parse_balls(row)
                
                # 检查号码格式
                if len(front_balls) != 5 or len(back_balls) != 2:
                    logger_manager.error(f"号码格式错误: 前区 {len(front_balls)}/5, 后区 {len(back_balls)}/2")
                    return False
        except Exception as e:
            logger_manager.error(f"数据格式检查失败: {e}")
            return False

    def detect_new_data(self) -> bool:
        """
        检测是否有新数据

        Returns:
            是否有新数据
        """
        try:
            # 获取当前最新期号
            current_latest = self._get_current_latest_issue()

            # 获取远程最新期号
            remote_latest = self._get_remote_latest_issue()

            if remote_latest is None:
                logger_manager.warning("无法获取远程最新期号")
                return False

            if current_latest is None:
                logger_manager.info("本地无数据，检测到新数据")
                return True

            # 比较期号
            has_new_data = int(remote_latest) > int(current_latest)

            if has_new_data:
                logger_manager.info(f"检测到新数据: 本地最新期号 {current_latest}, 远程最新期号 {remote_latest}")
            else:
                logger_manager.info(f"无新数据: 本地最新期号 {current_latest}, 远程最新期号 {remote_latest}")

            return has_new_data

        except Exception as e:
            logger_manager.error(f"检测新数据失败: {e}")
            return False

    def _get_current_latest_issue(self) -> Optional[str]:
        """获取当前最新期号"""
        try:
            if not os.path.exists(self.data_file):
                return None

            df = pd.read_csv(self.data_file)
            if df.empty:
                return None

            # 数据按期号倒序排列，第一行是最新期
            return str(df.iloc[0]['issue'])

        except Exception as e:
            logger_manager.error(f"获取当前最新期号失败: {e}")
            return None

    def _get_remote_latest_issue(self) -> Optional[str]:
        """获取远程最新期号"""
        try:
            # 使用爬虫获取最新一期数据
            from crawlers import ZhcwCrawler
            crawler = ZhcwCrawler()

            # 获取第一页数据
            page_data = crawler.crawl_page(1)

            if page_data:
                # 返回第一条数据的期号（最新期）
                return str(page_data[0]['issue'])

            return None

        except Exception as e:
            logger_manager.error(f"获取远程最新期号失败: {e}")
            return None

    def get_incremental_data(self, max_periods: int = 100) -> List[Dict]:
        """
        获取增量数据

        Args:
            max_periods: 最大获取期数

        Returns:
            增量数据列表
        """
        try:
            current_latest = self._get_current_latest_issue()

            # 使用爬虫获取数据
            from crawlers import ZhcwCrawler
            crawler = ZhcwCrawler()

            new_data = []
            page = 1

            while len(new_data) < max_periods:
                page_data = crawler.crawl_page(page)

                if not page_data:
                    break

                # 筛选新数据
                for item in page_data:
                    if current_latest is None or int(item['issue']) > int(current_latest):
                        new_data.append(item)
                    else:
                        # 遇到已存在的期号，停止获取
                        logger_manager.info(f"遇到已存在期号 {item['issue']}，停止获取")
                        return new_data[:max_periods]

                page += 1

                # 避免无限循环
                if page > 10:
                    break

            logger_manager.info(f"获取到 {len(new_data)} 期增量数据")
            return new_data[:max_periods]

        except Exception as e:
            logger_manager.error(f"获取增量数据失败: {e}")
            return []
        
        return True
    
    def _calculate_fingerprints(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        计算数据指纹
        
        Args:
            data: 数据
            
        Returns:
            数据指纹字典，键为期号，值为指纹
        """
        fingerprints = {}
        
        for _, row in data.iterrows():
            issue = str(row['issue'])
            
            # 计算行数据的指纹
            row_data = f"{row['issue']}_{row['date']}_{row['front_balls']}_{row['back_balls']}"
            fingerprint = hashlib.md5(row_data.encode()).hexdigest()
            
            fingerprints[issue] = fingerprint
        
        return fingerprints
    
    def _load_fingerprints(self) -> Dict[str, str]:
        """
        加载数据指纹
        
        Returns:
            数据指纹字典
        """
        fingerprint_file = os.path.join(self.cache_dir, "data_fingerprints.json")
        
        if os.path.exists(fingerprint_file):
            try:
                with open(fingerprint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger_manager.warning(f"加载数据指纹失败: {e}")
        
        # 如果加载失败或文件不存在，计算当前数据的指纹
        if self.df is not None:
            fingerprints = self._calculate_fingerprints(self.df)
            self._save_fingerprints(fingerprints)
            return fingerprints
        
        return {}
    
    def _save_fingerprints(self, fingerprints: Optional[Dict[str, str]] = None) -> None:
        """
        保存数据指纹
        
        Args:
            fingerprints: 数据指纹字典，如果为None则使用当前指纹
        """
        if fingerprints is None:
            fingerprints = self.data_fingerprints
        
        fingerprint_file = os.path.join(self.cache_dir, "data_fingerprints.json")
        
        try:
            with open(fingerprint_file, 'w', encoding='utf-8') as f:
                json.dump(fingerprints, f, ensure_ascii=False, indent=2)
            
            logger_manager.debug(f"数据指纹已保存，共 {len(fingerprints)} 条")
        except Exception as e:
            logger_manager.error(f"保存数据指纹失败: {e}")
    
    def _backup_data(self) -> None:
        """备份原始数据"""
        if self.df is None:
            return
        
        try:
            # 保存到缓存
            self.cache_manager.save_cache("data", "dlt_data_backup", self.df)
            logger_manager.debug("原始数据已备份")
        except Exception as e:
            logger_manager.error(f"备份原始数据失败: {e}")
    
    def _save_updated_data(self, updated_df: pd.DataFrame) -> None:
        """
        保存更新后的数据
        
        Args:
            updated_df: 更新后的数据
        """
        try:
            # 保存到文件
            data_file = os.path.join(self.data_dir, "dlt_data_all.csv")
            updated_df.to_csv(data_file, index=False)
            
            # 保存到缓存
            self.cache_manager.save_cache("data", "dlt_data_all", updated_df)
            
            logger_manager.info(f"更新后的数据已保存，数据量: {len(updated_df)}")
        except Exception as e:
            logger_manager.error(f"保存更新后的数据失败: {e}")
    
    def restore_backup(self) -> pd.DataFrame:
        """
        恢复备份数据
        
        Returns:
            恢复后的数据
        """
        try:
            # 从缓存加载备份
            backup_df = self.cache_manager.load_cache("data", "dlt_data_backup")
            
            if backup_df is None or len(backup_df) == 0:
                logger_manager.warning("没有可用的备份数据")
                return self.df
            
            # 保存到文件
            data_file = os.path.join(self.data_dir, "dlt_data_all.csv")
            backup_df.to_csv(data_file, index=False)
            
            # 更新内存中的数据
            self.df = backup_df
            
            logger_manager.info(f"数据已恢复到备份，数据量: {len(backup_df)}")
            
            return backup_df
        except Exception as e:
            logger_manager.error(f"恢复备份数据失败: {e}")
            return self.df
    
    def create_update_trigger(self, trigger_file: str) -> bool:
        """
        创建更新触发器
        
        Args:
            trigger_file: 触发器文件路径
            
        Returns:
            是否创建成功
        """
        try:
            # 创建触发器文件
            with open(trigger_file, 'w', encoding='utf-8') as f:
                f.write(f"update_trigger_{datetime.now().isoformat()}")
            
            logger_manager.info(f"更新触发器已创建: {trigger_file}")
            return True
        except Exception as e:
            logger_manager.error(f"创建更新触发器失败: {e}")
            return False
    
    def check_update_trigger(self, trigger_file: str) -> bool:
        """
        检查更新触发器
        
        Args:
            trigger_file: 触发器文件路径
            
        Returns:
            是否需要更新
        """
        if not os.path.exists(trigger_file):
            return False
        
        try:
            # 读取触发器文件
            with open(trigger_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # 检查是否是有效的触发器
            if content.startswith("update_trigger_"):
                # 删除触发器文件
                os.remove(trigger_file)
                logger_manager.info(f"检测到更新触发器: {trigger_file}")
                return True
        except Exception as e:
            logger_manager.error(f"检查更新触发器失败: {e}")
        
        return False


class AutoIncrementalUpdater(IncrementalDataUpdater):
    """自动增量更新器"""
    
    def __init__(self, data_dir: str = "data", cache_dir: str = "artifacts/cache/incremental",
               update_interval: int = 24, max_retries: int = 3):
        """
        初始化自动增量更新器
        
        Args:
            data_dir: 数据目录
            cache_dir: 缓存目录
            update_interval: 更新间隔（小时）
            max_retries: 最大重试次数
        """
        super().__init__(data_dir, cache_dir)
        self.update_interval = update_interval
        self.max_retries = max_retries
        self.last_update = self._load_last_update()
        self.update_history = self._load_update_history()
        
        logger_manager.info(f"自动增量更新器初始化完成，更新间隔: {update_interval} 小时")
    
    def _load_last_update(self) -> Optional[datetime]:
        """
        加载上次更新时间
        
        Returns:
            上次更新时间
        """
        last_update_file = os.path.join(self.cache_dir, "last_update.txt")
        
        if os.path.exists(last_update_file):
            try:
                with open(last_update_file, 'r', encoding='utf-8') as f:
                    timestamp = f.read().strip()
                    return datetime.fromisoformat(timestamp)
            except Exception as e:
                logger_manager.warning(f"加载上次更新时间失败: {e}")
        
        return None
    
    def _save_last_update(self, timestamp: Optional[datetime] = None) -> None:
        """
        保存上次更新时间
        
        Args:
            timestamp: 时间戳，如果为None则使用当前时间
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        last_update_file = os.path.join(self.cache_dir, "last_update.txt")
        
        try:
            with open(last_update_file, 'w', encoding='utf-8') as f:
                f.write(timestamp.isoformat())
            
            self.last_update = timestamp
            logger_manager.debug(f"上次更新时间已保存: {timestamp}")
        except Exception as e:
            logger_manager.error(f"保存上次更新时间失败: {e}")
    
    def _load_update_history(self) -> List[Dict[str, Any]]:
        """
        加载更新历史
        
        Returns:
            更新历史列表
        """
        history_file = os.path.join(self.cache_dir, "update_history.json")
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger_manager.warning(f"加载更新历史失败: {e}")
        
        return []
    
    def _save_update_history(self) -> None:
        """保存更新历史"""
        history_file = os.path.join(self.cache_dir, "update_history.json")
        
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.update_history, f, ensure_ascii=False, indent=2)
            
            logger_manager.debug(f"更新历史已保存，共 {len(self.update_history)} 条")
        except Exception as e:
            logger_manager.error(f"保存更新历史失败: {e}")
    
    def _add_update_history(self, update_info: Dict[str, Any]) -> None:
        """
        添加更新历史
        
        Args:
            update_info: 更新信息
        """
        # 添加时间戳
        update_info['timestamp'] = datetime.now().isoformat()
        
        # 添加到历史
        self.update_history.append(update_info)
        
        # 限制历史大小
        if len(self.update_history) > 100:
            self.update_history = self.update_history[-100:]
        
        # 保存历史
        self._save_update_history()
    
    def should_update(self) -> bool:
        """
        检查是否应该更新
        
        Returns:
            是否应该更新
        """
        if self.last_update is None:
            return True
        
        # 计算距离上次更新的时间
        elapsed = datetime.now() - self.last_update
        elapsed_hours = elapsed.total_seconds() / 3600
        
        return elapsed_hours >= self.update_interval
    
    def auto_update(self, new_data_source: Optional[Callable[[], pd.DataFrame]] = None) -> Dict[str, Any]:
        """
        自动更新
        
        Args:
            new_data_source: 新数据源函数，如果为None则使用默认数据源
            
        Returns:
            更新结果
        """
        if not self.should_update():
            logger_manager.info("不需要更新")
            return {'status': 'skipped', 'message': '不需要更新'}
        
        # 获取新数据
        if new_data_source is None:
            # 使用默认数据源
            # 这里应该实现一个默认的数据源，例如从网络获取最新数据
            # 由于没有具体的数据源，这里只是一个示例
            logger_manager.error("没有提供数据源")
            return {'status': 'error', 'message': '没有提供数据源'}
        
        # 尝试获取新数据
        retries = 0
        while retries < self.max_retries:
            try:
                new_data = new_data_source()
                break
            except Exception as e:
                retries += 1
                logger_manager.error(f"获取新数据失败 ({retries}/{self.max_retries}): {e}")
                if retries >= self.max_retries:
                    return {'status': 'error', 'message': f'获取新数据失败: {e}'}
        
        # 更新数据
        _, update_info = self.update_data(new_data)
        
        # 更新上次更新时间
        self._save_last_update()
        
        # 添加更新历史
        self._add_update_history(update_info)
        
        return update_info
    
    def get_update_history(self) -> List[Dict[str, Any]]:
        """
        获取更新历史
        
        Returns:
            更新历史列表
        """
        return self.update_history

    def batch_update_data(self, data_batches: List[pd.DataFrame]) -> Dict[str, Any]:
        """
        批量更新数据

        Args:
            data_batches: 数据批次列表

        Returns:
            更新结果统计
        """
        try:
            total_new_records = 0
            successful_batches = 0
            failed_batches = 0

            for i, batch in enumerate(data_batches):
                logger_manager.info(f"处理第 {i+1}/{len(data_batches)} 批数据")

                _, update_info = self.update_data(batch)
                if update_info['status'] == 'success':
                    successful_batches += 1
                    total_new_records += update_info.get('added_count', 0)
                else:
                    failed_batches += 1

            result = {
                'total_batches': len(data_batches),
                'successful_batches': successful_batches,
                'failed_batches': failed_batches,
                'total_new_records': total_new_records,
                'success_rate': successful_batches / len(data_batches) if data_batches else 0
            }

            logger_manager.info(f"批量更新完成: {result}")
            return result

        except Exception as e:
            logger_manager.error(f"批量更新失败: {e}")
            return {}

    def incremental_update_with_validation(self, new_data: pd.DataFrame,
                                         validation_rules: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        带验证的增量更新

        Args:
            new_data: 新数据
            validation_rules: 验证规则

        Returns:
            更新结果详情
        """
        try:
            result = {
                'success': False,
                'new_records': 0,
                'validation_errors': [],
                'data_quality_score': 0.0,
                'update_time': None
            }

            if new_data.empty:
                result['validation_errors'].append("新数据为空")
                return result

            # 应用验证规则
            if validation_rules:
                validation_result = self._apply_validation_rules(new_data, validation_rules)
                result['validation_errors'] = validation_result['errors']
                result['data_quality_score'] = validation_result['quality_score']

                if validation_result['quality_score'] < 0.8:
                    logger_manager.warning(f"数据质量分数过低: {validation_result['quality_score']}")
                    return result

            # 执行更新
            _, update_info = self.update_data(new_data)
            if update_info['status'] == 'success':
                result['success'] = True
                result['new_records'] = update_info.get('added_count', 0)
                result['update_time'] = datetime.now()

            return result

        except Exception as e:
            logger_manager.error(f"带验证的增量更新失败: {e}")
            result['validation_errors'].append(str(e))
            return result

    def _apply_validation_rules(self, data: pd.DataFrame, rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用验证规则

        Args:
            data: 待验证数据
            rules: 验证规则

        Returns:
            验证结果
        """
        try:
            errors = []
            quality_score = 1.0

            # 检查必需字段
            required_fields = rules.get('required_fields', ['issue', 'date', 'front_balls', 'back_balls'])
            for field in required_fields:
                if field not in data.columns:
                    errors.append(f"缺少必需字段: {field}")
                    quality_score -= 0.2
                elif data[field].isnull().any():
                    errors.append(f"字段 {field} 包含空值")
                    quality_score -= 0.1

            # 检查数据范围
            if 'data_ranges' in rules:
                for field, (min_val, max_val) in rules['data_ranges'].items():
                    if field in data.columns:
                        if data[field].min() < min_val or data[field].max() > max_val:
                            errors.append(f"字段 {field} 超出有效范围 [{min_val}, {max_val}]")
                            quality_score -= 0.1

            # 检查数据格式
            if 'format_checks' in rules:
                for field, pattern in rules['format_checks'].items():
                    if field in data.columns:
                        import re
                        invalid_count = sum(1 for val in data[field] if not re.match(pattern, str(val)))
                        if invalid_count > 0:
                            errors.append(f"字段 {field} 有 {invalid_count} 条记录格式不正确")
                            quality_score -= 0.05 * invalid_count / len(data)

            # 检查重复数据
            if rules.get('check_duplicates', True):
                duplicate_count = data.duplicated().sum()
                if duplicate_count > 0:
                    errors.append(f"发现 {duplicate_count} 条重复记录")
                    quality_score -= 0.05 * duplicate_count / len(data)

            quality_score = max(0.0, quality_score)

            return {
                'errors': errors,
                'quality_score': quality_score
            }

        except Exception as e:
            logger_manager.error(f"验证规则应用失败: {e}")
            return {
                'errors': [str(e)],
                'quality_score': 0.0
            }


if __name__ == "__main__":
    # 测试增量数据更新器
    print("🔄 测试增量数据更新器...")
    
    # 创建增量数据更新器
    updater = IncrementalDataUpdater()
    
    # 创建模拟新数据
    if updater.df is not None and len(updater.df) > 0:
        # 使用现有数据的前几行作为模拟新数据
        mock_new_data = updater.df.head(5).copy()
        
        # 修改期号，模拟新数据
        for i, row in mock_new_data.iterrows():
            mock_new_data.at[i, 'issue'] = f"mock_{row['issue']}"
        
        # 更新数据
        updated_df, update_info = updater.update_data(mock_new_data)
        
        print(f"更新信息: {update_info}")
        print(f"更新后数据量: {len(updated_df)}")
    
    # 测试自动增量更新器
    print("\n🔄 测试自动增量更新器...")
    
    # 创建自动增量更新器
    auto_updater = AutoIncrementalUpdater(update_interval=24)
    
    # 检查是否应该更新
    should_update = auto_updater.should_update()
    print(f"是否应该更新: {should_update}")
    
    # 模拟自动更新
    def mock_data_source():
        if updater.df is not None and len(updater.df) > 0:
            mock_data = updater.df.head(3).copy()
            for i, row in mock_data.iterrows():
                mock_data.at[i, 'issue'] = f"auto_{row['issue']}"
            return mock_data
        return pd.DataFrame()
    
    update_result = auto_updater.auto_update(mock_data_source)
    print(f"自动更新结果: {update_result}")
    
    # 获取更新历史
    history = auto_updater.get_update_history()
    print(f"更新历史: {len(history)} 条")
    
    print("增量数据更新器测试完成")