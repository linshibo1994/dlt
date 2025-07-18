#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增量训练数据更新器
实现增量更新训练数据的功能
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from datetime import datetime
import hashlib
import json

from core_modules import logger_manager, data_manager, cache_manager


class IncrementalDataUpdater:
    """增量数据更新器"""
    
    def __init__(self, data_dir: str = "data", cache_dir: str = "cache/incremental"):
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
    
    def __init__(self, data_dir: str = "data", cache_dir: str = "cache/incremental",
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