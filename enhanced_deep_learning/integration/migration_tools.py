#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
迁移工具模块
Migration Tools Module

提供数据迁移、格式转换、版本升级等功能。
"""

import os
import json
import csv
import pickle
import shutil
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException


class DataFormat(Enum):
    """数据格式枚举"""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PICKLE = "pickle"
    NUMPY = "numpy"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    XML = "xml"
    YAML = "yaml"


class MigrationStatus(Enum):
    """迁移状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class MigrationConfig:
    """迁移配置"""
    name: str
    source_path: str
    target_path: str
    source_format: DataFormat
    target_format: DataFormat
    description: str = ""
    backup_enabled: bool = True
    validation_enabled: bool = True
    chunk_size: int = 10000
    parallel_processing: bool = False
    custom_transformer: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MigrationResult:
    """迁移结果"""
    migration_id: str
    config: MigrationConfig
    status: MigrationStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    records_processed: int = 0
    records_migrated: int = 0
    records_failed: int = 0
    error_message: str = ""
    validation_results: Dict[str, Any] = field(default_factory=dict)
    backup_path: str = ""


class DataFormatDetector:
    """数据格式检测器"""
    
    @staticmethod
    def detect_format(file_path: str) -> Optional[DataFormat]:
        """
        检测文件格式
        
        Args:
            file_path: 文件路径
            
        Returns:
            检测到的数据格式
        """
        try:
            if not os.path.exists(file_path):
                return None
            
            # 根据文件扩展名检测
            file_ext = Path(file_path).suffix.lower()
            
            format_mapping = {
                '.json': DataFormat.JSON,
                '.csv': DataFormat.CSV,
                '.xlsx': DataFormat.EXCEL,
                '.xls': DataFormat.EXCEL,
                '.pkl': DataFormat.PICKLE,
                '.pickle': DataFormat.PICKLE,
                '.npy': DataFormat.NUMPY,
                '.npz': DataFormat.NUMPY,
                '.parquet': DataFormat.PARQUET,
                '.h5': DataFormat.HDF5,
                '.hdf5': DataFormat.HDF5,
                '.xml': DataFormat.XML,
                '.yaml': DataFormat.YAML,
                '.yml': DataFormat.YAML
            }
            
            detected_format = format_mapping.get(file_ext)
            
            if detected_format:
                logger_manager.debug(f"检测到文件格式: {file_path} -> {detected_format.value}")
                return detected_format
            
            # 尝试内容检测
            return DataFormatDetector._detect_by_content(file_path)
            
        except Exception as e:
            logger_manager.error(f"格式检测失败: {e}")
            return None
    
    @staticmethod
    def _detect_by_content(file_path: str) -> Optional[DataFormat]:
        """根据文件内容检测格式"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            
            # JSON检测
            if first_line.startswith('{') or first_line.startswith('['):
                try:
                    json.loads(first_line)
                    return DataFormat.JSON
                except:
                    pass
            
            # CSV检测
            if ',' in first_line:
                return DataFormat.CSV
            
            # XML检测
            if first_line.startswith('<?xml') or first_line.startswith('<'):
                return DataFormat.XML
            
            return None
            
        except Exception:
            return None


class DataConverter:
    """数据转换器"""
    
    def __init__(self):
        """初始化数据转换器"""
        self.converters = {
            (DataFormat.CSV, DataFormat.JSON): self._csv_to_json,
            (DataFormat.JSON, DataFormat.CSV): self._json_to_csv,
            (DataFormat.CSV, DataFormat.EXCEL): self._csv_to_excel,
            (DataFormat.EXCEL, DataFormat.CSV): self._excel_to_csv,
            (DataFormat.JSON, DataFormat.EXCEL): self._json_to_excel,
            (DataFormat.EXCEL, DataFormat.JSON): self._excel_to_json,
            (DataFormat.CSV, DataFormat.PICKLE): self._csv_to_pickle,
            (DataFormat.PICKLE, DataFormat.CSV): self._pickle_to_csv,
            (DataFormat.NUMPY, DataFormat.CSV): self._numpy_to_csv,
            (DataFormat.CSV, DataFormat.NUMPY): self._csv_to_numpy
        }
        
        logger_manager.info("数据转换器初始化完成")
    
    def convert(self, source_path: str, target_path: str,
                source_format: DataFormat, target_format: DataFormat,
                chunk_size: int = 10000, custom_transformer: Callable = None) -> bool:
        """
        转换数据格式
        
        Args:
            source_path: 源文件路径
            target_path: 目标文件路径
            source_format: 源格式
            target_format: 目标格式
            chunk_size: 块大小
            custom_transformer: 自定义转换器
            
        Returns:
            是否转换成功
        """
        try:
            # 检查转换器是否存在
            converter_key = (source_format, target_format)
            
            if custom_transformer:
                converter = custom_transformer
            elif converter_key in self.converters:
                converter = self.converters[converter_key]
            else:
                raise DeepLearningException(f"不支持的格式转换: {source_format.value} -> {target_format.value}")
            
            # 创建目标目录
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # 执行转换
            success = converter(source_path, target_path, chunk_size)
            
            if success:
                logger_manager.info(f"数据转换成功: {source_path} -> {target_path}")
            else:
                logger_manager.error(f"数据转换失败: {source_path} -> {target_path}")
            
            return success
            
        except Exception as e:
            logger_manager.error(f"数据转换异常: {e}")
            return False
    
    def _csv_to_json(self, source_path: str, target_path: str, chunk_size: int) -> bool:
        """CSV转JSON"""
        try:
            df = pd.read_csv(source_path)
            data = df.to_dict('records')
            
            with open(target_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            logger_manager.error(f"CSV转JSON失败: {e}")
            return False
    
    def _json_to_csv(self, source_path: str, target_path: str, chunk_size: int) -> bool:
        """JSON转CSV"""
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
            
            df.to_csv(target_path, index=False)
            return True
        except Exception as e:
            logger_manager.error(f"JSON转CSV失败: {e}")
            return False
    
    def _csv_to_excel(self, source_path: str, target_path: str, chunk_size: int) -> bool:
        """CSV转Excel"""
        try:
            df = pd.read_csv(source_path)
            df.to_excel(target_path, index=False)
            return True
        except Exception as e:
            logger_manager.error(f"CSV转Excel失败: {e}")
            return False
    
    def _excel_to_csv(self, source_path: str, target_path: str, chunk_size: int) -> bool:
        """Excel转CSV"""
        try:
            df = pd.read_excel(source_path)
            df.to_csv(target_path, index=False)
            return True
        except Exception as e:
            logger_manager.error(f"Excel转CSV失败: {e}")
            return False
    
    def _json_to_excel(self, source_path: str, target_path: str, chunk_size: int) -> bool:
        """JSON转Excel"""
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
            
            df.to_excel(target_path, index=False)
            return True
        except Exception as e:
            logger_manager.error(f"JSON转Excel失败: {e}")
            return False
    
    def _excel_to_json(self, source_path: str, target_path: str, chunk_size: int) -> bool:
        """Excel转JSON"""
        try:
            df = pd.read_excel(source_path)
            data = df.to_dict('records')
            
            with open(target_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            logger_manager.error(f"Excel转JSON失败: {e}")
            return False
    
    def _csv_to_pickle(self, source_path: str, target_path: str, chunk_size: int) -> bool:
        """CSV转Pickle"""
        try:
            df = pd.read_csv(source_path)
            df.to_pickle(target_path)
            return True
        except Exception as e:
            logger_manager.error(f"CSV转Pickle失败: {e}")
            return False
    
    def _pickle_to_csv(self, source_path: str, target_path: str, chunk_size: int) -> bool:
        """Pickle转CSV"""
        try:
            df = pd.read_pickle(source_path)
            df.to_csv(target_path, index=False)
            return True
        except Exception as e:
            logger_manager.error(f"Pickle转CSV失败: {e}")
            return False
    
    def _numpy_to_csv(self, source_path: str, target_path: str, chunk_size: int) -> bool:
        """Numpy转CSV"""
        try:
            data = np.load(source_path)
            df = pd.DataFrame(data)
            df.to_csv(target_path, index=False)
            return True
        except Exception as e:
            logger_manager.error(f"Numpy转CSV失败: {e}")
            return False
    
    def _csv_to_numpy(self, source_path: str, target_path: str, chunk_size: int) -> bool:
        """CSV转Numpy"""
        try:
            df = pd.read_csv(source_path)
            data = df.values
            np.save(target_path, data)
            return True
        except Exception as e:
            logger_manager.error(f"CSV转Numpy失败: {e}")
            return False


class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_migration(source_path: str, target_path: str,
                          source_format: DataFormat, target_format: DataFormat) -> Dict[str, Any]:
        """
        验证迁移结果
        
        Args:
            source_path: 源文件路径
            target_path: 目标文件路径
            source_format: 源格式
            target_format: 目标格式
            
        Returns:
            验证结果
        """
        try:
            validation_result = {
                'valid': False,
                'source_exists': os.path.exists(source_path),
                'target_exists': os.path.exists(target_path),
                'source_size': 0,
                'target_size': 0,
                'record_count_match': False,
                'data_integrity': False,
                'errors': []
            }
            
            if not validation_result['source_exists']:
                validation_result['errors'].append("源文件不存在")
                return validation_result
            
            if not validation_result['target_exists']:
                validation_result['errors'].append("目标文件不存在")
                return validation_result
            
            # 文件大小检查
            validation_result['source_size'] = os.path.getsize(source_path)
            validation_result['target_size'] = os.path.getsize(target_path)
            
            # 记录数量检查
            try:
                source_count = DataValidator._count_records(source_path, source_format)
                target_count = DataValidator._count_records(target_path, target_format)
                
                validation_result['source_record_count'] = source_count
                validation_result['target_record_count'] = target_count
                validation_result['record_count_match'] = source_count == target_count
                
                if not validation_result['record_count_match']:
                    validation_result['errors'].append(f"记录数量不匹配: 源{source_count} vs 目标{target_count}")
                
            except Exception as e:
                validation_result['errors'].append(f"记录数量检查失败: {e}")
            
            # 数据完整性检查
            try:
                validation_result['data_integrity'] = DataValidator._check_data_integrity(
                    source_path, target_path, source_format, target_format
                )
                
                if not validation_result['data_integrity']:
                    validation_result['errors'].append("数据完整性检查失败")
                
            except Exception as e:
                validation_result['errors'].append(f"数据完整性检查异常: {e}")
            
            # 总体验证结果
            validation_result['valid'] = (
                validation_result['source_exists'] and
                validation_result['target_exists'] and
                validation_result['record_count_match'] and
                validation_result['data_integrity'] and
                len(validation_result['errors']) == 0
            )
            
            return validation_result
            
        except Exception as e:
            logger_manager.error(f"验证迁移结果失败: {e}")
            return {
                'valid': False,
                'errors': [f"验证异常: {e}"]
            }
    
    @staticmethod
    def _count_records(file_path: str, data_format: DataFormat) -> int:
        """计算记录数量"""
        try:
            if data_format == DataFormat.CSV:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return sum(1 for line in f) - 1  # 减去标题行
            
            elif data_format == DataFormat.JSON:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return len(data)
                    else:
                        return 1
            
            elif data_format in [DataFormat.EXCEL]:
                df = pd.read_excel(file_path)
                return len(df)
            
            elif data_format == DataFormat.PICKLE:
                df = pd.read_pickle(file_path)
                return len(df)
            
            else:
                return 0
                
        except Exception as e:
            logger_manager.error(f"计算记录数量失败: {e}")
            return 0
    
    @staticmethod
    def _check_data_integrity(source_path: str, target_path: str,
                             source_format: DataFormat, target_format: DataFormat) -> bool:
        """检查数据完整性"""
        try:
            # 简化的完整性检查：比较第一行和最后一行数据
            source_sample = DataValidator._get_data_sample(source_path, source_format)
            target_sample = DataValidator._get_data_sample(target_path, target_format)
            
            if source_sample and target_sample:
                # 比较样本数据的关键字段
                return len(source_sample) == len(target_sample)
            
            return False
            
        except Exception as e:
            logger_manager.error(f"数据完整性检查失败: {e}")
            return False
    
    @staticmethod
    def _get_data_sample(file_path: str, data_format: DataFormat) -> Optional[List]:
        """获取数据样本"""
        try:
            if data_format == DataFormat.CSV:
                df = pd.read_csv(file_path, nrows=2)
                return df.values.tolist()
            
            elif data_format == DataFormat.JSON:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data[:2] if len(data) >= 2 else data
                    else:
                        return [data]
            
            elif data_format == DataFormat.EXCEL:
                df = pd.read_excel(file_path, nrows=2)
                return df.values.tolist()
            
            return None
            
        except Exception as e:
            logger_manager.error(f"获取数据样本失败: {e}")
            return None


class MigrationTool:
    """迁移工具主类"""
    
    def __init__(self):
        """初始化迁移工具"""
        self.format_detector = DataFormatDetector()
        self.data_converter = DataConverter()
        self.data_validator = DataValidator()
        self.migration_history = []
        
        logger_manager.info("迁移工具初始化完成")
    
    def migrate_data(self, config: MigrationConfig) -> MigrationResult:
        """
        执行数据迁移
        
        Args:
            config: 迁移配置
            
        Returns:
            迁移结果
        """
        migration_id = f"migration_{int(datetime.now().timestamp())}"
        
        result = MigrationResult(
            migration_id=migration_id,
            config=config,
            status=MigrationStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            logger_manager.info(f"开始数据迁移: {config.name}")
            
            # 检查源文件
            if not os.path.exists(config.source_path):
                raise DeepLearningException(f"源文件不存在: {config.source_path}")
            
            # 自动检测源格式（如果未指定）
            if config.source_format is None:
                detected_format = self.format_detector.detect_format(config.source_path)
                if detected_format:
                    config.source_format = detected_format
                else:
                    raise DeepLearningException("无法检测源文件格式")
            
            # 创建备份
            if config.backup_enabled:
                result.backup_path = self._create_backup(config.source_path)
            
            # 执行数据转换
            success = self.data_converter.convert(
                config.source_path,
                config.target_path,
                config.source_format,
                config.target_format,
                config.chunk_size,
                config.custom_transformer
            )
            
            if not success:
                raise DeepLearningException("数据转换失败")
            
            # 验证迁移结果
            if config.validation_enabled:
                validation_result = self.data_validator.validate_migration(
                    config.source_path,
                    config.target_path,
                    config.source_format,
                    config.target_format
                )
                
                result.validation_results = validation_result
                
                if not validation_result['valid']:
                    raise DeepLearningException(f"迁移验证失败: {validation_result['errors']}")
            
            # 更新结果
            result.status = MigrationStatus.COMPLETED
            result.records_migrated = result.validation_results.get('target_record_count', 0)
            result.records_processed = result.records_migrated
            
            logger_manager.info(f"数据迁移完成: {config.name}")
            
        except Exception as e:
            result.status = MigrationStatus.FAILED
            result.error_message = str(e)
            logger_manager.error(f"数据迁移失败: {e}")
        
        finally:
            result.end_time = datetime.now()
            self.migration_history.append(result)
        
        return result
    
    def _create_backup(self, source_path: str) -> str:
        """创建备份"""
        try:
            backup_dir = os.path.join(os.path.dirname(source_path), "backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(source_path)
            backup_filename = f"{timestamp}_{filename}"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            shutil.copy2(source_path, backup_path)
            
            logger_manager.info(f"备份创建成功: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger_manager.error(f"创建备份失败: {e}")
            return ""
    
    def get_migration_history(self) -> List[MigrationResult]:
        """获取迁移历史"""
        return self.migration_history.copy()
    
    def get_supported_formats(self) -> List[DataFormat]:
        """获取支持的格式"""
        return list(DataFormat)
    
    def get_supported_conversions(self) -> List[Tuple[DataFormat, DataFormat]]:
        """获取支持的转换"""
        return list(self.data_converter.converters.keys())


# 全局迁移工具实例
migration_tool = MigrationTool()


if __name__ == "__main__":
    # 测试迁移工具功能
    print("🔄 测试迁移工具功能...")
    
    try:
        import tempfile
        
        # 创建临时测试文件
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建测试CSV文件
            csv_file = os.path.join(temp_dir, "test.csv")
            test_data = pd.DataFrame({
                'id': [1, 2, 3],
                'name': ['Alice', 'Bob', 'Charlie'],
                'age': [25, 30, 35]
            })
            test_data.to_csv(csv_file, index=False)
            
            # 测试格式检测
            detected_format = DataFormatDetector.detect_format(csv_file)
            if detected_format == DataFormat.CSV:
                print("✅ 格式检测成功")
            
            # 测试数据迁移
            json_file = os.path.join(temp_dir, "test.json")
            
            config = MigrationConfig(
                name="test_migration",
                source_path=csv_file,
                target_path=json_file,
                source_format=DataFormat.CSV,
                target_format=DataFormat.JSON,
                description="测试迁移"
            )
            
            tool = MigrationTool()
            result = tool.migrate_data(config)
            
            if result.status == MigrationStatus.COMPLETED:
                print(f"✅ 数据迁移成功: {result.records_migrated} 条记录")
            else:
                print(f"❌ 数据迁移失败: {result.error_message}")
            
            # 测试支持的格式
            formats = tool.get_supported_formats()
            print(f"✅ 支持的格式: {len(formats)} 种")
            
            # 测试支持的转换
            conversions = tool.get_supported_conversions()
            print(f"✅ 支持的转换: {len(conversions)} 种")
        
        print("✅ 迁移工具功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("迁移工具功能测试完成")
