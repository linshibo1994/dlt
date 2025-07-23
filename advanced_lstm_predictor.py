#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高级LSTM预测器
基于enhanced_deep_learning模块的LSTM神经网络预测器包装器
提供与主程序兼容的接口
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

# 检查TensorFlow可用性
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# 导入核心模块
from core_modules import logger_manager, data_manager, cache_manager

# 导入enhanced_deep_learning模块
if TENSORFLOW_AVAILABLE:
    try:
        from enhanced_deep_learning.models.lstm_predictor import LSTMPredictor
        from enhanced_deep_learning.models.base_model import ModelMetadata
        ENHANCED_LSTM_AVAILABLE = True
    except ImportError:
        ENHANCED_LSTM_AVAILABLE = False
        logger_manager.warning("Enhanced LSTM模块导入失败，将使用简化实现")
else:
    ENHANCED_LSTM_AVAILABLE = False


class AdvancedLSTMPredictor:
    """高级LSTM预测器
    
    基于enhanced_deep_learning模块的LSTM神经网络预测器
    提供完整的深度学习预测功能
    """
    
    def __init__(self, data_file: str = None):
        """
        初始化高级LSTM预测器
        
        Args:
            data_file: 数据文件路径，默认为None使用系统默认数据
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow未安装，无法使用LSTM预测器")
            
        self.data_file = data_file
        self.lstm_predictor = None
        self.model_cache_dir = os.path.join('cache', 'models', 'lstm')
        
        # 确保缓存目录存在
        os.makedirs(self.model_cache_dir, exist_ok=True)
        
        # 初始化LSTM预测器
        self._init_lstm_predictor()
        
        logger_manager.info("高级LSTM预测器初始化完成")
    
    def _init_lstm_predictor(self):
        """初始化LSTM预测器"""
        if ENHANCED_LSTM_AVAILABLE:
            # 使用enhanced_deep_learning模块的LSTM实现
            config = {
                'sequence_length': 50,
                'lstm_units': [256, 128, 64],
                'dropout_rate': 0.3,
                'learning_rate': 0.0005,
                'batch_size': 64,
                'epochs': 200,
                'use_bidirectional': True,
                'use_attention': True,
                'use_residual': True,
                'attention_heads': 8,
                'l1_reg': 0.005,
                'l2_reg': 0.005,
                'gradient_clip_norm': 1.0
            }
            
            metadata = ModelMetadata(
                name="AdvancedLSTMPredictor",
                version="2.0.0",
                description="高级LSTM神经网络预测器",
                dependencies=["tensorflow", "scikit-learn", "numpy", "pandas"]
            )
            
            self.lstm_predictor = LSTMPredictor(config=config, metadata=metadata)
            logger_manager.info("使用Enhanced LSTM实现")
        else:
            # 如果enhanced模块不可用，抛出错误
            raise ImportError("Enhanced LSTM模块不可用，无法创建LSTM预测器")
    
    def lstm_predict(self, count: int = 3, periods: int = 500) -> List[Tuple[List[int], List[int]]]:
        """
        使用LSTM模型进行预测
        
        Args:
            count: 预测数量
            periods: 使用的历史数据期数
            
        Returns:
            预测结果列表，每个元素为(前区号码列表, 后区号码列表)
        """
        try:
            logger_manager.info(f"开始LSTM预测: 注数={count}, 分析期数={periods}")
            
            # 获取历史数据
            historical_data = data_manager.get_data()
            if historical_data is None or len(historical_data) == 0:
                raise ValueError("无法获取历史数据")
            
            # 使用指定期数的最新数据
            if len(historical_data) > periods:
                historical_data = historical_data.tail(periods)
            
            logger_manager.info(f"使用{len(historical_data)}期历史数据进行LSTM训练和预测")
            
            # 使用enhanced LSTM预测器进行预测
            if self.lstm_predictor is not None:
                predictions = self.lstm_predictor.predict(historical_data, count=count)
                
                # 转换为标准格式
                results = []
                for pred in predictions:
                    if isinstance(pred, tuple) and len(pred) == 2:
                        front_balls, back_balls = pred
                        results.append((list(front_balls), list(back_balls)))
                    elif isinstance(pred, dict):
                        front_balls = pred.get('front', pred.get('front_balls', []))
                        back_balls = pred.get('back', pred.get('back_balls', []))
                        results.append((list(front_balls), list(back_balls)))
                
                logger_manager.info(f"LSTM预测完成，生成{len(results)}注预测")
                return results
            else:
                raise ValueError("LSTM预测器未初始化")
                
        except Exception as e:
            logger_manager.error(f"LSTM预测失败: {e}")
            # 返回空结果而不是抛出异常，保持系统稳定性
            return []
    
    def train_model(self, periods: int = 1000) -> Dict[str, Any]:
        """
        训练LSTM模型
        
        Args:
            periods: 训练数据期数
            
        Returns:
            训练结果信息
        """
        try:
            logger_manager.info(f"开始LSTM模型训练，使用{periods}期数据")
            
            # 获取训练数据
            historical_data = data_manager.get_data()
            if historical_data is None or len(historical_data) == 0:
                raise ValueError("无法获取训练数据")
            
            # 使用指定期数的数据
            if len(historical_data) > periods:
                historical_data = historical_data.tail(periods)
            
            # 使用enhanced LSTM预测器进行训练
            if self.lstm_predictor is not None:
                training_result = self.lstm_predictor.train(historical_data)
                logger_manager.info("LSTM模型训练完成")
                return training_result
            else:
                raise ValueError("LSTM预测器未初始化")
                
        except Exception as e:
            logger_manager.error(f"LSTM模型训练失败: {e}")
            return {"error": str(e)}
    
    def save_model(self, model_path: str = None) -> bool:
        """
        保存LSTM模型
        
        Args:
            model_path: 模型保存路径
            
        Returns:
            是否保存成功
        """
        try:
            if self.lstm_predictor is not None:
                if model_path is None:
                    model_path = os.path.join(self.model_cache_dir, "advanced_lstm_model")
                
                self.lstm_predictor.save_model(model_path)
                logger_manager.info(f"LSTM模型已保存到: {model_path}")
                return True
            else:
                logger_manager.error("LSTM预测器未初始化，无法保存模型")
                return False
                
        except Exception as e:
            logger_manager.error(f"保存LSTM模型失败: {e}")
            return False
    
    def load_model(self, model_path: str = None) -> bool:
        """
        加载LSTM模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            是否加载成功
        """
        try:
            if self.lstm_predictor is not None:
                if model_path is None:
                    model_path = os.path.join(self.model_cache_dir, "advanced_lstm_model")
                
                if os.path.exists(model_path):
                    self.lstm_predictor.load_model(model_path)
                    logger_manager.info(f"LSTM模型已从{model_path}加载")
                    return True
                else:
                    logger_manager.warning(f"模型文件不存在: {model_path}")
                    return False
            else:
                logger_manager.error("LSTM预测器未初始化，无法加载模型")
                return False
                
        except Exception as e:
            logger_manager.error(f"加载LSTM模型失败: {e}")
            return False


# 如果直接运行此模块，进行测试
if __name__ == "__main__":
    if TENSORFLOW_AVAILABLE and ENHANCED_LSTM_AVAILABLE:
        try:
            print("🧠 测试高级LSTM预测器...")
            predictor = AdvancedLSTMPredictor()
            
            # 测试预测
            results = predictor.lstm_predict(count=3, periods=100)
            
            print(f"\n✅ LSTM预测结果 (共{len(results)}注):")
            for i, (front, back) in enumerate(results, 1):
                print(f"预测 {i}: 前区 {front}, 后区 {back}")
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    else:
        print("❌ TensorFlow或Enhanced LSTM模块不可用，无法测试")
