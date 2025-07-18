#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强深度学习模型
基于Transformer和GAN的深度学习预测模型
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from datetime import datetime

# 检查TensorFlow可用性
try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model, optimizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.models import load_model, save_model
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# 尝试导入核心模块
try:
    from core_modules import logger_manager, data_manager, cache_manager
except ImportError:
    # 如果在不同目录运行，添加父目录到路径
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core_modules import logger_manager, data_manager, cache_manager


class TransformerLotteryPredictor:
    """基于Transformer的彩票预测模型"""
    
    def __init__(self, sequence_length=20, d_model=128, num_heads=8, num_layers=4):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow未安装，无法使用Transformer预测器")
        
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.feature_dim = 21  # 与LSTM预测器保持一致
        
        # 模型组件
        self.scaler = MinMaxScaler()
        self.model = None
        self.is_trained = False
        
        # 获取数据
        self.df = data_manager.get_data()
        if self.df is None:
            logger_manager.error("数据未加载")
    
    def _extract_features(self, df_subset) -> np.ndarray:
        """提取深度特征（与LSTM预测器保持一致）"""
        features = []
        
        for _, row in df_subset.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            
            # 基础特征 (7维)
            feature_vector = front_balls + back_balls
            
            # 统计特征 (8维)
            front_sum = sum(front_balls)
            back_sum = sum(back_balls)
            total_sum = front_sum + back_sum
            front_mean = np.mean(front_balls)
            back_mean = np.mean(back_balls)
            front_std = np.std(front_balls)
            back_std = np.std(back_balls)
            span = max(front_balls) - min(front_balls)
            
            # 模式特征 (6维)
            odd_count = sum(1 for x in front_balls if x % 2 == 1)
            even_count = 5 - odd_count
            big_count = sum(1 for x in front_balls if x > 17)  # 大号(18-35)
            small_count = 5 - big_count  # 小号(1-17)
            consecutive_count = self._count_consecutive(front_balls)
            prime_count = sum(1 for x in front_balls if self._is_prime(x))
            
            # 组合所有特征
            all_features = (
                feature_vector +  # 7维
                [front_sum, back_sum, total_sum, front_mean, back_mean, front_std, back_std, span] +  # 8维
                [odd_count, even_count, big_count, small_count, consecutive_count, prime_count]  # 6维
            )
            
            features.append(all_features)
        
        return np.array(features)
    
    def _count_consecutive(self, numbers):
        """计算连续号码数量"""
        sorted_nums = sorted(numbers)
        consecutive = 0
        for i in range(len(sorted_nums) - 1):
            if sorted_nums[i+1] - sorted_nums[i] == 1:
                consecutive += 1
        return consecutive
    
    def _is_prime(self, n):
        """判断是否为质数"""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def _prepare_sequences(self, features):
        """准备时间序列数据"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            y.append(features[i][:7])  # 只预测前5个前区号码和2个后区号码
        
        return np.array(X), np.array(y)
    
    def _build_model(self):
        """构建Transformer模型"""
        # 输入层
        inputs = layers.Input(shape=(self.sequence_length, self.feature_dim))
        
        # 位置编码
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        positions = layers.Embedding(self.sequence_length, self.d_model)(positions)
        
        # 输入嵌入
        x = layers.Dense(self.d_model)(inputs)
        x = x + positions
        
        # Transformer层
        for _ in range(self.num_layers):
            # 多头注意力
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, 
                key_dim=self.d_model
            )(x, x)
            x = layers.Add()([x, attention_output])
            x = layers.LayerNormalization()(x)
            
            # 前馈网络
            ffn_output = layers.Dense(self.d_model * 4, activation='relu')(x)
            ffn_output = layers.Dense(self.d_model)(ffn_output)
            x = layers.Add()([x, ffn_output])
            x = layers.LayerNormalization()(x)
        
        # 全局平均池化
        x = layers.GlobalAveragePooling1D()(x)
        
        # 全连接层
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # 输出层 - 使用单一输出，前5个值为前区，后2个值为后区
        outputs = layers.Dense(7, activation='sigmoid')(x)
        
        # 构建模型
        model = Model(inputs=inputs, outputs=outputs)
        
        # 编译模型
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, epochs=100, validation_split=0.2, batch_size=32):
        """训练Transformer模型"""
        logger_manager.info("开始训练Transformer模型...")
        
        # 提取特征
        features = self._extract_features(self.df)
        
        # 数据标准化
        features_scaled = self.scaler.fit_transform(features)
        
        # 准备序列数据
        X, y = self._prepare_sequences(features_scaled)
        
        if len(X) == 0:
            logger_manager.error("序列数据不足，无法训练模型")
            return False
        
        # 构建模型
        self.model = self._build_model()
        
        # 设置回调
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0001)
        ]
        
        # 训练模型
        history = self.model.fit(
            X, y,
            epochs=epochs,
            validation_split=validation_split,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        logger_manager.info("Transformer模型训练完成")
        
        # 保存模型
        self._save_model()
        
        return True
    
    def _save_model(self):
        """保存模型"""
        try:
            # 创建模型目录
            model_dir = os.path.join("cache", "models")
            os.makedirs(model_dir, exist_ok=True)
            
            # 保存模型
            model_path = os.path.join(model_dir, "transformer_model.h5")
            self.model.save(model_path)
            
            # 保存缩放器
            scaler_path = os.path.join(model_dir, "transformer_scaler.npy")
            np.save(scaler_path, {
                'scale_': self.scaler.scale_,
                'min_': self.scaler.min_,
                'data_min_': self.scaler.data_min_,
                'data_max_': self.scaler.data_max_,
                'data_range_': self.scaler.data_range_
            })
            
            logger_manager.info(f"Transformer模型已保存到 {model_path}")
        except Exception as e:
            logger_manager.error(f"保存Transformer模型失败: {e}")
    
    def _load_model(self):
        """加载模型"""
        try:
            # 模型路径
            model_path = os.path.join("cache", "models", "transformer_model.h5")
            scaler_path = os.path.join("cache", "models", "transformer_scaler.npy")
            
            # 检查文件是否存在
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                logger_manager.warning("Transformer模型文件不存在，需要重新训练")
                return False
            
            # 加载模型
            self.model = load_model(model_path)
            
            # 加载缩放器
            scaler_data = np.load(scaler_path, allow_pickle=True).item()
            self.scaler.scale_ = scaler_data['scale_']
            self.scaler.min_ = scaler_data['min_']
            self.scaler.data_min_ = scaler_data['data_min_']
            self.scaler.data_max_ = scaler_data['data_max_']
            self.scaler.data_range_ = scaler_data['data_range_']
            
            self.is_trained = True
            logger_manager.info("Transformer模型加载成功")
            return True
        except Exception as e:
            logger_manager.error(f"加载Transformer模型失败: {e}")
            return False
    
    def predict(self, count=1) -> List[Tuple[List[int], List[int]]]:
        """使用Transformer进行预测"""
        # 尝试加载已有模型
        if not self.is_trained:
            if not self._load_model():
                logger_manager.info("模型未训练，开始训练...")
                if not self.train_model():
                    logger_manager.error("模型训练失败")
                    return []
        
        # 获取最近的序列数据
        recent_features = self._extract_features(self.df.head(self.sequence_length))
        recent_scaled = self.scaler.transform(recent_features)
        
        # 准备输入序列
        input_sequence = recent_scaled.reshape(1, self.sequence_length, self.feature_dim)
        
        predictions = []
        
        for _ in range(count):
            # 预测
            pred_scaled = self.model.predict(input_sequence, verbose=0)
            
            # 反标准化
            # 创建完整特征向量用于反标准化
            full_pred = np.zeros((1, self.feature_dim))
            full_pred[0, :7] = pred_scaled[0]
            pred_original = self.scaler.inverse_transform(full_pred)[0, :7]
            
            # 转换为彩票号码
            front_balls = [max(1, min(35, int(round(x)))) for x in pred_original[:5]]
            back_balls = [max(1, min(12, int(round(x)))) for x in pred_original[5:7]]
            
            # 确保号码唯一性
            front_balls = self._ensure_unique_numbers(front_balls, 1, 35, 5)
            back_balls = self._ensure_unique_numbers(back_balls, 1, 12, 2)
            
            predictions.append((sorted(front_balls), sorted(back_balls)))
            
            # 更新输入序列用于下一次预测
            new_feature = np.concatenate([pred_original, recent_scaled[-1, 7:]])
            input_sequence = np.roll(input_sequence, -1, axis=1)
            input_sequence[0, -1] = new_feature
        
        return predictions
    
    def _ensure_unique_numbers(self, numbers, min_val, max_val, target_count):
        """确保号码唯一性"""
        unique_numbers = list(set(numbers))
        
        # 如果数量不足，随机补充
        while len(unique_numbers) < target_count:
            candidate = np.random.randint(min_val, max_val + 1)
            if candidate not in unique_numbers:
                unique_numbers.append(candidate)
        
        return unique_numbers[:target_count]


class GAN_LotteryPredictor:
    """基于GAN的彩票号码生成器"""
    
    def __init__(self, latent_dim=100):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow未安装，无法使用GAN预测器")
        
        self.latent_dim = latent_dim
        self.generator = None
        self.discriminator = None
        self.gan = None
        self.is_trained = False
        
        # 获取数据
        self.df = data_manager.get_data()
        if self.df is None:
            logger_manager.error("数据未加载")
    
    def _build_generator(self):
        """构建生成器"""
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=self.latent_dim),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(7, activation='sigmoid')  # 5前区+2后区
        ])
        return model
    
    def _build_discriminator(self):
        """构建判别器"""
        model = tf.keras.Sequential([
            layers.Dense(512, activation='relu', input_dim=7),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_gan(self):
        """构建GAN"""
        self.discriminator.trainable = False
        
        gan_input = layers.Input(shape=(self.latent_dim,))
        generated = self.generator(gan_input)
        validity = self.discriminator(generated)
        
        model = Model(gan_input, validity)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy'
        )
        
        return model
    
    def _prepare_training_data(self):
        """准备训练数据"""
        # 提取号码数据
        real_samples = []
        
        for _, row in self.df.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            
            # 归一化到0-1范围
            normalized_front = [(x - 1) / 34 for x in front_balls]  # 1-35 -> 0-1
            normalized_back = [(x - 1) / 11 for x in back_balls]    # 1-12 -> 0-1
            
            real_samples.append(normalized_front + normalized_back)
        
        return np.array(real_samples)
    
    def train_model(self, epochs=2000, batch_size=32, sample_interval=100):
        """训练GAN模型"""
        logger_manager.info("开始训练GAN模型...")
        
        # 构建模型
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.gan = self._build_gan()
        
        # 准备训练数据
        real_samples = self._prepare_training_data()
        
        if len(real_samples) == 0:
            logger_manager.error("训练数据不足，无法训练模型")
            return False
        
        # 训练GAN
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            # 训练判别器
            idx = np.random.randint(0, real_samples.shape[0], batch_size)
            real_batch = real_samples[idx]
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_samples = self.generator.predict(noise, verbose=0)
            
            d_loss_real = self.discriminator.train_on_batch(real_batch, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_samples, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # 训练生成器
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, valid)
            
            # 打印进度
            if epoch % sample_interval == 0:
                logger_manager.info(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
        
        self.is_trained = True
        logger_manager.info("GAN模型训练完成")
        
        # 保存模型
        self._save_model()
        
        return True
    
    def _save_model(self):
        """保存模型"""
        try:
            # 创建模型目录
            model_dir = os.path.join("cache", "models")
            os.makedirs(model_dir, exist_ok=True)
            
            # 保存生成器
            generator_path = os.path.join(model_dir, "gan_generator.h5")
            self.generator.save(generator_path)
            
            # 保存判别器
            discriminator_path = os.path.join(model_dir, "gan_discriminator.h5")
            self.discriminator.save(discriminator_path)
            
            logger_manager.info(f"GAN模型已保存到 {model_dir}")
        except Exception as e:
            logger_manager.error(f"保存GAN模型失败: {e}")
    
    def _load_model(self):
        """加载模型"""
        try:
            # 模型路径
            generator_path = os.path.join("cache", "models", "gan_generator.h5")
            discriminator_path = os.path.join("cache", "models", "gan_discriminator.h5")
            
            # 检查文件是否存在
            if not os.path.exists(generator_path) or not os.path.exists(discriminator_path):
                logger_manager.warning("GAN模型文件不存在，需要重新训练")
                return False
            
            # 加载模型
            self.generator = load_model(generator_path)
            self.discriminator = load_model(discriminator_path)
            
            # 重建GAN
            self.gan = self._build_gan()
            
            self.is_trained = True
            logger_manager.info("GAN模型加载成功")
            return True
        except Exception as e:
            logger_manager.error(f"加载GAN模型失败: {e}")
            return False
    
    def predict(self, count=1) -> List[Tuple[List[int], List[int]]]:
        """使用GAN生成预测号码"""
        # 尝试加载已有模型
        if not self.is_trained:
            if not self._load_model():
                logger_manager.info("模型未训练，开始训练...")
                if not self.train_model(epochs=1000):  # 减少训练轮数以加快速度
                    logger_manager.error("模型训练失败")
                    return []
        
        predictions = []
        
        # 生成多组预测
        for _ in range(count):
            # 生成随机噪声
            noise = np.random.normal(0, 1, (1, self.latent_dim))
            
            # 生成号码
            generated = self.generator.predict(noise, verbose=0)[0]
            
            # 转换为实际号码
            front_balls = [int(round(x * 34 + 1)) for x in generated[:5]]  # 0-1 -> 1-35
            back_balls = [int(round(x * 11 + 1)) for x in generated[5:7]]  # 0-1 -> 1-12
            
            # 确保号码在有效范围内
            front_balls = [max(1, min(35, x)) for x in front_balls]
            back_balls = [max(1, min(12, x)) for x in back_balls]
            
            # 确保号码唯一性
            front_balls = self._ensure_unique_numbers(front_balls, 1, 35, 5)
            back_balls = self._ensure_unique_numbers(back_balls, 1, 12, 2)
            
            predictions.append((sorted(front_balls), sorted(back_balls)))
        
        return predictions
    
    def _ensure_unique_numbers(self, numbers, min_val, max_val, target_count):
        """确保号码唯一性"""
        unique_numbers = list(set(numbers))
        
        # 如果数量不足，随机补充
        while len(unique_numbers) < target_count:
            candidate = np.random.randint(min_val, max_val + 1)
            if candidate not in unique_numbers:
                unique_numbers.append(candidate)
        
        return unique_numbers[:target_count]


# 兼容性检查
if not TENSORFLOW_AVAILABLE:
    class TransformerLotteryPredictor:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow未安装，无法使用Transformer预测器")
        
        def predict(self, *args, **kwargs):
            return []
    
    class GAN_LotteryPredictor:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow未安装，无法使用GAN预测器")
        
        def predict(self, *args, **kwargs):
            return []


if __name__ == "__main__":
    # 测试Transformer预测器
    if TENSORFLOW_AVAILABLE:
        print("🧠 测试Transformer预测器...")
        transformer = TransformerLotteryPredictor()
        
        # 训练模型
        transformer.train_model(epochs=50)
        
        # 进行预测
        predictions = transformer.predict(3)
        
        print("Transformer预测结果:")
        for i, (front, back) in enumerate(predictions):
            front_str = ' '.join([str(b).zfill(2) for b in front])
            back_str = ' '.join([str(b).zfill(2) for b in back])
            print(f"第 {i+1} 注: {front_str} + {back_str}")
        
        # 测试GAN预测器
        print("\n🎮 测试GAN预测器...")
        gan = GAN_LotteryPredictor()
        
        # 训练模型
        gan.train_model(epochs=500, sample_interval=100)
        
        # 进行预测
        predictions = gan.predict(3)
        
        print("GAN预测结果:")
        for i, (front, back) in enumerate(predictions):
            front_str = ' '.join([str(b).zfill(2) for b in front])
            back_str = ' '.join([str(b).zfill(2) for b in back])
            print(f"第 {i+1} 注: {front_str} + {back_str}")
    else:
        print("❌ TensorFlow未安装，无法测试深度学习模型")