#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GAN预测器
基于生成对抗网络的深度学习预测模型
"""

import os
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Any
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import datetime
import functools

from .base_model import BaseModel as BaseDeepPredictor
from ..utils.config import DEFAULT_GAN_CONFIG
from ..utils.exceptions import ModelInitializationError, handle_model_error
from core_modules import logger_manager


class GANPredictor(BaseDeepPredictor):
    """基于GAN的彩票预测模型"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化GAN预测器
        
        Args:
            config: 配置参数字典
        """
        # 合并默认配置和用户配置
        merged_config = DEFAULT_GAN_CONFIG.copy()
        if config:
            merged_config.update(config)
        
        super().__init__(name="GAN", config=merged_config)
        
        # 从配置中提取参数
        self.latent_dim = self.config.get('latent_dim', 128)
        self.generator_layers = self.config.get('generator_layers', [256, 512, 256, 128])
        self.discriminator_layers = self.config.get('discriminator_layers', [128, 256, 128, 64])
        self.generator_lr = self.config.get('generator_lr', 0.0001)
        self.discriminator_lr = self.config.get('discriminator_lr', 0.0004)
        self.beta1 = self.config.get('beta1', 0.0)
        self.beta2 = self.config.get('beta2', 0.9)

        # 高级GAN参数
        self.gan_type = self.config.get('gan_type', 'conditional')  # 'vanilla', 'conditional', 'wgan', 'wgan-gp'
        self.use_self_attention = self.config.get('use_self_attention', True)
        self.use_spectral_norm = self.config.get('use_spectral_norm', True)
        self.gradient_penalty_weight = self.config.get('gradient_penalty_weight', 10.0)
        self.n_critic = self.config.get('n_critic', 5)  # WGAN中判别器训练次数
        self.label_smoothing = self.config.get('label_smoothing', 0.1)

        # 条件GAN参数
        self.num_conditions = self.config.get('num_conditions', 10)  # 条件向量维度

        # GAN特有属性
        self.generator = None
        self.discriminator = None
        self.gan = None

        logger_manager.info(f"初始化增强GAN预测器: type={self.gan_type}, latent_dim={self.latent_dim}, "
                          f"self_attention={self.use_self_attention}")
    
    def _build_model(self):
        """构建增强GAN模型"""
        try:
            if self.gan_type == 'conditional':
                return self._build_conditional_gan()
            elif self.gan_type == 'wgan':
                return self._build_wgan()
            elif self.gan_type == 'wgan-gp':
                return self._build_wgan_gp()
            else:
                return self._build_vanilla_gan()
        except Exception as e:
            raise ModelInitializationError(f"Enhanced_GAN_{self.gan_type}", str(e))

    def _build_conditional_gan(self):
        """构建条件GAN"""
        # 构建条件生成器
        self.generator = self._build_conditional_generator()

        # 构建条件判别器
        self.discriminator = self._build_conditional_discriminator()

        # 构建条件GAN
        self.gan = self._build_conditional_gan_model()

        logger_manager.info("条件GAN模型构建完成")
        return self.generator

    def _build_wgan(self):
        """构建WGAN"""
        # 构建WGAN生成器
        self.generator = self._build_wgan_generator()

        # 构建WGAN判别器（评论家）
        self.discriminator = self._build_wgan_critic()

        # 构建WGAN
        self.gan = self._build_wgan_model()

        logger_manager.info("WGAN模型构建完成")
        return self.generator

    def _build_wgan_gp(self):
        """构建WGAN-GP（带梯度惩罚）"""
        # 构建WGAN-GP生成器
        self.generator = self._build_wgan_generator()

        # 构建WGAN-GP判别器
        self.discriminator = self._build_wgan_gp_critic()

        # 构建WGAN-GP
        self.gan = self._build_wgan_gp_model()

        logger_manager.info("WGAN-GP模型构建完成")
        return self.generator

    def _build_vanilla_gan(self):
        """构建标准GAN"""
        # 构建生成器
        self.generator = self._build_generator()

        # 构建判别器
        self.discriminator = self._build_discriminator()

        # 构建GAN
        self.gan = self._build_gan()

        logger_manager.info("标准GAN模型构建完成")
        return self.generator

    def _build_conditional_generator(self):
        """构建条件生成器"""
        # 噪声输入
        noise_input = layers.Input(shape=(self.latent_dim,), name='noise_input')

        # 条件输入（历史数据特征）
        condition_input = layers.Input(shape=(self.num_conditions,), name='condition_input')

        # 合并噪声和条件
        merged_input = layers.Concatenate(name='merge_inputs')([noise_input, condition_input])

        x = merged_input

        # 生成器网络
        for i, units in enumerate(self.generator_layers):
            x = layers.Dense(units, name=f'gen_dense_{i}')(x)
            x = layers.BatchNormalization(name=f'gen_bn_{i}')(x)
            x = layers.LeakyReLU(alpha=0.2, name=f'gen_leaky_{i}')(x)
            x = layers.Dropout(0.3, name=f'gen_dropout_{i}')(x)

        # 自注意力机制
        if self.use_self_attention:
            x = self._add_self_attention(x, 'generator')

        # 输出层 - 7个数字（5前区 + 2后区）
        outputs = layers.Dense(7, activation='sigmoid', name='gen_output')(x)

        # 构建模型
        generator = Model(inputs=[noise_input, condition_input], outputs=outputs, name='Conditional_Generator')

        return generator

    def _build_conditional_discriminator(self):
        """构建条件判别器"""
        # 真实/生成数据输入
        data_input = layers.Input(shape=(7,), name='data_input')

        # 条件输入
        condition_input = layers.Input(shape=(self.num_conditions,), name='condition_input')

        # 合并数据和条件
        merged_input = layers.Concatenate(name='merge_inputs')([data_input, condition_input])

        x = merged_input

        # 判别器网络
        for i, units in enumerate(self.discriminator_layers):
            x = layers.Dense(units, name=f'disc_dense_{i}')(x)
            if self.use_spectral_norm:
                # 简化的谱归一化实现
                x = layers.BatchNormalization(name=f'disc_bn_{i}')(x)
            x = layers.LeakyReLU(alpha=0.2, name=f'disc_leaky_{i}')(x)
            x = layers.Dropout(0.3, name=f'disc_dropout_{i}')(x)

        # 自注意力机制
        if self.use_self_attention:
            x = self._add_self_attention(x, 'discriminator')

        # 输出层
        outputs = layers.Dense(1, activation='sigmoid', name='disc_output')(x)

        # 构建模型
        discriminator = Model(inputs=[data_input, condition_input], outputs=outputs, name='Conditional_Discriminator')

        return discriminator

    def _build_conditional_gan_model(self):
        """构建条件GAN模型"""
        # 冻结判别器权重
        self.discriminator.trainable = False

        # 输入
        noise_input = layers.Input(shape=(self.latent_dim,), name='gan_noise_input')
        condition_input = layers.Input(shape=(self.num_conditions,), name='gan_condition_input')

        # 生成器输出
        generated_data = self.generator([noise_input, condition_input])

        # 判别器判断
        validity = self.discriminator([generated_data, condition_input])

        # 构建GAN模型
        gan = Model(inputs=[noise_input, condition_input], outputs=validity, name='Conditional_GAN')

        # 编译GAN
        gan.compile(
            optimizer=optimizers.Adam(learning_rate=self.generator_lr, beta_1=self.beta1, beta_2=self.beta2),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # 编译判别器
        self.discriminator.trainable = True
        self.discriminator.compile(
            optimizer=optimizers.Adam(learning_rate=self.discriminator_lr, beta_1=self.beta1, beta_2=self.beta2),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return gan

    def _add_self_attention(self, x, layer_name):
        """添加自注意力机制"""
        # 简化的自注意力实现
        # 在实际应用中，这里会实现更复杂的自注意力机制

        # 将1D特征重塑为2D以便使用MultiHeadAttention
        seq_len = 1
        feature_dim = x.shape[-1]

        # 重塑为序列格式
        x_reshaped = layers.Reshape((seq_len, feature_dim), name=f'{layer_name}_reshape')(x)

        # 多头自注意力
        attention_output = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=feature_dim // 4,
            name=f'{layer_name}_self_attention'
        )(x_reshaped, x_reshaped)

        # 残差连接
        x_reshaped = layers.Add(name=f'{layer_name}_attention_add')([x_reshaped, attention_output])

        # 重塑回原始形状
        x = layers.Reshape((feature_dim,), name=f'{layer_name}_reshape_back')(x_reshaped)

        return x
    
    def _build_generator(self):
        """构建生成器"""
        model = tf.keras.Sequential(name="Generator")
        
        # 输入层
        model.add(layers.Input(shape=(self.latent_dim,)))
        
        # 隐藏层
        for i, units in enumerate(self.generator_layers):
            model.add(layers.Dense(units, name=f"generator_dense_{i}"))
            model.add(layers.BatchNormalization(name=f"generator_bn_{i}"))
            model.add(layers.LeakyReLU(alpha=0.2, name=f"generator_leaky_{i}"))
        
        # 输出层 - 7个输出（5前区+2后区）
        model.add(layers.Dense(7, activation='sigmoid', name="generator_output"))
        
        # 编译模型
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta1),
            loss='binary_crossentropy'
        )
        
        # 打印模型摘要
        model.summary()
        
        return model
    
    def _build_discriminator(self):
        """构建判别器"""
        model = tf.keras.Sequential(name="Discriminator")
        
        # 输入层
        model.add(layers.Input(shape=(7,)))
        
        # 隐藏层
        for i, units in enumerate(self.discriminator_layers):
            model.add(layers.Dense(units, name=f"discriminator_dense_{i}"))
            model.add(layers.LeakyReLU(alpha=0.2, name=f"discriminator_leaky_{i}"))
            model.add(layers.Dropout(0.3, name=f"discriminator_dropout_{i}"))
        
        # 输出层 - 单一输出（真/假）
        model.add(layers.Dense(1, activation='sigmoid', name="discriminator_output"))
        
        # 编译模型
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta1),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # 打印模型摘要
        model.summary()
        
        return model
    
    def _build_gan(self):
        """构建GAN"""
        # 冻结判别器权重
        self.discriminator.trainable = False
        
        # 创建GAN模型
        gan_input = layers.Input(shape=(self.latent_dim,))
        generated = self.generator(gan_input)
        validity = self.discriminator(generated)
        
        model = Model(gan_input, validity, name="GAN")
        
        # 编译模型
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta1),
            loss='binary_crossentropy'
        )
        
        # 打印模型摘要
        model.summary()
        
        return model
    
    def _prepare_training_data(self):
        """准备训练数据"""
        from .data_manager import DeepLearningDataManager
        
        # 创建数据管理器
        data_manager = DeepLearningDataManager()
        
        # 提取号码数据
        real_samples = []
        
        for _, row in self.df.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            
            # 归一化到0-1范围
            normalized_front = [(x - 1) / 34 for x in front_balls]  # 1-35 -> 0-1
            normalized_back = [(x - 1) / 11 for x in back_balls]    # 1-12 -> 0-1
            
            real_samples.append(normalized_front + normalized_back)
        
        # 转换为numpy数组
        real_samples = np.array(real_samples)
        
        # 数据增强
        if self.config.get('data_augmentation', True):
            real_samples = data_manager.augment_data(real_samples, factor=1.5)
        
        # 检测和处理异常数据
        normal_samples, anomaly_samples = data_manager.detect_anomalies(real_samples)
        
        if len(anomaly_samples) > 0:
            logger_manager.info(f"检测到 {len(anomaly_samples)} 个异常样本，已排除")
            real_samples = normal_samples
        
        return real_samples
    
    def _save_model(self):
        """保存模型"""
        try:
            # 保存生成器
            generator_path = os.path.join(self.model_dir, f"{self.name.lower()}_generator.h5")
            self.generator.save(generator_path)
            
            # 保存判别器
            discriminator_path = os.path.join(self.model_dir, f"{self.name.lower()}_discriminator.h5")
            self.discriminator.save(discriminator_path)
            
            logger_manager.info(f"{self.name}模型已保存")
        except Exception as e:
            logger_manager.error(f"保存{self.name}模型失败: {e}")
    
    def _load_model(self):
        """加载模型"""
        try:
            # 模型路径
            generator_path = os.path.join(self.model_dir, f"{self.name.lower()}_generator.h5")
            discriminator_path = os.path.join(self.model_dir, f"{self.name.lower()}_discriminator.h5")
            
            # 检查文件是否存在
            if not os.path.exists(generator_path) or not os.path.exists(discriminator_path):
                logger_manager.warning(f"{self.name}模型文件不存在，需要重新训练")
                return False
            
            # 加载模型
            self.generator = tf.keras.models.load_model(generator_path)
            self.discriminator = tf.keras.models.load_model(discriminator_path)
            
            # 重建GAN
            self.gan = self._build_gan()
            
            self.is_trained = True
            logger_manager.info(f"{self.name}模型加载成功")
            return True
        except Exception as e:
            logger_manager.error(f"加载{self.name}模型失败: {e}")
            return False
    
    def get_confidence(self) -> float:
        """
        获取预测置信度
        
        Returns:
            置信度分数 (0.0-1.0)
        """
        if not self.is_trained:
            return 0.0
        
        # GAN的置信度通常低于其他模型
        base_confidence = 0.6
        
        # 根据生成器复杂度调整
        complexity_factor = min(1.0, len(self.generator_layers) / 4)
        
        # 根据训练数据量调整
        data_factor = min(1.0, len(self.df) / 1000)
        
        confidence = base_confidence * complexity_factor * data_factor
        
        return min(0.85, confidence)  # GAN的最高置信度限制在0.85
    
    def use_fallback_config(self):
        """使用备用配置"""
        logger_manager.info("使用GAN备用配置")
        
        # 简化模型配置
        self.latent_dim = 50
        self.generator_layers = [64, 128, 64]
        self.discriminator_layers = [64, 32]
        
        # 更新配置字典
        self.config.update({
            'latent_dim': self.latent_dim,
            'generator_layers': self.generator_layers,
            'discriminator_layers': self.discriminator_layers
        })
    
    def use_simple_model(self):
        """使用简单模型"""
        logger_manager.info("使用简单GAN模型")
        
        # 极简配置
        self.latent_dim = 20
        self.generator_layers = [32, 64]
        self.discriminator_layers = [32]
        
        # 更新配置字典
        self.config.update({
            'latent_dim': self.latent_dim,
            'generator_layers': self.generator_layers,
            'discriminator_layers': self.discriminator_layers
        })


    @handle_model_error
    def train(self, epochs=None, batch_size=None, sample_interval=100):
        """
        训练GAN模型
        
        Args:
            epochs: 训练轮数，如果为None则使用配置中的值
            batch_size: 批处理大小，如果为None则使用配置中的值
            sample_interval: 采样间隔，每隔多少轮输出一次状态
            
        Returns:
            训练是否成功
        """
        from .training_utils import TrainingProgressCallback, TrainingVisualizer
        
        if epochs is None:
            epochs = self.config.get('epochs', 200)
        
        if batch_size is None:
            batch_size = self.config.get('batch_size', 64)
        
        logger_manager.info(f"开始训练GAN模型: epochs={epochs}, batch_size={batch_size}")
        
        try:
            # 构建模型
            if self.generator is None or self.discriminator is None or self.gan is None:
                self._build_model()
            
            # 准备训练数据
            real_samples = self._prepare_training_data()
            
            if len(real_samples) == 0:
                logger_manager.error("训练数据不足，无法训练模型")
                return False
            
            # 创建进度回调
            progress_callback = TrainingProgressCallback(self.name, epochs)
            progress_callback.on_train_begin()
            
            # 创建可视化器
            visualizer = TrainingVisualizer(self.name)
            
            # 训练历史记录
            history = {
                'd_loss': [],
                'd_accuracy': [],
                'g_loss': []
            }
            
            # 创建真假标签
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))
            
            # 训练GAN
            for epoch in range(epochs):
                # 训练判别器
                
                # 随机选择真实样本
                idx = np.random.randint(0, real_samples.shape[0], batch_size)
                real_batch = real_samples[idx]
                
                # 生成假样本
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                gen_samples = self.generator.predict(noise, verbose=0)
                
                # 添加噪声到标签（标签平滑化）
                valid_smooth = valid - 0.1 * np.random.random(valid.shape)
                fake_smooth = fake + 0.1 * np.random.random(fake.shape)
                
                # 训练判别器
                d_loss_real = self.discriminator.train_on_batch(real_batch, valid_smooth)
                d_loss_fake = self.discriminator.train_on_batch(gen_samples, fake_smooth)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # 训练生成器
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                g_loss = self.gan.train_on_batch(noise, valid)
                
                # 记录历史
                history['d_loss'].append(d_loss[0])
                history['d_accuracy'].append(d_loss[1])
                history['g_loss'].append(g_loss)
                
                # 更新进度
                if epoch % sample_interval == 0 or epoch == epochs - 1:
                    metrics_str = f"D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}%, G loss: {g_loss:.4f}"
                    progress_callback.on_epoch_end(epoch, {
                        'd_loss': d_loss[0],
                        'd_accuracy': d_loss[1],
                        'g_loss': g_loss
                    })
                
                # 实现早停
                if epoch > 50 and np.mean(history['d_accuracy'][-20:]) > 0.95:
                    logger_manager.info("判别器准确率过高，提前停止训练")
                    break
                
                # 判别器重置（如果判别器太强）
                if epoch > 10 and np.mean(history['d_accuracy'][-10:]) > 0.9:
                    logger_manager.info("判别器过强，重置判别器权重")
                    self.discriminator = self._build_discriminator()
                    self.gan = self._build_gan()
            
            # 完成训练
            progress_callback.on_train_end({
                'd_loss': np.mean(history['d_loss'][-10:]),
                'd_accuracy': np.mean(history['d_accuracy'][-10:]),
                'g_loss': np.mean(history['g_loss'][-10:])
            })
            
            # 可视化训练历史
            visualizer.plot_training_history(history)
            
            self.is_trained = True
            
            # 保存模型
            self._save_model()
            
            logger_manager.info(f"{self.name}模型训练完成")
            
            return True
        
        except Exception as e:
            logger_manager.error(f"{self.name}模型训练失败: {e}")
            return False
    
    def _generate_samples(self, count=1, noise_std=None):
        """
        生成样本
        
        Args:
            count: 生成样本数量
            noise_std: 噪声标准差，如果为None则使用配置中的值
            
        Returns:
            生成的样本数组
        """
        if noise_std is None:
            noise_std = self.config.get('noise_std', 0.1)
        
        # 生成随机噪声
        noise = np.random.normal(0, noise_std, (count, self.latent_dim))
        
        # 生成样本
        return self.generator.predict(noise, verbose=0)
    
    def _select_best_samples(self, samples, count=1):
        """
        选择最佳样本
        
        Args:
            samples: 生成的样本数组
            count: 选择数量
            
        Returns:
            选择的最佳样本
        """
        # 如果样本数量不足，直接返回
        if len(samples) <= count:
            return samples
        
        # 计算每个样本的质量分数
        scores = []
        
        for sample in samples:
            # 前区号码
            front = sample[:5]
            
            # 后区号码
            back = sample[5:7]
            
            # 计算分布均匀性（理想情况下，号码应该分布均匀）
            front_std = np.std(front)
            back_std = np.std(back)
            
            # 计算重复性（理想情况下，号码不应该重复）
            front_unique = len(np.unique(np.round(front * 34 + 1)))
            back_unique = len(np.unique(np.round(back * 11 + 1)))
            
            # 计算总分数（越高越好）
            score = (front_std * 0.3 + back_std * 0.2) + (front_unique * 0.3 + back_unique * 0.2)
            scores.append(score)
        
        # 选择分数最高的样本
        best_indices = np.argsort(scores)[-count:]
        return samples[best_indices]
    
    def _apply_gradient_penalty(self, real_samples, fake_samples):
        """
        应用梯度惩罚（Wasserstein GAN-GP）
        
        Args:
            real_samples: 真实样本
            fake_samples: 生成的样本
            
        Returns:
            梯度惩罚损失
        """
        batch_size = real_samples.shape[0]
        
        # 创建随机插值
        alpha = np.random.random((batch_size, 1))
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            predictions = self.discriminator(interpolated)
        
        # 计算梯度
        gradients = tape.gradient(predictions, interpolated)
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        gradient_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)
        
        return gradient_penalty


    @handle_model_error
    def predict(self, count=1, verbose=True) -> List[Tuple[List[int], List[int]]]:
        """
        生成预测结果
        
        Args:
            count: 预测注数
            verbose: 是否显示详细信息
            
        Returns:
            预测结果列表，每个元素为(前区号码列表, 后区号码列表)
        """
        from .prediction_utils import PredictionProcessor
        
        # 尝试加载已有模型
        if not self.is_trained:
            if not self._load_model():
                logger_manager.info(f"{self.name}模型未训练，开始训练...")
                if not self.train():
                    logger_manager.error(f"{self.name}模型训练失败")
                    return []
        
        # 创建预测处理器
        processor = PredictionProcessor()
        
        if verbose:
            logger_manager.info(f"使用{self.name}模型生成{count}注预测...")
        
        # 生成更多样本，然后选择最佳的
        gen_count = max(count * 3, 10)  # 生成3倍数量的样本
        
        # 生成样本
        raw_samples = self._generate_samples(gen_count)
        
        # 选择最佳样本
        best_samples = self._select_best_samples(raw_samples, count)
        
        # 处理预测结果
        predictions = []
        
        for i, sample in enumerate(best_samples):
            # 处理预测结果
            front_balls, back_balls = processor.process_raw_prediction(sample)
            predictions.append((front_balls, back_balls))
            
            if verbose:
                formatted = processor.format_prediction((front_balls, back_balls))
                logger_manager.info(f"预测 {i+1}/{count}: {formatted}")
        
        # 计算预测置信度
        confidence = processor.calculate_confidence(predictions)
        
        if verbose:
            logger_manager.info(f"{self.name}预测完成，置信度: {confidence:.2f}")
        
        return predictions
    
    def predict_with_details(self, count=1) -> Dict[str, Any]:
        """
        生成带详细信息的预测结果
        
        Args:
            count: 预测注数
            
        Returns:
            包含预测结果和详细信息的字典
        """
        from .prediction_utils import PredictionProcessor
        
        # 执行预测
        predictions = self.predict(count, verbose=False)
        
        # 创建预测处理器
        processor = PredictionProcessor()
        
        # 计算置信度
        confidence = processor.calculate_confidence(predictions)
        
        # 格式化预测结果
        formatted_predictions = []
        for i, pred in enumerate(predictions):
            formatted = processor.format_prediction(pred)
            formatted_predictions.append({
                'index': i + 1,
                'front_balls': pred[0],
                'back_balls': pred[1],
                'formatted': formatted
            })
        
        # 返回详细结果
        return {
            'model_name': self.name,
            'count': count,
            'predictions': formatted_predictions,
            'confidence': confidence,
            'model_config': {
                'latent_dim': self.latent_dim,
                'generator_layers': self.generator_layers,
                'discriminator_layers': self.discriminator_layers
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def evaluate_predictions(self, predictions: List[Tuple[List[int], List[int]]], 
                           actuals: List[Tuple[List[int], List[int]]]) -> Dict[str, Any]:
        """
        评估预测结果
        
        Args:
            predictions: 预测结果列表
            actuals: 实际结果列表
            
        Returns:
            评估结果字典
        """
        from .prediction_utils import PredictionEvaluator
        
        evaluator = PredictionEvaluator()
        return evaluator.evaluate_multiple_predictions(predictions, actuals)


if __name__ == "__main__":
    # 测试GAN预测器
    print("🎮 测试GAN预测器...")
    
    # 创建预测器
    gan = GANPredictor()
    
    # 构建模型
    gan._build_model()
    
    # 训练模型（小规模测试）
    gan.train(epochs=10, batch_size=16, sample_interval=2)
    
    # 进行预测
    predictions = gan.predict(3)
    
    print("GAN预测结果:")
    for i, (front, back) in enumerate(predictions):
        front_str = ' '.join([str(b).zfill(2) for b in front])
        back_str = ' '.join([str(b).zfill(2) for b in back])
        print(f"第 {i+1} 注: {front_str} + {back_str}")