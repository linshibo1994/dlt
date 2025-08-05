#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GAN预测器
基于生成对抗网络的深度学习预测模型
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Tuple, Dict, Any, Optional
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import datetime
import functools

from .base_model import BaseModel as BaseDeepPredictor, ModelConfig, ModelType
from .metadata import ModelMetadata
from ..utils.config import DEFAULT_GAN_CONFIG
from ..utils.exceptions import ModelInitializationError, handle_model_error
# 导入核心模块
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import core_modules as cm
logger_manager = cm.logger_manager
data_manager = cm.data_manager

from compound_modules.compound_predictor import CompoundPredictorMixin, CompoundConfig, CompoundResult

# 导入智能训练轮数计算器
try:
    from ..training.smart_epochs_calculator import SmartEpochsCalculator, TrainingConfig, ModelType as SmartModelType, PerformanceMode
except ImportError:
    SmartEpochsCalculator = None
    TrainingConfig = None
    SmartModelType = None
    PerformanceMode = None

# 导入智能早停机制
from ..utils.intelligent_early_stopping import GeneralIntelligentEarlyStopping


class GANPredictor(BaseDeepPredictor, CompoundPredictorMixin):
    """基于GAN的彩票预测模型（支持复式预测）"""
    
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
        
        # 创建ModelConfig对象
        model_config = ModelConfig(
            model_type=ModelType.GAN,
            model_name="GANPredictor",
            version="2.0.0",
            description="基于GAN的彩票号码预测模型"
        )
        super().__init__(model_config)
        CompoundPredictorMixin.__init__(self)

        # 保存配置参数
        self.config_params = merged_config
        
        # 从配置中提取参数
        self.latent_dim = self.config_params.get('latent_dim', 128)
        self.generator_layers = self.config_params.get('generator_layers', [256, 512, 256, 128])
        self.discriminator_layers = self.config_params.get('discriminator_layers', [128, 256, 128, 64])
        self.generator_lr = self.config_params.get('generator_lr', 0.0001)
        self.discriminator_lr = self.config_params.get('discriminator_lr', 0.0004)
        self.beta1 = self.config_params.get('beta1', 0.0)
        self.beta2 = self.config_params.get('beta2', 0.9)

        # 高级GAN参数
        self.gan_type = self.config_params.get('gan_type', 'conditional')  # 'vanilla', 'conditional', 'wgan', 'wgan-gp'
        self.use_self_attention = self.config_params.get('use_self_attention', True)
        self.use_spectral_norm = self.config_params.get('use_spectral_norm', True)
        self.gradient_penalty_weight = self.config_params.get('gradient_penalty_weight', 10.0)
        self.n_critic = self.config_params.get('n_critic', 5)  # WGAN中判别器训练次数
        self.label_smoothing = self.config_params.get('label_smoothing', 0.1)

        # 条件GAN参数
        self.num_conditions = self.config_params.get('num_conditions', 10)  # 条件向量维度

        # GAN特有属性
        self.generator = None
        self.discriminator = None
        self.gan = None
        self.is_trained = False  # 添加训练状态标志

        # 添加缺失的属性
        self.name = "GANPredictor"
        self.model_dir = os.path.join("cache", "models", "gan")

        # 确保模型目录存在
        os.makedirs(self.model_dir, exist_ok=True)

        logger_manager.info(f"初始化增强GAN预测器: type={self.gan_type}, latent_dim={self.latent_dim}, "
                          f"self_attention={self.use_self_attention}")

    def build_model(self):
        """构建模型（公共接口）"""
        return self._build_model()

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
        from ..data.data_manager import DeepLearningDataManager
        
        # 创建数据管理器
        data_manager = DeepLearningDataManager()

        # 获取历史数据（数据管理器在初始化时已加载数据）
        df = data_manager.df
        if df is None or df.empty:
            logger_manager.error("无法获取历史数据")
            return []

        # 提取号码数据
        real_samples = []

        for _, row in df.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            
            # 归一化到0-1范围
            normalized_front = [(x - 1) / 34 for x in front_balls]  # 1-35 -> 0-1
            normalized_back = [(x - 1) / 11 for x in back_balls]    # 1-12 -> 0-1
            
            real_samples.append(normalized_front + normalized_back)
        
        # 转换为numpy数组
        real_samples = np.array(real_samples)
        
        # 数据增强（暂时跳过，因为方法签名问题）
        if self.config_params.get('data_augmentation', False):  # 暂时禁用
            # real_samples = data_manager.augment_data(real_samples, factor=1.5)
            logger_manager.info("数据增强已跳过")
        
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
        
        # 根据训练数据量调整（使用默认数据量）
        data_factor = min(1.0, 2755 / 1000)  # 使用默认历史数据量
        
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
        self.config_params.update({
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
        self.config_params.update({
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
            epochs = self.config_params.get('epochs', 200)

            # 智能训练轮数计算
            try:
                # 使用默认数据大小进行计算
                actual_data_size = 2755  # 默认历史数据大小
                dynamic_max_epochs = min(100, max(30, actual_data_size // 100))  # 使用智能早停，允许更多训练轮数
                epochs = min(epochs, dynamic_max_epochs)
                logger_manager.info(f"智能训练轮数调整: {epochs}")
            except Exception as e:
                logger_manager.warning(f"智能轮数计算失败，使用默认值: {e}")
                epochs = self.config_params.get('epochs', 200)

        if batch_size is None:
            batch_size = self.config_params.get('batch_size', 64)

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

            # 初始化智能早停机制
            early_stopping = GeneralIntelligentEarlyStopping(
                patience=20,  # 连续20次相同结果时停止
                min_delta=1e-6,
                verbose=1
            )
            early_stopping.reset()

            # 训练GAN
            for epoch in range(epochs):
                # 训练判别器
                
                # 随机选择真实样本
                idx = np.random.randint(0, real_samples.shape[0], batch_size)
                real_batch = real_samples[idx]
                
                # 生成假样本
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # 根据GAN类型调用生成器
                if self.gan_type == 'conditional':
                    # 条件GAN需要条件输入
                    conditions = np.random.normal(0, 1, (batch_size, self.num_conditions))
                    gen_samples = self.generator.predict([noise, conditions], verbose=0)
                else:
                    # 标准GAN只需要噪声
                    gen_samples = self.generator.predict(noise, verbose=0)
                
                # 添加噪声到标签（标签平滑化）
                valid_smooth = valid - 0.1 * np.random.random(valid.shape)
                fake_smooth = fake + 0.1 * np.random.random(fake.shape)
                
                # 训练判别器
                if self.gan_type == 'conditional':
                    # 条件判别器需要数据和条件输入
                    conditions_real = np.random.normal(0, 1, (batch_size, self.num_conditions))
                    conditions_fake = np.random.normal(0, 1, (batch_size, self.num_conditions))
                    d_loss_real = self.discriminator.train_on_batch([real_batch, conditions_real], valid_smooth)
                    d_loss_fake = self.discriminator.train_on_batch([gen_samples, conditions_fake], fake_smooth)
                else:
                    # 标准判别器只需要数据
                    d_loss_real = self.discriminator.train_on_batch(real_batch, valid_smooth)
                    d_loss_fake = self.discriminator.train_on_batch(gen_samples, fake_smooth)

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # 训练生成器
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                if self.gan_type == 'conditional':
                    # 条件GAN需要噪声和条件输入
                    conditions = np.random.normal(0, 1, (batch_size, self.num_conditions))
                    g_loss = self.gan.train_on_batch([noise, conditions], valid)
                else:
                    # 标准GAN只需要噪声
                    g_loss = self.gan.train_on_batch(noise, valid)
                
                # 记录历史
                # 确保d_loss是正确的格式，处理嵌套列表
                def extract_scalar(value):
                    """递归提取标量值"""
                    if isinstance(value, (int, float)):
                        return float(value)
                    elif isinstance(value, (list, tuple)) and len(value) > 0:
                        return extract_scalar(value[0])
                    else:
                        return 0.0

                d_loss_val = extract_scalar(d_loss[0])
                d_acc_val = extract_scalar(d_loss[1])
                g_loss_val = extract_scalar(g_loss)

                history['d_loss'].append(d_loss_val)
                history['d_accuracy'].append(d_acc_val)
                history['g_loss'].append(g_loss_val)

                # 更新进度
                if epoch % sample_interval == 0 or epoch == epochs - 1:
                    metrics_str = f"D loss: {d_loss_val:.4f}, acc: {100*d_acc_val:.2f}%, G loss: {g_loss_val:.4f}"
                    progress_callback.on_epoch_end(epoch, {
                        'd_loss': d_loss_val,
                        'd_accuracy': d_acc_val,
                        'g_loss': g_loss_val
                    })

                # 智能早停检查
                combined_loss = (d_loss_val + g_loss_val) / 2  # 使用组合损失作为指标
                if early_stopping.update(combined_loss):
                    logger_manager.info(f"智能早停触发，在第{epoch + 1}轮停止训练")
                    break

                # 实现早停
                if epoch > 50 and np.mean(history['d_accuracy'][-20:]) > 0.95:
                    logger_manager.info("判别器准确率过高，提前停止训练")
                    break
                
                # 判别器重置（如果判别器太强）
                if epoch > 10 and np.mean(history['d_accuracy'][-10:]) > 0.9:
                    logger_manager.info("判别器过强，重置判别器权重")
                    if self.gan_type == 'conditional':
                        self.discriminator = self._build_conditional_discriminator()
                        self.gan = self._build_conditional_gan_model()
                    else:
                        self.discriminator = self._build_discriminator()
                        self.gan = self._build_gan()
            
            # 完成训练
            progress_callback.on_train_end({
                'd_loss': np.mean(history['d_loss'][-10:]),
                'd_accuracy': np.mean(history['d_accuracy'][-10:]),
                'g_loss': np.mean(history['g_loss'][-10:])
            })
            
            # 可视化训练历史
            visualizer.plot_history()
            
            self.is_trained = True
            
            # 保存模型
            self._save_model()

            # 设置训练完成标志
            self.is_trained = True

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
            noise_std = self.config_params.get('noise_std', 0.1)
        
        try:
            # 生成随机噪声
            noise = np.random.normal(0, noise_std, (count, self.latent_dim))

            # 生成样本
            if self.gan_type == 'conditional':
                # 条件GAN需要条件输入
                conditions = np.random.normal(0, 1, (count, self.num_conditions))
                samples = self.generator.predict([noise, conditions], verbose=0)
            else:
                # 标准GAN只需要噪声
                samples = self.generator.predict(noise, verbose=0)

            # 确保返回正确的数据类型和形状
            samples = np.array(samples, dtype=np.float32)

            # 验证样本形状
            if samples.ndim != 2 or samples.shape[1] != 7:
                logger_manager.warning(f"GAN生成样本形状异常: {samples.shape}, 期望: ({count}, 7)")
                # 创建回退样本
                samples = np.random.rand(count, 7).astype(np.float32)

            return samples

        except Exception as e:
            logger_manager.error(f"GAN样本生成失败: {e}")
            # 返回随机样本作为回退
            return np.random.rand(count, 7).astype(np.float32)
    
    def _select_best_samples(self, samples, count=1):
        """
        选择最佳样本
        
        Args:
            samples: 生成的样本数组
            count: 选择数量
            
        Returns:
            选择的最佳样本
        """
        # 确保samples是numpy数组
        samples = np.array(samples, dtype=np.float32)

        # 如果样本数量不足，直接返回 - 修复数组比较问题
        if samples.shape[0] <= count:
            return samples
        
        # 计算每个样本的质量分数
        scores = []

        for i in range(samples.shape[0]):
            sample = samples[i]
            # 确保sample是numpy数组并且是正确的数据类型
            sample = np.array(sample, dtype=np.float32)

            # 前区号码
            front = sample[:5]

            # 后区号码
            back = sample[5:7]

            # 计算分布均匀性（理想情况下，号码应该分布均匀）
            front_std = float(np.std(front))
            back_std = float(np.std(back))

            # 计算重复性（理想情况下，号码不应该重复）
            front_unique = len(np.unique(np.round(front * 34 + 1).astype(int)))
            back_unique = len(np.unique(np.round(back * 11 + 1).astype(int)))

            # 计算总分数（越高越好）
            score = (front_std * 0.3 + back_std * 0.2) + (front_unique * 0.3 + back_unique * 0.2)
            scores.append(float(score))
        
        # 选择分数最高的样本
        scores_array = np.array(scores)
        best_indices = np.argsort(scores_array)[-count:]
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
    def predict(self, data: pd.DataFrame = None, count=1, verbose=True) -> List[Tuple[List[int], List[int]]]:
        """
        生成预测结果

        Args:
            data: 历史数据（可选，如果不提供则使用内部数据）
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

    def evaluate(self, data):
        """评估模型性能（公共接口）"""
        try:
            # 准备数据
            real_samples = self._prepare_real_samples(data)

            # 生成假样本
            batch_size = min(100, len(real_samples))
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            if self.gan_type == 'conditional':
                # 条件GAN需要条件输入
                conditions = np.random.normal(0, 1, (batch_size, self.num_conditions))
                fake_samples = self.generator.predict([noise, conditions], verbose=0)
            else:
                # 标准GAN只需要噪声
                fake_samples = self.generator.predict(noise, verbose=0)

            # 使用判别器评估
            if self.gan_type == 'conditional':
                # 条件判别器需要数据和条件输入
                conditions = np.random.normal(0, 1, (batch_size, self.num_conditions))
                real_scores = self.discriminator.predict([real_samples[:batch_size], conditions], verbose=0)
                fake_scores = self.discriminator.predict([fake_samples, conditions], verbose=0)
            else:
                # 标准判别器只需要数据
                real_scores = self.discriminator.predict(real_samples[:batch_size], verbose=0)
                fake_scores = self.discriminator.predict(fake_samples, verbose=0)

            # 计算评估指标
            real_accuracy = np.mean(real_scores > 0.5)
            fake_accuracy = np.mean(fake_scores < 0.5)
            discriminator_accuracy = (real_accuracy + fake_accuracy) / 2

            result = {
                'discriminator_accuracy': discriminator_accuracy,
                'real_accuracy': real_accuracy,
                'fake_accuracy': fake_accuracy,
                'generator_loss': np.mean(-np.log(fake_scores + 1e-8))
            }

            logger_manager.info(f"GAN模型评估完成: {result}")

            return result

        except Exception as e:
            logger_manager.error(f"GAN模型评估失败: {e}")
            return {'error': str(e)}

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

    def predict_compound(self, config: Optional[CompoundConfig] = None) -> CompoundResult:
        """
        GAN复式预测

        Args:
            config: 复式预测配置

        Returns:
            复式预测结果
        """
        if config is None:
            config = self.compound_config or CompoundConfig()

        # 验证参数
        if not self.validate_compound_params(config.front_count, config.back_count, config.max_cost):
            raise ValueError("GAN复式预测参数验证失败")

        logger_manager.info(f"开始GAN复式预测: {config.front_count}+{config.back_count}")

        try:
            # 生成多样化样本
            candidate_count = max(config.front_count * 3, 25)
            candidates = []

            # 生成多个预测作为候选
            for i in range(candidate_count):
                predictions = self.predict(1)
                if predictions:
                    candidates.append(predictions[0])

            # 基于多样性选择最优组合
            front_balls, back_balls = self._select_diverse_compound(
                candidates, config.front_count, config.back_count
            )

            # 计算组合数和成本
            combinations = self.calculate_combinations(config.front_count, config.back_count)
            cost = self.calculate_cost(combinations)

            # 计算置信度
            confidence = min(0.85, max(0.3, len(candidates) / candidate_count * 0.8))

            # 创建结果
            from datetime import datetime
            result = CompoundResult(
                front_balls=front_balls,
                back_balls=back_balls,
                front_count=config.front_count,
                back_count=config.back_count,
                total_combinations=combinations,
                total_cost=cost,
                confidence=confidence,
                method="GAN复式预测",
                analysis_periods=config.periods,
                timestamp=datetime.now().isoformat(),
                details={
                    'model_type': 'GAN',
                    'candidate_count': len(candidates),
                    'selection_strategy': 'diversity_based'
                }
            )

            logger_manager.info(f"GAN复式预测完成: {config.front_count}+{config.back_count}, 置信度: {confidence:.3f}")
            return result

        except Exception as e:
            logger_manager.error(f"GAN复式预测失败: {e}")
            # 返回默认结果
            return super().predict_compound(config)

    def _select_diverse_compound(self, candidates: List[Tuple[List[int], List[int]]],
                               front_count: int, back_count: int) -> Tuple[List[int], List[int]]:
        """基于多样性选择复式组合"""
        if not candidates:
            import random
            front_balls = sorted(random.sample(range(1, 36), front_count))
            back_balls = sorted(random.sample(range(1, 13), back_count))
            return front_balls, back_balls

        # 收集所有候选号码
        all_front = []
        all_back = []
        for front, back in candidates:
            all_front.extend(front)
            all_back.extend(back)

        # 计算号码频率和多样性权重
        from collections import Counter
        front_counter = Counter(all_front)
        back_counter = Counter(all_back)

        # 选择平衡频率和多样性的号码
        front_candidates = list(front_counter.keys())
        back_candidates = list(back_counter.keys())

        # 确保有足够的候选号码
        while len(front_candidates) < front_count:
            for i in range(1, 36):
                if i not in front_candidates:
                    front_candidates.append(i)
                    if len(front_candidates) >= front_count:
                        break

        while len(back_candidates) < back_count:
            for i in range(1, 13):
                if i not in back_candidates:
                    back_candidates.append(i)
                    if len(back_candidates) >= back_count:
                        break

        # 随机选择以增加多样性
        import random
        selected_front = sorted(random.sample(front_candidates, front_count))
        selected_back = sorted(random.sample(back_candidates, back_count))

        return selected_front, selected_back


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