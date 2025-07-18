#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
深度学习模型配置
定义各种深度学习模型的默认配置参数
"""

# Transformer模型默认配置
DEFAULT_TRANSFORMER_CONFIG = {
    "name": "Transformer",
    "sequence_length": 20,
    "d_model": 128,
    "num_heads": 8,
    "num_layers": 4,
    "dff": 512,
    "dropout_rate": 0.1,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 100,
    "early_stopping_patience": 10
}

# GAN模型默认配置
DEFAULT_GAN_CONFIG = {
    "name": "GAN",
    "latent_dim": 100,
    "generator_layers": [128, 256, 128],
    "discriminator_layers": [128, 64, 32],
    "learning_rate": 0.0002,
    "beta1": 0.5,
    "batch_size": 64,
    "epochs": 200,
    "noise_std": 0.1
}

# 集成管理器默认配置
DEFAULT_ENSEMBLE_CONFIG = {
    "weights": {
        "transformer": 0.4,
        "gan": 0.3,
        "lstm": 0.3
    },
    "voting_threshold": 0.5,
    "confidence_threshold": 0.6
}

# 元学习优化器默认配置
DEFAULT_META_LEARNING_CONFIG = {
    "history_size": 50,
    "update_interval": 10,
    "learning_rate": 0.01,
    "performance_threshold": 0.5,
    "retrain_threshold": 0.3
}

# 数据管理器默认配置
DEFAULT_DATA_MANAGER_CONFIG = {
    "window_sizes": [100, 300, 500, 1000],
    "default_window": 500,
    "cache_enabled": True,
    "augmentation_factor": 1.5,
    "normalization": "minmax"
}

# 性能优化器默认配置
DEFAULT_PERFORMANCE_CONFIG = {
    "gpu_enabled": True,
    "batch_size": 32,
    "quantization_enabled": False,
    "cache_intermediate": True,
    "monitor_interval": 5
}