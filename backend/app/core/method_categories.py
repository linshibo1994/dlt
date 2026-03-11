"""
预测方法分类常量定义

统一管理所有预测方法的分类，避免硬编码列表散落在各处代码中。
"""

# 深度学习方法（需要 TensorFlow/PyTorch 等深度学习框架）
DEEP_LEARNING_METHODS = [
    'lstm',           # LSTM 长短期记忆网络
    'transformer',    # Transformer 注意力机制
    'gan',            # GAN 生成对抗网络
]

# 集成学习方法（可能使用 GPU 加速，但不依赖深度学习框架）
ENSEMBLE_METHODS = [
    'stacking',           # 堆叠集成
    'adaptive_ensemble',  # 自适应集成
    'ultimate_ensemble',  # 终极集成
]

# 传统统计方法
TRADITIONAL_METHODS = [
    'frequency',   # 频率分析
    'hot_cold',    # 冷热分析
    'missing',     # 遗漏分析
    'bayesian',    # 贝叶斯分析
]

# 马尔可夫链方法
MARKOV_METHODS = [
    'markov',           # 一阶马尔可夫
    'markov_2nd',       # 二阶马尔可夫
    'markov_3rd',       # 三阶马尔可夫
    'adaptive_markov',  # 自适应马尔可夫
]

# 高级算法
ADVANCED_METHODS = [
    'ensemble',     # 集成预测（加权平均）
    'clustering',   # 聚类预测
    'consensus_halving',  # 交集递减融合预测
    'nine_models',  # 九模型融合
]

# 智能增强方法
SMART_METHODS = [
    'super',      # 超级预测
    'adaptive',   # 自适应预测
    'enhanced',   # 增强预测
]

# 复式投注方法
COMPOUND_METHODS = [
    'compound',             # 复式投注
    'duplex',               # 胆拖投注
    'markov_compound',      # 马尔可夫复式
    'nine_models_compound', # 九模型复式
]

# 所有需要深度学习支持的方法（深度学习 + 部分集成学习）
REQUIRES_DEEP_LEARNING = DEEP_LEARNING_METHODS + ENSEMBLE_METHODS

# 所有传统方法（不需要深度学习框架）
ALL_TRADITIONAL = TRADITIONAL_METHODS + MARKOV_METHODS + ADVANCED_METHODS

# 所有方法的完整列表
ALL_METHODS = (
    TRADITIONAL_METHODS +
    MARKOV_METHODS +
    DEEP_LEARNING_METHODS +
    ENSEMBLE_METHODS +
    ADVANCED_METHODS +
    SMART_METHODS +
    COMPOUND_METHODS
)


def is_deep_learning_method(method: str) -> bool:
    """判断方法是否需要深度学习支持"""
    return method in REQUIRES_DEEP_LEARNING


def is_traditional_method(method: str) -> bool:
    """判断方法是否为传统方法"""
    return method in ALL_TRADITIONAL


def is_compound_method(method: str) -> bool:
    """判断方法是否为复式投注方法"""
    return method in COMPOUND_METHODS


def get_method_category(method: str) -> str:
    """获取方法的分类名称"""
    if method in TRADITIONAL_METHODS:
        return '传统统计'
    elif method in MARKOV_METHODS:
        return '马尔可夫链'
    elif method in DEEP_LEARNING_METHODS:
        return '深度学习'
    elif method in ENSEMBLE_METHODS:
        return '集成学习'
    elif method in ADVANCED_METHODS:
        return '高级算法'
    elif method in SMART_METHODS:
        return '智能增强'
    elif method in COMPOUND_METHODS:
        return '复式投注'
    else:
        return '未知类别'
