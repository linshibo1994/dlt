#!/bin/bash

# 测试新功能脚本
# 用于测试模型评估框架的新功能

# 设置日期
DATE=$(date +"%Y%m%d")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 创建输出目录
OUTPUT_DIR="output/new_features_${DATE}"
mkdir -p ${OUTPUT_DIR}

echo "🚀 开始测试新功能流程 (${DATE})"

# 测试模型基准测试框架
echo "🔍 测试模型基准测试框架..."
python3 examples/test_model_benchmark.py

# 测试模型优化器
echo "🔧 测试模型优化器..."
python3 examples/test_model_optimization.py

# 测试增强马尔可夫链
echo "⛓️ 测试增强马尔可夫链..."
python3 -c "
try:
    from improvements.enhanced_markov import get_markov_predictor
    
    print('🔄 测试增强马尔可夫链...')
    predictor = get_markov_predictor()
    
    # 测试二阶马尔可夫链
    print('📊 测试二阶马尔可夫链...')
    results = predictor.multi_order_markov_predict(count=3, periods=300, order=2)
    print('✅ 二阶马尔可夫链测试成功')
    for i, (front, back) in enumerate(results):
        print(f'  第 {i+1} 注: {sorted(front)} + {sorted(back)}')
    
    # 测试自适应马尔可夫链
    print('📊 测试自适应马尔可夫链...')
    results = predictor.adaptive_order_markov_predict(count=3, periods=300)
    print('✅ 自适应马尔可夫链测试成功')
    for i, pred in enumerate(results):
        print(f'  第 {i+1} 注: {sorted(pred[\"front_balls\"])} + {sorted(pred[\"back_balls\"])}')
        print(f'  阶数权重: {pred[\"order_weights\"]}')
except ImportError:
    print('❌ 增强马尔可夫链模块未找到')
"

# 测试LSTM深度学习
echo "🧠 测试LSTM深度学习..."
python3 -c "
try:
    from advanced_lstm_predictor import AdvancedLSTMPredictor, TENSORFLOW_AVAILABLE
    
    if TENSORFLOW_AVAILABLE:
        print('🧠 测试LSTM深度学习...')
        predictor = AdvancedLSTMPredictor()
        
        # 测试LSTM预测
        print('📊 测试LSTM预测...')
        results = predictor.lstm_predict(count=3)
        print('✅ LSTM预测测试成功')
        for i, (front, back) in enumerate(results):
            print(f'  第 {i+1} 注: {sorted(front)} + {sorted(back)}')
    else:
        print('❌ TensorFlow未安装，无法测试LSTM预测器')
except ImportError:
    print('❌ LSTM预测器模块未找到')
"

# 测试高级集成预测器
echo "🔄 测试高级集成预测器..."
python3 -c "
try:
    from improvements.advanced_ensemble import AdvancedEnsemblePredictor
    from predictor_modules import get_traditional_predictor, get_advanced_predictor
    
    print('🔄 测试高级集成预测器...')
    ensemble = AdvancedEnsemblePredictor()
    
    # 注册基础预测器
    traditional = get_traditional_predictor()
    advanced = get_advanced_predictor()
    
    ensemble.register_predictor('frequency', traditional, weight=0.3)
    ensemble.register_predictor('markov', advanced, weight=0.4)
    ensemble.register_predictor('bayesian', advanced, weight=0.3)
    
    # 测试Stacking集成
    print('📊 测试Stacking集成...')
    results = ensemble.stacking_predict(count=3)
    print('✅ Stacking集成测试成功')
    for i, (front, back) in enumerate(results):
        print(f'  第 {i+1} 注: {sorted(front)} + {sorted(back)}')
    
    # 测试加权集成
    print('📊 测试加权集成...')
    results = ensemble.weighted_ensemble_predict(count=3)
    print('✅ 加权集成测试成功')
    for i, (front, back) in enumerate(results):
        print(f'  第 {i+1} 注: {sorted(front)} + {sorted(back)}')
    
    # 测试自适应集成
    print('📊 测试自适应集成...')
    results = ensemble.adaptive_ensemble_predict(count=3)
    print('✅ 自适应集成测试成功')
    for i, (front, back) in enumerate(results):
        print(f'  第 {i+1} 注: {sorted(front)} + {sorted(back)}')
except ImportError:
    print('❌ 高级集成预测器模块未找到')
"

# 测试模型评估命令行工具
echo "🛠️ 测试模型评估命令行工具..."
python3 improvements/model_evaluation_cli.py evaluate \
  --register-default \
  --evaluate-all \
  --test-periods 10 \
  --compare \
  --report "${OUTPUT_DIR}/evaluation_report_${TIMESTAMP}.md" \
  --visualize-comparison "${OUTPUT_DIR}/model_comparison_${TIMESTAMP}.png" \
  --output-dir ${OUTPUT_DIR}

echo "✅ 新功能测试流程完成"
echo "📁 结果保存在: ${OUTPUT_DIR}"