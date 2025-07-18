#!/bin/bash

# 自动学习脚本
# 用于运行自适应学习和模型优化

# 设置日期
DATE=$(date +"%Y%m%d")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 创建输出目录
OUTPUT_DIR="output/learning_${DATE}"
mkdir -p ${OUTPUT_DIR}

echo "🧠 开始自动学习流程 (${DATE})"

# 运行自适应学习
echo "🎓 运行自适应学习..."
python3 dlt_main.py learn -p 1000 -t 300 --algorithm ucb1 --save "${OUTPUT_DIR}/learning_ucb1_${TIMESTAMP}.json"
python3 dlt_main.py learn -p 1000 -t 300 --algorithm epsilon_greedy --save "${OUTPUT_DIR}/learning_epsilon_greedy_${TIMESTAMP}.json"
python3 dlt_main.py learn -p 1000 -t 300 --algorithm thompson_sampling --save "${OUTPUT_DIR}/learning_thompson_sampling_${TIMESTAMP}.json"

# 运行模型优化
echo "🔧 运行模型优化..."
python3 improvements/model_evaluation_cli.py optimize \
  --module-path "examples.test_model_optimization" \
  --class-name "FrequencyPredictor" \
  --param-space "examples/param_space_example.json" \
  --method "grid" \
  --train-periods 300 \
  --val-periods 50 \
  --metric "accuracy" \
  --visualize-process "${OUTPUT_DIR}/frequency_optimization_process_${TIMESTAMP}.png" \
  --visualize-importance "${OUTPUT_DIR}/frequency_parameter_importance_${TIMESTAMP}.png" \
  --save-results "${OUTPUT_DIR}/frequency_optimization_${TIMESTAMP}.json" \
  --benchmark \
  --compare-baseline

# 使用优化后的模型生成预测
echo "🎯 使用优化后的模型生成预测..."
python3 dlt_main.py smart -c 5 --load "${OUTPUT_DIR}/learning_ucb1_${TIMESTAMP}.json" --save "${OUTPUT_DIR}/predictions_smart_${TIMESTAMP}.json"

echo "✅ 自动学习流程完成"
echo "📁 结果保存在: ${OUTPUT_DIR}"