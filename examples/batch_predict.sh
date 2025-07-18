#!/bin/bash

# 批量预测脚本
# 用于运行多种预测算法并比较结果

# 设置日期
DATE=$(date +"%Y%m%d")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 创建输出目录
OUTPUT_DIR="output/batch_${DATE}"
mkdir -p ${OUTPUT_DIR}

echo "🚀 开始批量预测流程 (${DATE})"

# 运行模型基准测试
echo "🔍 运行模型基准测试..."
python3 improvements/model_evaluation_cli.py benchmark \
  --register-default \
  --predictor-config examples/predictors_config.json \
  --evaluate-all \
  --test-periods 20 \
  --compare \
  --report "${OUTPUT_DIR}/benchmark_report_${TIMESTAMP}.md" \
  --visualize-comparison "${OUTPUT_DIR}/model_comparison_${TIMESTAMP}.png" \
  --save-results "${OUTPUT_DIR}/benchmark_results_${TIMESTAMP}.json" \
  --output-dir ${OUTPUT_DIR}

# 获取最佳模型
BEST_MODEL=$(python3 -c "
import json
with open('${OUTPUT_DIR}/benchmark_results_${TIMESTAMP}.json', 'r') as f:
    data = json.load(f)
if 'comparison' in data and 'overall_ranking' in data['comparison']:
    print(next(iter(data['comparison']['overall_ranking'].keys()), 'ensemble'))
else:
    print('ensemble')
")

echo "🏆 最佳模型: ${BEST_MODEL}"

# 使用最佳模型生成预测
echo "🎯 使用最佳模型生成预测..."
python3 dlt_main.py predict -m ${BEST_MODEL} -c 5 --save "${OUTPUT_DIR}/predictions_best_${TIMESTAMP}.json"

# 生成复式投注
echo "🎲 生成复式投注..."
python3 dlt_main.py predict -m compound --front-count 8 --back-count 4 --save "${OUTPUT_DIR}/predictions_compound_${TIMESTAMP}.json"

# 生成9种数学模型预测
echo "🧮 生成9种数学模型预测..."
python3 dlt_main.py predict -m nine_models -c 5 --save "${OUTPUT_DIR}/predictions_nine_models_${TIMESTAMP}.json"

echo "✅ 批量预测流程完成"
echo "📁 结果保存在: ${OUTPUT_DIR}"