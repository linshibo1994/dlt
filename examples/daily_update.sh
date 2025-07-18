#!/bin/bash

# 每日更新脚本
# 用于更新数据、运行预测和生成报告

# 设置日期
DATE=$(date +"%Y%m%d")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 创建输出目录
OUTPUT_DIR="output/daily_${DATE}"
mkdir -p ${OUTPUT_DIR}

echo "🔄 开始每日更新流程 (${DATE})"

# 更新数据
echo "📊 更新数据..."
python3 dlt_main.py data update --incremental

# 运行模型评估
echo "🔍 运行模型评估..."
python3 improvements/model_evaluation_cli.py evaluate \
  --register-default \
  --predictor-config examples/predictors_config.json \
  --evaluate-all \
  --test-periods 20 \
  --compare \
  --report "${OUTPUT_DIR}/evaluation_report_${TIMESTAMP}.md" \
  --visualize-comparison "${OUTPUT_DIR}/model_comparison_${TIMESTAMP}.png" \
  --output-dir ${OUTPUT_DIR}

# 生成预测
echo "🎯 生成预测..."
python3 dlt_main.py predict -m ensemble -c 5 --save "${OUTPUT_DIR}/predictions_ensemble_${TIMESTAMP}.json"
python3 dlt_main.py predict -m nine_models -c 5 --save "${OUTPUT_DIR}/predictions_nine_models_${TIMESTAMP}.json"

echo "✅ 每日更新流程完成"
echo "📁 结果保存在: ${OUTPUT_DIR}"