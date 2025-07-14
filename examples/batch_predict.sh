#!/bin/bash
# 批量预测脚本
# 使用方法: bash examples/batch_predict.sh

echo "🎯 开始批量预测..."

# 创建输出目录
mkdir -p output/batch_predictions

# 传统方法预测
echo "📊 传统方法预测..."
python3 dlt_main.py predict -m frequency -c 5 --save batch_frequency.json
python3 dlt_main.py predict -m hot_cold -c 5 --save batch_hot_cold.json
python3 dlt_main.py predict -m missing -c 5 --save batch_missing.json

# 高级方法预测
echo "🧠 高级方法预测..."
python3 dlt_main.py predict -m markov -c 5 --save batch_markov.json
python3 dlt_main.py predict -m bayesian -c 5 --save batch_bayesian.json
python3 dlt_main.py predict -m ensemble -c 5 --save batch_ensemble.json

# 混合策略预测
echo "🎯 混合策略预测..."
python3 dlt_main.py predict -m mixed_strategy -c 5 --strategy conservative --save batch_conservative.json
python3 dlt_main.py predict -m mixed_strategy -c 5 --strategy aggressive --save batch_aggressive.json
python3 dlt_main.py predict -m mixed_strategy -c 5 --strategy balanced --save batch_balanced.json

# 复式投注预测
echo "💰 复式投注预测..."
python3 dlt_main.py predict -m compound --front-count 8 --back-count 4 --save batch_compound_8_4.json
python3 dlt_main.py predict -m compound --front-count 10 --back-count 5 --save batch_compound_10_5.json

echo "✅ 批量预测完成！"
echo "📁 结果保存在 output/predictions/ 目录"
echo "📊 生成的预测文件："
ls -la output/predictions/batch_*.json

# 生成汇总报告
echo "📋 生成汇总报告..."
cat > output/batch_predictions/summary.txt << EOF
批量预测汇总报告
================

预测时间: $(date)
预测方法: 9种
每种方法注数: 5注
总预测注数: 45注

预测方法列表:
1. 频率预测 (frequency)
2. 冷热号预测 (hot_cold)
3. 遗漏预测 (missing)
4. 马尔可夫链预测 (markov)
5. 贝叶斯预测 (bayesian)
6. 集成预测 (ensemble)
7. 保守策略 (conservative)
8. 激进策略 (aggressive)
9. 平衡策略 (balanced)
10. 复式投注 8+4
11. 复式投注 10+5

文件位置: output/predictions/
EOF

echo "📄 汇总报告已生成: output/batch_predictions/summary.txt"
echo "🎉 批量预测任务完成！"
