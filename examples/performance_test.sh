#!/bin/bash

# 性能测试脚本
# 用于测试不同模型的性能和效率

# 设置日期
DATE=$(date +"%Y%m%d")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 创建输出目录
OUTPUT_DIR="output/performance_${DATE}"
mkdir -p ${OUTPUT_DIR}

echo "⚡ 开始性能测试流程 (${DATE})"

# 测试参数
TEST_PERIODS=(10 20 50 100)
CATEGORIES=("traditional" "advanced" "enhanced" "deep_learning" "ensemble")

# 运行模型基准测试
for periods in "${TEST_PERIODS[@]}"; do
    echo "🔍 运行模型基准测试 (测试期数: ${periods})..."
    python3 improvements/model_evaluation_cli.py benchmark \
      --register-default \
      --predictor-config examples/predictors_config.json \
      --evaluate-all \
      --test-periods ${periods} \
      --compare \
      --report "${OUTPUT_DIR}/benchmark_report_${periods}_${TIMESTAMP}.md" \
      --visualize-comparison "${OUTPUT_DIR}/model_comparison_${periods}_${TIMESTAMP}.png" \
      --save-results "${OUTPUT_DIR}/benchmark_results_${periods}_${TIMESTAMP}.json" \
      --output-dir ${OUTPUT_DIR}
done

# 比较不同类别的模型
for category in "${CATEGORIES[@]}"; do
    echo "🔄 比较 ${category} 类别的模型..."
    python3 improvements/model_evaluation_cli.py benchmark \
      --load-results "${OUTPUT_DIR}/benchmark_results_50_${TIMESTAMP}.json" \
      --compare \
      --categories ${category} \
      --visualize-comparison "${OUTPUT_DIR}/model_comparison_${category}_${TIMESTAMP}.png" \
      --output-dir ${OUTPUT_DIR}
done

# 生成性能报告
echo "📊 生成性能报告..."
python3 -c "
import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# 加载不同期数的测试结果
results = {}
for periods in [10, 20, 50, 100]:
    try:
        with open('${OUTPUT_DIR}/benchmark_results_{}_${TIMESTAMP}.json'.format(periods), 'r') as f:
            results[periods] = json.load(f)
    except:
        print('无法加载 {} 期测试结果'.format(periods))

# 提取性能数据
performance_data = {}
for periods, result in results.items():
    if 'results' in result:
        for model_name, model_info in result['results'].items():
            if model_name not in performance_data:
                performance_data[model_name] = {'periods': [], 'accuracy': [], 'time': [], 'category': model_info.get('category', 'unknown')}
            
            performance_data[model_name]['periods'].append(periods)
            performance_data[model_name]['accuracy'].append(model_info.get('metrics', {}).get('accuracy', 0))
            performance_data[model_name]['time'].append(model_info.get('avg_prediction_time', 0))

# 创建性能报告
report = ['# 模型性能测试报告', '', '## 测试日期: ${TIMESTAMP}', '']

# 添加准确率随测试期数变化的表格
report.append('## 准确率随测试期数变化')
report.append('')
report.append('| 模型名称 | 类别 | 10期 | 20期 | 50期 | 100期 |')
report.append('| --- | --- | --- | --- | --- | --- |')

for model_name, data in sorted(performance_data.items(), key=lambda x: x[0]):
    accuracy_values = []
    for periods in [10, 20, 50, 100]:
        if periods in data['periods']:
            idx = data['periods'].index(periods)
            accuracy_values.append('{:.4f}'.format(data['accuracy'][idx]))
        else:
            accuracy_values.append('N/A')
    
    report.append('| {} | {} | {} | {} | {} | {} |'.format(
        model_name, data['category'], *accuracy_values
    ))

# 添加预测时间随测试期数变化的表格
report.append('')
report.append('## 预测时间随测试期数变化 (秒)')
report.append('')
report.append('| 模型名称 | 类别 | 10期 | 20期 | 50期 | 100期 |')
report.append('| --- | --- | --- | --- | --- | --- |')

for model_name, data in sorted(performance_data.items(), key=lambda x: x[0]):
    time_values = []
    for periods in [10, 20, 50, 100]:
        if periods in data['periods']:
            idx = data['periods'].index(periods)
            time_values.append('{:.4f}'.format(data['time'][idx]))
        else:
            time_values.append('N/A')
    
    report.append('| {} | {} | {} | {} | {} | {} |'.format(
        model_name, data['category'], *time_values
    ))

# 保存报告
with open('${OUTPUT_DIR}/performance_report_${TIMESTAMP}.md', 'w') as f:
    f.write('\n'.join(report))

# 创建准确率对比图
plt.figure(figsize=(12, 8))
for model_name, data in performance_data.items():
    if len(data['periods']) >= 2:  # 至少有两个数据点才能画线
        plt.plot(data['periods'], data['accuracy'], marker='o', label=model_name)

plt.title('模型准确率随测试期数变化')
plt.xlabel('测试期数')
plt.ylabel('准确率')
plt.grid(True, alpha=0.3)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig('${OUTPUT_DIR}/accuracy_comparison_${TIMESTAMP}.png', dpi=300)

# 创建预测时间对比图
plt.figure(figsize=(12, 8))
for model_name, data in performance_data.items():
    if len(data['periods']) >= 2:  # 至少有两个数据点才能画线
        plt.plot(data['periods'], data['time'], marker='o', label=model_name)

plt.title('模型预测时间随测试期数变化')
plt.xlabel('测试期数')
plt.ylabel('预测时间 (秒)')
plt.grid(True, alpha=0.3)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig('${OUTPUT_DIR}/time_comparison_${TIMESTAMP}.png', dpi=300)

print('✅ 性能报告已生成')
"

echo "✅ 性能测试流程完成"
echo "📁 结果保存在: ${OUTPUT_DIR}"