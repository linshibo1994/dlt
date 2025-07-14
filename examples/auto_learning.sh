#!/bin/bash
# 自动化学习脚本
# 使用方法: bash examples/auto_learning.sh

echo "🧠 开始自动化学习..."

# 学习参数配置
START_PERIOD=100
TEST_PERIODS=1000
ALGORITHMS=("ucb1" "epsilon_greedy" "thompson_sampling")

# 创建输出目录
mkdir -p output/auto_learning

echo "📊 学习参数:"
echo "  起始期数: $START_PERIOD"
echo "  测试期数: $TEST_PERIODS"
echo "  算法数量: ${#ALGORITHMS[@]}"

# 多种算法学习
for algo in "${ALGORITHMS[@]}"; do
    echo ""
    echo "🔄 开始 $algo 算法学习..."
    
    # 执行学习
    python3 dlt_main.py learn -s $START_PERIOD -t $TEST_PERIODS --algorithm $algo --save ${algo}_results.json
    
    if [ $? -eq 0 ]; then
        echo "✅ $algo 学习完成"
        
        # 基于学习结果进行智能预测
        echo "🎯 基于 $algo 结果进行智能预测..."
        python3 dlt_main.py smart -c 5 --load output/learning/${algo}_results.json --save ${algo}_smart_pred.json
        
        if [ $? -eq 0 ]; then
            echo "✅ $algo 智能预测完成"
        else
            echo "❌ $algo 智能预测失败"
        fi
    else
        echo "❌ $algo 学习失败"
    fi
done

# 生成学习对比报告
echo ""
echo "📋 生成学习对比报告..."

cat > output/auto_learning/learning_comparison.txt << EOF
自动化学习对比报告
==================

学习时间: $(date)
起始期数: $START_PERIOD
测试期数: $TEST_PERIODS

算法对比:
EOF

# 分析每个算法的学习结果
for algo in "${ALGORITHMS[@]}"; do
    result_file="output/learning/${algo}_results.json"
    if [ -f "$result_file" ]; then
        echo "  $algo: 学习完成 ✅" >> output/auto_learning/learning_comparison.txt
    else
        echo "  $algo: 学习失败 ❌" >> output/auto_learning/learning_comparison.txt
    fi
done

cat >> output/auto_learning/learning_comparison.txt << EOF

文件位置:
- 学习结果: output/learning/
- 智能预测: output/predictions/
- 对比报告: output/auto_learning/

使用建议:
1. 查看各算法的中奖率对比
2. 选择表现最好的算法进行后续预测
3. 定期重新学习以适应新的数据模式
EOF

echo "📄 学习对比报告已生成: output/auto_learning/learning_comparison.txt"

# 显示学习结果文件
echo ""
echo "📁 生成的学习结果文件:"
ls -la output/learning/*_results.json 2>/dev/null || echo "  无学习结果文件"

echo ""
echo "📁 生成的智能预测文件:"
ls -la output/predictions/*_smart_pred.json 2>/dev/null || echo "  无智能预测文件"

echo ""
echo "🎉 自动化学习任务完成！"
echo "💡 建议: 使用 'python3 dlt_main.py smart -c 5 --load output/learning/最佳算法_results.json' 进行智能预测"
