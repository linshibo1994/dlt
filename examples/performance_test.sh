#!/bin/bash
# 算法性能测试脚本
# 使用方法: bash examples/performance_test.sh

echo "📈 开始算法性能测试..."

# 测试参数配置
START_PERIOD=100
TEST_PERIODS=500
METHODS=("frequency" "hot_cold" "missing" "markov" "bayesian" "ensemble")

# 创建输出目录
mkdir -p output/performance_test

echo "📊 测试参数:"
echo "  起始期数: $START_PERIOD"
echo "  测试期数: $TEST_PERIODS"
echo "  测试方法: ${#METHODS[@]}种"

# 性能测试
for method in "${METHODS[@]}"; do
    echo ""
    echo "🔄 测试 $method 算法性能..."
    
    # 记录开始时间
    start_time=$(date +%s)
    
    # 执行回测
    python3 dlt_main.py backtest -s $START_PERIOD -t $TEST_PERIODS -m $method > output/performance_test/backtest_${method}.log 2>&1
    
    # 记录结束时间
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    if [ $? -eq 0 ]; then
        echo "✅ $method 测试完成 (耗时: ${duration}秒)"
        
        # 提取中奖率信息
        win_rate=$(grep "中奖率:" output/performance_test/backtest_${method}.log | tail -1 | awk '{print $2}')
        echo "  中奖率: $win_rate"
    else
        echo "❌ $method 测试失败"
        win_rate="N/A"
    fi
    
    # 记录测试结果
    echo "$method,$win_rate,$duration" >> output/performance_test/results.csv
done

# 生成性能对比报告
echo ""
echo "📋 生成性能对比报告..."

cat > output/performance_test/performance_report.txt << EOF
算法性能测试报告
================

测试时间: $(date)
起始期数: $START_PERIOD
测试期数: $TEST_PERIODS

测试结果:
EOF

# 添加表头
echo "算法名称,中奖率,耗时(秒)" >> output/performance_test/performance_report.txt
echo "------------------------" >> output/performance_test/performance_report.txt

# 添加测试结果
if [ -f "output/performance_test/results.csv" ]; then
    cat output/performance_test/results.csv >> output/performance_test/performance_report.txt
fi

cat >> output/performance_test/performance_report.txt << EOF

性能分析:
1. 中奖率: 反映算法预测准确性
2. 耗时: 反映算法计算效率
3. 综合评分 = 中奖率 × 权重1 + (1/耗时) × 权重2

建议:
- 高中奖率算法适合实际投注
- 快速算法适合批量测试
- 平衡算法适合日常使用

详细日志: output/performance_test/backtest_*.log
EOF

echo "📄 性能报告已生成: output/performance_test/performance_report.txt"

# 显示测试结果摘要
echo ""
echo "📊 测试结果摘要:"
echo "算法名称     中奖率    耗时(秒)"
echo "--------------------------------"
if [ -f "output/performance_test/results.csv" ]; then
    while IFS=',' read -r method win_rate duration; do
        printf "%-12s %-8s %-8s\n" "$method" "$win_rate" "$duration"
    done < output/performance_test/results.csv
fi

# 找出最佳算法
echo ""
echo "🏆 推荐算法:"
if [ -f "output/performance_test/results.csv" ]; then
    # 按中奖率排序（简单排序）
    best_method=$(sort -t',' -k2 -nr output/performance_test/results.csv | head -1 | cut -d',' -f1)
    echo "  最高中奖率: $best_method"
    
    # 按速度排序
    fastest_method=$(sort -t',' -k3 -n output/performance_test/results.csv | head -1 | cut -d',' -f1)
    echo "  最快速度: $fastest_method"
fi

echo ""
echo "📁 生成的文件:"
ls -la output/performance_test/

echo ""
echo "🎉 性能测试完成！"
echo "💡 建议: 查看详细报告选择最适合的算法"
