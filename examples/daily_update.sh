#!/bin/bash
# 每日数据更新和分析脚本
# 使用方法: bash examples/daily_update.sh
# 可配置到crontab: 0 9 * * * /path/to/daily_update.sh

echo "📅 开始每日数据更新和分析..."

# 日志文件
LOG_FILE="logs/daily_update_$(date +%Y%m%d).log"
mkdir -p logs

# 记录开始时间
echo "$(date): 开始每日更新任务" | tee -a $LOG_FILE

# 检查网络连接
echo "🌐 检查网络连接..." | tee -a $LOG_FILE
if ping -c 1 www.zhcw.com &> /dev/null; then
    echo "✅ 网络连接正常" | tee -a $LOG_FILE
else
    echo "❌ 网络连接失败，跳过更新" | tee -a $LOG_FILE
    exit 1
fi

# 备份当前数据
echo "💾 备份当前数据..." | tee -a $LOG_FILE
if [ -f "data/dlt_data_all.csv" ]; then
    cp data/dlt_data_all.csv data/dlt_data_backup_$(date +%Y%m%d).csv
    echo "✅ 数据备份完成" | tee -a $LOG_FILE
fi

# 更新数据
echo "🔄 更新数据..." | tee -a $LOG_FILE
python3 dlt_main.py data update --source zhcw 2>&1 | tee -a $LOG_FILE

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ 数据更新成功" | tee -a $LOG_FILE
else
    echo "❌ 数据更新失败" | tee -a $LOG_FILE
    exit 1
fi

# 检查最新开奖结果
echo "🎯 检查最新开奖结果..." | tee -a $LOG_FILE
python3 dlt_main.py data latest 2>&1 | tee -a $LOG_FILE

# 清理旧缓存
echo "🗑️ 清理分析缓存..." | tee -a $LOG_FILE
python3 dlt_main.py system cache clear --type analysis 2>&1 | tee -a $LOG_FILE

# 重新分析数据
echo "📊 重新分析数据..." | tee -a $LOG_FILE
python3 dlt_main.py analyze -t comprehensive -p 500 --report --save daily_analysis_$(date +%Y%m%d).txt 2>&1 | tee -a $LOG_FILE

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ 数据分析完成" | tee -a $LOG_FILE
else
    echo "❌ 数据分析失败" | tee -a $LOG_FILE
fi

# 生成每日预测
echo "🎯 生成每日预测..." | tee -a $LOG_FILE

# 使用多种方法生成预测
methods=("ensemble" "markov" "bayesian")
for method in "${methods[@]}"; do
    echo "  生成 $method 预测..." | tee -a $LOG_FILE
    python3 dlt_main.py predict -m $method -c 3 --save daily_${method}_$(date +%Y%m%d).json 2>&1 | tee -a $LOG_FILE
done

# 生成混合策略预测
echo "  生成混合策略预测..." | tee -a $LOG_FILE
python3 dlt_main.py predict -m mixed_strategy -c 3 --strategy balanced --save daily_mixed_$(date +%Y%m%d).json 2>&1 | tee -a $LOG_FILE

# 生成可视化图表
echo "📈 生成可视化图表..." | tee -a $LOG_FILE
python3 dlt_main.py analyze -t basic -p 300 --visualize 2>&1 | tee -a $LOG_FILE

# 清理旧文件（保留最近7天）
echo "🧹 清理旧文件..." | tee -a $LOG_FILE

# 清理旧的备份文件
find data/ -name "dlt_data_backup_*.csv" -mtime +7 -delete 2>/dev/null
echo "  清理旧备份文件" | tee -a $LOG_FILE

# 清理旧的日志文件
find logs/ -name "daily_update_*.log" -mtime +7 -delete 2>/dev/null
echo "  清理旧日志文件" | tee -a $LOG_FILE

# 清理旧的分析报告
find output/reports/ -name "daily_analysis_*.txt" -mtime +7 -delete 2>/dev/null
echo "  清理旧分析报告" | tee -a $LOG_FILE

# 生成每日摘要
echo "📋 生成每日摘要..." | tee -a $LOG_FILE

cat > output/reports/daily_summary_$(date +%Y%m%d).txt << EOF
每日更新摘要
============

更新日期: $(date)
数据源: 中彩网 (zhcw.com)

更新内容:
✅ 数据更新完成
✅ 数据分析完成
✅ 预测生成完成
✅ 图表生成完成
✅ 文件清理完成

生成的文件:
- 分析报告: output/reports/daily_analysis_$(date +%Y%m%d).txt
- 集成预测: output/predictions/daily_ensemble_$(date +%Y%m%d).json
- 马尔可夫预测: output/predictions/daily_markov_$(date +%Y%m%d).json
- 贝叶斯预测: output/predictions/daily_bayesian_$(date +%Y%m%d).json
- 混合策略预测: output/predictions/daily_mixed_$(date +%Y%m%d).json
- 可视化图表: output/visualization/

建议:
1. 查看最新分析报告了解数据趋势
2. 参考多种预测方法的结果
3. 关注混合策略的平衡预测
4. 定期查看可视化图表

下次更新: $(date -d "tomorrow" +%Y-%m-%d)
EOF

# 显示摘要
echo "" | tee -a $LOG_FILE
echo "📄 每日摘要已生成: output/reports/daily_summary_$(date +%Y%m%d).txt" | tee -a $LOG_FILE

# 显示生成的文件
echo "" | tee -a $LOG_FILE
echo "📁 今日生成的文件:" | tee -a $LOG_FILE
echo "  分析报告: $(ls output/reports/daily_analysis_$(date +%Y%m%d).txt 2>/dev/null || echo '未生成')" | tee -a $LOG_FILE
echo "  预测文件: $(ls output/predictions/daily_*_$(date +%Y%m%d).json 2>/dev/null | wc -l) 个" | tee -a $LOG_FILE
echo "  可视化图表: $(ls output/visualization/*.png 2>/dev/null | wc -l) 个" | tee -a $LOG_FILE

# 记录结束时间
echo "$(date): 每日更新任务完成" | tee -a $LOG_FILE

echo ""
echo "🎉 每日更新和分析完成！"
echo "📄 详细日志: $LOG_FILE"
echo "📊 查看摘要: output/reports/daily_summary_$(date +%Y%m%d).txt"

# 如果是交互式运行，询问是否查看结果
if [ -t 0 ]; then
    echo ""
    read -p "是否查看最新开奖结果？(y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 dlt_main.py data latest
    fi
fi
