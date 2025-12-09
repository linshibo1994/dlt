#!/bin/bash

# 大乐透预测方法测试脚本启动器
# 使用方法: ./start_test.sh [模式]

echo "🎯 大乐透预测方法测试工具"
echo "================================"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3，请先安装 Python 3.8+"
    exit 1
fi

# 切换到脚本目录
cd "$(dirname "$0")"

# 如果没有指定参数，显示菜单
if [ $# -eq 0 ]; then
    echo ""
    echo "请选择测试模式:"
    echo "1. 系统检查"
    echo "2. 快速测试 (5-10分钟)"
    echo "3. 全面测试 (2-8小时)"
    echo "4. 优化测试 (随机参数)"
    echo "5. 自定义测试"
    echo "6. 查看配置"
    echo "7. 创建配置文件"
    echo ""
    read -p "请输入选项 (1-7): " choice
    
    case $choice in
        1)
            echo "正在执行系统检查..."
            python3 test_predictor.py check
            ;;
        2)
            echo "正在启动快速测试..."
            python3 test_predictor.py quick
            ;;
        3)
            echo "正在启动全面测试..."
            echo "⚠️  注意: 此模式可能需要数小时完成"
            read -p "确认继续? (y/N): " confirm
            if [[ $confirm =~ ^[Yy]$ ]]; then
                python3 test_predictor.py comprehensive
            else
                echo "已取消"
            fi
            ;;
        4)
            echo "正在启动优化测试..."
            python3 test_predictor.py optimization
            ;;
        5)
            echo "自定义测试模式"
            echo "格式: 方法名 起始期数 结束期数 [步长] [注数]"
            echo "示例: markov 10 2000 50 1"
            echo ""
            read -p "请输入预测方法: " method
            read -p "起始期数: " start_periods
            read -p "结束期数: " end_periods
            read -p "步长 (默认50): " step
            read -p "注数 (默认1): " count
            
            step=${step:-50}
            count=${count:-1}
            
            echo "正在执行自定义测试: $method $start_periods-$end_periods (步长$step, 每次$count注)..."
            python3 test_predictor.py custom "$method" "$start_periods" "$end_periods" "$step" "$count"
            ;;
        6)
            python3 test_predictor.py config
            ;;
        7)
            python3 test_predictor.py create-config
            ;;
        *)
            echo "无效选项"
            exit 1
            ;;
    esac
else
    # 直接执行命令行参数
    case $1 in
        "check"|"quick"|"comprehensive"|"optimization"|"config"|"create-config")
            python3 test_predictor.py "$@"
            ;;
        "custom")
            if [ $# -lt 4 ]; then
                echo "❌ 自定义测试需要参数: custom <方法> <起始期数> <结束期数> [步长] [注数]"
                echo "示例: $0 custom markov 10 2000 50 1"
                exit 1
            fi
            python3 test_predictor.py "$@"
            ;;
        "help"|"-h"|"--help")
            echo "使用方法:"
            echo "  $0                    # 交互式菜单"
            echo "  $0 check              # 系统检查"
            echo "  $0 quick              # 快速测试"
            echo "  $0 comprehensive      # 全面测试"
            echo "  $0 optimization       # 优化测试"
            echo "  $0 custom <方法> <起始期数> <结束期数> [步长] [注数]"
            echo "  $0 config             # 查看配置"
            echo "  $0 create-config      # 创建配置文件"
            echo ""
            echo "更多信息请查看 test_predictor_guide.md"
            ;;
        *)
            echo "❌ 未知命令: $1"
            echo "使用 '$0 help' 查看帮助"
            exit 1
            ;;
    esac
fi

echo ""
echo "测试完成！结果保存在 results/ 目录中"
echo "查看 HTML 报告获取详细分析结果"