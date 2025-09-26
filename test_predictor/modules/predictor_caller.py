#!/usr/bin/env python3
"""
预测器调用模块
负责调用现有CLI预测接口并解析输出结果
"""

import subprocess
import re
import os
import sys
import time
from typing import List, Dict, Optional, Tuple
import json


class PredictorCaller:
    """预测器CLI调用器"""
    
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            # 默认为项目根目录
            self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        else:
            self.base_dir = base_dir
        
        self.main_script = os.path.join(self.base_dir, 'dlt_main.py')
        self.timeout = 60  # 默认超时时间60秒
        
        # 支持的预测方法
        self.available_methods = [
            'frequency', 'hot_cold', 'missing',
            'markov', 'markov_2nd', 'markov_3rd', 'adaptive_markov',
            'bayesian', 'ensemble', 'compound', 'duplex',
            'markov_compound', 'nine_models', 'lstm', 'transformer', 'gan'
        ]

    def validate_method(self, method: str) -> bool:
        """验证预测方法是否支持"""
        return method in self.available_methods

    def build_command(self, method: str, periods: int, count: int, **kwargs) -> List[str]:
        """构建CLI命令"""
        if not self.validate_method(method):
            raise ValueError(f"不支持的预测方法: {method}")
        
        cmd = [
            sys.executable,  # 使用当前Python解释器
            self.main_script,
            'predict',
            '-m', method,
            '-p', str(periods),
            '-c', str(count)
        ]
        
        # 添加额外参数
        for key, value in kwargs.items():
            if key == 'acceleration' and value:
                cmd.extend(['--acceleration', str(value)])
            elif key == 'compound' and value:
                cmd.append('--compound')
            elif key == 'front_count' and value:
                cmd.extend(['--front-count', str(value)])
            elif key == 'back_count' and value:
                cmd.extend(['--back-count', str(value)])
        
        return cmd

    def execute_command(self, cmd: List[str]) -> Tuple[bool, str, str]:
        """执行CLI命令"""
        try:
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding='utf-8'
            )
            
            success = result.returncode == 0
            stdout = result.stdout
            stderr = result.stderr
            
            return success, stdout, stderr
            
        except subprocess.TimeoutExpired:
            return False, "", f"命令执行超时 (>{self.timeout}秒)"
        except Exception as e:
            return False, "", f"命令执行错误: {str(e)}"

    def parse_prediction_output(self, output: str) -> List[Dict]:
        """解析预测输出，提取预测号码"""
        predictions = []
        
        # 实际的预测结果格式: "第 1 注: 24 29 32 34 35 + 09 10 (方法: frequency, 置信度: 0.500)"
        patterns = [
            # 主要格式：第 X 注: 前区号码 + 后区号码 (其他信息...)
            r'第\s*(\d+)\s*注[：:]\s*(\d{1,2}(?:\s+\d{1,2}){4})\s*[+\+]\s*(\d{1,2}(?:\s+\d{1,2}){1})',
            # 备选格式: 前区: 01,02,03,04,05 后区: 01,02
            r'前区[：:]\s*([0-9,\s]+)\s*后区[：:]?\s*([0-9,\s]+)',
            # 另一种格式: 01,02,03,04,05 + 01,02
            r'(\d{2}(?:,\d{2}){4})\s*[+\+]\s*(\d{2}(?:,\d{2}){1})',
            # JSON格式提取
            r'"front_balls":\s*\[([^\]]+)\].*?"back_balls":\s*\[([^\]]+)\]',
            # 其他可能的格式
            r'(\d{1,2}(?:,\s*\d{1,2}){4})\s*[|\丨]\s*(\d{1,2}(?:,\s*\d{1,2}){1})',
            # 通用空格分隔格式: "数字 数字 数字 数字 数字 + 数字 数字"
            r'(\d{1,2}(?:\s+\d{1,2}){4})\s*[+\+]\s*(\d{1,2}(?:\s+\d{1,2}){1})'
        ]
        
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 尝试匹配各种模式
            for i, pattern in enumerate(patterns):
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    try:
                        if i == 0:  # 主要格式，需要跳过第一个匹配组（注号）
                            _, front_str, back_str = match
                        else:
                            front_str, back_str = match
                        
                        # 清理和解析号码
                        front_numbers = self._parse_numbers(front_str)
                        back_numbers = self._parse_numbers(back_str)
                        
                        # 验证号码
                        if self._validate_numbers(front_numbers, back_numbers):
                            prediction = {
                                'front_balls': ','.join([f"{n:02d}" for n in sorted(front_numbers)]),
                                'back_balls': ','.join([f"{n:02d}" for n in sorted(back_numbers)]),
                                'front_balls_list': sorted(front_numbers),
                                'back_balls_list': sorted(back_numbers),
                                'raw_line': line,
                                'pattern_used': i  # 调试信息：记录使用了哪个模式
                            }
                            predictions.append(prediction)
                            
                    except Exception as e:
                        # 调试信息：记录解析失败的行
                        print(f"解析失败 (模式{i}): {line[:100]}... 错误: {str(e)}")
                        continue
        
        # 去重
        unique_predictions = []
        seen = set()
        for pred in predictions:
            key = (tuple(pred['front_balls_list']), tuple(pred['back_balls_list']))
            if key not in seen:
                seen.add(key)
                unique_predictions.append(pred)
        
        # 调试信息
        print(f"解析结果: 找到 {len(unique_predictions)} 个有效预测")
        for i, pred in enumerate(unique_predictions):
            print(f"  预测{i+1}: 前区{pred['front_balls']} 后区{pred['back_balls']} (使用模式{pred.get('pattern_used', 'unknown')})")
        
        return unique_predictions

    def _parse_numbers(self, numbers_str: str) -> List[int]:
        """解析号码字符串，支持逗号和空格分隔"""
        # 移除各种符号，保留数字、逗号和空格
        clean_str = re.sub(r'[^\d,\s]', '', numbers_str.strip())
        
        # 分割号码：优先按逗号分割，如果没有逗号则按空格分割
        if ',' in clean_str:
            # 逗号分隔格式
            number_strs = [s.strip() for s in clean_str.split(',') if s.strip()]
        else:
            # 空格分隔格式
            number_strs = [s.strip() for s in clean_str.split() if s.strip()]
        
        # 转换为整数
        numbers = []
        for num_str in number_strs:
            if num_str.isdigit():
                numbers.append(int(num_str))
        
        return numbers

    def _validate_numbers(self, front_numbers: List[int], back_numbers: List[int]) -> bool:
        """验证号码的有效性"""
        # 检查数量
        if len(front_numbers) != 5 or len(back_numbers) != 2:
            return False
        
        # 检查范围
        for num in front_numbers:
            if not (1 <= num <= 35):
                return False
        
        for num in back_numbers:
            if not (1 <= num <= 12):
                return False
        
        # 检查重复
        if len(set(front_numbers)) != 5 or len(set(back_numbers)) != 2:
            return False
        
        return True

    def predict(self, method: str, periods: int, count: int, **kwargs) -> Dict:
        """执行预测并返回结果"""
        start_time = time.time()
        
        try:
            # 构建命令
            cmd = self.build_command(method, periods, count, **kwargs)
            print(f"执行预测命令: {' '.join(cmd)}")
            
            # 执行命令
            success, stdout, stderr = self.execute_command(cmd)
            
            execution_time = time.time() - start_time
            
            result = {
                'success': success,
                'method': method,
                'periods': periods,
                'count': count,
                'execution_time': execution_time,
                'command': ' '.join(cmd),
                'timestamp': time.time()
            }
            
            if success:
                print(f"命令执行成功，开始解析输出...")
                # 解析预测结果
                predictions = self.parse_prediction_output(stdout)
                result.update({
                    'predictions': predictions,
                    'prediction_count': len(predictions),
                    'stdout': stdout,
                    'stderr': stderr
                })
                
                if not predictions:
                    print("警告: 无法解析出任何预测结果!")
                    print(f"原始输出前1000字符:\n{stdout[:1000]}")
                    result.update({
                        'success': False,
                        'error': '无法解析预测结果',
                        'debug_output': stdout,  # 保存完整输出用于调试
                        'raw_stdout': stdout,
                        'raw_stderr': stderr
                    })
                else:
                    print(f"成功解析出 {len(predictions)} 个预测结果")
            else:
                print(f"命令执行失败: {stderr}")
                print(f"标准输出: {stdout[:500] if stdout else '无'}")
                result.update({
                    'error': f"命令执行失败: {stderr}",
                    'stdout': stdout,
                    'stderr': stderr,
                    'debug_info': f"返回码非0，可能的原因: 1)方法不存在 2)参数错误 3)数据文件问题"
                })
            
            return result
            
        except Exception as e:
            print(f"预测调用发生异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'method': method,
                'periods': periods,
                'count': count,
                'execution_time': time.time() - start_time,
                'error': f"预测调用异常: {str(e)}",
                'timestamp': time.time(),
                'exception_traceback': traceback.format_exc()
            }

    def batch_predict(self, method_configs: List[Dict], max_retries: int = 2) -> List[Dict]:
        """批量预测"""
        results = []
        
        for i, config in enumerate(method_configs):
            method = config.get('method')
            periods = config.get('periods', 100)
            count = config.get('count', 1)
            kwargs = config.get('kwargs', {})
            
            print(f"执行预测 {i+1}/{len(method_configs)}: {method} (p={periods}, c={count})")
            
            # 重试机制
            for retry in range(max_retries + 1):
                result = self.predict(method, periods, count, **kwargs)
                
                if result['success']:
                    results.append(result)
                    break
                elif retry < max_retries:
                    print(f"  重试 {retry + 1}/{max_retries}...")
                    time.sleep(1)
                else:
                    print(f"  预测失败: {result.get('error', '未知错误')}")
                    results.append(result)
        
        return results

    def test_connection(self) -> bool:
        """测试与主程序的连接"""
        try:
            cmd = [sys.executable, self.main_script, '--help']
            success, stdout, stderr = self.execute_command(cmd)
            return success and 'predict' in stdout.lower()
        except:
            return False

    def get_available_methods(self) -> List[str]:
        """获取可用的预测方法列表"""
        return self.available_methods.copy()