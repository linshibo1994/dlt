#!/usr/bin/env python3
"""
大乐透中奖等级判断模块
实现基于2019年新规则的9个奖级判断
"""

import re
from typing import Tuple, Dict, List, Set


class LotteryJudge:
    """大乐透中奖等级判断器"""
    
    def __init__(self):
        self.prize_levels = {
            1: "一等奖",
            2: "二等奖", 
            3: "三等奖",
            4: "四等奖",
            5: "五等奖",
            6: "六等奖",
            7: "七等奖",
            8: "八等奖",
            9: "九等奖"
        }
        
        # 大乐透正确的中奖规则
        self.prize_conditions = {
            1: [(5, 2)],           # 前区5中5 + 后区2中2
            2: [(5, 1)],           # 前区5中5 + 后区2中1  
            3: [(5, 0)],           # 前区5中5 + 后区2中0
            4: [(4, 2)],           # 前区5中4 + 后区2中2
            5: [(4, 1), (3, 2)],   # 前区5中4后区2中1 或 前区5中3后区2中2
            6: [(4, 0), (3, 1), (2, 2)],  # 前区5中4后区2中0 或 前区5中3后区2中1 或 前区5中2后区2中2
            7: [(3, 0), (2, 1)],   # 前区5中3后区2中0 或 前区5中2后区2中1
            8: [(1, 2), (2, 0), (0, 2)],  # 前区5中1后区2中2 或 前区5中2后区2中0 或 前区5中0后区2中2
            9: [(1, 1), (1, 0), (0, 1)]   # 前区5中1后区2中1 或 前区5中1后区2中0 或 前区5中0后区2中1
        }

    def parse_numbers(self, numbers_str: str) -> List[int]:
        """解析号码字符串"""
        # 移除引号并按逗号分割
        clean_str = numbers_str.strip('"').strip("'")
        numbers = [int(x.strip()) for x in clean_str.split(',')]
        return numbers

    def parse_lottery_data(self, front_balls: str, back_balls: str) -> Tuple[Set[int], Set[int]]:
        """解析彩票数据"""
        front_set = set(self.parse_numbers(front_balls))
        back_set = set(self.parse_numbers(back_balls))
        return front_set, back_set

    def calculate_matches(self, predicted_front: Set[int], predicted_back: Set[int], 
                         winning_front: Set[int], winning_back: Set[int]) -> Tuple[int, int]:
        """计算前区和后区的匹配数量"""
        front_matches = len(predicted_front.intersection(winning_front))
        back_matches = len(predicted_back.intersection(winning_back))
        return front_matches, back_matches

    def judge_prize_level(self, front_matches: int, back_matches: int) -> int:
        """根据匹配数量判断中奖等级，按照从高到低等级顺序判断"""
        # 按等级从高到低顺序检查
        for level in sorted(self.prize_conditions.keys()):
            conditions = self.prize_conditions[level]
            if (front_matches, back_matches) in conditions:
                return level
        return 0  # 未中奖  # 未中奖

    def check_winning(self, predicted_front: str, predicted_back: str, 
                     winning_front: str, winning_back: str) -> Dict:
        """检查中奖情况"""
        try:
            # 解析预测号码和中奖号码
            pred_front_set, pred_back_set = self.parse_lottery_data(predicted_front, predicted_back)
            win_front_set, win_back_set = self.parse_lottery_data(winning_front, winning_back)
            
            # 验证号码数量
            if len(pred_front_set) != 5 or len(pred_back_set) != 2:
                raise ValueError(f"预测号码格式错误: 前区{len(pred_front_set)}个, 后区{len(pred_back_set)}个")
            
            if len(win_front_set) != 5 or len(win_back_set) != 2:
                raise ValueError(f"开奖号码格式错误: 前区{len(win_front_set)}个, 后区{len(win_back_set)}个")
            
            # 计算匹配数量
            front_matches, back_matches = self.calculate_matches(
                pred_front_set, pred_back_set, win_front_set, win_back_set
            )
            
            # 判断中奖等级
            prize_level = self.judge_prize_level(front_matches, back_matches)
            
            result = {
                'is_winning': prize_level > 0,
                'prize_level': prize_level,
                'prize_name': self.prize_levels.get(prize_level, "未中奖"),
                'front_matches': front_matches,
                'back_matches': back_matches,
                'match_combination': f"{front_matches}+{back_matches}",
                'predicted_front': sorted(list(pred_front_set)),
                'predicted_back': sorted(list(pred_back_set)),
                'winning_front': sorted(list(win_front_set)),
                'winning_back': sorted(list(win_back_set)),
                'matched_front_numbers': sorted(list(pred_front_set.intersection(win_front_set))),
                'matched_back_numbers': sorted(list(pred_back_set.intersection(win_back_set)))
            }
            
            return result
            
        except Exception as e:
            return {
                'error': True,
                'error_message': str(e),
                'is_winning': False,
                'prize_level': 0
            }

    def is_major_prize(self, prize_level: int) -> bool:
        """判断是否为重大奖项（一等奖或二等奖）"""
        return prize_level in [1, 2]

    def get_prize_description(self, prize_level: int) -> str:
        """获取奖级描述"""
        if prize_level == 0:
            return "未中奖"
        
        conditions = self.prize_conditions.get(prize_level, [])
        condition_strs = [f"{f}+{b}" for f, b in conditions]
        
        return f"{self.prize_levels[prize_level]} ({'/'.join(condition_strs)})"

    def validate_numbers(self, front_balls: str, back_balls: str) -> bool:
        """验证号码格式和范围"""
        try:
            front_nums = self.parse_numbers(front_balls)
            back_nums = self.parse_numbers(back_balls)
            
            # 检查数量
            if len(front_nums) != 5 or len(back_nums) != 2:
                return False
            
            # 检查范围
            for num in front_nums:
                if not (1 <= num <= 35):
                    return False
            
            for num in back_nums:
                if not (1 <= num <= 12):
                    return False
            
            # 检查重复
            if len(set(front_nums)) != 5 or len(set(back_nums)) != 2:
                return False
            
            return True
            
        except:
            return False


def test_lottery_judge():
    """测试函数"""
    judge = LotteryJudge()
    
    # 测试用例
    test_cases = [
        # 一等奖
        ("01,02,03,04,05", "01,02", "01,02,03,04,05", "01,02", 1),
        # 二等奖
        ("01,02,03,04,05", "01,02", "01,02,03,04,05", "01,03", 2),
        # 三等奖
        ("01,02,03,04,05", "01,02", "01,02,03,04,05", "03,04", 3),
        # 未中奖
        ("01,02,03,04,05", "01,02", "06,07,08,09,10", "03,04", 0),
    ]
    
    print("=== 大乐透中奖判断测试 ===")
    for i, (pred_f, pred_b, win_f, win_b, expected) in enumerate(test_cases):
        result = judge.check_winning(pred_f, pred_b, win_f, win_b)
        print(f"\n测试 {i+1}:")
        print(f"预测: 前区{pred_f} 后区{pred_b}")
        print(f"开奖: 前区{win_f} 后区{win_b}")
        print(f"结果: {result.get('prize_name', '错误')}")
        print(f"匹配: {result.get('match_combination', 'N/A')}")
        print(f"预期: {judge.prize_levels.get(expected, '未中奖')}")
        print(f"正确: {'✓' if result.get('prize_level') == expected else '✗'}")


if __name__ == "__main__":
    test_lottery_judge()