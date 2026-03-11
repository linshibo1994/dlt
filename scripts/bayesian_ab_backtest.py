#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
贝叶斯预测多窗口对比回测脚本

默认比较 legacy vs enhanced，支持多种子统计，避免手工重复运行。
"""

import os
import sys
import argparse
import random
import numpy as np


def _setup_env(project_root: str) -> None:
    mpl_dir = os.path.join(project_root, 'artifacts', 'mplconfig')
    xdg_dir = os.path.join(project_root, 'artifacts', 'cache')
    os.makedirs(mpl_dir, exist_ok=True)
    os.makedirs(xdg_dir, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', mpl_dir)
    os.environ.setdefault('XDG_CACHE_HOME', xdg_dir)


def _inject_paths(project_root: str) -> None:
    app_dir = os.path.join(project_root, 'backend', 'app')
    for p in [
        app_dir,
        os.path.join(app_dir, 'core'),
        os.path.join(app_dir, 'utils'),
        os.path.join(app_dir, 'predictors'),
        os.path.join(app_dir, 'analyzers'),
        os.path.join(app_dir, 'learning'),
        os.path.join(app_dir, 'improvements'),
        project_root,
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(',') if x.strip()]


def run_window(df, data_manager, tracker, periods: int, start: int, test: int,
               mode: str, base_seed: int, predictor) -> dict:
    total_predictions = 0
    total_wins = 0
    prize_stats: dict = {}

    for i in range(test):
        period_idx = start + i
        if period_idx >= len(df):
            break
        row = df.iloc[period_idx]
        actual_front, actual_back = data_manager.parse_balls(row)

        seed = base_seed + period_idx * 17
        random.seed(seed)
        np.random.seed(seed % (2**32))

        pred = predictor.bayesian_predict(1, periods=periods, use_enhanced=(mode == 'enhanced'))
        if not pred:
            continue
        predicted_front, predicted_back = pred[0]

        prize_level, _, _ = tracker._calculate_prize_level(
            predicted_front, predicted_back, actual_front, actual_back
        )
        total_predictions += 1
        if prize_level != '未中奖':
            total_wins += 1
        prize_stats[prize_level] = prize_stats.get(prize_level, 0) + 1

    win_rate = total_wins / total_predictions if total_predictions else 0.0
    return {
        'total': total_predictions,
        'wins': total_wins,
        'win_rate': win_rate,
        'prize_stats': prize_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="贝叶斯预测多窗口对比回测")
    parser.add_argument('--periods', type=int, default=500, help='分析期数')
    parser.add_argument('--test', type=int, default=500, help='测试期数')
    parser.add_argument('--starts', type=str, default='100,300,500,700', help='起始期数列表，逗号分隔')
    parser.add_argument('--seeds', type=str, default='101,202,303,404,505', help='随机种子列表，逗号分隔')
    parser.add_argument('--mode', choices=['compare', 'legacy', 'enhanced'], default='compare',
                        help='对比模式：compare=legacy+enhanced')
    parser.add_argument('--mix', type=float, default=None, help='Dirichlet混合权重')
    parser.add_argument('--conc', type=float, default=None, help='Dirichlet集中度')
    parser.add_argument('--decay-enabled', action='store_true', help='启用时间衰减')
    parser.add_argument('--decay-disabled', action='store_true', help='关闭时间衰减')
    parser.add_argument('--decay-half-life', type=float, default=None, help='衰减半衰期')
    parser.add_argument('--decay-min-weight', type=float, default=None, help='衰减最小权重')
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    _setup_env(project_root)
    _inject_paths(project_root)

    import core_modules as cm
    from predictors.predictor_modules import TraditionalPredictor
    from adaptive_learning_modules import AccuracyTracker
    from analyzer_modules import BayesianConfig, load_bayesian_config

    overrides = {}
    if args.mix is not None:
        overrides['dirichlet_mix_weight'] = args.mix
    if args.conc is not None:
        overrides['dirichlet_concentration'] = args.conc
    if args.decay_enabled:
        overrides['decay_enabled'] = True
    if args.decay_disabled:
        overrides['decay_enabled'] = False
    if args.decay_half_life is not None:
        overrides['decay_half_life'] = args.decay_half_life
    if args.decay_min_weight is not None:
        overrides['decay_min_weight'] = args.decay_min_weight

    load_bayesian_config(overrides or None)

    df = cm.data_manager.get_data()
    if df is None:
        print('ERROR 数据为空')
        return 1

    tracker = AccuracyTracker()
    starts = _parse_int_list(args.starts)
    seeds = _parse_int_list(args.seeds)

    modes = ['legacy', 'enhanced'] if args.mode == 'compare' else [args.mode]

    print('=== MULTI-WINDOW BACKTEST ===')
    print(f'periods={args.periods}, test={args.test}, starts={starts}, seeds={seeds}')
    print(f'config: mix={BayesianConfig.DIRICHLET_MIX_WEIGHT}, conc={BayesianConfig.DIRICHLET_CONCENTRATION}, '
          f'decay={BayesianConfig.DECAY_ENABLED}, half_life={BayesianConfig.DECAY_HALF_LIFE}, '
          f'min_weight={BayesianConfig.DECAY_MIN_WEIGHT}')

    for mode in modes:
        seed_avgs = []
        total_wins = 0
        total_preds = 0
        print(f'\nMODE={mode}')
        for base_seed in seeds:
            window_rates = []
            predictor = TraditionalPredictor()
            for start in starts:
                res = run_window(
                    df, cm.data_manager, tracker, args.periods, start, args.test,
                    mode, base_seed, predictor
                )
                window_rates.append(res['win_rate'])
                total_wins += res['wins']
                total_preds += res['total']
            seed_avg = sum(window_rates) / len(window_rates)
            seed_avgs.append(seed_avg)
            print(f'  seed={base_seed} avg_win_rate={seed_avg:.3f} window_rates={[round(r,3) for r in window_rates]}')

        mean = float(np.mean(seed_avgs))
        std = float(np.std(seed_avgs, ddof=1)) if len(seed_avgs) > 1 else 0.0
        overall = total_wins / total_preds if total_preds else 0.0
        print(f'  mean_seed_avg={mean:.3f} std_seed_avg={std:.3f} overall={overall:.3f} total={total_wins}/{total_preds}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
