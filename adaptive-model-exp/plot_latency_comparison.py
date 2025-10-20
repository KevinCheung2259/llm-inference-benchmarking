#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------- Global matplotlib style (paper-like) ----------
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'Arial'],
    'axes.linewidth': 1.2,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'legend.frameon': False,
    'figure.dpi': 160,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.04,
})

# ---------- Config ----------
DATA_FILES = [
    ('/home/paperspace/zhangy/llm-inference-bench-char/adaptive-model-exp-new/detailed_results_20250907_083620.csv', 'Native'),
    ('/home/paperspace/zhangy/llm-inference-bench-char/adaptive-model-exp-new/detailed_results_20250907_084451.csv', 'Pretrain'),
    ('/home/paperspace/zhangy/llm-inference-bench-char/adaptive-model-exp-new/detailed_results_20250907_113336.csv', 'Adaptive'),
]
OUT_DIR = Path('/home/paperspace/zhangy/llm-inference-bench-char/adaptive-model-exp-new')

# 显示区间：0–90s，不再剔除冷启动
TIME_MIN, TIME_MAX = 0, 90
BIN = 5
ROLL_WIN = 2
MARK_EVERY = 2

# 颜色/样式（论文图：三色清晰、Adaptive 更醒目）
COLORS = {'Native': 'blue', 'Pretrain': 'green', 'Adaptive': 'orange'}
LINESTYLES = {'Native': '-', 'Pretrain': '--', 'Adaptive': '-'}
MARKERS = {'Native': 'o', 'Pretrain': 's', 'Adaptive': '^'}

def _read_one(fp, name):
    df = pd.read_csv(fp)
    needed = {'send_time', 'total_time'}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")

    df['latency'] = df['total_time'] - df['send_time']
    if 'ttft' not in df.columns: df['ttft'] = np.nan
    if 'tpot' not in df.columns: df['tpot'] = np.nan

    # 相对时间（从0开始）
    start = df['send_time'].min()
    df['t_rel'] = df['send_time'] - start

    # 只截取到 TIME_MAX，不再剔除前若干秒
    df = df[df['t_rel'] <= TIME_MAX].copy()
    # 归一化确保横轴从0开始（如果数据起点>0也安全）
    df['t_rel'] = df['t_rel'] - df['t_rel'].min()

    df = df[['t_rel', 'latency', 'ttft', 'tpot']].copy()
    df['method'] = name
    return df

def load_data():
    return [_read_one(fp, name) for fp, name in DATA_FILES]

def aggregate(data_list, bin_sec=BIN):
    """中位数 + IQR（Q25~Q75）"""
    agg = []
    for df in data_list:
        df['bin'] = (df['t_rel'] // bin_sec).astype(int) * bin_sec
        g = df.groupby('bin')
        q = g['latency'].quantile([0.25, 0.5, 0.75]).unstack()
        q.columns = ['q25', 'q50', 'q75']
        cnt = g['latency'].size().rename('count')
        out = pd.concat([q, cnt], axis=1).reset_index()
        out['method'] = df['method'].iloc[0]
        out = out[out['count'] >= 3]  # 小样本窗口丢弃
        if len(out) >= 3:
            for col in ('q50', 'q25', 'q75'):
                out[col] = out[col].rolling(ROLL_WIN, min_periods=1).median()
        agg.append(out)
    return agg

def _robust_ylim(dfs):
    vals = np.concatenate([d['q50'].values for d in dfs if len(d)])
    if len(vals) == 0: return (0, 1)
    lo, hi = np.nanpercentile(vals, 2), np.nanpercentile(vals, 98)
    pad = (hi - lo) * 0.12 if hi > lo else 1.0
    return max(0.0, lo - pad), hi + pad

def _improvement_between(df_ad, left_end, right_start, right_end):
    """比较 [0,left_end] 与 [right_start,right_end] 两阶段中位数的改善百分比"""
    early = df_ad[df_ad['bin'] <= left_end]['q50'].mean()
    late  = df_ad[(df_ad['bin'] >= right_start) & (df_ad['bin'] <= right_end)]['q50'].mean()
    if pd.isna(early) or pd.isna(late) or early <= 0:
        return np.nan, early, late
    return (early - late) / early * 100.0, early, late

def plot_latency(aggs, split=25):
    fig, ax = plt.subplots(figsize=(4.5, 2.8), constrained_layout=True)

    ymin, ymax = _robust_ylim(aggs)

    # 背景阶段：0–split 学习期，split–TIME_MAX 稳定期
    ax.axvspan(0, split,  facecolor="#f2dede", alpha=0.35, zorder=0)
    ax.axvspan(split, TIME_MAX, facecolor="#dff0d8", alpha=0.35, zorder=0)
    ax.axvline(x=split, color='k', linestyle=':', linewidth=1.0, alpha=0.8)

    # === 阶段文字说明（顶部居中，不随 y 轴缩放而移动）===
    phase1_mid = split / 2.0
    phase2_mid = (split + TIME_MAX) / 2.0
    # 使用 x=数据坐标，y=轴坐标（0=底部, 1=顶部）
    ax.text(phase1_mid, 0.98, f"Learning Period",
            transform=ax.get_xaxis_transform(),  # x: data, y: axes
            ha='center', va='top', fontsize=8, color='#444444', fontweight='bold')
    ax.text(phase2_mid, 0.98, f"Stable Period",
            transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=8, color='#444444', fontweight='bold')

    # 先画阴影，再画主线
    for df in aggs:
        m = df['method'].iloc[0]
        # 横坐标用窗口中心
        x = df['bin'] + BIN / 2.0
        ax.fill_between(x, df['q25'], df['q75'],
                        color=COLORS.get(m, '#444444'), alpha=0.15, linewidth=0, zorder=1)
        ax.plot(x, df['q50'],
                LINESTYLES.get(m, '-'),
                color=COLORS.get(m, '#444444'),
                marker=MARKERS.get(m, 'o'),
                markevery=max(1, MARK_EVERY),
                linewidth=2.0, markersize=4.0,
                label=m, alpha=0.98, zorder=2)

    # ---- 坐标范围设置 ----
    ax.set_xlim(-2, 92)                      # 留一点边距（比 0–90 更美观）
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Latency (s)')
    ax.grid(True, axis='y', linestyle='-', linewidth=0.4, alpha=0.25)

    # 图例横向上方（不遮挡）
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
              ncol=len(COLORS), fontsize=10,
              frameon=False, handlelength=1.8, handletextpad=0.5, columnspacing=1.2)

    # 不要标题：论文由图注说明
    return fig, ax

def print_stats(aggs):
    print("\n=== Latency (median by 5s bins) ===")
    print(f"{'Method':<10} {'Median@all':>12} {'P25@all':>10} {'P75@all':>10} {'MaxMedBin':>10} {'MinMedBin':>10}")
    for df in aggs:
        m = df['method'].iloc[0]
        med = df['q50'].median()
        p25 = df['q50'].quantile(0.25)
        p75 = df['q50'].quantile(0.75)
        mx  = df['q50'].max()
        mn  = df['q50'].min()
        print(f"{m:<10} {med:>12.3f} {p25:>10.3f} {p75:>10.3f} {mx:>10.3f} {mn:>10.3f}")

def main():
    print("Starting latency data analysis…")
    data = load_data()
    aggs = aggregate(data, bin_sec=BIN)
    print_stats(aggs)

    print("Creating latency comparison plot…")
    fig, ax = plot_latency(aggs, split=27)   # split 点可改：25/30/45 都可

    pdf = OUT_DIR / 'latency_comparison_paper.pdf'
    fig.savefig(pdf)
    print(f"Saved:\n- {pdf}")

    plt.show()

if __name__ == "__main__":
    main()
