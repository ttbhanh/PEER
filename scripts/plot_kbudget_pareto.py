#!/usr/bin/env python
from __future__ import annotations

"""k-budget Pareto figure: traces PEER, PRAG, and ERRA-R through semantic-F1
x aspect-F1 space as the evidence budget k varies, per platform and pooled.
Consumes the aggregate CSV from scripts/evaluate_evidence.py (columns
dataset, method, k, sem_f1, aspect_f1, n_cases); the pooled panel is a
case-count-weighted average across platforms."""

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

STYLE = {
    'PEER': {'color': '#0072B2', 'marker': 'o'},
    'PRAG': {'color': '#D55E00', 'marker': 's'},
    'ERRA': {'color': '#009E73', 'marker': '^'},
}
DATASET_TITLE = {'baby': 'Amazon Baby', 'yelp': 'Yelp', 'googlelocal': 'Google Local'}
K_ORDER = ['1', '3', '5', '7', '10', '15', '20', '25', '30', '35', '40']
K_LABELS = ['5', '40']
K_SKIP = {('PRAG', '5')}
K_OFFSET = {
    'PEER': {'5': (10, 16), '40': (10, -26)},
    'PRAG': {'5': (22, -30), '40': (24, 14)},
    'ERRA': {'5': (-18, 12), '40': (-32, -18)},
}
LABEL_FS, TICK_FS, SUBTITLE_FS, KLABEL_FS, LEGEND_FS = 25, 21, 23, 21, 21


def weighted_pool(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in df.groupby('method'):
        for k, gk in g.groupby('k'):
            w = gk['n_cases'].values
            rows.append({
                'dataset': 'pooled', 'method': method, 'k': k,
                'sem_f1': float(np.average(gk['sem_f1'], weights=w)),
                'aspect_f1': float(np.average(gk['aspect_f1'], weights=w)),
                'n_cases': int(w.sum()),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evaluation', required=True, help='Aggregate CSV from scripts/evaluate_evidence.py over the full k grid')
    ap.add_argument('--methods', nargs='+', default=['peer_full:PEER', 'prag:PRAG', 'erra_r:ERRA'],
                     help='method_column_value:DisplayName pairs')
    ap.add_argument('--datasets', nargs='+', default=['baby', 'yelp', 'googlelocal'])
    ap.add_argument('--output', required=True, help='Output path (.pdf or .png)')
    args = ap.parse_args()

    method_map = dict(m.split(':') for m in args.methods)
    df = pd.read_csv(args.evaluation)
    df['k'] = df['k'].astype(str)
    df = df[df['method'].isin(method_map)].copy()
    df['method'] = df['method'].map(method_map)
    df = df[df['dataset'].isin(args.datasets)]

    pooled = weighted_pool(df)
    df = pd.concat([df, pooled], ignore_index=True)

    panels = [(ds, DATASET_TITLE.get(ds, ds)) for ds in args.datasets] + [('pooled', 'Pooled')]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6.3))
    if len(panels) == 1:
        axes = [axes]

    for ax, (ds, title) in zip(axes, panels):
        sub_ds = df[df['dataset'] == ds]
        for method in method_map.values():
            sub = sub_ds[sub_ds['method'] == method].set_index('k').reindex(K_ORDER)
            st = STYLE.get(method, {'color': '#333333', 'marker': 'o'})
            ax.plot(sub['sem_f1'].values, sub['aspect_f1'].values, '-', color=st['color'],
                    marker=st['marker'], markersize=7, linewidth=2, label=method, zorder=3)
            for k in K_LABELS:
                if k in sub.index and (method, k) not in K_SKIP:
                    kx, ky = sub.loc[k, 'sem_f1'], sub.loc[k, 'aspect_f1']
                    dx, dy = K_OFFSET.get(method, {}).get(k, (6, 6))
                    ax.annotate(f'k={k}', (kx, ky), textcoords='offset points', xytext=(dx, dy),
                                fontsize=KLABEL_FS, color=st['color'],
                                arrowprops=dict(arrowstyle='-', color=st['color'], lw=0.8, alpha=0.6,
                                                 shrinkA=0, shrinkB=2))
        ax.set_title(title, fontsize=SUBTITLE_FS)
        ax.set_xlabel('Semantic', fontsize=LABEL_FS)
        ax.set_ylabel('Aspect', fontsize=LABEL_FS)
        ax.tick_params(axis='both', labelsize=TICK_FS)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=LEGEND_FS,
               bbox_to_anchor=(0.5, 1.06), frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = _ProjectPath(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches='tight', dpi=200)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
