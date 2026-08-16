#!/usr/bin/env python
from __future__ import annotations

"""Label-weight Pareto figure (Figure "labelweight-pareto"): plots every
configuration from scripts/search_label_weights.py (run on the test split,
see README) in semantic-F1 x aspect-F1 space against PRAG's and ERRA-R's
fixed operating points from the main comparison table.
"""

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PEER_COLOR = '#0072B2'
PRAG_COLOR = '#D55E00'
ERRA_COLOR = '#009E73'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--search-results', required=True, help='CSV from scripts/search_label_weights.py (test split)')
    ap.add_argument('--shipped-config', default='sem0.7_orig_3_2_1', help='"name" of the shipped config in --search-results')
    ap.add_argument('--prag', nargs=2, type=float, metavar=('SEM_F1', 'ASPECT_F1'), required=True,
                     help='PRAG pooled sem_f1/aspect_f1 from the main comparison table')
    ap.add_argument('--erra', nargs=2, type=float, metavar=('SEM_F1', 'ASPECT_F1'), required=True,
                     help='ERRA-R pooled sem_f1/aspect_f1 from the main comparison table')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.search_results).sort_values('semantic')
    shipped = df[df['name'] == args.shipped_config].iloc[0]

    fig, ax = plt.subplots(figsize=(7.5, 6))

    ax.plot(df['sem_f1_uavg'], df['aspect_f1_uavg'], '-', color=PEER_COLOR, alpha=0.5, linewidth=1.5, zorder=2)
    ax.scatter(df['sem_f1_uavg'], df['aspect_f1_uavg'], color=PEER_COLOR, s=45, zorder=3)
    ax.scatter([shipped['sem_f1_uavg']], [shipped['aspect_f1_uavg']], facecolors='none',
               edgecolors=PEER_COLOR, s=260, linewidths=2.5, zorder=4)
    ax.annotate('PEER (shipped)', (shipped['sem_f1_uavg'], shipped['aspect_f1_uavg']),
                textcoords='offset points', xytext=(14, 0), fontsize=18, color=PEER_COLOR,
                fontweight='bold', va='center')

    ax.scatter([args.prag[0]], [args.prag[1]], color=PRAG_COLOR, marker='s', s=140, zorder=3)
    ax.annotate('PRAG', args.prag, textcoords='offset points', xytext=(12, 0), fontsize=18,
                color=PRAG_COLOR, fontweight='bold', va='center')

    ax.scatter([args.erra[0]], [args.erra[1]], color=ERRA_COLOR, marker='^', s=140, zorder=3)
    ax.annotate('ERRA-R', args.erra, textcoords='offset points', xytext=(12, 0), fontsize=18,
                color=ERRA_COLOR, fontweight='bold', va='center')

    ax.set_xlabel('Semantic', fontsize=21)
    ax.set_ylabel('Aspect', fontsize=21)
    ax.tick_params(axis='both', labelsize=17)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = _ProjectPath(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches='tight', dpi=200)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
