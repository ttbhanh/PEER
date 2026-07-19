#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import pandas as pd
from peer.utils import ensure_dir


def md_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_markdown(index=False, floatfmt='.4f')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--output', default='paper_ready_summary.md')
    args = ap.parse_args()
    rdir = Path(args.results_dir)
    sections = ['# PEER Experimental Summary\n']
    for name in ['dataset_stats.csv','evaluation.csv','ablation.csv','significance.csv','k_sensitivity.csv']:
        p = rdir / name
        if p.exists():
            df = pd.read_csv(p)
            sections.append(f'\n## {name}\n')
            sections.append(md_table(df))
            sections.append('\n')
    out = Path(args.output)
    ensure_dir(out.parent if out.parent != Path('.') else '.')
    out.write_text('\n'.join(sections), encoding='utf-8')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
