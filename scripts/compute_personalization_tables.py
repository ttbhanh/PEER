#!/usr/bin/env python
from __future__ import annotations

"""Aggregates scripts/personalization_swap_test.py's per-(anchor, method,
swap_name) rows into the paper's personalization tables: cross-user
divergence, self-vs-real_other alignment gain, %anchors with Gain>0, mean/
median Gain, E[sem_f1 | Gain>0], and Corr(Gain, sem_f1) -- each with a
paired (2000-resample) bootstrap significance test of PEER against every
other method, per platform and pooled.
"""

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse

import numpy as np
import pandas as pd

from peer.utils import ensure_dir


def paired_p(a, b, rng, n_boot):
    diff = a - b
    n = len(diff)
    if n == 0:
        return float('nan')
    boots = np.array([np.mean(rng.choice(diff, size=n, replace=True)) for _ in range(n_boot)])
    return 2 * min((boots <= 0).mean(), (boots >= 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', nargs='+', required=True,
                     help='One or more scripts/personalization_swap_test.py output CSVs (one per platform)')
    ap.add_argument('--main-method', default='peer_full')
    ap.add_argument('--methods', nargs='+', default=['peer_full', 'prag', 'erra_r'])
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', default='results/personalization_tables.log')
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(p) for p in args.results], ignore_index=True)
    datasets = sorted(df['dataset'].unique())
    print(f'total rows: {len(df)}')

    self_df = df[df['swap_name'] == 'self'].copy()
    self_df['gain'] = self_df['gain_realother']  # paper's Gain = self vs real_other only
    print(f'self-only rows: {len(self_df)}')
    for ds in datasets:
        n = self_df[(self_df['dataset'] == ds) & (self_df['method'] == args.main_method)]['anchor_cid'].nunique()
        print(f'{ds}: N={n}')

    rng = np.random.default_rng(args.seed)
    lines = []

    def emit(s):
        print(s)
        lines.append(s)

    def scope_iter():
        for ds in datasets:
            yield ds, self_df[self_df['dataset'] == ds]
        yield 'pooled', self_df

    emit('=' * 80)
    emit('TABLE: Cross-user divergence (Div = 1 - Jaccard(S(self), S(real_other)))')
    emit('=' * 80)
    for ds, sub in scope_iter():
        emit(f'-- {ds} (N={sub[sub.method==args.main_method]["anchor_cid"].nunique()}) --')
        for m in args.methods:
            s = sub[sub['method'] == m]
            emit(f'  {m:10s}: {s["div_self_other"].mean():.4f} +/- {s["div_self_other"].std():.3f}')
    emit('\nSignificance (main vs. baseline) on Div:')
    for ds, sub in scope_iter():
        piv = sub.pivot(index='anchor_cid', columns='method', values='div_self_other')
        for m in args.methods:
            if m == args.main_method:
                continue
            p = paired_p(piv[args.main_method].values, piv[m].values, rng, args.n_boot)
            emit(f'  {ds:12s} vs {m:10s} p={p:.4f}')

    emit('\n' + '=' * 80)
    emit('TABLE: Self-other alignment gain (Gain = self_match - realother_match)')
    emit('=' * 80)
    for ds, sub in scope_iter():
        emit(f'-- {ds} --')
        for m in args.methods:
            s = sub[sub['method'] == m]
            emit(f'  {m:10s}: {s["gain"].mean():.4f} +/- {s["gain"].std():.3f}')
    emit('\nSignificance (main vs. baseline) on Gain:')
    for ds, sub in scope_iter():
        piv = sub.pivot(index='anchor_cid', columns='method', values='gain')
        for m in args.methods:
            if m == args.main_method:
                continue
            p = paired_p(piv[args.main_method].values, piv[m].values, rng, args.n_boot)
            emit(f'  {ds:12s} vs {m:10s} p={p:.4f}')

    emit('\n' + '=' * 80)
    emit('TABLE: %% anchors with Gain > 0')
    emit('=' * 80)
    for ds, sub in scope_iter():
        emit(f'-- {ds} --')
        for m in args.methods:
            s = sub[sub['method'] == m]
            emit(f'  {m:10s}: {(s["gain"] > 0).mean() * 100:.1f}%')
    emit('\nSignificance (main vs. baseline) on Gain>0 rate:')
    for ds, sub in scope_iter():
        piv = sub.pivot(index='anchor_cid', columns='method', values='gain')
        ind = (piv > 0).astype(float)
        for m in args.methods:
            if m == args.main_method:
                continue
            p = paired_p(ind[args.main_method].values, ind[m].values, rng, args.n_boot)
            emit(f'  {ds:12s} vs {m:10s} p={p:.4f}')

    emit('\n' + '=' * 80)
    emit('TABLE: mean / median Gain (unconditional, all anchors)')
    emit('=' * 80)
    for ds, sub in scope_iter():
        emit(f'-- {ds} --')
        for m in args.methods:
            s = sub[sub['method'] == m]
            emit(f'  {m:10s}: mean={s["gain"].mean():.4f}  median={s["gain"].median():.4f}')
    emit('\nSignificance (main vs. baseline) on mean Gain:')
    for ds, sub in scope_iter():
        piv = sub.pivot(index='anchor_cid', columns='method', values='gain')
        for m in args.methods:
            if m == args.main_method:
                continue
            p = paired_p(piv[args.main_method].values, piv[m].values, rng, args.n_boot)
            emit(f'  {ds:12s} vs {m:10s} p={p:.4f}')

    emit('\n' + '=' * 80)
    emit('TABLE: E[sem_f1 | Gain > 0]  (conditional outcome quality)')
    emit('=' * 80)
    for ds, sub in scope_iter():
        emit(f'-- {ds} --')
        for m in args.methods:
            s = sub[(sub['method'] == m) & (sub['gain'] > 0)]
            emit(f'  {m:10s}: n={len(s):5d}  E[sem_f1|Gain>0]={s["sem_f1"].mean():.4f}')
    emit('\nSignificance (main vs. baseline) on E[sem_f1|Gain>0] -- unpaired (different')
    emit('anchor sets since the conditioning event differs per method), bootstrap of the mean diff:')
    for ds, sub in scope_iter():
        a = sub[(sub['method'] == args.main_method) & (sub['gain'] > 0)]['sem_f1'].values
        for m in args.methods:
            if m == args.main_method:
                continue
            b = sub[(sub['method'] == m) & (sub['gain'] > 0)]['sem_f1'].values
            if len(a) == 0 or len(b) == 0:
                emit(f'  {ds:12s} vs {m}: insufficient data')
                continue
            boots = np.array([
                np.mean(rng.choice(a, size=len(a), replace=True)) - np.mean(rng.choice(b, size=len(b), replace=True))
                for _ in range(args.n_boot)
            ])
            p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
            emit(f'  {ds:12s} vs {m}: diff={a.mean()-b.mean():+.4f} p={p:.4f}')

    emit('\n' + '=' * 80)
    emit('TABLE: Corr(Gain, sem_f1)')
    emit('=' * 80)
    for ds, sub in scope_iter():
        emit(f'-- {ds} --')
        for m in args.methods:
            s = sub[sub['method'] == m]
            emit(f'  {m:10s}: corr={s["gain"].corr(s["sem_f1"]):.3f}')
    emit('\nSignificance (main vs. baseline) on Corr(Gain,sem_f1) via bootstrap of the correlation diff:')
    for ds, sub in scope_iter():
        for m in args.methods:
            if m == args.main_method:
                continue
            piv_g = sub.pivot(index='anchor_cid', columns='method', values='gain')
            piv_s = sub.pivot(index='anchor_cid', columns='method', values='sem_f1')
            n = len(piv_g)
            idx_all = np.arange(n)
            diffs = []
            for _ in range(args.n_boot):
                idx = rng.choice(idx_all, size=n, replace=True)
                c1 = np.corrcoef(piv_g[args.main_method].values[idx], piv_s[args.main_method].values[idx])[0, 1]
                c2 = np.corrcoef(piv_g[m].values[idx], piv_s[m].values[idx])[0, 1]
                diffs.append(c1 - c2)
            diffs = np.array(diffs)
            p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
            emit(f'  {ds:12s} vs {m}: p={p:.4f}')

    out = _ProjectPath(args.output)
    ensure_dir(out.parent)
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
