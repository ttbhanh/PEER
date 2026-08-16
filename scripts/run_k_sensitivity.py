#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import subprocess
import sys


def run(cmd):
    print('+', ' '.join(map(str, cmd)))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs-dir', default='data/processed/pairs')
    ap.add_argument('--model', default='models/peer_ltr.pkl')
    ap.add_argument('--embeddings', default='embeddings/embeddings.npz')
    ap.add_argument('--output-dir', default='outputs/predictions/k_sensitivity')
    ap.add_argument('--results-output', default='results/k_sensitivity.csv')
    ap.add_argument('--k-list', nargs='+', default=['1','3','5','user_avg'])
    args = ap.parse_args()
    run([sys.executable, 'scripts/run_baselines.py', '--pairs-dir', args.pairs_dir, '--split', 'test', '--methods', 'sbert_user_metadata', 'mmr_user_metadata', '--k-list', *args.k_list, '--embeddings', args.embeddings, '--output', args.output_dir])
    run([sys.executable, 'scripts/select_topk.py', '--pairs-dir', args.pairs_dir, '--model', args.model, '--splits', 'test', '--method-name', 'peer_full', '--k-list', *args.k_list, '--embeddings', args.embeddings, '--output', args.output_dir])
    run([sys.executable, 'scripts/evaluate_evidence.py', '--pred-dir', args.output_dir, '--embeddings', args.embeddings, '--output', args.results_output])


if __name__ == '__main__':
    main()
