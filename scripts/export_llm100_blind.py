#!/usr/bin/env python
from __future__ import annotations

"""Splits the saved 100-case sample (scripts/sample_llm100_cases.py output)
into N batch files containing ONLY what a blind zero-shot selection task may
see: case_id, k, metadata_text, user_history_text, candidates. Ground truth
is deliberately omitted so these files are safe to hand to a fresh
LLM-selection agent."""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='results/llm100_cases.json')
    ap.add_argument('--batch-size', type=int, default=10)
    ap.add_argument('--out-dir', default='results/llm100_blind_batches')
    args = ap.parse_args()

    cases = json.load(open(args.cases))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_batches = 0
    for i in range(0, len(cases), args.batch_size):
        batch = cases[i:i + args.batch_size]
        blind = [{
            'case_id': c['case_id'],
            'k': c['k'],
            'metadata_text': c['metadata_text'],
            'user_history_text': c['user_history_text'],
            'candidates': c['candidates'],
        } for c in batch]
        out_path = out_dir / f'batch_{n_batches:02d}.json'
        with open(out_path, 'w') as f:
            json.dump(blind, f, indent=1)
        print(f'{out_path}: {len(blind)} cases')
        n_batches += 1
    print(f'wrote {n_batches} batch files -> {out_dir}')


if __name__ == '__main__':
    main()
