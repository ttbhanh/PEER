#!/usr/bin/env python
from __future__ import annotations

"""Grouped feature-removal ablation (paper, Sec. "Discussion and Limitations":
"the grouped ablation that removes item semantic similarity and item aspect
salience is the only feature-group removal that causes a clear sem-F1
decrease"). Retrains the ranker with each named feature group dropped, then
re-selects and lets evaluate_evidence.py/significance_test.py report the
resulting sem-F1/aspect-F1 so that specific comparative claim can be checked."""

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import subprocess
import sys
from pathlib import Path

from peer.utils import ensure_dir

DATASETS = ['baby', 'yelp', 'googlelocal']

VARIANT_DROPS = {
    'full': [],
    'no_user': ['user_sem_sim', 'user_aspect_overlap'],
    'no_item_salience': ['item_sem_sim', 'item_aspect_salience'],
    'no_target_emb': ['target_emb_sim'],
}
SELECTOR_PARAMS = {
    'full': dict(lambda_coverage=0.1, mu_redundancy=0.0, eta_noise=0.0),
    'no_coverage': dict(lambda_coverage=0.0, mu_redundancy=0.0, eta_noise=0.0),
}


def run(cmd):
    print('+', ' '.join(map(str, cmd)))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=['full', 'no_user', 'no_item_salience', 'no_target_emb', 'no_coverage'])
    ap.add_argument('--pairs-dir', default='data/processed/pairs')
    ap.add_argument('--embeddings', default='embeddings/embeddings.npz')
    ap.add_argument('--models-dir', default='models/ablation')
    ap.add_argument('--output', default='outputs/predictions/ablation')
    ap.add_argument('--backend', default='auto')
    ap.add_argument('--k-list', nargs='+', default=['1', '3', '5', 'user_avg'])
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    args = ap.parse_args()
    ensure_dir(args.models_dir)
    ensure_dir(args.output)

    for v in args.variants:
        if v in VARIANT_DROPS:
            model_path = str(Path(args.models_dir) / f'peer_{v}.pkl')
            run([sys.executable, 'scripts/train_ltr.py', '--train', f'{args.pairs_dir}/train.parquet',
                 '--valid', f'{args.pairs_dir}/valid.parquet', '--backend', args.backend,
                 '--output', model_path, '--drop-features', *VARIANT_DROPS[v]])
            params = SELECTOR_PARAMS['full']
        elif v in SELECTOR_PARAMS:
            model_path = str(Path(args.models_dir) / 'peer_full.pkl')
            if not Path(model_path).exists():
                run([sys.executable, 'scripts/train_ltr.py', '--train', f'{args.pairs_dir}/train.parquet',
                     '--valid', f'{args.pairs_dir}/valid.parquet', '--backend', args.backend, '--output', model_path])
            params = SELECTOR_PARAMS[v]
        else:
            raise ValueError(f'Unknown ablation variant: {v}')

        for ds in args.datasets:
            run([
                sys.executable, 'scripts/select_topk.py',
                '--pairs-dir', args.pairs_dir,
                '--model', model_path,
                '--splits', 'test',
                '--method-name', f'peer_{v}',
                '--k-list', *args.k_list,
                '--lambda-coverage', str(params['lambda_coverage']),
                '--mu-redundancy', str(params['mu_redundancy']),
                '--eta-noise', str(params['eta_noise']),
                '--embeddings', args.embeddings,
                '--output', args.output,
                '--dataset', ds,
            ])
        combined_path = Path(args.output) / f'peer_{v}_test.jsonl'
        with open(combined_path, 'w') as out_f:
            for ds in args.datasets:
                part_path = Path(args.output) / f'peer_{v}_test__{ds}.jsonl'
                out_f.write(part_path.read_text())
                part_path.unlink()
        print(f'Merged per-dataset predictions -> {combined_path}')


if __name__ == '__main__':
    main()
