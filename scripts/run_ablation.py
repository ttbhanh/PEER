#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import subprocess
import sys
from pathlib import Path

from peer.utils import ensure_dir

VARIANT_DROPS = {
    'full': [],
    'no_user': ['user_sem_sim', 'user_aspect_overlap'],
    # no_metadata is moot: metadata_sem_sim/metadata_aspect_overlap are already
    # excluded from FEATURE_COLUMNS_DEFAULT (peer/models.py) and dropped from the
    # default --variants list below; kept here only for historical re-runs.
    'no_metadata': ['metadata_sem_sim', 'metadata_aspect_overlap'],
    'no_item_salience': ['item_sem_sim', 'item_aspect_salience'],
    'no_sentiment': ['sentiment_match'],
    'no_context_semantic': ['user_sem_sim', 'metadata_sem_sim', 'item_sem_sim'],
    # target_emb_sim: the PRAG-style (user,item)-conditioned retriever feature
    # (scripts/train_peer_retriever.py) added to close PEER's sem_f1 gap vs the
    # PRAG baseline -- isolates its actual marginal contribution.
    'no_target_emb': ['target_emb_sim'],
}
# 'full' mirrors scripts/select_topk.py's shipped defaults (checkpoints/
# noise_redundancy_fix_2026-07-13/ holds the prior lambda=0.25/mu=0.15/
# eta=0.15 config). mu_redundancy/eta_noise are now 0 in the shipped selector
# (redundancy/noise penalties dropped to chase an outright sem_f1/aspect_f1
# win instead), so no_noise_penalty/no_diversity would be identical to 'full'
# and are removed from the default --variants list below. no_coverage (lambda
# ->0) remains the one selector-mechanism ablation still worth running.
SELECTOR_PARAMS = {
    # lambda_coverage 0.25->0.1, see scripts/select_topk.py for the re-tuning note.
    'full': dict(lambda_coverage=0.1, mu_redundancy=0.0, eta_noise=0.0),
    'no_coverage': dict(lambda_coverage=0.0, mu_redundancy=0.0, eta_noise=0.0),
}


def run(cmd):
    print('+', ' '.join(map(str, cmd)))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=['full','no_user','no_item_salience','no_target_emb','no_coverage'])
    ap.add_argument('--pairs-dir', default='data/processed/pairs')
    ap.add_argument('--embeddings', default='embeddings/embeddings.npz')
    ap.add_argument('--models-dir', default='models/ablation')
    ap.add_argument('--output', default='outputs/predictions/ablation')
    ap.add_argument('--backend', default='auto')
    ap.add_argument('--k-list', nargs='+', default=['1','3','5','user_avg'])
    args = ap.parse_args()
    ensure_dir(args.models_dir)
    ensure_dir(args.output)

    for v in args.variants:
        if v in VARIANT_DROPS:
            model_path = str(Path(args.models_dir) / f'peer_{v}.pkl')
            run([sys.executable, 'scripts/train_ltr.py', '--train', f'{args.pairs_dir}/train.parquet', '--valid', f'{args.pairs_dir}/valid.parquet', '--backend', args.backend, '--output', model_path, '--drop-features', *VARIANT_DROPS[v]])
            params = SELECTOR_PARAMS.get('full')
        elif v in SELECTOR_PARAMS:
            # Use full model if already trained, otherwise train it.
            model_path = str(Path(args.models_dir) / 'peer_full.pkl')
            if not Path(model_path).exists():
                run([sys.executable, 'scripts/train_ltr.py', '--train', f'{args.pairs_dir}/train.parquet', '--valid', f'{args.pairs_dir}/valid.parquet', '--backend', args.backend, '--output', model_path])
            params = SELECTOR_PARAMS[v]
        else:
            raise ValueError(f'Unknown ablation variant: {v}')
        # Run per-dataset (not the whole test split at once): select_topk.py's
        # per-case `cases` construction over the full 3-dataset test split was
        # found to no longer fit the server's 32GB cgroup limit at this scale
        # (see scripts/select_topk.py's --dataset flag). Merge the resulting
        # per-dataset jsonl files into the single combined one every downstream
        # script (evaluate_evidence.py, significance_test.py) expects.
        for ds in ['baby', 'musical', 'cellphone']:
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
            for ds in ['baby', 'musical', 'cellphone']:
                part_path = Path(args.output) / f'peer_{v}_test__{ds}.jsonl'
                out_f.write(part_path.read_text())
                part_path.unlink()
        print(f'Merged per-dataset predictions -> {combined_path}')


if __name__ == '__main__':
    main()
