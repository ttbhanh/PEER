#!/usr/bin/env python
from __future__ import annotations

"""Independent-encoder / independent-extractor robustness check (paper, Sec.
"Independent Evaluation against Circularity"). PEER's own ranker features and
composite training label share one embedding space (all-mpnet-base-v2) with
sem-F1, and one aspect extractor (hybrid spaCy noun-chunking) with aspect-F1
-- a real circularity risk. This script re-scores every method's ALREADY
SELECTED evidence (no re-training, no re-selection) with a second,
independent measurement never used anywhere in PEER's features/label/selector:

  semantic  BAAI/bge-base-en-v1.5 (different base architecture and training
            recipe from all-mpnet-base-v2, no shared checkpoint lineage) in
            place of the shared encoder, recomputing sem-F1.
  aspect    YAKE (unsupervised, statistical, no parsing, no corpus-level
            vocabulary -- the opposite design axis from the shared hybrid
            spaCy+frequency pipeline) in place of the shared extractor,
            recomputing aspect-F1.

Usage:
  python scripts/independent_check.py semantic --pred-dir outputs/predictions \\
      --cases data/cases --output results/independent_semantic.csv
  python scripts/independent_check.py aspect --pred-dir outputs/predictions \\
      --cases data/cases --output results/independent_aspect.csv

Both subcommands also write a paired-bootstrap significance table (PEER vs.
each of --baselines, per dataset and pooled, 2,000 resamples -- same protocol
as significance_test.py) to --sig-output.
"""

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from peer.aspects import aspect_f1 as aspect_f1_fn
from peer.utils import ensure_dir, read_jsonl


def iter_prediction_files(pred_dir: Path):
    yield from sorted(Path(pred_dir).rglob('*.jsonl'))


def load_case_gt_sentences(cases_dir: str) -> dict[str, dict]:
    """case_id -> {'gt_sentences': [text, ...], 'gt_ids': [sentence_id, ...]}"""
    m = {}
    for split in ['train', 'valid', 'test']:
        p = Path(cases_dir) / f'cases_{split}.jsonl'
        if not p.exists():
            continue
        for c in read_jsonl(p):
            m[c['case_id']] = {
                'gt_sentences': c['ground_truth_sentences'],
                'gt_ids': [f"{c['case_id']}_gt_{i}" for i in range(len(c['ground_truth_sentences']))],
            }
    return m


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42):
    diff = a - b
    rng = np.random.default_rng(seed)
    boots = np.array([np.mean(rng.choice(diff, size=len(diff), replace=True)) for _ in range(n_boot)])
    p = float(2 * min(np.mean(boots <= 0), np.mean(boots >= 0)))
    return float(np.mean(diff)), p


def run_significance(per_case: pd.DataFrame, metric_col: str, main_method: str, baselines: list[str], output: str):
    rows = []
    for ds in [*sorted(per_case['dataset'].unique()), 'pooled']:
        sub = per_case if ds == 'pooled' else per_case[per_case['dataset'] == ds]
        piv = sub.pivot_table(index='case_id', columns='method', values=metric_col)
        if main_method not in piv.columns:
            continue
        for base in baselines:
            if base not in piv.columns:
                continue
            m = piv[[main_method, base]].dropna()
            if len(m) == 0:
                continue
            d, p = bootstrap_diff(m[main_method].values, m[base].values)
            rows.append({'dataset': ds, 'baseline': base, 'n': len(m), 'delta': d, 'p': p})
    out = pd.DataFrame(rows)
    ensure_dir(Path(output).parent)
    out.to_csv(output, index=False)
    print(f'Wrote significance -> {output}')
    print(out.to_string(index=False))


def cmd_semantic(args):
    from sentence_transformers import SentenceTransformer

    case_gt = load_case_gt_sentences(args.cases)
    print(f'Loading independent encoder: {args.model}')
    model = SentenceTransformer(args.model)

    print('Collecting selected + ground-truth sentences from all prediction files...')
    texts: dict[str, str] = {}
    pred_rows = []
    for f in iter_prediction_files(args.pred_dir):
        for p in read_jsonl(f):
            if str(p.get('k')) != str(args.k):
                continue
            for sid, text in zip(p.get('selected_sentence_ids', []), p.get('selected_texts', [])):
                texts[sid] = text
            pred_rows.append(p)
    for gt in case_gt.values():
        for sid, text in zip(gt['gt_ids'], gt['gt_sentences']):
            texts[sid] = text
    print(f'{len(texts)} unique sentences to encode')

    ids = list(texts.keys())
    embs_arr = model.encode([texts[i] for i in ids], batch_size=args.batch_size,
                             show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    embs = {sid: embs_arr[i] for i, sid in enumerate(ids)}

    rows = []
    for p in tqdm(pred_rows, desc='Scoring sem-F1 (independent encoder)'):
        gt = case_gt.get(p['case_id'], {})
        e_emb = np.asarray([embs[sid] for sid in p.get('selected_sentence_ids', []) if sid in embs], dtype=np.float32)
        g_emb = np.asarray([embs[sid] for sid in gt.get('gt_ids', []) if sid in embs], dtype=np.float32)
        if len(e_emb) and len(g_emb):
            sims = (e_emb @ g_emb.T)
            prec = float(np.mean(np.max(sims, axis=1)))
            rec = float(np.mean(np.max(sims, axis=0)))
            f1 = 2 * prec * rec / (prec + rec + 1e-12)
        else:
            f1 = 0.0
        rows.append({'case_id': p['case_id'], 'dataset': p['dataset'], 'method': p['method'], 'sem_f1_indep': f1})

    per_case = pd.DataFrame(rows)
    ensure_dir(Path(args.output).parent)
    per_case.to_csv(args.output, index=False)
    print(f'Wrote per-case sem_f1 -> {args.output}')
    print(per_case.groupby(['dataset', 'method'])['sem_f1_indep'].mean().to_string())
    run_significance(per_case, 'sem_f1_indep', args.main_method, args.baselines, args.sig_output)


def cmd_aspect(args):
    import yake

    case_gt = load_case_gt_sentences(args.cases)
    extractor = yake.KeywordExtractor(lan='en', n=args.max_ngram, top=args.top_n, dedupLim=0.9)

    def tag(text: str) -> list[str]:
        if not text:
            return []
        return [kw for kw, _ in extractor.extract_keywords(text)]

    print('Collecting selected + ground-truth sentences from all prediction files...')
    texts: dict[str, str] = {}
    pred_rows = []
    for f in iter_prediction_files(args.pred_dir):
        for p in read_jsonl(f):
            if str(p.get('k')) != str(args.k):
                continue
            for sid, text in zip(p.get('selected_sentence_ids', []), p.get('selected_texts', [])):
                texts[sid] = text
            pred_rows.append(p)
    for gt in case_gt.values():
        for sid, text in zip(gt['gt_ids'], gt['gt_sentences']):
            texts[sid] = text
    print(f'{len(texts)} unique sentences to tag with YAKE')

    yake_tags = {sid: tag(t) for sid, t in tqdm(texts.items(), desc='YAKE tagging')}

    rows = []
    for p in tqdm(pred_rows, desc='Scoring aspect-F1 (independent extractor)'):
        gt = case_gt.get(p['case_id'], {})
        pred_aspects = [a for sid in p.get('selected_sentence_ids', []) for a in yake_tags.get(sid, [])]
        gt_aspects = [a for sid in gt.get('gt_ids', []) for a in yake_tags.get(sid, [])]
        _, _, f1 = aspect_f1_fn(pred_aspects, gt_aspects)
        rows.append({'case_id': p['case_id'], 'dataset': p['dataset'], 'method': p['method'], 'aspect_f1_yake': f1})

    per_case = pd.DataFrame(rows)
    ensure_dir(Path(args.output).parent)
    per_case.to_csv(args.output, index=False)
    print(f'Wrote per-case aspect_f1 -> {args.output}')
    print(per_case.groupby(['dataset', 'method'])['aspect_f1_yake'].mean().to_string())
    run_significance(per_case, 'aspect_f1_yake', args.main_method, args.baselines, args.sig_output)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='check', required=True)

    p_sem = sub.add_parser('semantic', help='Re-score sem-F1 with an independent encoder')
    p_sem.add_argument('--pred-dir', default='outputs/predictions')
    p_sem.add_argument('--cases', default='data/cases')
    p_sem.add_argument('--model', default='BAAI/bge-base-en-v1.5')
    p_sem.add_argument('--batch-size', type=int, default=256)
    p_sem.add_argument('--k', default='user_avg')
    p_sem.add_argument('--main-method', default='peer_full')
    p_sem.add_argument('--baselines', nargs='+', default=['prag', 'erra_r', 'a2spr', 'bm25_user', 'hrdr', 'narre', 'random'])
    p_sem.add_argument('--output', default='results/independent_semantic_per_case.csv')
    p_sem.add_argument('--sig-output', default='results/independent_semantic_significance.csv')
    p_sem.set_defaults(func=cmd_semantic)

    p_asp = sub.add_parser('aspect', help='Re-tag aspects with an independent extractor (YAKE)')
    p_asp.add_argument('--pred-dir', default='outputs/predictions')
    p_asp.add_argument('--cases', default='data/cases')
    p_asp.add_argument('--top-n', type=int, default=5, help='Keywords retained per sentence')
    p_asp.add_argument('--max-ngram', type=int, default=2, help='Max tokens per YAKE keyword')
    p_asp.add_argument('--k', default='user_avg')
    p_asp.add_argument('--main-method', default='peer_full')
    p_asp.add_argument('--baselines', nargs='+', default=['prag', 'erra_r', 'a2spr', 'bm25_user', 'hrdr', 'narre', 'random'])
    p_asp.add_argument('--output', default='results/independent_aspect_per_case.csv')
    p_asp.add_argument('--sig-output', default='results/independent_aspect_significance.csv')
    p_asp.set_defaults(func=cmd_aspect)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
