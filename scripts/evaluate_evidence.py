#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from peer.aspects import aspect_f1
from peer.metrics import semantic_prf, redundancy, aspect_noise, rouge_scores, bleu_score, aggregate_metrics
from peer.utils import ensure_dir, read_jsonl, write_jsonl


def load_embeddings(path: str | Path):
    p = Path(path)
    npy_path = p.with_suffix('.npy')
    if not npy_path.exists():
        return {}
    # Plain .npy (mmap-able) + sibling ids.json, not .npz -- see
    # peer/embeddings.py::_npy_and_ids_paths.
    emb = np.load(npy_path, mmap_mode='r')
    with open(p.with_suffix('.ids.json')) as f:
        ids = json.load(f)
    return {str(ids[i]): emb[i] for i in range(len(ids))}


def load_case_gt(cases_dir: str | Path):
    m = {}
    for split in ['train','valid','test']:
        p = Path(cases_dir) / f'cases_{split}.jsonl'
        if not p.exists():
            continue
        for c in read_jsonl(p):
            m[c['case_id']] = {
                'gt_sentence_ids': [f"{c['case_id']}_gt_{i}" for i in range(len(c['ground_truth_sentences']))],
                'gt_sentences': c['ground_truth_sentences'],
                'ground_truth_text': c['ground_truth_text'],
            }
    return m


def iter_prediction_files(pred_dir: Path):
    if pred_dir.is_file():
        yield pred_dir
    else:
        yield from sorted(pred_dir.rglob('*.jsonl'))


def eval_row(pred, embs, case_gt):
    selected_ids = pred.get('selected_sentence_ids') or []
    selected_texts = pred.get('selected_texts') or []
    selected_aspects = list(dict.fromkeys(pred.get('selected_aspects') or []))
    gt_aspects = list(dict.fromkeys(pred.get('gt_aspects') or []))
    user_aspects = list(dict.fromkeys(pred.get('user_aspects') or []))
    gt_info = case_gt.get(pred['case_id'], {})
    gt_ids = gt_info.get('gt_sentence_ids', [])
    gt_text = pred.get('ground_truth_text') or gt_info.get('ground_truth_text', '')
    e_emb = np.asarray([embs[sid] for sid in selected_ids if sid in embs], dtype=np.float32)
    g_emb = np.asarray([embs[sid] for sid in gt_ids if sid in embs], dtype=np.float32)
    if len(e_emb) and len(g_emb):
        sem_p, sem_r, sem_f1 = semantic_prf(e_emb, g_emb)
    else:
        sem_p = sem_r = sem_f1 = 0.0
    asp_p, asp_r, asp_f1 = aspect_f1(selected_aspects, gt_aspects)
    pred_text = ' '.join(selected_texts)
    rouge = rouge_scores(pred_text, gt_text) if pred_text and gt_text else {'rouge1':0.0,'rouge2':0.0,'rougeL':0.0}
    return {
        'case_id': pred['case_id'],
        'dataset': pred.get('dataset',''),
        'method': pred['method'],
        'k': pred.get('k',''),
        'sem_p': sem_p,
        'sem_r': sem_r,
        'sem_f1': sem_f1,
        'aspect_p': asp_p,
        'aspect_r': asp_r,
        'aspect_f1': asp_f1,
        'noise': aspect_noise(selected_aspects, gt_aspects, user_aspects),
        'redundancy': redundancy(e_emb) if len(e_emb) else 0.0,
        'aspect_diversity': len(set(selected_aspects)) / max(1, len(selected_texts)),
        'rouge1': rouge['rouge1'],
        'rouge2': rouge['rouge2'],
        'rougeL': rouge['rougeL'],
        'bleu': bleu_score(pred_text, gt_text) if pred_text and gt_text else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-dir', default='outputs/predictions')
    ap.add_argument('--cases', default='data/cases')
    ap.add_argument('--embeddings', default='embeddings/embeddings.npz')
    ap.add_argument('--output', default='results/evaluation.csv')
    ap.add_argument('--per-case-output', default=None)
    args = ap.parse_args()

    embs = load_embeddings(args.embeddings)
    case_gt = load_case_gt(args.cases)
    rows = []
    for f in iter_prediction_files(Path(args.pred_dir)):
        preds = read_jsonl(f)
        for p in tqdm(preds, desc=f'Eval {f.name}'):
            rows.append(eval_row(p, embs, case_gt))
    per_case = pd.DataFrame(rows)
    if args.per_case_output:
        Path(args.per_case_output).parent.mkdir(parents=True, exist_ok=True)
        per_case.to_csv(args.per_case_output, index=False)
    agg = pd.DataFrame(aggregate_metrics(rows, ['dataset','method','k']))
    ensure_dir(Path(args.output).parent)
    agg.to_csv(args.output, index=False)
    print(f'Wrote aggregate metrics -> {args.output}')
    print(agg.sort_values(['dataset','k','sem_f1'], ascending=[True, True, False]).head(20).to_string(index=False))


if __name__ == '__main__':
    main()
