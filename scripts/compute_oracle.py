#!/usr/bin/env python
from __future__ import annotations

"""Oracle upper-bound analysis: for every test case, use the held-out review
directly (never as a feature or training signal for any deployed method) to
greedily select k=user_avg candidates from that case's actual candidate pool,
maximizing sem-F1 and aspect-F1 respectively. This is a (1-1/e)-approximate
greedy solution to the underlying submodular coverage objective, not an exact
combinatorial optimum, so reported values are a conservative estimate of the
true ceiling. Also reports the candidate-pool aspect recall ceiling."""

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from peer.aspects import aspect_f1
from peer.metrics import semantic_prf
from peer.utils import cosine_matrix, ensure_dir, read_jsonl, resolve_cases_path


def load_embeddings(path: str):
    ids = json.load(open(f'{path}.ids.json'))
    id2idx = {sid: i for i, sid in enumerate(ids)}
    arr = np.load(f'{path}.npy', mmap_mode='r')
    return id2idx, arr


def greedy_sem_oracle(cand_ids, cand_embs, gt_embs, k):
    """Greedily add candidates maximizing the resulting sem-F1 at each step."""
    n = len(cand_ids)
    if n == 0 or len(gt_embs) == 0:
        return [], 0.0
    sim = cosine_matrix(cand_embs, gt_embs)  # (n_cand, n_gt)
    own_max = sim.max(axis=1)  # each candidate's own best match to any GT sentence
    selected = []
    selected_mask = np.zeros(n, dtype=bool)
    running_max_per_gt = np.zeros(sim.shape[1], dtype=np.float32)
    running_own_sum = 0.0
    best_f1 = 0.0
    for _ in range(min(k, n)):
        remaining = np.where(~selected_mask)[0]
        if len(remaining) == 0:
            break
        # tentative recall if each remaining candidate were added
        tentative_max = np.maximum(running_max_per_gt[None, :], sim[remaining])  # (n_rem, n_gt)
        tentative_recall = tentative_max.mean(axis=1)
        tentative_precision = (running_own_sum + own_max[remaining]) / (len(selected) + 1)
        tentative_f1 = 2 * tentative_precision * tentative_recall / (tentative_precision + tentative_recall + 1e-12)
        best_local = int(np.argmax(tentative_f1))
        chosen = remaining[best_local]
        selected.append(chosen)
        selected_mask[chosen] = True
        running_max_per_gt = tentative_max[best_local]
        running_own_sum += own_max[chosen]
        best_f1 = float(tentative_f1[best_local])
    return [cand_ids[i] for i in selected], best_f1


def greedy_aspect_oracle(cand_ids, cand_aspects, gt_aspects_set, k):
    """Greedily add candidates maximizing the resulting aspect-F1 at each step."""
    n = len(cand_ids)
    if n == 0 or not gt_aspects_set:
        return [], 0.0
    selected = []
    selected_idx = set()
    covered = set()
    n_selected = 0
    best_f1 = 0.0
    for _ in range(min(k, n)):
        best_gain_f1 = -1.0
        best_i = None
        best_new_covered = None
        for i in range(n):
            if i in selected_idx:
                continue
            tentative_covered = covered | set(cand_aspects[i])
            p = len(tentative_covered & gt_aspects_set) / max(1, len(tentative_covered))
            # precision denominator uses total distinct selected aspects (matches aspect_f1's set-based def)
            r = len(tentative_covered & gt_aspects_set) / max(1, len(gt_aspects_set))
            f1 = 2 * p * r / (p + r + 1e-12)
            if f1 > best_gain_f1:
                best_gain_f1 = f1
                best_i = i
        if best_i is None:
            break
        selected.append(cand_ids[best_i])
        selected_idx.add(best_i)
        covered |= set(cand_aspects[best_i])
        best_f1 = best_gain_f1
    return selected, best_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', required=True)
    ap.add_argument('--aspects', required=True)
    ap.add_argument('--embeddings', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    cases = list(read_jsonl(resolve_cases_path(args.cases, args.split)))
    print(f'{len(cases)} cases')

    adf = pd.read_parquet(args.aspects, columns=['case_id', 'kind', 'sentence_id', 'aspects'])
    adf = adf.drop_duplicates(subset='sentence_id')
    asp_by_sid = {sid: (list(a) if a is not None else []) for sid, a in zip(adf['sentence_id'], adf['aspects'])}
    del adf

    id2idx, emb_arr = load_embeddings(args.embeddings)

    def get_emb(sid):
        idx = id2idx.get(sid)
        return None if idx is None else np.asarray(emb_arr[idx], dtype=np.float32)

    rows = []
    for c in tqdm(cases, desc=f'{args.dataset} oracle'):
        case_id = c['case_id']
        k = int(c.get('user_avg_k', 3))
        cand_ids = [cnd['sentence_id'] for cnd in c['candidate_sentences']]
        cand_embs_list = [get_emb(sid) for sid in cand_ids]
        valid = [i for i, e in enumerate(cand_embs_list) if e is not None]
        cand_ids_v = [cand_ids[i] for i in valid]
        cand_embs = np.asarray([cand_embs_list[i] for i in valid], dtype=np.float32) if valid else np.zeros((0, 768), dtype=np.float32)
        cand_aspects_v = [asp_by_sid.get(sid, []) for sid in cand_ids_v]

        gt_sentences = c['ground_truth_sentences']
        gt_ids = [f'{case_id}_gt_{i}' for i in range(len(gt_sentences))]
        gt_embs = [get_emb(g) for g in gt_ids]
        gt_embs = np.asarray([e for e in gt_embs if e is not None], dtype=np.float32)
        gt_aspects = set()
        for g in gt_ids:
            gt_aspects.update(asp_by_sid.get(g, []))

        pool_aspects = set()
        for a in cand_aspects_v:
            pool_aspects.update(a)
        recall_pool_aspect = len(pool_aspects & gt_aspects) / max(1, len(gt_aspects))

        _, oracle_sem_f1 = greedy_sem_oracle(cand_ids_v, cand_embs, gt_embs, k)
        _, oracle_aspect_f1 = greedy_aspect_oracle(cand_ids_v, cand_aspects_v, gt_aspects, k)

        rows.append({
            'case_id': case_id, 'dataset': args.dataset, 'k': k,
            'recall_pool_aspect': recall_pool_aspect,
            'oracle_sem_f1': oracle_sem_f1,
            'oracle_aspect_f1': oracle_aspect_f1,
        })

    out = pd.DataFrame(rows)
    ensure_dir(_ProjectPath(args.output).parent)
    out.to_csv(args.output, index=False)
    print(f'saved {len(out)} rows -> {args.output}')
    print(out[['recall_pool_aspect', 'oracle_sem_f1', 'oracle_aspect_f1']].mean())


if __name__ == '__main__':
    main()
