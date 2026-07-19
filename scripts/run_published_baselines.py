#!/usr/bin/env python
from __future__ import annotations

"""Task-adapted reimplementations of 5 published baselines, each following the
*actual* mechanism of its paper (see configs/baselines/manifest.json for the
citations and per-method notes on what had to be adapted for PEER's sentence-
selection task, since none of the five were designed for it):

  a2spr    Huang et al. 2020 -- cosine similarity between per-aspect sentiment
           vectors (candidate sentence vs. user history).
  erra_r   ACL 2023 -- dense retrieval against a user+metadata query embedding,
           boosted for candidates covering the (user,item) pair's top-n salient
           aspects. (The paper's explanation-generation stage does not apply.)
  prag     EMNLP 2023 -- a small trained retriever estimates a target-review
           embedding from (user, metadata) and ranks candidates by similarity to
           it; a scaled-down stand-in for the paper's transformer retriever
           (see scripts/train_prag_retriever.py). Generation stage not applicable.
  narre    WWW 2018 -- real trained CNN+attention rating-prediction network
           (scripts/train_published_neural.py); we score candidates using its
           item-side attention (context = each candidate review's author),
           i.e. the paper's own "review usefulness" signal.
  hrdr     Neurocomputing 2020 -- same training entrypoint, --model hrdr;
           attention context is the frozen user-item rating matrix instead of
           review-author identity.

This script replaces the old flat linear-formula scorers (baselines/published/
scorers.py) which is kept only for tests/backward reference.
"""

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from peer.aspects import extract_aspect_phrases, canonicalize_aspect
from peer.aspect_sentiment import AspectVector, build_aspect_sentiment_vector, cosine_sparse
from peer.selectors import resolve_k, topk
from peer.utils import cosine_vec, ensure_dir, read_jsonl, write_jsonl
from baselines.published.neural.infer import NarreScorer, HrdrScorer
from scripts.train_prag_retriever import PragRetriever

ALL_METHODS = ['narre', 'hrdr', 'a2spr', 'erra_r', 'prag']


def load_embeddings(path: str, needed_ids: set[str] | None = None) -> dict[str, np.ndarray]:
    """Load embeddings from the plain .npy + sibling ids.json pair (see
    peer/embeddings.py::_npy_and_ids_paths) -- mmap'd, so rows are paged in
    from disk as reclaimable page cache instead of fully materialized as
    anon RAM. `needed_ids` (e.g. just one dataset's cases, via --dataset)
    further limits the dict actually built and retained for the rest of the
    script's run."""
    p = Path(path)
    emb = np.load(p.with_suffix('.npy'), mmap_mode='r')
    with open(p.with_suffix('.ids.json')) as f:
        ids = json.load(f)
    if needed_ids is None:
        return {str(sid): emb[i] for i, sid in enumerate(ids)}
    return {str(sid): emb[i] for i, sid in enumerate(ids) if sid in needed_ids}


def load_aspect_map(path: str, case_ids: set[str]) -> dict[tuple[str, str, str], list[str]]:
    import pandas as pd
    # Push the case_id filter down to parquet read time (pyarrow row-group
    # pruning) instead of materializing the full aspects table (millions of
    # rows, across every case in the dataset) and filtering afterward -- the
    # full read alone was found to add several GB on top of the embeddings
    # load, enough to cross the server's 32GB cgroup limit.
    df = pd.read_parquet(path, columns=['case_id', 'kind', 'sentence_id', 'aspects'],
                          filters=[('case_id', 'in', list(case_ids))])
    m: dict[tuple[str, str, str], list[str]] = {}
    for case_id, kind, sentence_id, aspects in zip(df['case_id'], df['kind'], df['sentence_id'], df['aspects']):
        m[(case_id, kind, sentence_id)] = list(aspects) if aspects is not None else []
    return m


def load_aspect_vocab(path: str) -> set[str]:
    return {json.loads(l)['aspect'] for l in open(path, 'r', encoding='utf-8')}


def user_hist_aspects(sentence: str, vocab: set[str]) -> list[str]:
    return [canonicalize_aspect(a, vocab) for a in extract_aspect_phrases(sentence, 'hybrid', 8)]


def parse_k_list(values: list[str]) -> list[Any]:
    return ['user_avg' if str(v).lower() in {'user', 'user_avg', 'user_avg_k'} else int(v) for v in values]


# ---------------------------------------------------------------- A2SPR ----
def score_a2spr(case: dict, cand_aspects: dict[str, list[str]], user_vec: AspectVector) -> dict[str, float]:
    scores = {}
    for cand in case['candidate_sentences']:
        sid = cand['sentence_id']
        cand_vec = build_aspect_sentiment_vector([(cand['text'], cand_aspects[sid])])
        scores[sid] = cosine_sparse(cand_vec, user_vec)
    return scores


# --------------------------------------------------------------- ERRA-R ----
def score_erra_r(case: dict, cand_aspects: dict[str, list[str]], embs: dict[str, np.ndarray],
                  user_aspect_counts: Counter, top_n: int = 5, boost: float = 1.3,
                  w_user: float = 0.5, w_meta: float = 0.5) -> dict[str, float]:
    case_id = case['case_id']
    user_emb = embs.get(f'{case_id}_user_history_text')
    meta_emb = embs.get(f'{case_id}_metadata_text')
    if user_emb is not None and meta_emb is not None:
        query = w_user * user_emb + w_meta * meta_emb
    else:
        query = user_emb if user_emb is not None else meta_emb

    item_aspect_counts = Counter(a for cand in case['candidate_sentences'] for a in cand_aspects[cand['sentence_id']])
    shared = user_aspect_counts.keys() & item_aspect_counts.keys()
    salience = {a: user_aspect_counts[a] + item_aspect_counts[a] for a in shared}
    top_aspects = set(sorted(salience, key=salience.get, reverse=True)[:top_n])

    scores = {}
    for cand in case['candidate_sentences']:
        sid = cand['sentence_id']
        e = embs.get(sid)
        base = cosine_vec(e, query) if (e is not None and query is not None) else 0.0
        has_top = bool(set(cand_aspects[sid]) & top_aspects)
        scores[sid] = base * boost if has_top else base
    return scores


# ----------------------------------------------------------------- PRAG ----
def score_prag(case: dict, embs: dict[str, np.ndarray], prag_model) -> dict[str, float]:
    case_id = case['case_id']
    user_emb = embs.get(f'{case_id}_user_history_text')
    meta_emb = embs.get(f'{case_id}_metadata_text')
    if user_emb is None or meta_emb is None or prag_model is None:
        return {c['sentence_id']: 0.0 for c in case['candidate_sentences']}
    with torch.no_grad():
        est = prag_model(torch.from_numpy(user_emb).unsqueeze(0), torch.from_numpy(meta_emb).unsqueeze(0)).squeeze(0).numpy()
    scores = {}
    for cand in case['candidate_sentences']:
        e = embs.get(cand['sentence_id'])
        scores[cand['sentence_id']] = float(cosine_vec(e, est)) if e is not None else 0.0
    return scores


# ---------------------------------------------------------------- NARRE ----
def score_narre(case: dict, scorer: NarreScorer | None) -> dict[str, float]:
    cands = case['candidate_sentences']
    if scorer is None:
        return {c['sentence_id']: 0.0 for c in cands}
    review_texts: dict[str, list[str]] = {}
    review_authors: dict[str, str] = {}
    for c in cands:
        rid = c['review_id']
        review_texts.setdefault(rid, []).append(c['text'])
        review_authors[rid] = c.get('user_id', '')
    rids = list(review_texts)
    atts = scorer.score_reviews([' '.join(review_texts[r]) for r in rids], [review_authors[r] for r in rids])
    review_score = dict(zip(rids, atts))
    return {c['sentence_id']: review_score.get(c['review_id'], 0.0) for c in cands}


# ----------------------------------------------------------------- HRDR ----
def score_hrdr(case: dict, scorer: HrdrScorer | None) -> dict[str, float]:
    cands = case['candidate_sentences']
    if scorer is None:
        return {c['sentence_id']: 0.0 for c in cands}
    review_texts: dict[str, list[str]] = {}
    for c in cands:
        review_texts.setdefault(c['review_id'], []).append(c['text'])
    rids = list(review_texts)
    atts = scorer.score_reviews([' '.join(review_texts[r]) for r in rids], case['item_id'])
    review_score = dict(zip(rids, atts))
    return {c['sentence_id']: review_score.get(c['review_id'], 0.0) for c in cands}


def load_prag_model(path: str):
    if not Path(path).exists():
        return None
    with open(path, 'rb') as f:
        ck = pickle.load(f)
    model = PragRetriever(emb_dim=ck['emb_dim'], hidden=ck['hidden'])
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='data/cases')
    ap.add_argument('--split', default='test')
    ap.add_argument('--aspects', default='data/processed/aspects/sentence_aspects.parquet')
    ap.add_argument('--aspect-vocab', default='data/processed/aspects/aspect_vocab.jsonl')
    ap.add_argument('--embeddings', default='embeddings/embeddings.npz')
    ap.add_argument('--methods', nargs='+', default=ALL_METHODS)
    ap.add_argument('--k-list', nargs='+', default=['1', '3', '5', 'user_avg'])
    ap.add_argument('--models-dir', default='models')
    ap.add_argument('--output', default='outputs/predictions/published')
    ap.add_argument('--dataset', default=None,
                     help='Restrict to one dataset (e.g. baby/musical/cellphone) instead of all cases in the '
                          'split at once. Loading embeddings.npz always decompresses the whole corpus regardless '
                          '(one monolithic compressed array), but filtering cases+aspects+the retained embeddings '
                          'dict down to a single dataset cuts peak memory enough to fit the server\'s 32GB cgroup '
                          'limit; output filenames get a __{dataset} suffix so per-dataset runs can be merged '
                          '(concatenate the jsonl files) after all datasets have been scored.')
    args = ap.parse_args()

    cases = read_jsonl(f'{args.cases}/cases_{args.split}.jsonl')
    if args.dataset is not None:
        cases = [c for c in cases if c['dataset'] == args.dataset]
    case_ids = {c['case_id'] for c in cases}
    datasets = {c['dataset'] for c in cases}

    aspect_map = load_aspect_map(args.aspects, case_ids)
    vocab = load_aspect_vocab(args.aspect_vocab)
    needed_ids: set[str] = set()
    for c in cases:
        needed_ids.add(f"{c['case_id']}_user_history_text")
        needed_ids.add(f"{c['case_id']}_metadata_text")
        for cand in c['candidate_sentences']:
            needed_ids.add(cand['sentence_id'])
    embs = load_embeddings(args.embeddings, needed_ids)
    k_list = parse_k_list(args.k_list)

    narre_scorers: dict[str, NarreScorer] = {}
    hrdr_scorers: dict[str, HrdrScorer] = {}
    if 'narre' in args.methods:
        for ds in datasets:
            p = Path(args.models_dir) / f'narre_{ds}.pt'
            if p.exists():
                narre_scorers[ds] = NarreScorer(str(p))
            else:
                print(f'WARNING: no NARRE checkpoint for dataset={ds} ({p}); scores will be 0')
    if 'hrdr' in args.methods:
        for ds in datasets:
            p = Path(args.models_dir) / f'hrdr_{ds}.pt'
            if p.exists():
                hrdr_scorers[ds] = HrdrScorer(str(p))
            else:
                print(f'WARNING: no HRDR checkpoint for dataset={ds} ({p}); scores will be 0')
    prag_model = load_prag_model(str(Path(args.models_dir) / 'prag_retriever.pt')) if 'prag' in args.methods else None

    predictions: dict[str, list[dict]] = {m: [] for m in args.methods}

    for case in tqdm(cases, desc=f'published baselines ({args.split})'):
        case_id = case['case_id']
        cands = case['candidate_sentences']
        cand_aspects = {c['sentence_id']: aspect_map.get((case_id, 'candidate', c['sentence_id']), []) for c in cands}

        user_hist_pairs = [(s, user_hist_aspects(s, vocab)) for s in case['user_history_sentences']]
        user_vec = build_aspect_sentiment_vector(user_hist_pairs)
        user_aspect_counts = Counter(a for _, asp in user_hist_pairs for a in asp)
        user_aspects_list = list(dict.fromkeys(a for _, asp in user_hist_pairs for a in asp))
        gt_aspects_list = list(dict.fromkeys(
            a for gi in range(len(case['ground_truth_sentences']))
            for a in aspect_map.get((case_id, 'ground_truth', f'{case_id}_gt_{gi}'), [])
        ))

        method_scores: dict[str, dict[str, float]] = {}
        if 'a2spr' in args.methods:
            method_scores['a2spr'] = score_a2spr(case, cand_aspects, user_vec)
        if 'erra_r' in args.methods:
            method_scores['erra_r'] = score_erra_r(case, cand_aspects, embs, user_aspect_counts)
        if 'prag' in args.methods:
            method_scores['prag'] = score_prag(case, embs, prag_model)
        if 'narre' in args.methods:
            method_scores['narre'] = score_narre(case, narre_scorers.get(case['dataset']))
        if 'hrdr' in args.methods:
            method_scores['hrdr'] = score_hrdr(case, hrdr_scorers.get(case['dataset']))

        cand_by_id = {c['sentence_id']: c for c in cands}

        for method, scores in method_scores.items():
            cand_objs = [{'sentence_id': c['sentence_id'], 'text': c['text'],
                          'score': float(scores.get(c['sentence_id'], 0.0)),
                          'aspects': cand_aspects[c['sentence_id']]} for c in cands]
            for kv in k_list:
                k = resolve_k(kv, int(case.get('user_avg_k', 3)), len(cand_objs))
                selected = topk(cand_objs, k)
                predictions[method].append({
                    'case_id': case_id,
                    'dataset': case['dataset'],
                    'method': method,
                    'k': str(kv),
                    'selected_sentence_ids': [s['sentence_id'] for s in selected],
                    'selected_texts': [s['text'] for s in selected],
                    'selected_aspects': [a for s in selected for a in s['aspects']],
                    'ground_truth_text': case['ground_truth_text'],
                    'gt_aspects': gt_aspects_list,
                    'user_aspects': user_aspects_list,
                    'user_avg_k': int(case.get('user_avg_k', 3)),
                })

    out_dir = ensure_dir(args.output)
    suffix = f'__{args.dataset}' if args.dataset is not None else ''
    for method, preds in predictions.items():
        path = out_dir / f'{method}_{args.split}{suffix}.jsonl'
        write_jsonl(preds, path)
        print(f'{method}: {len(preds)} -> {path}')


if __name__ == '__main__':
    main()
