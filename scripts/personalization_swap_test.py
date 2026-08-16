#!/usr/bin/env python
from __future__ import annotations

"""Controlled user-swap personalization test: for every test item with >=2
users, fix the item/candidates/timestamp/budget and re-run selection with
each of several histories substituted in turn (self, real second reviewer,
--n-foreign unrelated), so any difference in evidence is attributable only
to whose history was used. Produces one row per (anchor, method, swap_name)
for scripts/compute_personalization_tables.py."""

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import json
import pickle
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch

from peer.models import RankerModel
from peer.selectors import greedy_coverage_select, topk
from peer.metrics import semantic_prf
from peer.utils import ensure_dir, read_jsonl, resolve_cases_path
from scripts.train_peer_retriever import PeerTargetRetriever
from scripts.train_prag_retriever import PragRetriever


def cosine(a, b):
    if a is None or b is None:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def aspect_overlap(a, b):
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa))


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def rank_weighted_counts(aspects_ranked):
    n = len(aspects_ranked)
    return {a: (1.0 - i / max(1, n)) for i, a in enumerate(aspects_ranked)}


def sem_f1_of(selected, gt_embs):
    if not selected or len(gt_embs) == 0:
        return 0.0
    e_emb = np.asarray([s['embedding'] for s in selected], dtype=np.float32)
    _, _, f1 = semantic_prf(e_emb, gt_embs)
    return f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--cases', default='data/cases')
    ap.add_argument('--pairs', default='data/processed/pairs/test.parquet')
    ap.add_argument('--aspects', default='data/processed/aspects/sentence_aspects.parquet')
    ap.add_argument('--embeddings', default='embeddings/embeddings')
    ap.add_argument('--user-history-override', default='models/user_history_embeddings')
    ap.add_argument('--peer-ranker', default='models/peer_ltr.pkl')
    ap.add_argument('--peer-retriever', default='models/peer_target_retriever.pt')
    ap.add_argument('--prag-retriever', default='models/prag_retriever.pt')
    ap.add_argument('--lambda-coverage', type=float, default=0.1)
    ap.add_argument('--n-foreign', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', default='results/personalization_swap_test.csv')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print('loading ranker + retrievers...', flush=True)
    ranker = RankerModel.load(args.peer_ranker)
    ranker.model.set_params(n_jobs=1)

    with open(args.peer_retriever, 'rb') as f:
        ck = pickle.load(f)
    peer_retriever = PeerTargetRetriever(emb_dim=ck['emb_dim'], hidden=ck['hidden'])
    peer_retriever.load_state_dict(ck['state_dict'])
    peer_retriever.eval()

    with open(args.prag_retriever, 'rb') as f:
        ck2 = pickle.load(f)
    prag_retriever = PragRetriever(emb_dim=ck2['emb_dim'], hidden=ck2['hidden'])
    prag_retriever.load_state_dict(ck2['state_dict'])
    prag_retriever.eval()
    print('models loaded', flush=True)

    override_ids = json.load(open(f'{args.user_history_override}.ids.json'))
    override_vecs = np.load(f'{args.user_history_override}.npy')
    user_emb_by_case = {sid[:-len('_user_history_text')]: np.asarray(v, dtype=np.float32)
                         for sid, v in zip(override_ids, override_vecs)}
    print(f'user-history embeddings: {len(user_emb_by_case)}', flush=True)

    cases = read_jsonl(resolve_cases_path(args.cases, 'test'))
    case_by_id = {c['case_id']: c for c in cases}
    by_item = defaultdict(list)
    for c in cases:
        by_item[c['item_id']].append(c)

    anchors = []
    all_case_ids = [c['case_id'] for c in cases]
    for item_id, group in by_item.items():
        if len(group) < 2:
            continue
        group_sorted = sorted(group, key=lambda c: int(c['timestamp']))
        anchor_c, other_c = group_sorted[0], group_sorted[1]
        foreigns = []
        tries = 0
        while len(foreigns) < args.n_foreign and tries < 50:
            cand = all_case_ids[rng.integers(0, len(all_case_ids))]
            tries += 1
            if cand != anchor_c['case_id'] and cand != other_c['case_id'] and cand not in foreigns:
                foreigns.append(cand)
        anchors.append({'anchor_cid': anchor_c['case_id'], 'other_cid': other_c['case_id'], 'foreigns': foreigns})
    print(f'{len(anchors)} candidate anchors (before availability filtering)', flush=True)

    ids_json = json.load(open(f'{args.embeddings}.ids.json'))
    id2idx = {sid: i for i, sid in enumerate(ids_json)}
    emb_arr = np.load(f'{args.embeddings}.npy', mmap_mode='r')

    def get_emb(sid):
        idx = id2idx.get(sid)
        return None if idx is None else np.asarray(emb_arr[idx], dtype=np.float32)

    pairs = pd.read_parquet(args.pairs, columns=['case_id', 'sentence_id', 'text', 'aspects',
                                                  'item_sem_sim', 'item_aspect_salience'])
    pairs_by_case = {cid: g.reset_index(drop=True) for cid, g in pairs.groupby('case_id')}
    del pairs
    print('pairs grouped by case', flush=True)

    adf = pd.read_parquet(args.aspects, columns=['case_id', 'kind', 'sentence_id', 'aspects'],
                           filters=[('kind', '=', 'user_history')])
    user_aspect_map = {cid: (list(a) if a is not None else []) for cid, a in zip(adf['case_id'], adf['aspects'])}
    del adf
    print('user_aspect_map size:', len(user_aspect_map), flush=True)

    def get_user_emb(h_cid):
        return user_emb_by_case.get(h_cid)

    def select_peer(t_emb, k, cand_embs, group, h_cid):
        user_emb = get_user_emb(h_cid)
        user_aspects = user_aspect_map.get(h_cid, [])
        user_sem_sim = [cosine(ce, user_emb) for ce in cand_embs]
        target_emb_sim = [cosine(ce, t_emb) for ce in cand_embs]
        user_aspect_overlap_col = [aspect_overlap(list(a) if a is not None else [], user_aspects) for a in group['aspects']]
        g2 = group.assign(user_sem_sim=user_sem_sim, target_emb_sim=target_emb_sim, user_aspect_overlap=user_aspect_overlap_col)
        scores = ranker.predict(g2)
        cands = [{
            'sentence_id': g2.iloc[i]['sentence_id'], 'text': g2.iloc[i]['text'],
            'score': float(scores[i]), 'aspects': list(g2.iloc[i]['aspects']) if g2.iloc[i]['aspects'] is not None else [],
            'embedding': cand_embs[i] if cand_embs[i] is not None else np.zeros(768, dtype=np.float32),
        } for i in range(len(g2))]
        return greedy_coverage_select(cands, k, lambda_coverage=args.lambda_coverage)

    def select_prag(est, k, cand_embs, group):
        cands = [{
            'sentence_id': group.iloc[i]['sentence_id'], 'text': group.iloc[i]['text'],
            'score': cosine(cand_embs[i], est),
            'aspects': list(group.iloc[i]['aspects']) if group.iloc[i]['aspects'] is not None else [],
            'embedding': cand_embs[i] if cand_embs[i] is not None else np.zeros(768, dtype=np.float32),
        } for i in range(len(group))]
        return topk(cands, k)

    def select_erra_r(h_cid, k, cand_embs, group, item_aspect_cnt, meta_emb, top_n=5, boost=1.3, w_user=0.5, w_meta=0.5):
        user_emb = get_user_emb(h_cid)
        if user_emb is not None and meta_emb is not None:
            query = w_user * user_emb + w_meta * meta_emb
        elif user_emb is not None:
            query = user_emb
        else:
            query = meta_emb
        user_aspect_counts = rank_weighted_counts(user_aspect_map.get(h_cid, []))
        shared = user_aspect_counts.keys() & item_aspect_cnt.keys()
        salience = {a: user_aspect_counts[a] + item_aspect_cnt[a] for a in shared}
        top_aspects = set(sorted(salience, key=salience.get, reverse=True)[:top_n])
        cands = []
        for i in range(len(group)):
            row = group.iloc[i]
            aspects = list(row['aspects']) if row['aspects'] is not None else []
            base = cosine(cand_embs[i], query)
            has_top = bool(set(aspects) & top_aspects)
            score = base * boost if has_top else base
            cands.append({
                'sentence_id': row['sentence_id'], 'text': row['text'], 'score': score,
                'aspects': aspects, 'embedding': cand_embs[i] if cand_embs[i] is not None else np.zeros(768, dtype=np.float32),
            })
        return topk(cands, k)

    results = []
    t0 = time.time()
    n_ok = 0

    for ai, anc in enumerate(anchors):
        anchor_cid = anc['anchor_cid']
        anchor_case = case_by_id[anchor_cid]
        group = pairs_by_case.get(anchor_cid)
        if group is None or len(group) == 0:
            continue
        cand_ids = group['sentence_id'].tolist()
        cand_embs = [get_emb(sid) for sid in cand_ids]
        item_emb_vecs = [e for e in cand_embs if e is not None]
        if not item_emb_vecs:
            continue
        item_emb = np.mean(item_emb_vecs, axis=0)
        k_fixed = int(anchor_case.get('user_avg_k', 3))
        meta_emb = get_emb(f'{anchor_cid}_metadata_text')

        gt_sents = anchor_case.get('ground_truth_sentences', [])
        gt_ids = [f'{anchor_cid}_gt_{i}' for i in range(len(gt_sents))]
        gt_embs = [get_emb(g) for g in gt_ids]
        gt_embs = np.asarray([e for e in gt_embs if e is not None], dtype=np.float32)

        item_aspect_cnt = Counter()
        for row_aspects in group['aspects']:
            item_aspect_cnt.update(list(row_aspects) if row_aspects is not None else [])

        swap_sources = {'self': anchor_cid, 'real_other': anc['other_cid']}
        for i, f in enumerate(anc['foreigns']):
            swap_sources[f'foreign_{i}'] = f

        if not all(h_cid in user_emb_by_case for h_cid in swap_sources.values()):
            continue
        if item_emb is None or len(gt_embs) == 0:
            continue
        n_ok += 1

        hist_aspects = {name: set(user_aspect_map.get(h_cid, [])) for name, h_cid in swap_sources.items()}

        for method in ['peer_full', 'prag', 'erra_r']:
            sels, sel_aspects_by_name, sem_f1_by_name = {}, {}, {}
            for name, h_cid in swap_sources.items():
                user_emb = get_user_emb(h_cid)
                if method == 'peer_full':
                    with torch.no_grad():
                        t_emb = peer_retriever(
                            torch.from_numpy(user_emb).unsqueeze(0), torch.from_numpy(item_emb).unsqueeze(0)
                        ).squeeze(0).numpy()
                    selected = select_peer(t_emb, k_fixed, cand_embs, group, h_cid)
                elif method == 'prag':
                    if meta_emb is None:
                        selected = []
                    else:
                        with torch.no_grad():
                            est = prag_retriever(
                                torch.from_numpy(user_emb).unsqueeze(0), torch.from_numpy(meta_emb).unsqueeze(0)
                            ).squeeze(0).numpy()
                        selected = select_prag(est, k_fixed, cand_embs, group)
                else:
                    selected = select_erra_r(h_cid, k_fixed, cand_embs, group, item_aspect_cnt, meta_emb)

                sels[name] = {s['sentence_id'] for s in selected}
                sel_aspects_by_name[name] = [a for s in selected for a in s['aspects']]
                sem_f1_by_name[name] = sem_f1_of(selected, gt_embs)

            names = list(sels.keys())
            div_self_other = 1 - jaccard(sels['self'], sels['real_other'])

            for name in names:
                self_j = jaccard(set(sel_aspects_by_name[name]), hist_aspects[name])
                others = [jaccard(set(sel_aspects_by_name[name]), hist_aspects[n2]) for n2 in names if n2 != name]
                other_j = float(np.mean(others)) if others else 0.0
                realother_j = jaccard(set(sel_aspects_by_name[name]), hist_aspects['real_other'])
                results.append({
                    'dataset': args.dataset, 'anchor_cid': anchor_cid, 'method': method, 'swap_name': name,
                    'div_self_other': div_self_other if name == 'self' else np.nan,
                    'self_match': self_j, 'other_match': other_j, 'gain': self_j - other_j,
                    'realother_match': realother_j, 'gain_realother': self_j - realother_j,
                    'sem_f1': sem_f1_by_name[name],
                })

        if ai % 100 == 0:
            print(ai, f'{time.time()-t0:.1f}s ok={n_ok}', flush=True)

    print(f'total time {time.time()-t0:.1f}s, usable anchors {n_ok}', flush=True)
    df = pd.DataFrame(results)
    ensure_dir(_ProjectPath(args.output).parent)
    df.to_csv(args.output, index=False)
    print(f'saved {len(df)} rows -> {args.output}', flush=True)


if __name__ == '__main__':
    main()
