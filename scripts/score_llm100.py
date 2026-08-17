#!/usr/bin/env python
from __future__ import annotations

"""Scores the zero-shot-LLM 100-case sample (scripts/sample_llm100_cases.py +
scripts/export_llm100_blind.py + a blind LLM-selection pass) against the
leakage-free per-platform models: sem-F1/aspect-F1 for the LLM's selections,
PEER's own 4 ranker features evaluated on both PEER's and the LLM's picks,
and PEER/PRAG/ERRA restricted to the identical 100 cases for a matched
comparison, with paired-bootstrap significance of PEER vs. the LLM."""

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import json
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import torch

from peer.aspects import aspect_f1
from peer.metrics import semantic_prf
from peer.utils import cosine_vec, ensure_dir, read_jsonl
from scripts.train_peer_retriever import PeerTargetRetriever


def load_embeddings(path):
    ids = json.load(open(f'{path}.ids.json'))
    id2idx = {sid: i for i, sid in enumerate(ids)}
    arr = np.load(f'{path}.npy', mmap_mode='r')
    return id2idx, arr


def paired_bootstrap_p(diff, rng, n_boot=2000):
    n = len(diff)
    boots = np.array([np.mean(rng.choice(diff, size=n, replace=True)) for _ in range(n_boot)])
    return 2 * min((boots <= 0).mean(), (boots >= 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='results/llm100_cases.json')
    ap.add_argument('--llm-results-dir', default='results/llm100_blind_batches')
    ap.add_argument('--embeddings-template', default='embeddings/embeddings_{ds}',
                     help='{ds}-templated path stem for scripts/cache_embeddings.py output')
    ap.add_argument('--aspects-template', default='data/processed_{ds}/aspects/sentence_aspects.parquet')
    ap.add_argument('--retriever-template', default='models/peer_target_retriever_{ds}.pt')
    ap.add_argument('--pred-template', default='outputs/predictions_{ds}/peer/peer_full_test.jsonl')
    ap.add_argument('--pairs-template', default='data/processed_{ds}/pairs/test.parquet')
    ap.add_argument('--percase-template', default='results/per_case_{ds}.csv')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', default='results/llm100_results.json')
    args = ap.parse_args()

    cases = json.load(open(args.cases))
    case_by_id = {c['case_id']: c for c in cases}
    print(f'{len(cases)} sampled cases')

    llm_selected: dict[str, list[str]] = {}
    for p in sorted(_ProjectPath(args.llm_results_dir).glob('result_*.json')):
        for r in json.load(open(p)):
            llm_selected[r['case_id']] = r['selected_ids']
    missing = [c['case_id'] for c in cases if c['case_id'] not in llm_selected]
    if missing:
        raise RuntimeError(f'{len(missing)} cases missing LLM selections: {missing[:5]}...')
    print(f'{len(llm_selected)} LLM selections loaded')

    ds_list = sorted(set(c['dataset'] for c in cases))

    # per-platform: embeddings, aspects, retriever
    id2idx: dict[str, dict] = {}
    emb_arr: dict[str, np.ndarray] = {}
    asp_by_sid: dict[str, dict] = {}
    retrievers: dict[str, PeerTargetRetriever] = {}
    for ds in ds_list:
        id2idx[ds], emb_arr[ds] = load_embeddings(args.embeddings_template.format(ds=ds))
        adf = pd.read_parquet(args.aspects_template.format(ds=ds),
                               columns=['sentence_id', 'aspects'])
        adf = adf.drop_duplicates(subset='sentence_id')
        asp_by_sid[ds] = {sid: (list(a) if a is not None else []) for sid, a in zip(adf['sentence_id'], adf['aspects'])}
        ck = pickle.load(open(args.retriever_template.format(ds=ds), 'rb'))
        r = PeerTargetRetriever(emb_dim=ck['emb_dim'], hidden=ck['hidden'])
        r.load_state_dict(ck['state_dict'])
        r.eval()
        retrievers[ds] = r

    def get_emb(ds, sid):
        idx = id2idx[ds].get(sid)
        return None if idx is None else np.asarray(emb_arr[ds][idx], dtype=np.float32)

    llm_rows = []
    for c in cases:
        ds, cid = c['dataset'], c['case_id']
        sel_ids = llm_selected[cid]
        cand_by_id = {cnd['id']: cnd['text'] for cnd in c['candidates']}
        sel_texts = [cand_by_id[i] for i in sel_ids if i in cand_by_id]
        sel_aspects = [asp_by_sid[ds].get(i, []) for i in sel_ids if i in cand_by_id]

        gt_sents = c['ground_truth_sentences']
        gt_ids = [f'{cid}_gt_{i}' for i in range(len(gt_sents))]
        gt_embs = np.asarray([e for e in (get_emb(ds, g) for g in gt_ids) if e is not None], dtype=np.float32)
        gt_aspects = set()
        for g in gt_ids:
            gt_aspects.update(asp_by_sid[ds].get(g, []))

        sel_embs = np.asarray([e for e in (get_emb(ds, i) for i in sel_ids) if e is not None], dtype=np.float32)
        _, _, sem_f1 = semantic_prf(sel_embs, gt_embs) if len(sel_embs) and len(gt_embs) else (0.0, 0.0, 0.0)
        pred_aspects_flat = [a for al in sel_aspects for a in al]
        _, _, asp_f1 = aspect_f1(pred_aspects_flat, gt_aspects)

        # PEER-feature values of the LLM's own selection
        user_emb = get_emb(ds, f'{cid}_user_history_text')
        meta_emb = get_emb(ds, f'{cid}_metadata_text')
        item_embs = [e for e in (get_emb(ds, cnd['id']) for cnd in c['candidates']) if e is not None]
        item_emb = np.mean(item_embs, axis=0) if item_embs else None
        target_emb = None
        if user_emb is not None and item_emb is not None:
            with torch.no_grad():
                target_emb = retrievers[ds](
                    torch.from_numpy(user_emb).unsqueeze(0), torch.from_numpy(item_emb).unsqueeze(0)
                ).squeeze(0).numpy()
        item_aspect_cnt = Counter()
        for cnd in c['candidates']:
            item_aspect_cnt.update(asp_by_sid[ds].get(cnd['id'], []))
        max_item_count = max(item_aspect_cnt.values()) if item_aspect_cnt else 1
        user_aspects = asp_by_sid[ds].get(f'{cid}_user_history_text', [])
        if not user_aspects:
            user_aspects = asp_by_sid[ds].get(f'{cid}_user_history', [])

        user_sem_sims, target_sims, salvals, overlaps = [], [], [], []
        for i in sel_ids:
            s_emb = get_emb(ds, i)
            s_aspects = asp_by_sid[ds].get(i, [])
            user_sem_sims.append(cosine_vec(s_emb, user_emb) if s_emb is not None and user_emb is not None else 0.0)
            target_sims.append(cosine_vec(s_emb, target_emb) if s_emb is not None and target_emb is not None else 0.0)
            salvals.append(np.mean([item_aspect_cnt[a] / max_item_count for a in s_aspects]) if s_aspects else 0.0)
            sa, ua = set(s_aspects), set(user_aspects)
            overlaps.append(len(sa & ua) / max(1, len(sa)) if sa else 0.0)

        history_shares_aspect = bool(set(user_aspects) & gt_aspects)

        llm_rows.append({
            'case_id': cid, 'dataset': ds,
            'sem_f1': sem_f1, 'aspect_f1': asp_f1,
            'user_sem_sim': float(np.mean(user_sem_sims)) if user_sem_sims else 0.0,
            'target_emb_sim': float(np.mean(target_sims)) if target_sims else 0.0,
            'item_aspect_salience': float(np.mean(salvals)) if salvals else 0.0,
            'user_aspect_overlap': float(np.mean(overlaps)) if overlaps else 0.0,
            'history_shares_aspect': history_shares_aspect,
        })

    llm_df = pd.DataFrame(llm_rows)
    print('\nLLM pooled: sem_f1=%.4f aspect_f1=%.4f' % (llm_df['sem_f1'].mean(), llm_df['aspect_f1'].mean()))

    # PEER's own selections + features, restricted to the 100 cases, from existing per-platform artifacts
    case_ids_by_ds = {ds: [c['case_id'] for c in cases if c['dataset'] == ds] for ds in ds_list}
    peer_feat_rows = []
    for ds in ds_list:
        pred_path = args.pred_template.format(ds=ds)
        preds = {r['case_id']: r for r in read_jsonl(pred_path) if r.get('k') == 'user_avg'}
        pairs = pd.read_parquet(args.pairs_template.format(ds=ds),
                                 columns=['case_id', 'sentence_id', 'user_sem_sim', 'target_emb_sim',
                                          'item_aspect_salience', 'user_aspect_overlap'])
        pairs_idx = pairs.set_index(['case_id', 'sentence_id'])
        for cid in case_ids_by_ds[ds]:
            r = preds.get(cid)
            if r is None:
                continue
            sel_ids = r['selected_sentence_ids']
            vals = {'user_sem_sim': [], 'target_emb_sim': [], 'item_aspect_salience': [], 'user_aspect_overlap': []}
            for sid in sel_ids:
                try:
                    row = pairs_idx.loc[(cid, sid)]
                except KeyError:
                    continue
                for k in vals:
                    vals[k].append(float(row[k]))
            peer_feat_rows.append({'case_id': cid, 'dataset': ds,
                                    **{k: (float(np.mean(v)) if v else 0.0) for k, v in vals.items()}})
    peer_feat_df = pd.DataFrame(peer_feat_rows)

    print('PEER pooled features:', peer_feat_df[['user_sem_sim', 'target_emb_sim',
          'item_aspect_salience', 'user_aspect_overlap']].mean().to_dict())
    print('LLM  pooled features:', llm_df[['user_sem_sim', 'target_emb_sim',
          'item_aspect_salience', 'user_aspect_overlap']].mean().to_dict())

    # PEER / PRAG / ERRA sem_f1 & aspect_f1 restricted to the 100 cases, from existing percase eval files
    method_rows = []
    for ds in ds_list:
        percase = pd.read_csv(args.percase_template.format(ds=ds))
        percase = percase[percase['k'] == 'user_avg']
        for m in ['peer_full', 'prag', 'erra_r']:
            sub = percase[(percase['method'] == m) & (percase['case_id'].isin(case_ids_by_ds[ds]))]
            for _, row in sub.iterrows():
                method_rows.append({'dataset': ds, 'method': m, 'case_id': row['case_id'],
                                     'sem_f1': row['sem_f1'], 'aspect_f1': row['aspect_f1']})
    method_df = pd.DataFrame(method_rows)

    rng = np.random.default_rng(args.seed)
    llm_by_case = llm_df.set_index('case_id')
    sig = {}
    for m in ['peer_full', 'prag', 'erra_r']:
        sub = method_df[method_df['method'] == m].set_index('case_id').loc[llm_df['case_id']]
        diff_sem = sub['sem_f1'].values - llm_by_case.loc[sub.index, 'sem_f1'].values
        diff_asp = sub['aspect_f1'].values - llm_by_case.loc[sub.index, 'aspect_f1'].values
        win_sem = int((diff_sem > 0).sum())
        win_asp = int((diff_asp > 0).sum())
        p_sem = paired_bootstrap_p(diff_sem, rng)
        p_asp = paired_bootstrap_p(diff_asp, rng)
        sig[m] = {'mean_sem': float(sub['sem_f1'].mean()), 'mean_asp': float(sub['aspect_f1'].mean()),
                  'diff_sem': float(diff_sem.mean()), 'diff_asp': float(diff_asp.mean()),
                  'win_sem': win_sem, 'win_asp': win_asp, 'n': len(sub),
                  'p_sem': float(p_sem), 'p_asp': float(p_asp)}
        print(f'{m}: sem_f1={sig[m]["mean_sem"]:.4f} (+{sig[m]["diff_sem"]:+.4f}, win {win_sem}/{len(sub)}, p={p_sem:.4f})  '
              f'aspect_f1={sig[m]["mean_asp"]:.4f} (+{sig[m]["diff_asp"]:+.4f}, win {win_asp}/{len(sub)}, p={p_asp:.4f})')

    # feature-gap subset split by whether history shares an aspect with the held-out review
    subset_rows = []
    for shares in [True, False]:
        llm_sub = llm_df[llm_df['history_shares_aspect'] == shares]
        cids = llm_sub['case_id'].tolist()
        peer_sub = peer_feat_df[peer_feat_df['case_id'].isin(cids)]
        subset_rows.append({'shares_aspect': shares, 'n': len(cids),
                             'peer_target_emb_sim': float(peer_sub['target_emb_sim'].mean()),
                             'llm_target_emb_sim': float(llm_sub['target_emb_sim'].mean())})
        print(f'shares_aspect={shares} n={len(cids)}: PEER target_emb_sim={subset_rows[-1]["peer_target_emb_sim"]:.4f} '
              f'LLM target_emb_sim={subset_rows[-1]["llm_target_emb_sim"]:.4f}')

    out = {
        'llm_percase': llm_df.to_dict(orient='records'),
        'peer_features_percase': peer_feat_df.to_dict(orient='records'),
        'method_percase': method_df.to_dict(orient='records'),
        'significance': sig,
        'feature_gap_subset': subset_rows,
        'llm_pooled_sem_f1': float(llm_df['sem_f1'].mean()),
        'llm_pooled_sem_f1_std': float(llm_df['sem_f1'].std()),
        'llm_pooled_aspect_f1': float(llm_df['aspect_f1'].mean()),
        'llm_pooled_aspect_f1_std': float(llm_df['aspect_f1'].std()),
        'llm_pooled_features': llm_df[['user_sem_sim', 'target_emb_sim', 'item_aspect_salience', 'user_aspect_overlap']].mean().to_dict(),
        'peer_pooled_features': peer_feat_df[['user_sem_sim', 'target_emb_sim', 'item_aspect_salience', 'user_aspect_overlap']].mean().to_dict(),
    }
    out_path = _ProjectPath(args.output)
    ensure_dir(out_path.parent)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'\nwrote -> {out_path}')


if __name__ == '__main__':
    main()
