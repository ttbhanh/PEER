#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from baselines.published.neural.data import RatingCorpus, kcore_filter, load_temporal_safe_pool, cap_by_user, cap_entities, tokenize_records
from baselines.published.neural.narre import NARRE
from baselines.published.neural.hrdr import HRDR
from peer.utils import ensure_dir, read_jsonl


def temporal_cutoff(cases_dir: str, dataset: str) -> int:
    p = Path(cases_dir) / 'cases_test.jsonl'
    ts = [c['timestamp'] for c in read_jsonl(p) if c['dataset'] == dataset]
    if not ts:
        raise ValueError(f'No test cases found for dataset={dataset} in {p}')
    return min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['narre', 'hrdr'])
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--raw-dir', default='data/raw')
    ap.add_argument('--cases', default='data/cases')
    ap.add_argument('--user-col', default='user_id')
    ap.add_argument('--item-col', default='parent_asin',
                     help='Raw review JSONL field holding the item/business id (Amazon: parent_asin; '
                          'Yelp: business_id; Google Local: gmap_id).')
    ap.add_argument('--max-rows', type=int, default=300_000)
    ap.add_argument('--temporal-pool-cap', type=int, default=20_000_000,
                     help='Reservoir-sampling safety valve for load_temporal_safe_pool -- the raw, '
                          'pre-kcore-filter pool held in memory before capping down to --max-rows. Each '
                          'record costs roughly ~900B-1KB, so the default (20M) can need ~18-20GB of RAM '
                          'on its own; lower this (e.g. 3-5M) on RAM-constrained hosts/containers -- check '
                          'the actual cgroup memory limit, not just `free -h`, which often reports the '
                          'host machine total rather than the container quota.')
    ap.add_argument('--kcore', type=int, default=5)
    ap.add_argument('--max-entities', type=int, default=15_000,
                     help='HRDR only: cap distinct users/items to the most-reviewed N (its rating-pattern MLP scales '
                          'quadratically with entity count -- an uncapped catalog can OOM a single Linear layer). Ignored for narre.')
    ap.add_argument('--review-len', type=int, default=60)
    ap.add_argument('--review-num', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cutoff = temporal_cutoff(args.cases, args.dataset)
    print(f'[{args.dataset}] temporal cutoff (test-split min timestamp): {cutoff}')

    t0 = time.time()
    pool = load_temporal_safe_pool(args.raw_dir, args.dataset, cutoff, hard_cap=args.temporal_pool_cap, seed=args.seed,
                                    user_col=args.user_col, item_col=args.item_col)
    print(f'[{args.dataset}] temporal-safe pool: {len(pool)} reviews in {time.time()-t0:.1f}s')
    records = kcore_filter(pool, k=args.kcore)
    del pool
    print(f'[{args.dataset}] after {args.kcore}-core filter: {len(records)} reviews, '
          f'{len({r[0] for r in records})} users, {len({r[1] for r in records})} items')
    if len(records) < 200:
        raise ValueError(f'Too few reviews survive {args.kcore}-core filtering ({len(records)}); lower --kcore')
    if args.model == 'hrdr' and (len({r[0] for r in records}) > args.max_entities or len({r[1] for r in records}) > args.max_entities):
        records = cap_entities(records, max_users=args.max_entities, max_items=args.max_entities, seed=args.seed)
        records = kcore_filter(records, k=args.kcore)
        print(f'[{args.dataset}] after capping to <= {args.max_entities} users/items + re-{args.kcore}-core: '
              f'{len(records)} reviews, {len({r[0] for r in records})} users, {len({r[1] for r in records})} items')
    records = cap_by_user(records, args.max_rows, seed=args.seed)
    print(f'[{args.dataset}] after capping to ~{args.max_rows} rows by user: {len(records)} reviews, '
          f'{len({r[0] for r in records})} users, {len({r[1] for r in records})} items')
    records = tokenize_records(records, review_len=args.review_len)

    corpus = RatingCorpus(records, review_len=args.review_len, review_num=args.review_num)
    print(f'[{args.dataset}] corpus: {corpus.n_users} users, {corpus.n_items} items, vocab={corpus.vocab_size}, samples={len(corpus.samples)}')

    n = len(corpus.samples)
    idx = list(range(n))
    random.shuffle(idx)
    n_val = max(1, int(0.05 * n))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    device = torch.device(args.device)
    if args.model == 'narre':
        model = NARRE(corpus.vocab_size, corpus.n_users, corpus.n_items).to(device)
    else:
        model = HRDR(corpus.vocab_size, corpus.n_users, corpus.n_items).to(device)
        rating_matrix = torch.from_numpy(corpus.build_rating_matrix()).to(device)
        print(f'[{args.dataset}] rating matrix: {tuple(rating_matrix.shape)} ({rating_matrix.numel()*4/1e6:.1f}MB)')

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    def run_epoch(indices, train: bool):
        model.train(train)
        random.shuffle(indices) if train else None
        total_loss, total_mae, n_seen = 0.0, 0.0, 0
        for start in range(0, len(indices), args.batch_size):
            batch_idx = indices[start:start + args.batch_size]
            b = corpus.batch(batch_idx)
            input_u = torch.from_numpy(b['input_u']).to(device)
            input_i = torch.from_numpy(b['input_i']).to(device)
            uid = torch.from_numpy(b['uid']).to(device)
            iid = torch.from_numpy(b['iid']).to(device)
            rating = torch.from_numpy(b['rating']).to(device)
            if args.model == 'narre':
                input_reuid = torch.from_numpy(b['input_reuid']).to(device)
                input_reiid = torch.from_numpy(b['input_reiid']).to(device)
                pred, _, _ = model(input_u, input_reuid, input_i, input_reiid, uid, iid)
            else:
                pred = model(input_u, input_i, uid, iid, rating_matrix)
            loss = loss_fn(pred, rating)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            bs = len(batch_idx)
            total_loss += loss.item() * bs
            total_mae += (pred - rating).abs().sum().item()
            n_seen += bs
        return total_loss / max(1, n_seen), total_mae / max(1, n_seen)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_mae = run_epoch(train_idx, train=True)
        val_loss, val_mae = run_epoch(val_idx, train=False)
        print(f'[{args.dataset}][{args.model}] epoch {epoch}/{args.epochs} '
              f'train_mse={train_loss:.4f} train_mae={train_mae:.4f} '
              f'val_mse={val_loss:.4f} val_mae={val_mae:.4f} ({time.time()-t0:.1f}s)', flush=True)

    out_path = Path(args.output) if args.output else Path('models') / f'{args.model}_{args.dataset}.pt'
    ensure_dir(out_path.parent)
    checkpoint = {
        'model': args.model,
        'state_dict': model.state_dict(),
        'token2id': corpus.token2id,
        'user2idx': corpus.user2idx,
        'item2idx': corpus.item2idx,
        'review_len': args.review_len,
        'review_num': args.review_num,
        'n_users': corpus.n_users,
        'n_items': corpus.n_items,
        'vocab_size': corpus.vocab_size,
    }
    if args.model == 'hrdr':
        checkpoint['rating_matrix'] = corpus.build_rating_matrix()
    with open(out_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f'Saved -> {out_path}')


if __name__ == '__main__':
    main()
