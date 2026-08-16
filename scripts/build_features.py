#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import multiprocessing as mp
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from peer.aspects import aspect_overlap
from peer.sentiment import sentiment_match
from peer.utils import ensure_dir, cosine_vec, count_jsonl_lines, jsonl_open, resolve_cases_path
from scripts.train_peer_retriever import PeerTargetRetriever

PAIRS_SCHEMA = pa.schema([
    ('case_id', pa.string()), ('dataset', pa.string()), ('split', pa.string()),
    ('user_id', pa.string()), ('item_id', pa.string()), ('sentence_id', pa.string()),
    ('review_id', pa.string()), ('candidate_user_id', pa.string()), ('text', pa.string()),
    ('rating', pa.float64()), ('candidate_rating', pa.float64()),
    ('timestamp', pa.int64()), ('candidate_timestamp', pa.int64()),
    ('aspects', pa.list_(pa.string())), ('user_aspects', pa.list_(pa.string())),
    ('metadata_aspects', pa.list_(pa.string())), ('gt_aspects', pa.list_(pa.string())),
    ('ground_truth_text', pa.string()), ('user_avg_k', pa.int64()), ('label', pa.float64()),
    ('semantic_to_gt', pa.float64()), ('aspect_match_to_gt', pa.float64()),
    ('coverage_gain', pa.float64()), ('user_sem_sim', pa.float64()),
    ('metadata_sem_sim', pa.float64()), ('item_sem_sim', pa.float64()),
    ('target_emb_sim', pa.float64()),
    ('user_aspect_overlap', pa.float64()),
    ('item_aspect_salience', pa.float64()), ('sentiment_match', pa.float64()),
    ('helpfulness_norm', pa.float64()), ('recency_norm', pa.float64()),
])


def iter_jsonl_batches(path: Path, batch_size: int):
    """Stream a cases_{split}.jsonl file in bounded batches instead of
    loading the whole split into memory at once."""
    batch: list[dict[str, Any]] = []
    with jsonl_open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def to_list(x):
    if x is None:
        return []
    try:
        import pandas as pd
        if pd.isna(x):
            return []
    except Exception:
        pass
    if isinstance(x, (list, tuple, set)):
        return list(x)
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    return [x]


class _AspectStore:
    """Read-only sentence_id -> aspect-list lookup backed by an interned
    vocabulary + flat int32 array + offsets (CSR layout), instead of
    dict[tuple, list[str]] -- avoids fork copy-on-write memory blowup in
    worker processes. Keyed by sentence_id alone (not case_id/kind): aspects
    are a pure function of a sentence's text, so a shared sentence_id across
    cases always carries the same aspect list."""
    __slots__ = ('index', 'aspect_ids', 'offsets', 'vocab')

    def __init__(self, index: dict[str, int], aspect_ids: np.ndarray, offsets: np.ndarray, vocab: list[str]):
        self.index = index
        self.aspect_ids = aspect_ids
        self.offsets = offsets
        self.vocab = vocab

    def get(self, sentence_id: str, default=None):
        idx = self.index.get(sentence_id)
        if idx is None:
            return default
        vocab = self.vocab
        return [vocab[i] for i in self.aspect_ids[self.offsets[idx]:self.offsets[idx + 1]]]


def load_aspect_maps(path: Path) -> _AspectStore:
    df = pd.read_parquet(path, columns=['sentence_id', 'aspects'])
    index: dict[str, int] = {}
    vocab_map: dict[str, int] = {}
    vocab: list[str] = []
    aspect_ids_list: list[int] = []
    offsets_list: list[int] = [0]
    for sentence_id, aspects_val in zip(df['sentence_id'], df['aspects']):
        if sentence_id in index:
            continue
        index[sentence_id] = len(offsets_list) - 1
        for a in to_list(aspects_val):
            vi = vocab_map.get(a)
            if vi is None:
                vi = len(vocab)
                vocab_map[a] = vi
                vocab.append(a)
            aspect_ids_list.append(vi)
        offsets_list.append(len(aspect_ids_list))
    return _AspectStore(
        index,
        np.asarray(aspect_ids_list, dtype=np.int32),
        np.asarray(offsets_list, dtype=np.int64),
        vocab,
    )


class _EmbeddingStore:
    """Read-only embedding lookup backed by one contiguous 2D array + a
    str->int index, instead of dict[str, np.ndarray] -- avoids fork
    copy-on-write memory blowup in worker processes."""
    __slots__ = ('matrix', 'index', 'override')

    def __init__(self, matrix: np.ndarray, index: dict[str, int], override: dict[str, np.ndarray] | None = None):
        self.matrix = matrix
        self.index = index
        self.override = override or {}

    def get(self, sid: str, default=None):
        if sid in self.override:
            return self.override[sid]
        idx = self.index.get(sid)
        return self.matrix[idx] if idx is not None else default


def load_embeddings(path: Path, user_history_override: Path | None = None) -> _EmbeddingStore:
    emb = np.load(path.with_suffix('.npy'), mmap_mode='r')  # mmap'd, not .npz
    with open(path.with_suffix('.ids.json')) as f:
        ids = json.load(f)
    override: dict[str, np.ndarray] = {}
    if user_history_override is not None:
        o_ids = json.load(open(user_history_override.with_suffix('.ids.json')))
        o_vecs = np.load(user_history_override.with_suffix('.npy'))
        override = {sid: np.asarray(vec, dtype=np.float32) for sid, vec in zip(o_ids, o_vecs)}
    return _EmbeddingStore(emb, {sid: i for i, sid in enumerate(ids)}, override)


def get_emb(embs: _EmbeddingStore, sid: str) -> np.ndarray | None:
    return embs.get(sid)


def max_sem(sent_emb: np.ndarray | None, gt_embs: list[np.ndarray]) -> float:
    if sent_emb is None or not gt_embs:
        return 0.0
    return max(cosine_vec(sent_emb, g) for g in gt_embs)


_W: dict[str, Any] = {}


def _process_case(item: tuple[str, dict]) -> list[dict[str, Any]]:
    """Build every feature row for one case. Runs in a forked worker process,
    reading the shared aspect map / embeddings set up in _W by the parent."""
    split, c = item
    aspects = _W['aspects']; embs = _W['embs']; w = _W['weights']; retriever = _W.get('peer_retriever')
    case_id = c['case_id']
    user_aspects = aspects.get(f'{case_id}_user_history', [])
    metadata_aspects = aspects.get(f'{case_id}_metadata', [])
    if not user_aspects:
        user_aspects = aspects.get(f'{case_id}_user_history_text', [])
    gt_aspects = []
    gt_embs = []
    gt_sentences = c['ground_truth_sentences']
    for gi, _ in enumerate(gt_sentences):
        sid = f'{case_id}_gt_{gi}'
        gt_aspects.extend(aspects.get(sid, []))
        e = get_emb(embs, sid)
        if e is not None:
            gt_embs.append(e)
    gt_aspects_unique = list(dict.fromkeys(gt_aspects))
    user_emb = get_emb(embs, f'{case_id}_user_history_text')
    meta_emb = get_emb(embs, f'{case_id}_metadata_text')
    item_profile_embs = [get_emb(embs, cand['sentence_id']) for cand in c['candidate_sentences']]
    item_profile_embs = [e for e in item_profile_embs if e is not None]
    item_emb = np.mean(item_profile_embs, axis=0) if item_profile_embs else None
    target_emb = None
    if retriever is not None and user_emb is not None and item_emb is not None:
        with torch.no_grad():
            target_emb = retriever(
                torch.from_numpy(user_emb).unsqueeze(0), torch.from_numpy(item_emb).unsqueeze(0)
            ).squeeze(0).numpy()
    item_aspect_cnt = Counter()
    for cand in c['candidate_sentences']:
        item_aspect_cnt.update(aspects.get(cand['sentence_id'], []))
    max_item_count = max(item_aspect_cnt.values()) if item_aspect_cnt else 1
    max_help = max([float(x.get('helpful_vote', 0.0)) for x in c['candidate_sentences']] + [1.0])
    t = int(c['timestamp'])
    min_t = min([int(x['timestamp']) for x in c['candidate_sentences']] + [t])
    span_t = max(1, t - min_t)

    rows: list[dict[str, Any]] = []
    covered_for_label = set()
    for cand in c['candidate_sentences']:
        sid = cand['sentence_id']
        s_aspects = aspects.get(sid, [])
        s_emb = get_emb(embs, sid)
        sem = max_sem(s_emb, gt_embs)
        asp_match = len(set(s_aspects) & set(gt_aspects_unique)) / max(1, len(set(s_aspects))) if s_aspects else 0.0
        sent_match = max([sentiment_match(cand['text'], gt) for gt in gt_sentences] + [0.5])
        new_gt = set(s_aspects) & set(gt_aspects_unique) - covered_for_label
        cov_gain = len(new_gt) / max(1, len(set(gt_aspects_unique)))
        covered_for_label.update(new_gt)
        label = w['semantic'] * sem + w['aspect'] * asp_match + w['sentiment'] * sent_match + w['coverage'] * cov_gain
        item_sal = np.mean([item_aspect_cnt[a] / max_item_count for a in s_aspects]) if s_aspects else 0.0
        rows.append({
            'case_id': case_id,
            'dataset': c['dataset'],
            'split': split,
            'user_id': c['user_id'],
            'item_id': c['item_id'],
            'sentence_id': sid,
            'review_id': cand.get('review_id', sid),
            'candidate_user_id': cand.get('user_id', ''),
            'text': cand['text'],
            'rating': float(c.get('rating', 0.0)),
            'candidate_rating': float(cand.get('rating', 0.0)),
            'timestamp': int(c['timestamp']),
            'candidate_timestamp': int(cand['timestamp']),
            'aspects': s_aspects,
            'user_aspects': user_aspects,
            'metadata_aspects': metadata_aspects,
            'gt_aspects': gt_aspects_unique,
            'ground_truth_text': c['ground_truth_text'],
            'user_avg_k': int(c.get('user_avg_k', 3)),
            'label': float(label),
            'semantic_to_gt': float(sem),
            'aspect_match_to_gt': float(asp_match),
            'coverage_gain': float(cov_gain),
            'user_sem_sim': cosine_vec(s_emb, user_emb) if s_emb is not None and user_emb is not None else 0.0,
            'metadata_sem_sim': cosine_vec(s_emb, meta_emb) if s_emb is not None and meta_emb is not None else 0.0,
            'item_sem_sim': cosine_vec(s_emb, item_emb) if s_emb is not None and item_emb is not None else 0.0,
            'target_emb_sim': cosine_vec(s_emb, target_emb) if s_emb is not None and target_emb is not None else 0.0,
            'user_aspect_overlap': aspect_overlap(s_aspects, user_aspects),
            'item_aspect_salience': float(item_sal),
            'sentiment_match': float(sent_match),
            'helpfulness_norm': float(cand.get('helpful_vote', 0.0)) / max(1.0, max_help),
            'recency_norm': (int(cand['timestamp']) - min_t) / span_t,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='data/cases')
    ap.add_argument('--aspects', default='data/processed/aspects/sentence_aspects.parquet')
    ap.add_argument('--embeddings', default='embeddings/embeddings.npz')
    ap.add_argument('--user-history-override', default=None,
                     help='Path stem from scripts/cache_user_history_embeddings.py; merged on top of --embeddings for user_sem_sim/target_emb_sim only')
    ap.add_argument('--output', default='data/processed/pairs')
    ap.add_argument('--semantic-weight', type=float, default=0.4)
    ap.add_argument('--aspect-weight', type=float, default=0.3)
    ap.add_argument('--sentiment-weight', type=float, default=0.2)
    ap.add_argument('--coverage-weight', type=float, default=0.1)
    ap.add_argument('--num-workers', type=int, default=min(os.cpu_count() or 1, 64),
                     help='Parallel worker processes for per-case feature building (fork-based)')
    ap.add_argument('--batch-size', type=int, default=3000,
                     help='Cases per batch, written to the output parquet incrementally instead of accumulating '
                          'the whole split in memory. Lower this on tighter RAM budgets.')
    ap.add_argument('--peer-retriever', default='models/peer_target_retriever.pt',
                     help='Checkpoint from scripts/train_peer_retriever.py for the target_emb_sim feature; skipped (feature=0) if missing')
    args = ap.parse_args()

    out_dir = ensure_dir(args.output)
    _W['aspects'] = load_aspect_maps(Path(args.aspects))
    override_path = Path(args.user_history_override) if args.user_history_override else None
    _W['embs'] = load_embeddings(Path(args.embeddings), override_path)
    retriever_path = Path(args.peer_retriever)
    if retriever_path.exists():
        with open(retriever_path, 'rb') as f:
            ck = pickle.load(f)
        retriever = PeerTargetRetriever(emb_dim=ck['emb_dim'], hidden=ck['hidden'])
        retriever.load_state_dict(ck['state_dict'])
        retriever.eval()
        _W['peer_retriever'] = retriever
        print(f'Loaded target-embedding retriever from {retriever_path} (val_cosdist={ck["val_cosdist"]:.4f})')
    else:
        _W['peer_retriever'] = None
        print(f'WARNING: no retriever checkpoint at {retriever_path}; target_emb_sim will be 0 for all rows')
    _W['weights'] = {
        'semantic': args.semantic_weight,
        'aspect': args.aspect_weight,
        'sentiment': args.sentiment_weight,
        'coverage': args.coverage_weight,
    }
    n_workers = max(1, args.num_workers)
    batch_size = max(1, args.batch_size)

    for split in ['train', 'valid', 'test']:
        path = resolve_cases_path(args.cases, split)
        if path is None:
            continue
        n_cases = count_jsonl_lines(path)
        out_path = out_dir / f'{split}.parquet'
        writer: pq.ParquetWriter | None = None
        total_rows = 0
        ctx = mp.get_context('fork')
        pool = ctx.Pool(processes=n_workers) if n_workers > 1 else None
        try:
            with tqdm(total=n_cases, desc=f'Build features {split} ({n_workers}w, batch={batch_size})') as pbar:
                for batch in iter_jsonl_batches(path, batch_size):
                    items = [(split, c) for c in batch]
                    rows: list[dict[str, Any]] = []
                    if pool is None:
                        for it in items:
                            rows.extend(_process_case(it))
                            pbar.update(1)
                    else:
                        chunksize = max(1, min(200, len(items) // (n_workers * 8) or 1))
                        for result in pool.imap(_process_case, items, chunksize=chunksize):
                            rows.extend(result)
                            pbar.update(1)
                    if not rows:
                        continue
                    table = pa.Table.from_pandas(pd.DataFrame(rows), schema=PAIRS_SCHEMA, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(out_path, PAIRS_SCHEMA)
                    writer.write_table(table)
                    total_rows += len(rows)
                    del rows, table, items
        finally:
            if pool is not None:
                pool.close()
                pool.join()
            if writer is not None:
                writer.close()
        print(f'Wrote {total_rows} rows -> {out_path}')


if __name__ == '__main__':
    main()
