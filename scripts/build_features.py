#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import json
import multiprocessing as mp
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from peer.aspects import aspect_overlap
from peer.sentiment import sentiment_match
from peer.text import tokenize
from peer.utils import ensure_dir, cosine_vec
from scripts.train_peer_retriever import PeerTargetRetriever

# Explicit, fixed schema for the output pairs parquet -- every _process_case row
# dict always has exactly these keys/types, so writing batches with this schema
# (rather than letting pyarrow re-infer it per batch) guarantees every
# row-group in the incrementally-written file is consistent, even for a batch
# where e.g. every 'aspects' list happens to be empty.
PAIRS_SCHEMA = pa.schema([
    ('case_id', pa.string()), ('dataset', pa.string()), ('split', pa.string()),
    ('user_id', pa.string()), ('item_id', pa.string()), ('sentence_id', pa.string()),
    ('review_id', pa.string()), ('candidate_user_id', pa.string()), ('text', pa.string()),
    ('rating', pa.float64()), ('candidate_rating', pa.float64()),
    ('timestamp', pa.int64()), ('candidate_timestamp', pa.int64()),
    ('aspects', pa.list_(pa.string())), ('user_aspects', pa.list_(pa.string())),
    ('gt_aspects', pa.list_(pa.string())),
    ('ground_truth_text', pa.string()), ('user_avg_k', pa.int64()), ('label', pa.float64()),
    ('semantic_to_gt', pa.float64()), ('aspect_match_to_gt', pa.float64()),
    ('coverage_gain', pa.float64()), ('user_sem_sim', pa.float64()),
    ('item_sem_sim', pa.float64()), ('target_emb_sim', pa.float64()),
    ('user_aspect_overlap', pa.float64()),
    ('item_aspect_salience', pa.float64()), ('sentiment_match', pa.float64()),
    ('helpfulness_norm', pa.float64()), ('recency_norm', pa.float64()),
    ('sentence_len_norm', pa.float64()), ('aspect_count_norm', pa.float64()),
])


def iter_jsonl_batches(path: Path, batch_size: int):
    """Stream a cases_{split}.jsonl file in bounded batches instead of loading
    the whole split into memory at once -- keeps the per-batch working set
    small and bounded regardless of split size."""
    batch: list[dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
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
        if pd.isna(x):
            return []
    except Exception:
        pass
    if isinstance(x, (list, tuple, set)):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return [x]


class _AspectStore:
    """Read-only sentence_id -> aspect-list lookup backed by a small interned
    vocabulary + one flat int32 array + integer offsets (CSR layout), instead
    of a dict[tuple, list[str]] holding one Python list + several Python
    strings per sentence. Read by forked worker processes (_process_case, via
    multiprocessing.Pool below) once per candidate sentence, for every case --
    a dict-of-lists here is dense enough traffic to be the difference between
    fitting in a bounded-memory environment and OOMing.

    The original key was (case_id, kind, sentence_id); dropping case_id/kind
    down to sentence_id alone is safe: aspects are a pure function of a
    sentence's text, so every duplicate sentence_id across different cases'
    pools already carries byte-identical aspect lists."""
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
    str->int index, instead of a dict[str, np.ndarray] with one Python object
    per row -- see _AspectStore docstring for why this matters under
    multiprocessing.Pool forking."""
    __slots__ = ('matrix', 'index')

    def __init__(self, matrix: np.ndarray, index: dict[str, int]):
        self.matrix = matrix
        self.index = index

    def get(self, sid: str, default=None):
        idx = self.index.get(sid)
        return self.matrix[idx] if idx is not None else default


def load_embeddings(path: Path) -> _EmbeddingStore:
    emb = np.load(path.with_suffix('.npy'), mmap_mode='r')
    with open(path.with_suffix('.ids.json')) as f:
        ids = json.load(f)
    return _EmbeddingStore(emb, {sid: i for i, sid in enumerate(ids)})


def get_emb(embs: _EmbeddingStore, sid: str) -> np.ndarray | None:
    return embs.get(sid)


def max_sem(sent_emb: np.ndarray | None, gt_embs: list[np.ndarray]) -> float:
    if sent_emb is None or not gt_embs:
        return 0.0
    return max(cosine_vec(sent_emb, g) for g in gt_embs)


_W: dict[str, Any] = {}


def _process_case(item: tuple[str, dict]) -> list[dict[str, Any]]:
    """Build every feature row for one case. Runs in a forked worker process;
    reads the shared read-only aspect map and embedding store set up in _W by
    the parent before Pool creation (copy-on-write fork; no per-task pickling)."""
    split, c = item
    aspects = _W['aspects']; embs = _W['embs']; w = _W['weights']; retriever = _W.get('peer_retriever')
    case_id = c['case_id']
    user_aspects = aspects.get(f'{case_id}_user_history', [])
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
    item_profile_embs = [get_emb(embs, cand['sentence_id']) for cand in c['candidate_sentences']]
    item_profile_embs = [e for e in item_profile_embs if e is not None]
    item_emb = np.mean(item_profile_embs, axis=0) if item_profile_embs else None
    # PRAG-style estimate of the target review's own embedding, from (user, item)
    # context only (see scripts/train_peer_retriever.py). One forward pass per
    # case, reused below for every candidate's target_emb_sim feature.
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
            'gt_aspects': gt_aspects_unique,
            'ground_truth_text': c['ground_truth_text'],
            'user_avg_k': int(c.get('user_avg_k', 3)),
            'label': float(label),
            'semantic_to_gt': float(sem),
            'aspect_match_to_gt': float(asp_match),
            'coverage_gain': float(cov_gain),
            'user_sem_sim': cosine_vec(s_emb, user_emb) if s_emb is not None and user_emb is not None else 0.0,
            'item_sem_sim': cosine_vec(s_emb, item_emb) if s_emb is not None and item_emb is not None else 0.0,
            'target_emb_sim': cosine_vec(s_emb, target_emb) if s_emb is not None and target_emb is not None else 0.0,
            'user_aspect_overlap': aspect_overlap(s_aspects, user_aspects),
            'item_aspect_salience': float(item_sal),
            'sentiment_match': float(sent_match),
            'helpfulness_norm': float(cand.get('helpful_vote', 0.0)) / max(1.0, max_help),
            'recency_norm': (int(cand['timestamp']) - min_t) / span_t,
            'sentence_len_norm': min(1.0, len(tokenize(cand['text'])) / 80.0),
            'aspect_count_norm': min(1.0, len(set(s_aspects)) / 8.0),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='data/cases')
    ap.add_argument('--aspects', default='data/processed/aspects/sentence_aspects.parquet')
    ap.add_argument('--embeddings', default='embeddings/embeddings.npz')
    ap.add_argument('--output', default='data/processed/pairs')
    ap.add_argument('--semantic-weight', type=float, default=0.70)
    ap.add_argument('--aspect-weight', type=float, default=0.15)
    ap.add_argument('--sentiment-weight', type=float, default=0.10)
    ap.add_argument('--coverage-weight', type=float, default=0.05)
    ap.add_argument('--num-workers', type=int, default=min(os.cpu_count() or 1, 64),
                     help='Parallel worker processes for per-case feature building (fork-based)')
    ap.add_argument('--batch-size', type=int, default=3000,
                     help='Cases per batch; each batch is written to the output parquet immediately '
                          '(incremental row-groups) instead of accumulating the whole split in memory. '
                          'Lower this on tighter RAM budgets.')
    ap.add_argument('--peer-retriever', default='models/peer_target_retriever.pt',
                     help='Checkpoint from scripts/train_peer_retriever.py for the target_emb_sim feature; skipped (feature=0) if missing')
    args = ap.parse_args()

    out_dir = ensure_dir(args.output)
    _W['aspects'] = load_aspect_maps(Path(args.aspects))
    _W['embs'] = load_embeddings(Path(args.embeddings))
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
        path = Path(args.cases) / f'cases_{split}.jsonl'
        if not path.exists():
            continue
        n_cases = sum(1 for _ in open(path, 'r', encoding='utf-8') if _.strip())
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
