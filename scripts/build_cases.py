#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import gc
import gzip
import json
import multiprocessing as mp
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from peer.text import split_sentences, clean_text, compact_profile
from peer.utils import ensure_dir, normalize_timestamp, write_jsonl, set_seed


def metadata_to_text(row: dict[str, Any]) -> str:
    parts = []
    for k in ['title', 'main_category', 'store', 'brand']:
        if row.get(k):
            parts.append(str(row[k]))
    for k in ['categories', 'features', 'description']:
        v = row.get(k)
        if isinstance(v, list):
            parts.extend(str(x) for x in v[:20])
        elif isinstance(v, dict):
            parts.extend(str(x) for x in list(v.values())[:20])
        elif v:
            parts.append(str(v))
    return clean_text(' '.join(parts))


def infer_file(raw_dir: Path, dataset: str, kind: str) -> Path:
    patterns = [
        f'{dataset}_{kind}.jsonl', f'{dataset}-{kind}.jsonl', f'{dataset}.{kind}.jsonl',
        f'{kind}_{dataset}.jsonl', f'{kind}-{dataset}.jsonl', f'{dataset}_{kind}.json',
    ]
    for p in patterns:
        path = raw_dir / p
        if path.exists():
            return path
    if kind == 'reviews':
        matches = list(raw_dir.glob(f'*{dataset}*review*.jsonl')) + list(raw_dir.glob(f'*{dataset}*.jsonl'))
    else:
        matches = list(raw_dir.glob(f'*{dataset}*meta*.jsonl'))
    if matches:
        return matches[0]
    raise FileNotFoundError(f'Cannot infer {kind} file for dataset={dataset} in {raw_dir}')


def _code_of(s: str, code_map: dict[str, int], lookup: list[str]) -> int:
    """Intern a string to a small int code, to avoid fork copy-on-write memory
    blowup from refcounting Python strings in worker processes."""
    c = code_map.get(s)
    if c is None:
        c = len(lookup)
        code_map[s] = c
        lookup.append(s)
    return c


class _SentenceStore:
    """Read-only per-review sentence-text lookup backed by one shared string
    buffer + integer offset arrays, instead of list[list[str]] (one Python
    object per sentence) -- avoids fork copy-on-write memory blowup when many
    worker processes read it repeatedly."""
    __slots__ = ('text', 'starts', 'ends', 'review_start', 'review_count')

    def __init__(self, text: str, starts: np.ndarray, ends: np.ndarray,
                 review_start: np.ndarray, review_count: np.ndarray):
        self.text = text
        self.starts = starts
        self.ends = ends
        self.review_start = review_start
        self.review_count = review_count

    def get(self, ix: int) -> list[str]:
        s0 = int(self.review_start[ix])
        n = int(self.review_count[ix])
        text = self.text
        starts = self.starts
        ends = self.ends
        return [text[int(starts[s0 + k]):int(ends[s0 + k])] for k in range(n)]


class _GroupIndex:
    """Read-only int -> [int, ...] group lookup (e.g. item_code -> review row
    indices), backed by a sorted index array + offsets (CSR layout) instead
    of dict[int, list[int]] -- same fork/COW rationale as _SentenceStore."""
    __slots__ = ('sorted_idx', 'offsets')

    def __init__(self, sorted_idx: np.ndarray, offsets: np.ndarray):
        self.sorted_idx = sorted_idx
        self.offsets = offsets

    def get(self, code: int, default=()):
        if code < 0 or code >= len(self.offsets) - 1:
            return default
        start, end = int(self.offsets[code]), int(self.offsets[code + 1])
        if start == end:
            return default
        return self.sorted_idx[start:end]


def _stream_metadata(path: Path, item_col: str, max_rows: int | None) -> dict[str, str]:
    """Read metadata line by line, keeping only item_id -> compact text."""
    meta_map: dict[str, str] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            iid = r.get(item_col) or r.get('parent_asin') or r.get('asin')
            if iid:
                meta_map[str(iid)] = metadata_to_text(r)
    return meta_map


def _load_reviews_arrays(args, dataset: str, review_path: Path):
    """Stream-parse reviews JSONL into compact arrays (numpy for numeric
    fields, int codes for user/item ids) instead of one Python dict per raw
    review, to keep memory bounded on large files."""
    user_col, item_col, text_col = args.user_col, args.item_col, args.text_col
    rating_col, ts_col, helpful_col = args.rating_col, args.timestamp_col, args.helpful_col

    ts_list: list[int] = []
    rating_list: list[float] = []
    helpful_list: list[float] = []
    user_code_list: list[int] = []
    item_code_list: list[int] = []
    text_parts: list[str] = []
    sent_starts_list: list[int] = []
    sent_ends_list: list[int] = []
    review_sent_start_list: list[int] = []
    review_sent_count_list: list[int] = []
    running_offset = 0
    running_sent_idx = 0

    user_code_map: dict[str, int] = {}
    user_lookup: list[str] = []
    item_code_map: dict[str, int] = {}
    item_lookup: list[str] = []

    n_seen = 0
    with open(review_path, 'r', encoding='utf-8') as f:
        for line in f:
            if args.max_raw_rows is not None and n_seen >= args.max_raw_rows:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_seen += 1
            if user_col not in row or item_col not in row or text_col not in row or ts_col not in row:
                if n_seen == 1:
                    raise ValueError(f'Missing required columns in reviews; available={list(row.keys())}')
                continue
            sentences = split_sentences(row.get(text_col), min_words=args.min_sentence_words, max_words=args.max_sentence_words)
            if not sentences:
                continue
            ts_list.append(normalize_timestamp(row[ts_col]))
            try:
                rating_list.append(float(row.get(rating_col, 0.0) or 0.0))
            except (TypeError, ValueError):
                rating_list.append(0.0)
            try:
                helpful_list.append(float(row.get(helpful_col, 0.0) or 0.0))
            except (TypeError, ValueError):
                helpful_list.append(0.0)
            user_code_list.append(_code_of(str(row[user_col]), user_code_map, user_lookup))
            item_code_list.append(_code_of(str(row[item_col]), item_code_map, item_lookup))
            review_sent_start_list.append(running_sent_idx)
            review_sent_count_list.append(len(sentences))
            for s in sentences:
                text_parts.append(s)
                running_offset += len(s)
                sent_starts_list.append(running_offset - len(s))
                sent_ends_list.append(running_offset)
                running_sent_idx += 1
            del row

    ts_arr = np.asarray(ts_list, dtype=np.int64)
    rating_arr = np.asarray(rating_list, dtype=np.float32)
    helpful_arr = np.asarray(helpful_list, dtype=np.float32)
    user_code_arr = np.asarray(user_code_list, dtype=np.int32)
    item_code_arr = np.asarray(item_code_list, dtype=np.int32)
    del ts_list, rating_list, helpful_list, user_code_list, item_code_list
    big_text = ''.join(text_parts)
    del text_parts
    sentence_store = _SentenceStore(
        big_text,
        np.asarray(sent_starts_list, dtype=np.int64),
        np.asarray(sent_ends_list, dtype=np.int64),
        np.asarray(review_sent_start_list, dtype=np.int64),
        np.asarray(review_sent_count_list, dtype=np.int32),
    )
    del sent_starts_list, sent_ends_list, review_sent_start_list, review_sent_count_list
    return ts_arr, rating_arr, helpful_arr, user_code_arr, item_code_arr, sentence_store, user_lookup, item_lookup


_W: dict[str, Any] = {}


def _process_user(item: tuple[int, np.ndarray]) -> list[dict[str, Any]]:
    """Build every eligible case for one user. Runs in a forked worker process,
    reading shared read-only arrays set up in _W by the parent (relies on
    copy-on-write fork semantics, so no per-task pickling of the dataset)."""
    u_code, idxs = item
    ts_arr = _W['ts_arr']; item_code_arr = _W['item_code_arr']; user_code_arr = _W['user_code_arr']
    rating_arr = _W['rating_arr']; helpful_arr = _W['helpful_arr']
    sentence_store = _W['sentence_store']
    item_indices = _W['item_indices']; meta_map = _W['meta_map']; a = _W['args']
    dataset = _W['dataset']; user_lookup = _W['user_lookup']; item_lookup = _W['item_lookup']

    idxs = sorted(idxs, key=lambda ix: ts_arr[ix])
    out: list[dict[str, Any]] = []
    if len(idxs) <= a['min_user_history']:
        return out
    for pos in range(a['min_user_history'], len(idxs)):
        target_idx = idxs[pos]
        t = int(ts_arr[target_idx])
        item_code = int(item_code_arr[target_idx])
        history_idxs = [ix for ix in idxs[:pos] if item_code_arr[ix] != item_code]
        if len(history_idxs) < a['min_user_history']:
            continue
        history_sentences: list[str] = []
        hist_sentence_counts = []
        for ix in history_idxs[-a['max_user_history_reviews']:]:
            ss = sentence_store.get(ix)
            history_sentences.extend(ss)
            hist_sentence_counts.append(len(ss))
        history_sentences = history_sentences[-a['max_user_history_sentences']:]
        user_avg_k = max(1, round(sum(hist_sentence_counts) / max(1, len(hist_sentence_counts))))

        candidate_reviews = []
        for ix in item_indices.get(item_code, []):
            if user_code_arr[ix] == u_code:
                continue
            if ts_arr[ix] >= t:
                continue
            candidate_reviews.append(ix)
        if len(candidate_reviews) < a['min_candidate_reviews']:
            continue
        candidate_sentences = []
        item_id = item_lookup[item_code]
        for ix in candidate_reviews:
            helpful = float(helpful_arr[ix])
            review_id = f'{dataset}_r{ix}'
            for si, sent in enumerate(sentence_store.get(ix)):
                candidate_sentences.append({
                    'sentence_id': f'{review_id}_s{si}',
                    'review_id': review_id,
                    'user_id': user_lookup[user_code_arr[ix]],
                    'item_id': item_id,
                    'timestamp': int(ts_arr[ix]),
                    'rating': float(rating_arr[ix]),
                    'helpful_vote': helpful,
                    'text': sent,
                })
        if len(candidate_sentences) < a['min_candidate_sentences']:
            continue
        candidate_sentences = sorted(candidate_sentences, key=lambda x: x['timestamp'], reverse=True)[:a['max_candidate_sentences']]
        gt_sentences = sentence_store.get(target_idx)
        if not gt_sentences:
            continue
        out.append({
            'dataset': dataset,
            'user_id': user_lookup[u_code],
            'item_id': item_id,
            'timestamp': t,
            'rating': float(rating_arr[target_idx]),
            'user_history_sentences': history_sentences,
            'user_history_text': compact_profile(history_sentences),
            'user_avg_k': int(user_avg_k),
            'metadata_text': meta_map.get(item_id, ''),
            'candidate_sentences': candidate_sentences,
            'ground_truth_sentences': gt_sentences,
            'ground_truth_text': ' '.join(gt_sentences),
        })
    return out


class _DiskStreamCollector:
    """Streams every offered case to a temp JSONL file on disk instead of
    accumulating in memory (--max-cases-per-dataset not set): peak RAM is
    bounded by a lightweight (case_id, timestamp) index, used later to decide
    the temporal train/valid/test split without re-loading case bodies."""

    def __init__(self, dataset: str, tmp_path: Path):
        self.dataset = dataset
        self.tmp_path = tmp_path
        self._fh = gzip.open(tmp_path, 'wt', encoding='utf-8', compresslevel=4)
        self._counter = 0
        self.index: list[tuple[str, int]] = []  # (case_id, timestamp), kept in memory

    def offer(self, case: dict[str, Any]) -> None:
        case_id = f'{self.dataset}_{self._counter}'
        self._counter += 1
        case['case_id'] = case_id
        self._fh.write(json.dumps(case, ensure_ascii=False) + '\n')
        self.index.append((case_id, case['timestamp']))

    def close(self) -> None:
        self._fh.close()


class _ReservoirSampler:
    """Algorithm-R reservoir sampling: a uniform random sample of at most
    `cap` items from a stream of unknown length, in O(cap) memory."""

    def __init__(self, cap: int | None, rng: random.Random):
        self.cap = cap
        self.rng = rng
        self.seen = 0
        self.reservoir: list[dict[str, Any]] = []

    def offer(self, case: dict[str, Any]) -> None:
        if self.cap is None:
            self.reservoir.append(case)
            return
        self.seen += 1
        if len(self.reservoir) < self.cap:
            self.reservoir.append(case)
        else:
            j = self.rng.randint(0, self.seen - 1)
            if j < self.cap:
                self.reservoir[j] = case


def build_dataset_cases(args, dataset: str, rng: random.Random, tmp_dir: Path | None = None):
    """Returns ('memory', dataset, None, cases) if --max-cases-per-dataset is
    set (cases held in memory), else ('streamed', dataset, tmp_path, index)
    with every case written to tmp_path and index the lightweight
    [(case_id, timestamp), ...] list main() uses to decide the split."""
    raw_dir = Path(args.raw_dir)
    review_path = Path(args.reviews) if args.reviews else infer_file(raw_dir, dataset, 'reviews')
    metadata_path = Path(args.metadata) if args.metadata else None
    if metadata_path is None:
        try:
            metadata_path = infer_file(raw_dir, dataset, 'metadata')
        except FileNotFoundError:
            metadata_path = None

    streaming = args.max_cases_per_dataset is None

    print(f'[{dataset}] Loading reviews from {review_path}')
    (ts_arr, rating_arr, helpful_arr, user_code_arr, item_code_arr,
     sentence_store, user_lookup, item_lookup) = _load_reviews_arrays(args, dataset, review_path)
    if len(ts_arr) == 0:
        return ('streamed', dataset, None, []) if streaming else ('memory', dataset, None, [])
    print(f'[{dataset}] {len(ts_arr)} reviews, {len(user_lookup)} users, {len(item_lookup)} items')

    meta_map: dict[str, str] = {}
    if metadata_path and metadata_path.exists():
        print(f'[{dataset}] Loading metadata from {metadata_path}')
        meta_map = _stream_metadata(metadata_path, args.item_col, args.max_meta_rows)

    user_indices: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(user_code_arr)):
        user_indices[int(user_code_arr[idx])].append(idx)
    n_items = len(item_lookup)
    item_sort_order = np.argsort(item_code_arr, kind='stable').astype(np.int64)
    sorted_item_codes = item_code_arr[item_sort_order]
    item_offsets = np.searchsorted(sorted_item_codes, np.arange(n_items + 1)).astype(np.int64)
    item_indices = _GroupIndex(item_sort_order, item_offsets)

    _W.clear()
    _W['ts_arr'] = ts_arr
    _W['item_code_arr'] = item_code_arr
    _W['user_code_arr'] = user_code_arr
    _W['rating_arr'] = rating_arr
    _W['helpful_arr'] = helpful_arr
    _W['sentence_store'] = sentence_store
    _W['item_indices'] = item_indices
    _W['meta_map'] = meta_map
    _W['dataset'] = dataset
    _W['user_lookup'] = user_lookup
    _W['item_lookup'] = item_lookup
    _W['args'] = {
        'min_user_history': args.min_user_history,
        'min_candidate_reviews': args.min_candidate_reviews,
        'min_candidate_sentences': args.min_candidate_sentences,
        'max_candidate_sentences': args.max_candidate_sentences,
        'max_user_history_sentences': args.max_user_history_sentences,
        'max_user_history_reviews': args.max_user_history_reviews,
    }

    n_workers = max(1, args.num_workers)
    items = list(user_indices.items())
    if streaming:
        assert tmp_dir is not None, 'tmp_dir is required when --max-cases-per-dataset is not set'
        tmp_path = tmp_dir / f'.tmp_allcases_{dataset}.jsonl.gz'
        collector: _DiskStreamCollector | _ReservoirSampler = _DiskStreamCollector(dataset, tmp_path)
        print(f'[{dataset}] No --max-cases-per-dataset cap set; streaming every eligible case to '
              f'{tmp_path} as it is produced (memory-bounded by a lightweight index, not case content).')
    else:
        collector = _ReservoirSampler(args.max_cases_per_dataset, rng)

    if n_workers == 1 or len(items) < n_workers * 4:
        for it in tqdm(items, desc=f'[{dataset}] Build temporal cases'):
            for case in _process_user(it):
                collector.offer(case)
    else:
        chunksize = max(1, min(500, len(items) // (n_workers * 8) or 1))
        ctx = mp.get_context('fork')
        with ctx.Pool(processes=n_workers) as pool:
            for result in tqdm(pool.imap_unordered(_process_user, items, chunksize=chunksize), total=len(items), desc=f'[{dataset}] Build temporal cases ({n_workers}w)'):
                for case in result:
                    collector.offer(case)

    _W.clear()
    del ts_arr, rating_arr, helpful_arr, user_code_arr, item_code_arr, sentence_store, user_indices, item_indices, meta_map
    gc.collect()

    if streaming:
        collector.close()
        print(f'[{dataset}] {len(collector.index)} total eligible cases streamed to {collector.tmp_path}')
        return ('streamed', dataset, collector.tmp_path, collector.index)

    print(f'[{dataset}] {collector.seen} total eligible cases seen (kept {len(collector.reservoir)} via reservoir sampling)')
    cases = collector.reservoir
    for i, c in enumerate(cases):
        c['case_id'] = f'{dataset}_{i}'
    return ('memory', dataset, None, cases)


def assign_splits(cases: list[dict[str, Any]], valid_ratio: float, test_ratio: float) -> list[dict[str, Any]]:
    cases = sorted(cases, key=lambda x: x['timestamp'])
    n = len(cases)
    n_test = int(round(n * test_ratio))
    n_valid = int(round(n * valid_ratio))
    for idx, c in enumerate(cases):
        if idx < n - n_valid - n_test:
            c['split'] = 'train'
        elif idx < n - n_test:
            c['split'] = 'valid'
        else:
            c['split'] = 'test'
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', required=True)
    ap.add_argument('--raw-dir', default='data/raw')
    ap.add_argument('--reviews', default=None, help='Optional single review file; use only with one dataset')
    ap.add_argument('--metadata', default=None, help='Optional single metadata file; use only with one dataset')
    ap.add_argument('--output', default='data/cases')
    ap.add_argument('--user-col', default='user_id')
    ap.add_argument('--item-col', default='parent_asin')
    ap.add_argument('--text-col', default='text')
    ap.add_argument('--rating-col', default='rating')
    ap.add_argument('--timestamp-col', default='timestamp')
    ap.add_argument('--helpful-col', default='helpful_vote')
    ap.add_argument('--min-user-history', type=int, default=3)
    ap.add_argument('--min-candidate-reviews', type=int, default=5)
    ap.add_argument('--min-candidate-sentences', type=int, default=20)
    ap.add_argument('--max-candidate-sentences', type=int, default=300)
    ap.add_argument('--max-user-history-sentences', type=int, default=100)
    ap.add_argument('--max-user-history-reviews', type=int, default=20)
    ap.add_argument('--min-sentence-words', type=int, default=3)
    ap.add_argument('--max-sentence-words', type=int, default=80)
    ap.add_argument('--valid-ratio', type=float, default=0.1)
    ap.add_argument('--test-ratio', type=float, default=0.2)
    ap.add_argument('--max-cases-per-dataset', type=int, default=None,
                     help='Cap eligible cases per dataset via reservoir sampling (bounds peak memory; strongly recommended for full-scale raw data)')
    ap.add_argument('--max-raw-rows', type=int, default=None)
    ap.add_argument('--max-meta-rows', type=int, default=None)
    ap.add_argument('--num-workers', type=int, default=min(os.cpu_count() or 1, 64),
                     help='Parallel worker processes for per-user case building (fork-based). '
                          'Defaults to min(cpu_count, 64) to bound copy-on-write memory growth from many workers.')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.output)
    rng = random.Random(args.seed)

    streaming = args.max_cases_per_dataset is None
    tmp_dir = ensure_dir(out_dir / '.tmp_build_cases') if streaming else None

    results = [build_dataset_cases(args, ds, rng, tmp_dir) for ds in args.datasets]

    if not streaming:
        all_cases = []
        for _, ds, _, cases in results:
            print(f'[{ds}] Built {len(cases)} eligible cases')
            all_cases.extend(cases)
        split_cases = []
        for ds in sorted(set(c['dataset'] for c in all_cases)):
            ds_cases = [c for c in all_cases if c['dataset'] == ds]
            split_cases.extend(assign_splits(ds_cases, args.valid_ratio, args.test_ratio))
        all_cases = split_cases

        for split in ['train', 'valid', 'test']:
            rows = [c for c in all_cases if c['split'] == split]
            write_jsonl(rows, out_dir / f'cases_{split}.jsonl.gz')
            print(f'Wrote {len(rows)} {split} cases -> {out_dir / f"cases_{split}.jsonl.gz"}')

        stats = []
        for ds in sorted(set(c['dataset'] for c in all_cases)):
            subset = [c for c in all_cases if c['dataset'] == ds]
            stats.append({
                'dataset': ds,
                'n_cases': len(subset),
                'n_train': sum(c['split'] == 'train' for c in subset),
                'n_valid': sum(c['split'] == 'valid' for c in subset),
                'n_test': sum(c['split'] == 'test' for c in subset),
                'avg_candidates': sum(len(c['candidate_sentences']) for c in subset) / max(1, len(subset)),
                'avg_gt_sentences': sum(len(c['ground_truth_sentences']) for c in subset) / max(1, len(subset)),
                'avg_user_history_sentences': sum(len(c['user_history_sentences']) for c in subset) / max(1, len(subset)),
            })
        pd.DataFrame(stats).to_csv(out_dir / 'dataset_stats.csv', index=False)
        return

    split_map: dict[str, str] = {}
    for _, ds, _, index in results:
        idx_sorted = sorted(index, key=lambda x: x[1])
        n = len(idx_sorted)
        n_test = int(round(n * args.test_ratio))
        n_valid = int(round(n * args.valid_ratio))
        for pos, (case_id, _ts) in enumerate(idx_sorted):
            if pos < n - n_valid - n_test:
                split_map[case_id] = 'train'
            elif pos < n - n_test:
                split_map[case_id] = 'valid'
            else:
                split_map[case_id] = 'test'

    out_handles = {s: gzip.open(out_dir / f'cases_{s}.jsonl.gz', 'wt', encoding='utf-8', compresslevel=4) for s in ['train', 'valid', 'test']}
    counts = {s: 0 for s in ['train', 'valid', 'test']}
    stats_acc: dict[str, dict[str, float]] = {}
    for _, ds, tmp_path, index in results:
        print(f'[{ds}] Built {len(index)} eligible cases')
        acc = stats_acc.setdefault(ds, {'n_cases': 0, 'n_train': 0, 'n_valid': 0, 'n_test': 0,
                                         'sum_candidates': 0, 'sum_gt': 0, 'sum_hist': 0})
        if tmp_path is None:
            continue
        with gzip.open(tmp_path, 'rt', encoding='utf-8') as f:
            for line in f:
                case = json.loads(line)
                split = split_map[case['case_id']]
                case['split'] = split
                out_handles[split].write(json.dumps(case, ensure_ascii=False) + '\n')
                counts[split] += 1
                acc['n_cases'] += 1
                acc[f'n_{split}'] += 1
                acc['sum_candidates'] += len(case['candidate_sentences'])
                acc['sum_gt'] += len(case['ground_truth_sentences'])
                acc['sum_hist'] += len(case['user_history_sentences'])
        tmp_path.unlink()

    for s in ['train', 'valid', 'test']:
        out_handles[s].close()
        print(f'Wrote {counts[s]} {s} cases -> {out_dir / f"cases_{s}.jsonl.gz"}')
    tmp_dir.rmdir()

    stats = []
    for ds, acc in stats_acc.items():
        n = max(1, acc['n_cases'])
        stats.append({
            'dataset': ds,
            'n_cases': acc['n_cases'],
            'n_train': acc['n_train'],
            'n_valid': acc['n_valid'],
            'n_test': acc['n_test'],
            'avg_candidates': acc['sum_candidates'] / n,
            'avg_gt_sentences': acc['sum_gt'] / n,
            'avg_user_history_sentences': acc['sum_hist'] / n,
        })
    pd.DataFrame(stats).to_csv(out_dir / 'dataset_stats.csv', index=False)


if __name__ == '__main__':
    main()
