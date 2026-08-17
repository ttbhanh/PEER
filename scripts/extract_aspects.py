#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import multiprocessing as mp
import os
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from peer.aspects import extract_aspect_phrases, normalize_aspects_by_frequency, build_profile_aspects
from peer.utils import ensure_dir, read_jsonl, resolve_cases_path, write_jsonl


def iter_cases(cases_dir: Path):
    for split in ['train', 'valid', 'test']:
        path = resolve_cases_path(cases_dir, split)
        if path is not None:
            for c in read_jsonl(path):
                yield split, c


_METHOD = 'hybrid'


def _init_worker(method: str) -> None:
    global _METHOD
    _METHOD = method


def _extract_task(task: tuple[str, int]) -> list[str]:
    text, top_n = task
    return extract_aspect_phrases(text, _METHOD, top_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='data/cases')
    ap.add_argument('--output', default='data/processed/aspects')
    ap.add_argument('--method', default='hybrid', choices=['hybrid', 'spacy', 'fallback'])
    ap.add_argument('--min-count', type=int, default=2)
    ap.add_argument('--max-vocab', type=int, default=5000)
    ap.add_argument('--top-n-per-text', type=int, default=8)
    ap.add_argument('--num-workers', type=int, default=min(os.cpu_count() or 1, 64),
                     help='Parallel worker processes for per-sentence aspect extraction (fork-based)')
    args = ap.parse_args()
    cases_dir = Path(args.cases)
    out_dir = ensure_dir(args.output)
    cases = list(iter_cases(cases_dir))

    print('Collecting text spans...')
    tasks: list[tuple[str, int]] = []
    raw_rows_meta: list[dict] = []
    for split, c in cases:
        case_id = c['case_id']
        for cand in c['candidate_sentences']:
            tasks.append((cand['text'], args.top_n_per_text))
            raw_rows_meta.append({'case_id': case_id, 'split': split, 'kind': 'candidate', 'sentence_id': cand['sentence_id'], 'text': cand['text']})
        for gi, gt in enumerate(c['ground_truth_sentences']):
            tasks.append((gt, args.top_n_per_text))
            raw_rows_meta.append({'case_id': case_id, 'split': split, 'kind': 'ground_truth', 'sentence_id': f'{case_id}_gt_{gi}', 'text': gt})
        for kind, text in [('metadata', c.get('metadata_text', '')), ('user_history', c.get('user_history_text', ''))]:
            tasks.append((text, args.top_n_per_text * 4))
            raw_rows_meta.append({'case_id': case_id, 'split': split, 'kind': kind, 'sentence_id': f'{case_id}_{kind}', 'text': text})

    print(f'First pass: extract raw aspect phrases for {len(tasks)} text spans...')
    n_workers = max(1, args.num_workers)
    if n_workers == 1 or len(tasks) < n_workers * 8:
        _init_worker(args.method)
        per_row_aspects = [_extract_task(t) for t in tqdm(tasks)]
    else:
        chunksize = max(1, min(2000, len(tasks) // (n_workers * 20) or 1))
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=n_workers, initializer=_init_worker, initargs=(args.method,)) as pool:
            per_row_aspects = list(tqdm(pool.imap(_extract_task, tasks, chunksize=chunksize), total=len(tasks)))

    # Vocabulary built from train only, then applied to canonicalize valid/test too.
    train_raw_aspects = [a for meta, asp in zip(raw_rows_meta, per_row_aspects) if meta['split'] == 'train' for a in asp]
    mapping = normalize_aspects_by_frequency(train_raw_aspects, min_count=args.min_count, max_vocab=args.max_vocab)
    vocab_counter = Counter(mapping.get(a, a) for a in train_raw_aspects)
    vocab = [{'aspect': a, 'count': c} for a, c in vocab_counter.most_common(args.max_vocab)]
    write_jsonl(vocab, out_dir / 'aspect_vocab.jsonl')

    print('Second pass: canonicalize aspects...')
    out_rows = []
    for meta, raw_aspects in zip(raw_rows_meta, per_row_aspects):
        aspects = []
        for a in raw_aspects:
            ca = mapping.get(a, a)
            if ca:
                aspects.append(ca)
        seen = set()
        aspects = [a for a in aspects if not (a in seen or seen.add(a))]
        out_rows.append(meta | {'aspects': aspects})
    pd.DataFrame(out_rows).to_parquet(out_dir / 'sentence_aspects.parquet', index=False)
    print(f'Wrote {len(out_rows)} aspect rows -> {out_dir / "sentence_aspects.parquet"}')


if __name__ == '__main__':
    main()
