#!/usr/bin/env python
from __future__ import annotations

"""PEER's own user-history representation h_u: each history sentence encoded
individually and mean-pooled, instead of concatenated into one blob and
encoded as a single string (scripts/cache_embeddings.py's approach, which
truncates for users whose history exceeds the encoder's token limit). Used
only by PEER's own ranker features via --user-history-override in
build_features.py / train_peer_retriever.py; baselines keep using the blob
encoding unchanged. Output is keyed by the same f'{case_id}_user_history_text'
id as the main embeddings."""

import argparse
import json
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import numpy as np
from sentence_transformers import SentenceTransformer

from peer.text import clean_text
from peer.utils import ensure_dir, read_jsonl, resolve_cases_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='data/cases')
    ap.add_argument('--model', default='sentence-transformers/all-mpnet-base-v2')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--output', default='models/user_history_embeddings')
    ap.add_argument('--device', default=None, help='e.g. mps, cuda, cpu; default lets sentence-transformers auto-pick')
    args = ap.parse_args()

    encoder = SentenceTransformer(args.model, device=args.device)
    print(f'encoder loaded on device={encoder.device}, max_seq_length={encoder.max_seq_length}', flush=True)

    all_cases = []
    for split in ['train', 'valid', 'test']:
        p = resolve_cases_path(args.cases, split)
        if p is None:
            continue
        cs = read_jsonl(p)
        all_cases.extend(cs)
        print(f'{split}: {len(cs)} cases', flush=True)

    all_sentences: list[str] = []
    case_slice: dict[str, tuple[int, int]] = {}
    for c in all_cases:
        cid = c['case_id']
        sents = [clean_text(s) for s in c.get('user_history_sentences', [])]
        sents = [s for s in sents if s]
        start = len(all_sentences)
        all_sentences.extend(sents)
        case_slice[cid] = (start, len(all_sentences))
    print(f'total individual history sentences to encode: {len(all_sentences)}', flush=True)

    embs = encoder.encode(all_sentences, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)
    print('encoding done', flush=True)

    ids_out = []
    vecs_out = []
    n_empty = 0
    for cid, (s0, s1) in case_slice.items():
        if s1 <= s0:
            n_empty += 1
            continue
        vecs_out.append(embs[s0:s1].mean(axis=0))
        ids_out.append(f'{cid}_user_history_text')
    print(f'{len(ids_out)} cases with a user-history embedding, {n_empty} had no history sentences (skipped)', flush=True)

    out_path = _ProjectPath(args.output)
    ensure_dir(out_path.parent)
    np.save(out_path.with_suffix('.npy'), np.stack(vecs_out).astype(np.float32))
    with open(out_path.with_suffix('.ids.json'), 'w') as f:
        json.dump(ids_out, f)
    print(f'saved -> {out_path}.npy / .ids.json', flush=True)


if __name__ == '__main__':
    main()
