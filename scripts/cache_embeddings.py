#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from tqdm import tqdm

from peer.embeddings import TextEmbedder, save_embeddings
from peer.utils import ensure_dir, read_jsonl, resolve_cases_path


def collect_texts(cases_dir: Path):
    rows = []
    seen = set()
    for split in ['train', 'valid', 'test']:
        p = resolve_cases_path(cases_dir, split)
        if p is None:
            continue
        for c in read_jsonl(p):
            for cand in c['candidate_sentences']:
                sid = cand['sentence_id']
                if sid not in seen:
                    rows.append((sid, cand['text']))
                    seen.add(sid)
            for gi, gt in enumerate(c['ground_truth_sentences']):
                sid = f"{c['case_id']}_gt_{gi}"
                rows.append((sid, gt))
            for kind in ['user_history_text', 'metadata_text', 'ground_truth_text']:
                sid = f"{c['case_id']}_{kind}"
                rows.append((sid, c.get(kind, '') or ''))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='data/cases')
    ap.add_argument('--model', default='sentence-transformers/all-mpnet-base-v2')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--no-fallback-tfidf', action='store_true')
    ap.add_argument('--force-tfidf', action='store_true', help='Skip sentence-transformers and use TF-IDF/SVD fallback immediately')
    ap.add_argument('--tfidf-components', type=int, default=128)
    ap.add_argument('--output', default='embeddings/embeddings.npz')
    args = ap.parse_args()

    rows = collect_texts(Path(args.cases))
    rows = sorted(rows, key=lambda r: len(r[1]))
    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    print(f'Encoding {len(texts)} texts with {args.model}')
    
    model_name = 'tfidf-forced' if args.force_tfidf else args.model
    embedder = TextEmbedder(model_name, device=args.device, fp16=args.fp16, fallback_tfidf=not args.no_fallback_tfidf)
    if embedder.kind == 'tfidf_svd':
        print('WARNING: sentence-transformers unavailable or failed; using TF-IDF/SVD fallback.')
        embedder.fit(texts, n_components=args.tfidf_components)
    emb = embedder.encode(texts, batch_size=args.batch_size)
    out_path = Path(args.output)
    save_embeddings(ids, emb, out_path)
    embedder.save(out_path.parent / 'embedder')
    print(f'Wrote embeddings {emb.shape} -> {out_path}')


if __name__ == '__main__':
    main()
