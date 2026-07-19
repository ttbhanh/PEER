#!/usr/bin/env python
from __future__ import annotations

"""Fine-tune a MiniLM cross-encoder to close PEER's remaining sem_f1 gap against
PRAG, per the user's chosen direction: "Cross-encoder reranking top-50, train
bang max sentence-level ground-truth similarity va hard negatives lay tu
top-ranked PRAG candidates."

Training pair construction, per case:
  - query  = user_history_text (truncated)
  - answer = one candidate sentence's text
  - label  = max_g cosine(candidate_emb, gt_emb) -- the same max-sentence-level
    similarity semantic_prf() uses at eval time (peer/metrics.py), so the
    regression target is aligned with what sem_f1 actually measures (train/eval
    objective mismatch was flagged as a real risk during design).

Per case we sample:
  - up to 2 positives: the candidates with the highest actual label.
  - up to 2 PRAG-hard-negatives: candidates PRAG's own retriever (the ORIGINAL
    metadata-conditioned one, models/prag_retriever.pt -- we want PRAG's real
    failure mode, not our own metadata-free variant) ranks in its own top-5,
    but whose actual label is well below the case's best candidate. This is
    exactly "PRAG likes it, but it's not actually close to the target" --
    training on many such pairs is what should teach the reranker to correct
    PRAG's specific mistake, not just learn semantic similarity in general.
  - up to 1 random negative for stability.

Loss: MSELoss (regression to the float label) -- both positives and hard
negatives use their true label, so oversampling hard negatives in the training
set is what shapes the ranking, rather than a separate margin/classification
objective. Base checkpoint: cross-encoder/ms-marco-MiniLM-L-6-v2 (small enough
to fine-tune on CPU/MPS in this project's local-only setup).
"""

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset

from peer.utils import cosine_vec, read_jsonl
from scripts.train_prag_retriever import PragRetriever

_MAX_QUERY_CHARS = 600


def load_embeddings(path: str) -> dict[str, np.ndarray]:
    # Plain .npy (mmap-able) + sibling ids.json, not .npz -- see
    # peer/embeddings.py::_npy_and_ids_paths.
    p = Path(path)
    emb = np.load(p.with_suffix('.npy'), mmap_mode='r')
    with open(p.with_suffix('.ids.json')) as f:
        ids = json.load(f)
    return {str(ids[i]): emb[i] for i in range(len(ids))}


def load_prag(path: str) -> tuple[PragRetriever, int]:
    with open(path, 'rb') as f:
        ck = pickle.load(f)
    m = PragRetriever(emb_dim=ck['emb_dim'], hidden=ck['hidden'])
    m.load_state_dict(ck['state_dict'])
    m.eval()
    return m


def build_pairs(cases_path: str, embs: dict[str, np.ndarray], prag: PragRetriever, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for c in read_jsonl(cases_path):
        cid = c['case_id']
        user_emb = embs.get(f'{cid}_user_history_text')
        meta_emb = embs.get(f'{cid}_metadata_text')
        gt_embs = [embs[f'{cid}_gt_{i}'] for i in range(len(c['ground_truth_sentences'])) if f'{cid}_gt_{i}' in embs]
        if user_emb is None or meta_emb is None or not gt_embs:
            continue
        with torch.no_grad():
            prag_target = prag(torch.from_numpy(user_emb).unsqueeze(0), torch.from_numpy(meta_emb).unsqueeze(0)).squeeze(0).numpy()

        cands = c['candidate_sentences']
        scored = []
        for cand in cands:
            e = embs.get(cand['sentence_id'])
            if e is None:
                continue
            actual = max(cosine_vec(e, g) for g in gt_embs)
            prag_score = cosine_vec(e, prag_target)
            scored.append((cand['text'], actual, prag_score))
        if len(scored) < 4:
            continue

        query = c['user_history_text'][:_MAX_QUERY_CHARS]
        by_actual = sorted(scored, key=lambda x: -x[1])
        by_prag = sorted(scored, key=lambda x: -x[2])

        positives = by_actual[:2]
        top_actual_ids = {t[0] for t in positives}
        hard_negs = [t for t in by_prag[:5] if t[0] not in top_actual_ids and t[1] < positives[0][1] - 0.05][:2]
        remaining = [t for t in scored if t[0] not in top_actual_ids and t not in hard_negs]
        random_neg = [rng.choice(remaining)] if remaining else []

        for text, actual, _ in positives + hard_negs + random_neg:
            rows.append({'query': query, 'answer': text, 'label': float(actual)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='data/cases')
    ap.add_argument('--embeddings', default='embeddings/embeddings.npz')
    ap.add_argument('--prag-model', default='models/prag_retriever.pt')
    ap.add_argument('--base-model', default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', default='models/peer_cross_encoder')
    ap.add_argument('--hf-home', default='.hf_cache', help='Writable cache dir for downloading the base checkpoint (the default ~/.cache/huggingface is root-owned and unwritable on this machine)')
    args = ap.parse_args()

    if args.hf_home:
        os.environ['HF_HOME'] = str(_ProjectPath(args.hf_home).resolve())

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    embs = load_embeddings(args.embeddings)
    prag = load_prag(args.prag_model)

    print('Building training pairs (positives + PRAG-hard-negatives)...')
    train_rows = build_pairs(f'{args.cases}/cases_train.jsonl', embs, prag, seed=args.seed)
    valid_rows = build_pairs(f'{args.cases}/cases_valid.jsonl', embs, prag, seed=args.seed)
    print(f'train pairs={len(train_rows)} valid pairs={len(valid_rows)}')

    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import MSELoss

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = CrossEncoder(args.base_model, num_labels=1, device=device)
    # Labels are cosine similarities in ~[0,1] (actually [-1,1] but candidate/gt
    # embeddings are rarely anti-correlated in practice); the base checkpoint's
    # raw logits are on MS-MARCO's unbounded relevance scale, so without a
    # bounding activation MSE starts from a large, badly-scaled loss.
    loss = MSELoss(model, activation_fn=torch.nn.Sigmoid())

    train_ds = Dataset.from_list(train_rows)
    valid_ds = Dataset.from_list(valid_rows)

    out_path = _ProjectPath(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    train_args = CrossEncoderTrainingArguments(
        output_dir=str(out_path / '_trainer'),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy='epoch',
        save_strategy='no',
        logging_steps=50,
        seed=args.seed,
        report_to=[],
    )
    trainer = CrossEncoderTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        loss=loss,
    )
    trainer.train()
    model.save_pretrained(str(out_path))
    print(f'Saved fine-tuned cross-encoder -> {out_path}')


if __name__ == '__main__':
    main()
