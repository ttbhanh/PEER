#!/usr/bin/env python
from __future__ import annotations

"""PRAG (Xie et al., EMNLP 2023) baseline retriever mechanism: estimate a
target-review embedding from (user, metadata) via a small MLP (a scaled-down
stand-in for the paper's transformer blocks), rank candidates by similarity
to it. The generator stage does not apply to PEER's selection task."""

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from peer.utils import ensure_dir, read_jsonl, resolve_cases_path


class PragRetriever(nn.Module):
    def __init__(self, emb_dim: int = 768, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, user_emb: torch.Tensor, meta_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([user_emb, meta_emb], dim=-1)
        return F.normalize(self.net(x), dim=-1)


def load_embeddings(path: str) -> dict[str, np.ndarray]:
    p = Path(path)  # mmap'd .npy + ids.json, see peer/embeddings.py
    emb = np.load(p.with_suffix('.npy'), mmap_mode='r')
    with open(p.with_suffix('.ids.json')) as f:
        ids = json.load(f)
    return {str(ids[i]): emb[i] for i in range(len(ids))}


def build_dataset(cases_path: str, embs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    user_x, meta_x, target_y = [], [], []
    for c in read_jsonl(cases_path):
        cid = c['case_id']
        u = embs.get(f'{cid}_user_history_text')
        m = embs.get(f'{cid}_metadata_text')
        g = embs.get(f'{cid}_ground_truth_text')
        if u is None or m is None or g is None:
            continue
        user_x.append(u); meta_x.append(m); target_y.append(g)
    return np.stack(user_x), np.stack(meta_x), np.stack(target_y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='data/cases')
    ap.add_argument('--embeddings', default='embeddings/embeddings.npz')
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', default='models/prag_retriever.pt')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    embs = load_embeddings(args.embeddings)
    train_u, train_m, train_y = build_dataset(resolve_cases_path(args.cases, 'train'), embs)
    valid_u, valid_m, valid_y = build_dataset(resolve_cases_path(args.cases, 'valid'), embs)
    emb_dim = train_u.shape[1]
    print(f'train={len(train_u)} valid={len(valid_u)} emb_dim={emb_dim}')

    model = PragRetriever(emb_dim=emb_dim, hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def to_t(a):
        return torch.from_numpy(a)

    train_u_t, train_m_t, train_y_t = to_t(train_u), to_t(train_m), to_t(train_y)
    valid_u_t, valid_m_t, valid_y_t = to_t(valid_u), to_t(valid_m), to_t(valid_y)

    n = len(train_u)
    best_val = float('inf')
    best_state = None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        idx = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, args.batch_size):
            b = idx[start:start + args.batch_size]
            pred = model(train_u_t[b], train_m_t[b])
            loss = (1 - F.cosine_similarity(pred, train_y_t[b], dim=-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(b)
        train_loss = total_loss / n

        model.eval()
        with torch.no_grad():
            pred = model(valid_u_t, valid_m_t)
            val_loss = (1 - F.cosine_similarity(pred, valid_y_t, dim=-1)).mean().item()
            baseline_loss = (1 - F.cosine_similarity(valid_u_t, valid_y_t, dim=-1)).mean().item()
        marker = ''
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = '  <- best'
        print(f'epoch {epoch}/{args.epochs} train_cosdist={train_loss:.4f} val_cosdist={val_loss:.4f} '
              f'(baseline: user_emb alone={baseline_loss:.4f}) ({time.time()-t0:.1f}s){marker}', flush=True)

    out_path = _ProjectPath(args.output)
    ensure_dir(out_path.parent)
    with open(out_path, 'wb') as f:
        pickle.dump({'state_dict': best_state, 'emb_dim': emb_dim, 'hidden': args.hidden, 'val_cosdist': best_val}, f)
    print(f'Saved best checkpoint (val_cosdist={best_val:.4f}) -> {out_path}')


if __name__ == '__main__':
    main()
