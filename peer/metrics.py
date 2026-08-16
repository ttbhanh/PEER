from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None
try:
    import sacrebleu
except Exception:
    sacrebleu = None

from .aspects import aspect_f1
from .utils import cosine_matrix


def semantic_prf(e_emb: np.ndarray, g_emb: np.ndarray) -> tuple[float, float, float]:
    if len(e_emb) == 0 or len(g_emb) == 0:
        return 0.0, 0.0, 0.0
    sims = cosine_matrix(e_emb, g_emb)
    p = float(np.mean(np.max(sims, axis=1)))
    r = float(np.mean(np.max(sims, axis=0)))
    f1 = 2 * p * r / (p + r + 1e-12)
    return p, r, float(f1)


def redundancy(emb: np.ndarray) -> float:
    if len(emb) <= 1:
        return 0.0
    sims = cosine_matrix(emb, emb)
    tri = sims[np.triu_indices(len(emb), k=1)]
    return float(np.mean(tri)) if len(tri) else 0.0


def aspect_noise(pred_aspects: list[str], gold_aspects: list[str], user_aspects: list[str] | None = None) -> float:
    pred = set(pred_aspects)
    if not pred:
        return 0.0
    allowed = set(gold_aspects)
    if user_aspects:
        allowed |= set(user_aspects)
    irrelevant = pred - allowed
    return len(irrelevant) / max(1, len(pred))


def _token_f1(pred_text: str, gold_text: str) -> float:
    from collections import Counter
    p = pred_text.lower().split()
    g = gold_text.lower().split()
    if not p or not g:
        return 0.0
    cp, cg = Counter(p), Counter(g)
    overlap = sum((cp & cg).values())
    prec = overlap / max(1, len(p))
    rec = overlap / max(1, len(g))
    return float(2 * prec * rec / (prec + rec + 1e-12))


_ROUGE_SCORER = None


def rouge_scores(pred_text: str, gold_text: str) -> dict[str, float]:
    global _ROUGE_SCORER
    if rouge_scorer is None:
        f = _token_f1(pred_text, gold_text)
        return {'rouge1': f, 'rouge2': 0.0, 'rougeL': f}
    if _ROUGE_SCORER is None:
        _ROUGE_SCORER = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    sc = _ROUGE_SCORER.score(gold_text, pred_text)
    return {k: float(v.fmeasure) for k, v in sc.items()}


def bleu_score(pred_text: str, gold_text: str) -> float:
    if sacrebleu is None:
        return _token_f1(pred_text, gold_text)
    try:
        return float(sacrebleu.sentence_bleu(pred_text, [gold_text]).score / 100.0)
    except Exception:
        return 0.0


def aggregate_metrics(rows: list[dict[str, Any]], group_cols: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = tuple(r[c] for c in group_cols)
        groups[key].append(r)
    out = []
    numeric_keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float, np.floating))})
    for key, items in groups.items():
        row = {c: key[i] for i, c in enumerate(group_cols)}
        row['n_cases'] = len(items)
        for nk in numeric_keys:
            vals = [float(x[nk]) for x in items if nk in x and np.isfinite(x[nk])]
            if vals:
                row[nk] = float(np.mean(vals))
        out.append(row)
    return out
