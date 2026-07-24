from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .utils import cosine_vec


def resolve_k(k_value: str | int, user_avg_k: int, max_k: int) -> int:
    if isinstance(k_value, int):
        k = k_value
    elif str(k_value).lower() in {'user_avg', 'user_avg_k', 'user'}:
        k = user_avg_k
    else:
        k = int(k_value)
    return max(1, min(int(k), max_k))


def topk(candidates: list[dict[str, Any]], k: int, score_key: str = 'score') -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda x: float(x.get(score_key, 0.0)), reverse=True)[:k]


def greedy_coverage_select(
    candidates: list[dict[str, Any]],
    k: int,
    score_key: str = 'score',
    emb_key: str = 'embedding',
    aspect_key: str = 'aspects',
    lambda_coverage: float = 0.1,
    mu_redundancy: float = 0.0,
    eta_noise: float = 0.0,
    nu_aspect_repeat: float = 0.0,
    allowed_aspects: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Stage 2 (paper, Sec. "Greedy Evidence Construction"): at each step, pick

        argmax_c  utility(c) + lambda*C(c|S) - mu*R(c|S) - eta*N(c) - nu*P(c|S)

    where C is marginal aspect coverage (fraction of c's aspects not yet in the
    selected set S), R is the max cosine similarity to an already-selected
    sentence (redundancy), N is the fraction of c's aspects outside
    `allowed_aspects` (noise), and P is a log(1+count) penalty per aspect of c
    already covered by S (soft repetition penalty). The paper's reported
    numbers use lambda=0.1 and mu=eta=nu=0 (Sec. "Discussion and Limitations").
    """
    remaining = candidates.copy()
    selected: list[dict[str, Any]] = []
    allowed_aspects = allowed_aspects or set()
    covered_set: set[str] = set()
    covered_count: Counter = Counter()
    while remaining and len(selected) < k:
        best_idx = 0
        best_score = -1e18
        for idx, c in enumerate(remaining):
            aspects = set(c.get(aspect_key, []) or [])
            utility = float(c.get(score_key, 0.0))
            new_cov = len(aspects - covered_set) / len(aspects) if aspects else 0.0
            red = 0.0
            if selected and emb_key in c:
                red = max(cosine_vec(c[emb_key], s[emb_key]) for s in selected if emb_key in s)
            noise = 0.0
            if aspects and allowed_aspects:
                noise = len(aspects - allowed_aspects) / max(1, len(aspects))
            repeat = float(np.mean([np.log1p(covered_count[a]) for a in aspects])) if aspects else 0.0
            final = (utility + lambda_coverage * new_cov - mu_redundancy * red
                     - eta_noise * noise - nu_aspect_repeat * repeat)
            if final > best_score:
                best_score = final
                best_idx = idx
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        chosen_aspects = chosen.get(aspect_key, []) or []
        covered_set.update(chosen_aspects)
        covered_count.update(chosen_aspects)
    return selected
