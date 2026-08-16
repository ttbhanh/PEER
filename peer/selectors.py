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


def precision_gate(
    candidates: list[dict[str, Any]],
    k: int,
    score_key: str = 'score',
    percentile: float = 0.0,
    min_pool: int | None = None,
) -> list[dict[str, Any]]:
    """Stage 1 of two-stage selection: drop candidates whose utility falls below
    the `percentile`-th percentile of THIS case's own candidate utility
    distribution (relative, not an absolute cutoff -- utility scale drifts
    across cases/datasets). Coverage/diversity terms in stage 2 (see
    greedy_coverage_select) can then be given more room to reorder *within* an
    already-relevant pool without risking noise, instead of competing against
    the noise term to keep irrelevant candidates out one at a time.

    `percentile=0` is a no-op (keeps everyone) -- callers opt in. Always keeps
    at least `min_pool` (default max(k*3, 10)) candidates even if the gate would
    cut deeper, so a strict gate can't starve stage 2 of choices."""
    if percentile <= 0 or not candidates:
        return candidates
    min_pool = min_pool if min_pool is not None else max(k * 3, 10)
    scores = sorted(float(c.get(score_key, 0.0)) for c in candidates)
    cutoff = scores[int(len(scores) * percentile)] if scores else float('-inf')
    kept = [c for c in candidates if float(c.get(score_key, 0.0)) >= cutoff]
    if len(kept) < min(min_pool, len(candidates)):
        kept = sorted(candidates, key=lambda c: -float(c.get(score_key, 0.0)))[:min_pool]
    return kept


def mmr_select(candidates: list[dict[str, Any]], k: int, score_key: str = 'score', emb_key: str = 'embedding', lambda_rel: float = 0.7) -> list[dict[str, Any]]:
    remaining = candidates.copy()
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < k:
        best_idx = 0
        best_score = -1e18
        for idx, c in enumerate(remaining):
            rel = float(c.get(score_key, 0.0))
            div_penalty = 0.0
            if selected and emb_key in c:
                div_penalty = max(cosine_vec(c[emb_key], s[emb_key]) for s in selected if emb_key in s)
            score = lambda_rel * rel - (1 - lambda_rel) * div_penalty
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(remaining.pop(best_idx))
    return selected


def greedy_coverage_select(
    candidates: list[dict[str, Any]],
    k: int,
    score_key: str = 'score',
    emb_key: str = 'embedding',
    aspect_key: str = 'aspects',
    lambda_coverage: float = 0.2,
    mu_redundancy: float = 0.1,
    eta_noise: float = 0.1,
    nu_aspect_repeat: float = 0.0,
    allowed_aspects: set[str] | None = None,
    aspect_demand: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Greedy per-round argmax of utility + coverage - redundancy - noise -
    aspect_repeat (see module docstring in scripts/select_topk.py for the exact
    formula and where each term's inputs come from).

    Coverage has two modes:
      - default (aspect_demand=None): binary "seen this aspect at all yet" --
        once one sentence about an aspect is picked, later sentences of that
        same aspect get zero coverage credit, capping every aspect's reward at
        a single representative regardless of how salient it is or how large k is.
      - aspect_demand given (a dict aspect -> target sentence count, e.g. from
        peer.aspect_demand.build_demand_from_weights, which already gates *which*
        aspects are demand-eligible and caps *how deep* each one's quota goes):
        a SEER-style per-aspect *quota*, so an aspect worth 2-3 sentences keeps
        earning coverage credit until its quota is met, not just its first hit.

    `nu_aspect_repeat` is a separate, softer anti-redundancy signal on top of
    the demand cap: a log(1+count) penalty per already-selected occurrence of a
    candidate's aspects, so repetition costs a little even for aspects with
    demand still open, rather than being free until the cap hits.
    """
    remaining = candidates.copy()
    selected: list[dict[str, Any]] = []
    allowed_aspects = allowed_aspects or set()
    use_quota = bool(aspect_demand)
    covered_set: set[str] = set()
    covered_count: Counter = Counter()
    while remaining and len(selected) < k:
        best_idx = 0
        best_score = -1e18
        for idx, c in enumerate(remaining):
            aspects = set(c.get(aspect_key, []) or [])
            utility = float(c.get(score_key, 0.0))
            if not aspects:
                new_cov = 0.0
            elif use_quota:
                new_cov = sum(1 for a in aspects if covered_count[a] < aspect_demand.get(a, 0)) / len(aspects)
            else:
                new_cov = len(aspects - covered_set) / len(aspects)
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
        if use_quota:
            covered_count.update(chosen_aspects)
        else:
            covered_set.update(chosen_aspects)
            covered_count.update(chosen_aspects)
    return selected
