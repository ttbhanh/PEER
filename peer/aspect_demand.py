from __future__ import annotations

"""Shared per-aspect "how many sentences do we want about this" demand construction.

Two builders share one normalization routine:
  - `normalize_demand_to_k` rescales an *existing* demand distribution (e.g.
    SEER's own aspect multiset, see baselines/published/seer_algo.py) to sum to
    a fixed evaluation k.
  - `build_demand_from_weights` builds a demand distribution from scratch out of
    non-oracle importance weights (used by PEER's own selector, peer/selectors.py,
    to get a SEER-style per-aspect quota without reading the ground truth).
"""

from collections import Counter


def normalize_demand_to_k(demand: dict[str, int], aspect_avail: Counter, k: int) -> dict[str, int]:
    """Rescale `demand` to sum to k while never exceeding how many candidate
    sentences of that aspect actually exist (`aspect_avail`), preserving demand's
    relative aspect proportions as closely as an integer round-robin allows."""
    demand = {a: min(c, aspect_avail.get(a, 0)) for a, c in demand.items() if aspect_avail.get(a, 0) > 0}
    order = sorted(demand, key=lambda a: -demand[a]) or sorted(aspect_avail, key=lambda a: -aspect_avail[a])
    total = sum(demand.values())
    # pad: round-robin add to the (originally) most-demanded aspects first
    while total < k:
        progressed = False
        for a in order:
            cur = demand.get(a, 0)
            if cur < aspect_avail.get(a, 0):
                demand[a] = cur + 1
                total += 1
                progressed = True
                if total >= k:
                    break
        if not progressed:
            break
    # if still short, expand to any other aspect present among candidates
    if total < k:
        for a, avail in aspect_avail.most_common():
            if a in demand and demand[a] >= avail:
                continue
            cur = demand.get(a, 0)
            add = min(avail - cur, k - total)
            if add > 0:
                demand[a] = cur + add
                total += add
            if total >= k:
                break
    # trim: remove from the least-(originally)-demanded aspect first
    trim_order = sorted(demand, key=lambda a: (demand[a], -aspect_avail.get(a, 0)))
    i = 0
    while total > k and trim_order:
        a = trim_order[i % len(trim_order)]
        if demand.get(a, 0) > 0:
            demand[a] -= 1
            total -= 1
            if demand[a] == 0:
                del demand[a]
                trim_order = [x for x in trim_order if x != a]
                i = 0
                continue
        i += 1
    return {a: c for a, c in demand.items() if c > 0}


def build_demand_from_weights(
    weights: dict[str, float],
    aspect_avail: Counter,
    k: int,
    min_weight: float = 0.0,
    max_per_aspect: int | None = None,
) -> dict[str, int]:
    """Build an integer per-aspect quota summing to (at most) k directly from
    importance weights, favoring higher-weight aspects but spreading round-robin
    across the ranked list once their quota is filled -- so a single dominant
    aspect doesn't consume the whole budget when k is large enough to cover
    several. Never exceeds how many candidate sentences of that aspect exist.

    `min_weight` is an eligibility gate: an aspect only gets a quota slot at all
    if its combined (user-relevance, item-salience) weight clears this bar --
    "don't reward every new aspect, only ones the user plausibly cares about or
    the item is plausibly about." `max_per_aspect` caps how deep any single
    aspect's quota can go, independent of how much weight/availability it has --
    the guard against 3 selected sentences all being the same (very salient)
    aspect, which is depth SEER's own paper-derived demand doesn't cap either but
    hurts diversity/redundancy for a fixed small k."""
    candidates = [a for a in weights if weights[a] >= min_weight and aspect_avail.get(a, 0) > 0]
    if not candidates or k <= 0:
        return {}
    cap = {a: min(aspect_avail[a], max_per_aspect) if max_per_aspect is not None else aspect_avail[a] for a in candidates}
    order = sorted(candidates, key=lambda a: -weights[a])
    demand: dict[str, int] = {a: 0 for a in order}
    total = 0
    while total < k:
        progressed = False
        for a in order:
            if demand[a] < cap[a]:
                demand[a] += 1
                total += 1
                progressed = True
                if total >= k:
                    break
        if not progressed:
            break
    return {a: c for a, c in demand.items() if c > 0}
