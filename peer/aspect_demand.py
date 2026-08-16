from __future__ import annotations

"""Per-aspect "how many sentences do we want about this" demand construction:
normalize_demand_to_k rescales an existing demand distribution to sum to k;
build_demand_from_weights builds one from scratch out of importance weights
(used by PEER's own selector, peer/selectors.py)."""

from collections import Counter


def normalize_demand_to_k(demand: dict[str, int], aspect_avail: Counter, k: int) -> dict[str, int]:
    """Rescale `demand` to sum to k, never exceeding available candidates per
    aspect, preserving relative proportions as closely as round-robin allows."""
    demand = {a: min(c, aspect_avail.get(a, 0)) for a, c in demand.items() if aspect_avail.get(a, 0) > 0}
    order = sorted(demand, key=lambda a: -demand[a]) or sorted(aspect_avail, key=lambda a: -aspect_avail[a])
    total = sum(demand.values())
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
    """Build an integer per-aspect quota summing to at most k from importance
    weights, favoring higher-weight aspects but spreading round-robin once
    their quota fills, so one dominant aspect doesn't consume the whole
    budget. `min_weight` gates eligibility; `max_per_aspect` caps how many
    selected sentences may count toward a single aspect."""
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
