from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .sentiment import simple_sentiment

AspectVector = dict[str, float]


def build_aspect_sentiment_vector(sentences_with_aspects: Iterable[tuple[str, list[str]]]) -> AspectVector:
    """A2SPR-style per-aspect sentiment profile: for each aspect mentioned across a
    set of (text, aspects) pairs, average the sentence-level sentiment of every
    sentence that mentions it. Mirrors Huang et al. 2020's aspect-sentiment vector,
    minus the TF-IDF aspect extraction step (PEER already tags aspects upstream)."""
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for text, aspects in sentences_with_aspects:
        if not aspects:
            continue
        s = simple_sentiment(text)
        for a in aspects:
            sums[a] += s
            counts[a] += 1
    return {a: sums[a] / counts[a] for a in sums}


def cosine_sparse(a: AspectVector, b: AspectVector) -> float:
    """Cosine similarity between two sparse aspect->sentiment vectors."""
    if not a or not b:
        return 0.0
    keys = a.keys() & b.keys()
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    norm_a = sum(v * v for v in a.values()) ** 0.5
    norm_b = sum(v * v for v in b.values()) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
