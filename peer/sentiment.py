from __future__ import annotations

from .text import tokenize

POS_WORDS = set('good great excellent amazing perfect love loved nice easy fast useful comfortable durable cheap affordable best clear bright strong recommend wonderful awesome'.split())
NEG_WORDS = set('bad poor terrible awful hate hated broken difficult slow useless uncomfortable expensive worst blurry weak disappointed problem issue return returned'.split())


def simple_sentiment(text: str) -> float:
    toks = tokenize(text)
    if not toks:
        return 0.0
    pos = sum(t in POS_WORDS for t in toks)
    neg = sum(t in NEG_WORDS for t in toks)
    return (pos - neg) / max(1, pos + neg)


def sentiment_match(a: str, b: str) -> float:
    sa, sb = simple_sentiment(a), simple_sentiment(b)
    if abs(sa) < 0.1 or abs(sb) < 0.1:
        return 0.5
    return 1.0 if sa * sb > 0 else 0.0
