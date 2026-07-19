from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+')
STOP = set('''a an and are as at be by for from has have i in is it its of on or that the this to was were with you your we they them he she but if then so very really just my our their product item one two about after before into out more most much many some any not no can could would should will'''.split())


def clean_text(text: object) -> str:
    if text is None:
        return ''
    s = str(text)
    s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def split_sentences(text: object, min_words: int = 3, max_words: int = 80) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    out: list[str] = []
    for p in parts:
        p = clean_text(p)
        words = tokenize(p)
        if min_words <= len(words) <= max_words:
            out.append(p)
    return out


def tokenize(text: object) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(clean_text(text))]


def normalize_phrase(text: object) -> str:
    toks = [t for t in tokenize(text) if t not in STOP and len(t) > 1]
    return ' '.join(toks[:5])


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def compact_profile(sentences: list[str], max_chars: int = 2048) -> str:
    text = ' '.join(clean_text(s) for s in sentences if clean_text(s))
    return text[:max_chars]
