from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .text import normalize_phrase, tokenize, STOP


COMMON_ASPECT_HINTS = {
    'price', 'quality', 'sound', 'battery', 'life', 'screen', 'camera', 'size', 'fit', 'design',
    'material', 'color', 'shipping', 'delivery', 'packaging', 'performance', 'taste', 'smell',
    'durability', 'comfort', 'brand', 'feature', 'value', 'weight', 'volume', 'bass', 'tone',
    'display', 'charger', 'case', 'strap', 'button', 'software', 'app', 'service', 'warranty'
}


_SPACY_NLP = None
_SPACY_LOAD_FAILED = False


def _get_spacy_nlp():
    global _SPACY_NLP, _SPACY_LOAD_FAILED
    if _SPACY_NLP is not None or _SPACY_LOAD_FAILED:
        return _SPACY_NLP
    try:
        import spacy
        _SPACY_NLP = spacy.load('en_core_web_sm', disable=['ner', 'lemmatizer'])
    except Exception:
        _SPACY_LOAD_FAILED = True
    return _SPACY_NLP


def _noun_chunks_spacy(text: str) -> list[str]:
    nlp = _get_spacy_nlp()
    if nlp is None:
        return []
    try:
        doc = nlp(text[:10000])
        chunks = []
        for c in doc.noun_chunks:
            phrase = normalize_phrase(c.text)
            if phrase and 1 <= len(phrase.split()) <= 4:
                chunks.append(phrase)
        return chunks
    except Exception:
        return []


def _noun_phrases_fallback(text: str) -> list[str]:
    toks = [t for t in tokenize(text) if t not in STOP and len(t) > 2]
    phrases: list[str] = []
    for n in (1, 2, 3):
        for i in range(max(0, len(toks) - n + 1)):
            phrase = ' '.join(toks[i:i+n])
            if n == 1 and phrase not in COMMON_ASPECT_HINTS:
                continue
            phrases.append(phrase)
    return phrases


def extract_aspect_phrases(text: str, method: str = 'hybrid', top_n: int | None = None) -> list[str]:
    phrases: list[str] = []
    if method in {'spacy', 'hybrid'}:
        phrases.extend(_noun_chunks_spacy(text))
    if method in {'fallback', 'hybrid'} or not phrases:
        phrases.extend(_noun_phrases_fallback(text))
    cnt = Counter(p for p in phrases if p and len(p) <= 60)
    most = [p for p, _ in cnt.most_common(top_n)] if top_n else list(cnt.keys())
    return most


def canonicalize_aspect(raw: str, vocab: set[str]) -> str:
    """Map one raw aspect phrase onto the saved canonical vocabulary using the same
    rule as normalize_aspects_by_frequency (exact match -> singular -> head noun),
    for aspects extracted after the vocab was already built (e.g. user-history
    sentences scored on the fly at inference time, not part of the original corpus
    pass in extract_aspects.py)."""
    if raw in vocab:
        return raw
    if raw.endswith('s') and raw[:-1] in vocab:
        return raw[:-1]
    head = raw.split()[-1] if raw.split() else raw
    return head if head in vocab else raw


def normalize_aspects_by_frequency(aspects: Iterable[str], min_count: int = 2, max_vocab: int = 5000) -> dict[str, str]:
    """Simple canonicalizer: keep frequent phrases; map singular/plural-like forms."""
    cnt = Counter(a for a in aspects if a)
    vocab = [a for a, c in cnt.most_common(max_vocab) if c >= min_count]
    mapping: dict[str, str] = {}
    for a in cnt:
        if a in vocab:
            mapping[a] = a
        elif a.endswith('s') and a[:-1] in vocab:
            mapping[a] = a[:-1]
        else:
            # fallback to head noun if frequent
            head = a.split()[-1] if a.split() else a
            mapping[a] = head if head in vocab else a
    return mapping


def aspect_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa))


def aspect_f1(pred: Iterable[str], gold: Iterable[str]) -> tuple[float, float, float]:
    pset, gset = set(pred), set(gold)
    if not pset and not gset:
        return 0.0, 0.0, 0.0
    inter = len(pset & gset)
    p = inter / max(1, len(pset))
    r = inter / max(1, len(gset))
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(p), float(r), float(f1)


def build_profile_aspects(sentence_aspects: list[list[str]], max_aspects: int = 50) -> list[str]:
    cnt = Counter()
    for arr in sentence_aspects:
        cnt.update(arr)
    return [a for a, _ in cnt.most_common(max_aspects)]
