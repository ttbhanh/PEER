from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from peer.text import clean_text, tokenize
from peer.utils import normalize_timestamp

PAD, UNK = 0, 1
_MAX_CHARS = 600  # raw-text cap while the pool is still huge; word-level truncation happens later, post-kcore


def load_temporal_safe_pool(raw_dir: str, dataset: str, cutoff_ts: int,
                             hard_cap: int = 20_000_000, seed: int = 42,
                             user_col: str = 'user_id', item_col: str = 'parent_asin') -> list[tuple[str, str, float, str]]:
    """Stream dataset's raw reviews with timestamp < cutoff_ts (same temporal
    boundary as the PEER test split). Loads the whole pool rather than
    reservoir-sampling rows, since review data is sparse enough that a random
    row sample can leave k-core filtering (see kcore_filter) with no
    survivors; hard_cap is a safety valve for pools too large to hold in
    memory. Keeps raw text, not pre-tokenized ids -- tokenization happens
    later (tokenize_records) on the much smaller post-filter survivor set."""
    review_path = Path(raw_dir) / f'{dataset}_reviews.jsonl'
    rng = random.Random(seed)
    pool: list[tuple[str, str, float, str]] = []
    seen = 0
    with open(review_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ts = normalize_timestamp(row.get('timestamp'))
            if ts >= cutoff_ts:
                continue
            text = clean_text(row.get('text'))[:_MAX_CHARS]
            if not text:
                continue
            try:
                rating = float(row.get('rating', 0.0) or 0.0)
            except (TypeError, ValueError):
                rating = 0.0
            rec = (str(row.get(user_col)), str(row.get(item_col)), rating, text)
            seen += 1
            if len(pool) < hard_cap:
                pool.append(rec)
            else:
                j = rng.randint(0, seen - 1)
                if j < hard_cap:
                    pool[j] = rec
    return pool


def tokenize_records(records: list[tuple[str, str, float, str]], review_len: int = 60) -> list[tuple[str, str, float, list[str]]]:
    """Tokenize the (already kcore-filtered and user-capped, so much smaller) raw-text
    records into truncated token lists for RatingCorpus."""
    out = []
    for u, i, rating, text in records:
        toks = tokenize(text)[:review_len]
        if toks:
            out.append((u, i, rating, toks))
    return out


def cap_by_user(records: list[tuple[str, str, float, list[str]]], max_rows: int, seed: int = 42) -> list[tuple[str, str, float, list[str]]]:
    """Cap the pool by dropping whole users (not random rows), to preserve
    the density k-core filtering just established."""
    if len(records) <= max_rows:
        return records
    rng = random.Random(seed)
    users = list({r[0] for r in records})
    rng.shuffle(users)
    kept_users: set[str] = set()
    running = 0
    counts = Counter(r[0] for r in records)
    for u in users:
        if running >= max_rows:
            break
        kept_users.add(u)
        running += counts[u]
    return [r for r in records if r[0] in kept_users]


def cap_entities(records: list, max_users: int | None = None, max_items: int | None = None, seed: int = 42) -> list:
    """Bound distinct users/items to the most-reviewed ones: HRDR's
    rating-pattern MLP takes a raw (n_items,)/(n_users,) row/column as input,
    so an unfiltered large catalog can OOM its first Linear layer."""
    if max_items is not None:
        ic = Counter(r[1] for r in records)
        keep_items = {i for i, _ in ic.most_common(max_items)}
        records = [r for r in records if r[1] in keep_items]
    if max_users is not None:
        uc = Counter(r[0] for r in records)
        keep_users = {u for u, _ in uc.most_common(max_users)}
        records = [r for r in records if r[0] in keep_users]
    return records


def kcore_filter(records: list[tuple[str, str, float, list[str]]], k: int = 5) -> list[tuple[str, str, float, list[str]]]:
    """Standard k-core filtering (iteratively drop users/items with < k
    reviews until stable) -- also keeps HRDR's dense (n_users, n_items)
    rating matrix small enough to materialize."""
    recs = records
    while True:
        uc = Counter(r[0] for r in recs)
        ic = Counter(r[1] for r in recs)
        filtered = [r for r in recs if uc[r[0]] >= k and ic[r[1]] >= k]
        if len(filtered) == len(recs):
            return filtered
        recs = filtered
        if not recs:
            return recs


class RatingCorpus:
    """Indexed corpus for NARRE/HRDR training: per-review word-id sequences, plus
    for every user/item the (padded) list of *other* reviews they wrote / received,
    and which counterpart entity each of those reviews is about (input_reuid /
    input_reiid in the original NARRE code) -- the attention context signal."""

    def __init__(self, records: list[tuple[str, str, float, list[str]]],
                 review_len: int = 60, review_num: int = 10,
                 min_vocab_count: int = 3, max_vocab: int = 20000):
        self.review_len = review_len
        self.review_num = review_num

        token_counts = Counter(t for _, _, _, toks in records for t in toks)
        vocab_terms = [t for t, c in token_counts.most_common(max_vocab) if c >= min_vocab_count]
        self.token2id = {'<pad>': PAD, '<unk>': UNK}
        for t in vocab_terms:
            self.token2id[t] = len(self.token2id)
        self.vocab_size = len(self.token2id)

        self.user2idx: dict[str, int] = {}
        self.item2idx: dict[str, int] = {}
        user_reviews: dict[int, list[int]] = defaultdict(list)
        item_reviews: dict[int, list[int]] = defaultdict(list)
        self.review_words: list[np.ndarray] = []
        self.review_user: list[int] = []
        self.review_item: list[int] = []
        self.samples: list[tuple[int, int, float, int]] = []  # (user_idx, item_idx, rating, review_idx)

        for u, i, rating, toks in records:
            if u not in self.user2idx:
                self.user2idx[u] = len(self.user2idx)
            if i not in self.item2idx:
                self.item2idx[i] = len(self.item2idx)
            u_idx, i_idx = self.user2idx[u], self.item2idx[i]
            word_ids = self._encode(toks)
            r_idx = len(self.review_words)
            self.review_words.append(word_ids)
            self.review_user.append(u_idx)
            self.review_item.append(i_idx)
            user_reviews[u_idx].append(r_idx)
            item_reviews[i_idx].append(r_idx)
            self.samples.append((u_idx, i_idx, rating, r_idx))

        self.user_reviews = user_reviews
        self.item_reviews = item_reviews
        self.n_users = len(self.user2idx)
        self.n_items = len(self.item2idx)

    def _encode(self, toks: list[str]) -> np.ndarray:
        ids = [self.token2id.get(t, UNK) for t in toks[:self.review_len]]
        ids += [PAD] * (self.review_len - len(ids))
        return np.asarray(ids, dtype=np.int64)

    def _padded_review_block(self, review_idxs: list[int], exclude_review: int, counterpart: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Returns (review_num, review_len) word-id block and (review_num,) counterpart-id
        array, excluding the review being predicted (leave-one-out, avoids the model
        trivially reading the target review) and capped/padded to review_num."""
        others = [r for r in review_idxs if r != exclude_review][: self.review_num]
        block = np.zeros((self.review_num, self.review_len), dtype=np.int64)
        ctx = np.zeros((self.review_num,), dtype=np.int64)
        for k, r in enumerate(others):
            block[k] = self.review_words[r]
            ctx[k] = counterpart[r]
        return block, ctx

    def batch(self, indices: list[int]) -> dict[str, np.ndarray]:
        u_blocks, u_ctx, i_blocks, i_ctx, uids, iids, ratings = [], [], [], [], [], [], []
        for idx in indices:
            u_idx, i_idx, rating, r_idx = self.samples[idx]
            ub, uc = self._padded_review_block(self.user_reviews[u_idx], r_idx, self.review_item)
            ib, ic = self._padded_review_block(self.item_reviews[i_idx], r_idx, self.review_user)
            u_blocks.append(ub); u_ctx.append(uc)
            i_blocks.append(ib); i_ctx.append(ic)
            uids.append(u_idx); iids.append(i_idx); ratings.append(rating)
        return {
            'input_u': np.stack(u_blocks), 'input_reuid': np.stack(u_ctx),
            'input_i': np.stack(i_blocks), 'input_reiid': np.stack(i_ctx),
            'uid': np.asarray(uids, dtype=np.int64), 'iid': np.asarray(iids, dtype=np.int64),
            'rating': np.asarray(ratings, dtype=np.float32),
        }

    def encode_review_text(self, toks: list[str]) -> np.ndarray:
        return self._encode(toks)

    def build_rating_matrix(self) -> np.ndarray:
        """Dense (n_users, n_items) rating matrix for HRDR's rating-pattern context
        -- only tractable because the corpus has already been k-core filtered down
        to a dense, bounded entity set (see kcore_filter)."""
        m = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        for u_idx, i_idx, rating, _ in self.samples:
            m[u_idx, i_idx] = rating
        return m
