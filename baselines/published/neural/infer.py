from __future__ import annotations

"""Inference-time scoring of PEER candidate reviews using a trained NARRE/HRDR
checkpoint. We only need the attention sub-computation (per-review "usefulness"
weight), not the full rating-prediction head: NARRE's item-side attention context
is the *author* of each candidate review (this matches the official model, where
input_reiid is "which user wrote this review of item i", not the querying user --
personalization instead flows through combining review-attention with the
querying user's own review-based tower at prediction time, which PEER's
sentence-selection task has no use for). HRDR's context is the item's column of
the (frozen) rating matrix, unaffected by which user is asking."""

import io
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F

from peer.text import tokenize
from .narre import NARRE
from .hrdr import HRDR

PAD, UNK = 0, 1


class _CPUUnpickler(pickle.Unpickler):
    """Checkpoints here are saved with plain pickle.dump (not torch.save), so a
    CUDA-trained checkpoint's tensor storages embed a CUDA device tag that
    torch.load(..., map_location=...) does not propagate through the nested
    torch.storage._load_from_bytes call pickle triggers for each tensor. Force
    every such nested load to CPU explicitly."""

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        return super().find_class(module, name)


def _load_checkpoint_cpu(checkpoint_path: str):
    with open(checkpoint_path, 'rb') as f:
        return _CPUUnpickler(f).load()


def _encode(tokens: list[str], token2id: dict[str, int], review_len: int) -> list[int]:
    ids = [token2id.get(t, UNK) for t in tokens[:review_len]]
    ids += [PAD] * (review_len - len(ids))
    return ids


class NarreScorer:
    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        ck = _load_checkpoint_cpu(checkpoint_path)
        self.review_len = ck['review_len']
        self.token2id = ck['token2id']
        self.user2idx = ck['user2idx']
        self.oov_user = ck['n_users']  # reserved slot, see NARRE(n_users+1, ...)
        self.device = torch.device(device)
        self.model = NARRE(ck['vocab_size'], ck['n_users'], ck['n_items']).to(self.device)
        self.model.load_state_dict(ck['state_dict'])
        self.model.eval()

    @torch.no_grad()
    def score_reviews(self, review_texts: list[str], review_authors: list[str]) -> list[float]:
        """One attention weight per review (softmax over the given review set),
        higher = the trained model finds this review more useful/representative
        for this item, exactly NARRE's own review-usefulness signal."""
        if not review_texts:
            return []
        m = self.model
        word_ids = torch.tensor([_encode(tokenize(t), self.token2id, self.review_len) for t in review_texts],
                                 dtype=torch.long, device=self.device).unsqueeze(0)  # (1, R, L)
        author_idx = torch.tensor([self.user2idx.get(a, self.oov_user) for a in review_authors],
                                   dtype=torch.long, device=self.device).unsqueeze(0)  # (1, R)
        b, review_num, review_len = word_ids.shape
        emb = m.word_emb(word_ids.view(review_num, review_len))
        h_pool = m.cnn(emb).view(1, review_num, -1)
        # item-side attention context = the review's author (att_user_id), matching
        # the model's own _side(..., self.att_user_id, ...) call for i_feas/i_att
        ctx = F.relu(m.att_user_id(author_idx))
        att_logit = m.Wpi(F.relu(m.Wai(h_pool) + m.Wru_i(ctx))).squeeze(-1)
        att = F.softmax(att_logit, dim=1).squeeze(0)
        return att.tolist()


class HrdrScorer:
    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        ck = _load_checkpoint_cpu(checkpoint_path)
        self.review_len = ck['review_len']
        self.token2id = ck['token2id']
        self.item2idx = ck['item2idx']
        self.device = torch.device(device)
        self.model = HRDR(ck['vocab_size'], ck['n_users'], ck['n_items']).to(self.device)
        self.model.load_state_dict(ck['state_dict'])
        self.model.eval()
        self.rating_matrix = torch.from_numpy(ck['rating_matrix']).to(self.device)

    @torch.no_grad()
    def score_reviews(self, review_texts: list[str], item_id: str) -> list[float]:
        """Attention weight per review using the item's rating-pattern context
        (HRDR's context is the item, not the review author -- see module docstring)."""
        if not review_texts:
            return []
        m = self.model
        if item_id not in self.item2idx:
            return [1.0 / len(review_texts)] * len(review_texts)  # OOV item: uniform fallback
        i_idx = self.item2idx[item_id]
        word_ids = torch.tensor([_encode(tokenize(t), self.token2id, self.review_len) for t in review_texts],
                                 dtype=torch.long, device=self.device).unsqueeze(0)
        item_col = self.rating_matrix.t()[i_idx].unsqueeze(0)  # (1, n_users)
        tower = m.item_tower
        b, review_num, review_len = word_ids.shape
        emb = m.word_emb(word_ids.view(review_num, review_len))
        r_fea = tower.cnn(emb).view(1, review_num, -1)
        matrix_vec = tower.mlp(item_col)
        query = tower.base_query + matrix_vec
        att_logit = torch.bmm(r_fea, query.unsqueeze(2)).squeeze(-1)
        att = F.softmax(att_logit, dim=1).squeeze(0)
        return att.tolist()
