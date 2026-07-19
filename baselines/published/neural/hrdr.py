from __future__ import annotations

"""PyTorch port of HRDR (Liu et al., Neurocomputing 2020), ported from the
official implementation at github.com/cspjiao/HDRL (methods/multi_view.py,
models/models.py -- both already PyTorch, so this is a direct architectural port
rather than a framework translation). Two encoder towers (user, item), each: (1) a
rating-pattern context vector from an MLP over that entity's row/column of the
user x item rating matrix (frozen, not trained -- matches the original's
`requires_grad = False`), (2) a CNN review-text encoder, (3) attention over that
entity's reviews using a query built from a learned base vector *plus* the
rating-pattern context (the paper's "attention guided by rating patterns"), (4)
review feature fused with the rating-pattern context. The two towers are then
merged and passed through a small prediction head (not in the released repo
excerpt available to us; implemented here in the same NCF+bias spirit as NARRE's
own head, documented as our own completion of that missing piece)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .narre import _ReviewCNN


class _HRDRTower(nn.Module):
    def __init__(self, context_dim: int, embedding_size: int, r_filters_num: int,
                 filter_sizes: tuple[int, ...], dropout: float):
        super().__init__()
        num_filters = r_filters_num // len(filter_sizes)
        assert num_filters * len(filter_sizes) == r_filters_num, 'r_filters_num must be divisible by len(filter_sizes)'
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, max(context_dim // 2, 1)),
            nn.ReLU(),
            nn.Linear(max(context_dim // 2, 1), max(context_dim // 4, 1)),
            nn.ReLU(),
            nn.Linear(max(context_dim // 4, 1), r_filters_num),
            nn.LayerNorm(r_filters_num),
        )
        self.cnn = _ReviewCNN(embedding_size, num_filters, filter_sizes)
        self.base_query = nn.Parameter(torch.randn(1, r_filters_num) * 0.1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, word_emb_fn, review_words, rating_row):
        b, review_num, review_len = review_words.shape
        emb = word_emb_fn(review_words.view(b * review_num, review_len))
        r_fea = self.cnn(emb).view(b, review_num, -1)  # (b, review_num, r_filters_num)

        matrix_vec = self.mlp(rating_row)  # (b, r_filters_num) -- rating-pattern context
        query = self.base_query + matrix_vec  # (b, r_filters_num)

        att_logit = torch.bmm(r_fea, query.unsqueeze(2)).squeeze(-1)  # (b, review_num)
        mask = (review_words.abs().sum(dim=-1) != 0).float()
        att_logit = att_logit.masked_fill(mask == 0, float('-1e9'))
        att = F.softmax(att_logit, dim=1).unsqueeze(-1)
        r_fea_agg = (F.relu(r_fea) * att).sum(dim=1)  # (b, r_filters_num)

        all_feature = r_fea_agg + matrix_vec  # 'add' merge mode
        return self.dropout(all_feature)


class HRDR(nn.Module):
    def __init__(self, vocab_size: int, n_users: int, n_items: int,
                 embedding_size: int = 100, r_filters_num: int = 64,
                 filter_sizes: tuple[int, ...] = (2, 4), dropout: float = 0.5):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.user_tower = _HRDRTower(n_items, embedding_size, r_filters_num, filter_sizes, dropout)
        self.item_tower = _HRDRTower(n_users, embedding_size, r_filters_num, filter_sizes, dropout)
        self.predict = nn.Linear(r_filters_num, 1)
        self.user_bias = nn.Embedding(n_users + 1, 1)
        self.item_bias = nn.Embedding(n_items + 1, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, input_u, input_i, uid, iid, rating_matrix):
        user_row = rating_matrix[uid]          # (b, n_items)
        item_col = rating_matrix.t()[iid]       # (b, n_users)
        u_feat = self.user_tower(self.word_emb, input_u, user_row)
        i_feat = self.item_tower(self.word_emb, input_i, item_col)
        ui = F.relu(u_feat * i_feat)
        score = self.predict(ui).squeeze(-1)
        pred = score + self.user_bias(uid).squeeze(-1) + self.item_bias(iid).squeeze(-1) + self.global_bias
        return pred
