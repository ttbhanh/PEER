from __future__ import annotations

"""PyTorch port of NARRE (Chen et al., WWW 2018), ported from the official
TensorFlow-1 implementation at github.com/chenchongthu/NARRE (model/NARRE.py).
Kept architecturally faithful: per-review CNN encoder, two-sided attention (each
side's attention query is conditioned on the *counterpart* entity's ID embedding,
softmax-normalized over that user's/item's reviews), weighted-sum pooling, fused
with a separate MF-style ID latent factor, NCF-style interaction, plus user/item/
global bias terms. One deliberate simplification: a single shared word-embedding
table for user- and item-side text (the original uses two independently-sized
tables for what is, in practice, the same English review vocabulary)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ReviewCNN(nn.Module):
    def __init__(self, embedding_size: int, num_filters: int, filter_sizes: tuple[int, ...]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, kernel_size=(fs, embedding_size)) for fs in filter_sizes
        ])

    def forward(self, review_word_emb: torch.Tensor) -> torch.Tensor:
        # review_word_emb: (batch*review_num, review_len, embedding_size)
        x = review_word_emb.unsqueeze(1)  # (N, 1, L, E)
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(x)).squeeze(3)  # (N, num_filters, L-fs+1)
            pooled.append(F.max_pool1d(h, h.size(2)).squeeze(2))  # (N, num_filters)
        return torch.cat(pooled, dim=1)  # (N, num_filters_total)


class NARRE(nn.Module):
    def __init__(self, vocab_size: int, n_users: int, n_items: int,
                 embedding_size: int = 100, embedding_id: int = 32,
                 num_filters: int = 50, filter_sizes: tuple[int, ...] = (2, 3),
                 attention_size: int = 32, n_latent: int = 32, dropout: float = 0.5):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.cnn = _ReviewCNN(embedding_size, num_filters, filter_sizes)
        num_filters_total = num_filters * len(filter_sizes)

        # attention-context ID tables (distinct from the latent-factor tables below,
        # matching the original code's separate uidW/iidW vs uidmf/iidmf)
        self.att_user_id = nn.Embedding(n_users + 1, embedding_id)
        self.att_item_id = nn.Embedding(n_items + 1, embedding_id)

        self.Wau = nn.Linear(num_filters_total, attention_size)
        self.Wru_u = nn.Linear(embedding_id, attention_size, bias=False)
        self.Wpu = nn.Linear(attention_size, 1)
        self.Wai = nn.Linear(num_filters_total, attention_size)
        self.Wru_i = nn.Linear(embedding_id, attention_size, bias=False)
        self.Wpi = nn.Linear(attention_size, 1)

        self.Wu = nn.Linear(num_filters_total, n_latent)
        self.Wi = nn.Linear(num_filters_total, n_latent)
        self.user_mf = nn.Embedding(n_users + 1, n_latent)
        self.item_mf = nn.Embedding(n_items + 1, n_latent)

        self.dropout = nn.Dropout(dropout)
        self.Wmul = nn.Linear(n_latent, 1, bias=False)
        self.user_bias = nn.Embedding(n_users + 1, 1)
        self.item_bias = nn.Embedding(n_items + 1, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def _side(self, review_words, review_ctx_id, att_ctx_table, Wa, Wr, Wp, W_proj, mf_table, entity_id, review_num, review_len):
        b = review_words.size(0)
        emb = self.word_emb(review_words.view(b * review_num, review_len))
        h_pool = self.cnn(emb).view(b, review_num, -1)  # (b, review_num, F)
        ctx = F.relu(att_ctx_table(review_ctx_id))  # (b, review_num, embedding_id)
        att_logit = Wp(F.relu(Wa(h_pool) + Wr(ctx))).squeeze(-1)  # (b, review_num)
        mask = (review_words.abs().sum(dim=-1) != 0).float()  # zero-padded reviews get -inf
        att_logit = att_logit.masked_fill(mask == 0, float('-1e9'))
        att = F.softmax(att_logit, dim=1).unsqueeze(-1)  # (b, review_num, 1)
        feas = (att * h_pool).sum(dim=1)  # (b, F)
        feas = W_proj(feas) + mf_table(entity_id)
        return feas, att.squeeze(-1)

    def forward(self, input_u, input_reuid, input_i, input_reiid, uid, iid):
        review_num_u, review_len_u = input_u.size(1), input_u.size(2)
        review_num_i, review_len_i = input_i.size(1), input_i.size(2)

        u_feas, u_att = self._side(input_u, input_reuid, self.att_item_id, self.Wau, self.Wru_u, self.Wpu,
                                    self.Wu, self.user_mf, uid, review_num_u, review_len_u)
        i_feas, i_att = self._side(input_i, input_reiid, self.att_user_id, self.Wai, self.Wru_i, self.Wpi,
                                    self.Wi, self.item_mf, iid, review_num_i, review_len_i)

        u_feas = self.dropout(u_feas)
        i_feas = self.dropout(i_feas)
        fm = F.relu(u_feas * i_feas)
        fm = self.dropout(fm)
        score = self.Wmul(fm).squeeze(-1)
        pred = score + self.user_bias(uid).squeeze(-1) + self.item_bias(iid).squeeze(-1) + self.global_bias
        return pred, u_att, i_att
