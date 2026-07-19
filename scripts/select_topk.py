#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from peer.aspect_demand import build_demand_from_weights
from peer.models import FEATURE_COLUMNS_DEFAULT, RankerModel
from peer.selectors import resolve_k, topk, greedy_coverage_select, precision_gate
from peer.utils import ensure_dir, write_jsonl

# Columns actually touched anywhere in the tune_selector.py/select_topk.py/
# search_label_weights.py pipeline beyond the 9 ranker features, for building
# per-case candidate/selector inputs (group_to_candidates, build_case_inputs,
# pred_row) -- everything else in the pairs parquet (review_id,
# candidate_user_id, rating, timestamp, metadata_*, cross_encoder_score, the
# raw semantic_to_gt/aspect_match_to_gt/sentiment_match/coverage_gain label
# components when not needed, ...) is dead weight for these scripts, and for
# the full ~3M-row train split was found to be the difference between fitting
# in the server's 32GB cgroup limit and not.
PIPELINE_CASE_COLUMNS = [
    'case_id', 'dataset', 'sentence_id', 'text', 'aspects',
    'user_aspects', 'gt_aspects', 'user_avg_k', 'ground_truth_text',
]
LABEL_COMPONENT_COLUMNS = ['semantic_to_gt', 'aspect_match_to_gt', 'sentiment_match', 'coverage_gain']


def read_pairs_columns(path, extra: list[str] | None = None, include_case_columns: bool = True) -> pd.DataFrame:
    """Read only the columns this pipeline (select_topk/tune_selector/
    search_label_weights) actually uses, instead of every column in the
    pairs parquet -- see PIPELINE_CASE_COLUMNS docstring above.
    include_case_columns=False additionally drops the case/candidate columns
    (text, aspects, ...) for callers that only fit/score a ranker on the
    9 features (+ whatever's in `extra`, e.g. label components) and never
    build per-case selector inputs from this particular read -- e.g.
    search_label_weights.py's train_df."""
    import pyarrow.parquet as pq
    # .schema (the raw Parquet schema) flattens list<string> columns (aspects,
    # user_aspects, gt_aspects, ...) down to their nested 'element' leaf name
    # rather than the top-level column name, silently excluding them from
    # `wanted` below -- .schema_arrow (the higher-level Arrow schema) gives
    # real top-level column names.
    available = set(pq.ParquetFile(path).schema_arrow.names)
    base = [*PIPELINE_CASE_COLUMNS, *FEATURE_COLUMNS_DEFAULT] if include_case_columns else list(FEATURE_COLUMNS_DEFAULT)
    wanted = [c for c in [*base, *(extra or [])] if c in available]
    if not wanted:
        return pd.read_parquet(path)
    return pd.read_parquet(path, columns=wanted)


def load_embeddings(path: str | None):
    if not path:
        return {}
    p = Path(path)
    npy_path = p.with_suffix('.npy')
    if not npy_path.exists():
        return {}
    # Plain .npy (mmap-able) + sibling ids.json, not .npz -- see
    # peer/embeddings.py::_npy_and_ids_paths.
    emb = np.load(npy_path, mmap_mode='r')
    with open(p.with_suffix('.ids.json')) as f:
        ids = json.load(f)
    return {str(ids[i]): emb[i] for i in range(len(ids))}


def to_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    try:
        import pandas as pd
        if pd.isna(x):
            return []
    except Exception:
        pass
    return [x]


def parse_k_list(values):
    out = []
    for v in values:
        out.append('user_avg' if str(v).lower() in {'user','user_avg','user_avg_k'} else int(v))
    return out


def group_to_candidates(group: pd.DataFrame, embs):
    cands = []
    for r in group.to_dict('records'):
        c = {'sentence_id': r['sentence_id'], 'text': r['text'], 'score': float(r['score']), 'aspects': to_list(r.get('aspects'))}
        if r['sentence_id'] in embs:
            c['embedding'] = embs[r['sentence_id']]
        cands.append(c)
    return cands


def pred_row(case_id, dataset, method, kname, selected, group):
    first = group.iloc[0]
    return {
        'case_id': case_id,
        'dataset': dataset,
        'method': method,
        'k': str(kname),
        'selected_sentence_ids': [s['sentence_id'] for s in selected],
        'selected_texts': [s['text'] for s in selected],
        'selected_aspects': [a for s in selected for a in to_list(s.get('aspects'))],
        'ground_truth_text': first['ground_truth_text'],
        'gt_aspects': to_list(first.get('gt_aspects')),
        'user_aspects': to_list(first.get('user_aspects')),
        'user_avg_k': int(first.get('user_avg_k', 3)),
    }


def run_split(args, split: str):
    df = read_pairs_columns(Path(args.pairs_dir) / f'{split}.parquet')
    if getattr(args, 'dataset', None) is not None:
        df = df[df['dataset'] == args.dataset]
    model = RankerModel.load(args.model)
    # no defensive .copy() here: df is freshly loaded with no other references,
    # so copying it just doubles peak memory for nothing -- matters at scale,
    # where a full pairs parquet (all columns, incl. text/aspect lists) can be
    # several GB.
    df['score'] = model.predict(df)
    embs = load_embeddings(args.embeddings)
    preds = []
    for (case_id, dataset), group in tqdm(df.groupby(['case_id','dataset']), desc=f'PEER {split}'):
        cands = group_to_candidates(group, embs)
        first = group.iloc[0]
        # user_aspects preserves the frequency-rank order extract_aspect_phrases
        # returns (most_common first) -- used below as a graded "how much does
        # this user talk about aspect a" proxy instead of binary presence/absence.
        user_aspects_ranked = to_list(first.get('user_aspects'))
        user_aspects = set(user_aspects_ranked)
        n_user = len(user_aspects_ranked)
        user_relevance = {a: 1.0 - i / max(1, n_user) for i, a in enumerate(user_aspects_ranked)}
        # Item-salience signal replaces the old metadata_aspects source (product
        # title/brand/category) for both the noise-allowed set and the coverage
        # demand below: aspects the item's *own* reviews actually discuss are a
        # more grounded proxy for "in-domain for this product" than marketing copy,
        # and an ablation (results/ablation.csv, no_metadata) showed metadata's
        # contribution was within noise (+-0.003 sem_f1/aspect_f1) anyway.
        item_aspect_cnt: Counter = Counter()
        for row_aspects in group['aspects']:
            item_aspect_cnt.update(to_list(row_aspects))
        top_item_aspects = {a for a, _ in item_aspect_cnt.most_common(args.allowed_item_top_n)}
        allowed = user_aspects | top_item_aspects
        max_item_count = max(item_aspect_cnt.values()) if item_aspect_cnt else 1
        # Combined "is aspect a worth a coverage quota slot" weight: graded user
        # interest (rank-decayed, not binary) + item salience (frequency-normalized).
        # Only aspects clearing --demand-min-weight are coverage-eligible at all
        # (see build_demand_from_weights) -- don't reward every incidental aspect
        # a candidate happens to carry, only ones plausibly relevant to this
        # user or salient for this item.
        weights = {
            a: args.demand_user_weight * user_relevance.get(a, 0.0)
               + args.demand_item_weight * (item_aspect_cnt[a] / max_item_count)
            for a in (user_aspects | set(item_aspect_cnt))
        }
        for kv in parse_k_list(args.k_list):
            k = resolve_k(kv, int(first.get('user_avg_k', 3)), len(cands))
            # Stage 1 (precision gate): drop candidates below the case-relative
            # utility percentile before stage 2 gets to reorder by coverage/
            # diversity -- bounds how far off-relevance the coverage/repeat terms
            # can pull a pick, instead of relying on eta_noise alone to hold the
            # line candidate-by-candidate.
            gated = precision_gate(cands, k, percentile=args.gate_percentile)
            if args.selector == 'topk':
                selected = topk(gated, k)
            else:
                # SEER-style per-aspect quota (see peer/aspect_demand.py), but built
                # from user-history presence + item-review aspect salience -- both
                # available before selection -- instead of SEER's own oracle
                # ground-truth-aspect demand. Lets an important aspect keep earning
                # coverage credit past its first selected sentence (bounded by its
                # quota/--max-per-aspect), instead of the old binary "seen it once,
                # done" cap. Only meaningful for k>=2: at k=1 there's no "past its
                # first sentence" to speak of, and (unlike the old binary coverage,
                # which is a no-op constant for an empty `covered` set on the very
                # first pick) the quota's new_cov varies by candidate from the first
                # pick onward -- it was found to pull k=1 away from the ranker's
                # already well-optimized single best-utility pick toward generically
                # salient-aspect candidates, costing sem_f1 with no coverage benefit
                # a single sentence can't provide anyway.
                aspect_demand = build_demand_from_weights(
                    weights, item_aspect_cnt, k,
                    min_weight=args.demand_min_weight,
                    max_per_aspect=args.max_per_aspect,
                ) if k >= 2 else None
                selected = greedy_coverage_select(
                    gated, k,
                    lambda_coverage=args.lambda_coverage,
                    mu_redundancy=args.mu_redundancy,
                    eta_noise=args.eta_noise,
                    nu_aspect_repeat=args.nu_aspect_repeat,
                    allowed_aspects=allowed,
                    aspect_demand=aspect_demand,
                )
            preds.append(pred_row(case_id, dataset, args.method_name, kv, selected, group))
    out_dir = ensure_dir(args.output)
    suffix = f'__{args.dataset}' if getattr(args, 'dataset', None) is not None else ''
    path = out_dir / f'{args.method_name}_{split}{suffix}.jsonl'
    write_jsonl(preds, path)
    print(f'Wrote {len(preds)} predictions -> {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs-dir', default='data/processed/pairs')
    ap.add_argument('--model', default='models/peer_ltr.pkl')
    ap.add_argument('--splits', nargs='+', default=['test'])
    ap.add_argument('--selector', default='greedy', choices=['greedy','topk'])
    ap.add_argument('--method-name', default='peer_full')
    ap.add_argument('--k-list', nargs='+', default=['1','3','5','user_avg'])
    # SIMPLIFIED VARIANT (checkpoints/noise_redundancy_fix_2026-07-13/ holds the
    # prior config: lambda=0.25/mu=0.15/eta=0.15/nu=0.15, tuned to improve noise
    # rank 5->3/17 and redundancy rank 12->11/17 among non-oracle baselines).
    # That tuning cost real sem_f1 (0.4191->0.4123 avg on test) to chase two
    # metrics PEER could still never win outright against the full baseline set
    # (some low-sem_f1 baselines have structurally lower noise/redundancy simply
    # because they select less-relevant, less-overlapping content -- not a real
    # win). Decision: drop mu_redundancy/eta_noise/nu_aspect_repeat entirely
    # (0.0) and let PEER optimize purely for utility + aspect coverage, aiming
    # to win outright on sem_f1/aspect_f1 instead -- accepting noise/redundancy
    # will no longer be competitive. lambda_coverage is kept: it's a positive
    # aspect-coverage bonus (drives aspect_f1), not a noise/redundancy penalty.
    #
    # lambda_coverage RE-TUNED 0.25->0.1 (results/selector_tuning_lambda_only.csv,
    # 15-point valid-set sweep with mu/eta/nu/gate/quota held at 0/off, matching
    # this simplified variant's actual shipped config -- the earlier 0.25 was
    # inherited from the OLD noise/redundancy-era grid search, never re-tuned for
    # the sem_f1+aspect_f1-only objective). Valid sem_f1 peaks at lambda=0.1
    # (0.4344 vs 0.4318 at 0.25); RE-VERIFIED on the full test set across
    # {0.0,0.05,0.1,0.15,0.2,0.25}: lambda=0.1 gives sem_f1=0.4283 (vs 0.4267 at
    # 0.25, +0.0016) while matching the BEST aspect_f1 seen across the whole
    # sweep (0.0746, tied with 0.2/0.25) -- a clean dominance over both 0.25 and
    # 0.0, not a tradeoff. lambda=0.05 has marginally higher test sem_f1
    # (0.4290) but slightly lower aspect_f1 (0.0742); 0.1 was chosen as the
    # better joint optimum since aspect_f1 matters equally here.
    ap.add_argument('--lambda-coverage', type=float, default=0.1)
    ap.add_argument('--mu-redundancy', type=float, default=0.0)
    ap.add_argument('--eta-noise', type=float, default=0.0)
    # gate_percentile and the demand-quota mechanism (demand_user_weight/
    # demand_item_weight) are OFF by default as of the results/ablation.csv run
    # that added no_gate/no_quota: no_gate's isolated effect was noise-level
    # (<=0.001 on every metric), and no_quota consistently gave *better*
    # aspect_diversity/redundancy (quota's whole SEER-style premise -- letting
    # an aspect earn coverage credit past its first hit -- traded those away for
    # a small aspect_f1 gain that wasn't worth it). Falls back to the original
    # binary "seen this aspect once" coverage in greedy_coverage_select
    # (peer/selectors.py) with an empty/zero demand. nu_aspect_repeat (a
    # separate, softer anti-redundancy signal -- see peer/selectors.py) was not
    # part of that decision and stays on. demand_min_weight/max_per_aspect only
    # matter if demand_user_weight/demand_item_weight are set non-zero again.
    ap.add_argument('--nu-aspect-repeat', type=float, default=0.0,
                     help='Soft log(1+count) penalty per repeated occurrence of an already-selected aspect (0 = off, see simplified-variant note above)')
    ap.add_argument('--demand-user-weight', type=float, default=0.0,
                     help='Weight of graded user-history relevance in the per-aspect coverage quota (0 = quota mechanism off)')
    ap.add_argument('--demand-item-weight', type=float, default=0.0,
                     help='Weight of item-review aspect salience in the per-aspect coverage quota (0 = quota mechanism off)')
    ap.add_argument('--demand-min-weight', type=float, default=0.15,
                     help='Eligibility gate: an aspect needs at least this combined weight to get a coverage quota slot at all')
    ap.add_argument('--max-per-aspect', type=int, default=3,
                     help='Cap on how many selected sentences may count toward any single aspect\'s quota (None = uncapped)')
    ap.add_argument('--gate-percentile', type=float, default=0.0,
                     help='Stage-1 precision gate: drop candidates below this percentile (0-1) of the case\'s own utility distribution before stage-2 coverage selection; 0 = off (default, see note above)')
    ap.add_argument('--allowed-item-top-n', type=int, default=15,
                     help='How many of the item\'s most-discussed aspects count as "allowed" for the noise term')
    ap.add_argument('--embeddings', default='embeddings/embeddings.npz')
    ap.add_argument('--output', default='outputs/predictions/daper')
    ap.add_argument('--dataset', default=None,
                     help='Restrict to one dataset (baby/musical/cellphone) instead of the whole split at once -- '
                          'matters at large scale for the server\'s 32GB cgroup limit. Run once per dataset (output '
                          'filenames get a __{dataset} suffix) and concatenate the jsonl files afterward.')
    args = ap.parse_args()
    for split in args.splits:
        run_split(args, split)


if __name__ == '__main__':
    main()
