#!/usr/bin/env python
from __future__ import annotations

"""Reproduces the paper's zero-shot LLM case study (Sec. "Case Study: PEER vs.
General-Purpose LLMs"): one case, a fixed candidate pool, a fixed prompt
template, run independently against each LLM in a normal chat UI (no
programmatic API calls -- this keeps the protocol identical to how a user
would actually interact with each provider, and avoids pinning the release to
one LLM vendor's SDK).

  export  Given --case-id, write the filled-in prompt (history + numbered,
          ID-tagged candidate pool + instructions) to --output. Paste this
          into any LLM chat interface.
  score   Given --case-id and --response (a JSON array of chosen sentence IDs,
          as a file path or inline string), compute sem-F1 against the
          held-out review, the same metric reported in the paper's table.
"""

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import numpy as np

from peer.metrics import semantic_prf
from peer.selectors import resolve_k
from peer.utils import ensure_dir, read_jsonl

PROMPT_TEMPLATE = """You are helping a review site pick which existing review sentences to show a
specific shopper as "evidence" for a product, before that shopper has written
their own review. You will NOT see the shopper's actual future review -- your
job is to anticipate what would be most relevant to them, based only on their
own past review history and the candidate sentences below.

USER'S OWN PAST REVIEW SENTENCES (their history on this platform, across other products):
{HISTORY}

CANDIDATE SENTENCES (from other users' reviews of THIS product -- pick from this list only):
{CANDIDATES}

Select exactly {K} sentence(s) from the CANDIDATE list above that best
anticipate what this specific user would care about or write, based on their
own history. Copy the bracketed ID exactly as shown, e.g. [{EXAMPLE_ID}].

Respond with ONLY a JSON array of the {K} chosen IDs, nothing else, e.g.
["{EXAMPLE_ID}"]
"""


def find_case(cases_dir: str, case_id: str) -> dict:
    for split in ['train', 'valid', 'test']:
        p = Path(cases_dir) / f'cases_{split}.jsonl'
        if not p.exists():
            continue
        for c in read_jsonl(p):
            if c['case_id'] == case_id:
                return c
    raise ValueError(f'case_id {case_id} not found under {cases_dir}/cases_*.jsonl')


def load_embeddings(path: str):
    p = Path(path)
    emb = np.load(p.with_suffix('.npy'), mmap_mode='r')
    with open(p.with_suffix('.ids.json')) as f:
        ids = json.load(f)
    return {str(ids[i]): emb[i] for i in range(len(ids))}


def cmd_export(args):
    case = find_case(args.cases, args.case_id)
    k = resolve_k(args.k, int(case.get('user_avg_k', 3)), len(case['candidate_sentences']))
    history = '\n'.join(case['user_history_sentences'])
    candidates = case['candidate_sentences']
    cand_lines = '\n'.join(f"[{c['sentence_id']}] {c['text']}" for c in candidates)
    example_id = candidates[0]['sentence_id'] if candidates else 'sentence_id'
    prompt = PROMPT_TEMPLATE.format(HISTORY=history, CANDIDATES=cand_lines, K=k, EXAMPLE_ID=example_id)
    ensure_dir(Path(args.output).parent)
    Path(args.output).write_text(prompt, encoding='utf-8')
    print(f'Case {args.case_id}: {len(candidates)} candidates, k={k}')
    print(f'Wrote prompt -> {args.output}')


def cmd_score(args):
    case = find_case(args.cases, args.case_id)
    response = args.response
    if Path(response).exists():
        response = Path(response).read_text(encoding='utf-8')
    selected_ids = json.loads(response)
    if not isinstance(selected_ids, list):
        raise ValueError('--response must be a JSON array of sentence IDs')

    embs = load_embeddings(args.embeddings)
    cand_by_id = {c['sentence_id']: c for c in case['candidate_sentences']}
    unknown = [sid for sid in selected_ids if sid not in cand_by_id]
    if unknown:
        print(f'WARNING: {len(unknown)} returned ID(s) not in the candidate pool (hallucinated): {unknown}')

    e_emb = np.asarray([embs[sid] for sid in selected_ids if sid in embs], dtype=np.float32)
    gt_ids = [f"{case['case_id']}_gt_{i}" for i in range(len(case['ground_truth_sentences']))]
    g_emb = np.asarray([embs[sid] for sid in gt_ids if sid in embs], dtype=np.float32)
    sem_p, sem_r, sem_f1 = semantic_prf(e_emb, g_emb) if len(e_emb) and len(g_emb) else (0.0, 0.0, 0.0)

    print(f'Case {args.case_id}: {len(selected_ids)} selected, {len(unknown)} unknown/hallucinated ID(s)')
    print(f'sem_p={sem_p:.4f} sem_r={sem_r:.4f} sem_f1={sem_f1:.4f}')
    print('Selected text:')
    for sid in selected_ids:
        if sid in cand_by_id:
            print(f'  [{sid}] {cand_by_id[sid]["text"]}')
    print(f'\nGround truth: {case["ground_truth_text"]}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='mode', required=True)

    p_export = sub.add_parser('export', help='Write the filled-in prompt for a case')
    p_export.add_argument('--case-id', required=True)
    p_export.add_argument('--cases', default='data/cases')
    p_export.add_argument('--k', default='user_avg')
    p_export.add_argument('--output', default='outputs/case_studies/prompt.txt')
    p_export.set_defaults(func=cmd_export)

    p_score = sub.add_parser('score', help="Score an LLM's returned selection against ground truth")
    p_score.add_argument('--case-id', required=True)
    p_score.add_argument('--cases', default='data/cases')
    p_score.add_argument('--embeddings', default='embeddings/embeddings.npz')
    p_score.add_argument('--response', required=True,
                          help='JSON array of selected sentence IDs, as a file path or inline string')
    p_score.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
