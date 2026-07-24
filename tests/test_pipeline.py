import numpy as np

from peer.aspects import aspect_f1, aspect_overlap, extract_aspect_phrases
from peer.metrics import aspect_noise, redundancy, semantic_prf
from peer.selectors import greedy_coverage_select, resolve_k, topk
from peer.sentiment import sentiment_match, simple_sentiment
from peer.aspect_sentiment import build_aspect_sentiment_vector, cosine_sparse


def test_resolve_k():
    assert resolve_k(3, user_avg_k=5, max_k=10) == 3
    assert resolve_k('user_avg', user_avg_k=5, max_k=10) == 5
    assert resolve_k('user_avg', user_avg_k=99, max_k=10) == 10  # capped by pool size
    assert resolve_k(0, user_avg_k=5, max_k=10) == 1  # floor at 1


def test_topk_orders_by_score():
    cands = [{'sentence_id': 'a', 'score': 0.1}, {'sentence_id': 'b', 'score': 0.9}, {'sentence_id': 'c', 'score': 0.5}]
    selected = topk(cands, 2)
    assert [c['sentence_id'] for c in selected] == ['b', 'c']


def test_greedy_coverage_select_prefers_new_aspects():
    # Two candidates score the same on utility; the one covering a new aspect
    # should win once the first pick has already covered 'battery'.
    cands = [
        {'sentence_id': 'a', 'score': 0.5, 'aspects': ['battery']},
        {'sentence_id': 'b', 'score': 0.5, 'aspects': ['battery']},
        {'sentence_id': 'c', 'score': 0.5, 'aspects': ['screen']},
    ]
    selected = greedy_coverage_select(cands, k=2, lambda_coverage=0.5)
    ids = {c['sentence_id'] for c in selected}
    assert 'c' in ids  # the new-aspect candidate must be picked over a duplicate


def test_aspect_f1_perfect_and_disjoint():
    p, r, f1 = aspect_f1(['battery', 'screen'], ['battery', 'screen'])
    assert p == r == 1.0 and abs(f1 - 1.0) < 1e-6
    p, r, f1 = aspect_f1(['battery'], ['screen'])
    assert p == r == f1 == 0.0


def test_aspect_overlap_precision_like():
    assert aspect_overlap(['a', 'b'], ['a']) == 0.5
    assert aspect_overlap([], ['a']) == 0.0


def test_semantic_prf_identical_embeddings_gives_f1_one():
    e = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    p, r, f1 = semantic_prf(e, e)
    assert abs(f1 - 1.0) < 1e-6


def test_aspect_noise_penalizes_out_of_domain_aspects():
    assert aspect_noise(['battery', 'unrelated'], gold_aspects=['battery'], user_aspects=None) == 0.5
    assert aspect_noise(['battery'], gold_aspects=['battery'], user_aspects=None) == 0.0


def test_redundancy_zero_for_orthogonal_selection():
    e = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert redundancy(e) == 0.0


def test_simple_sentiment_and_match():
    assert simple_sentiment('this is great and amazing') > 0
    assert simple_sentiment('this is terrible and broken') < 0
    assert sentiment_match('great product', 'amazing quality') == 1.0
    assert sentiment_match('great product', 'terrible quality') == 0.0


def test_aspect_sentiment_vector_cosine():
    v1 = build_aspect_sentiment_vector([('great battery', ['battery'])])
    v2 = build_aspect_sentiment_vector([('amazing battery', ['battery'])])
    assert cosine_sparse(v1, v2) > 0
    assert cosine_sparse({}, v2) == 0.0


def test_extract_aspect_phrases_fallback_finds_hint_words():
    phrases = extract_aspect_phrases('The battery life and screen quality are great.', method='fallback')
    assert 'battery' in phrases or 'screen' in phrases
