import numpy as np
import pytest

from tripin_ai.evaluation.metrics import auc, bootstrap, ndcg_at_k, pair_correct, top_k


def test_pair_correct_counts_ties_as_half():
    assert pair_correct(0.9, 0.1) == 1.0
    assert pair_correct(0.5, 0.5) == 0.5
    assert pair_correct(0.1, 0.9) == 0.0


def test_auc_by_hand():
    # 음성 4개 중 2개보다 높고 1개와 동점 → (2 + 0.5) / 4
    assert auc(0.5, np.array([0.1, 0.2, 0.5, 0.9])) == pytest.approx(0.625)


def test_ndcg_by_hand():
    # 이득 [0, 2, 1]로 뽑힘, 이상적 순서는 [2, 1, 0]
    ranked = np.array([0.0, 2.0, 1.0])
    dcg = 2 / np.log2(3) + 1 / np.log2(4)
    idcg = 2 / np.log2(2) + 1 / np.log2(3)
    assert ndcg_at_k(ranked, np.array([2.0, 1.0, 0.0, 0.0]), 3) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(np.array([0.0]), np.array([0.0, 0.0]), 1) == 0.0


def test_top_k_breaks_ties_randomly_but_keeps_order():
    scores = np.array([1.0, 3.0, 3.0, 2.0])
    seen = {tuple(top_k(scores, 2, np.random.default_rng(s))) for s in range(20)}
    assert seen == {(1, 2), (2, 1)}
    assert top_k(scores, 3, np.random.default_rng(0))[2] == 3


def test_bootstrap_point_value_and_paired_difference():
    units = {
        "인기": [("a", 1, 2), ("b", 0, 2)],        # 1/4
        "모델": [("a", 2, 2), ("b", 1, 2)],        # 3/4
    }
    result = bootstrap(units, "인기", n_boot=200, seed=0)
    assert result["인기"]["value"] == pytest.approx(0.25)
    assert result["모델"]["value"] == pytest.approx(0.75)
    assert result["모델"]["diff"] == pytest.approx(0.5)
    assert result["모델"]["diff_lo"] > 0   # 두 여행자 모두 모델이 높으므로 모든 재표본에서 양수
    assert result["인기"]["n_travelers"] == 2


def test_popularity_matching_is_exact_for_small_counts_and_relative_for_large():
    from tripin_ai.evaluation.runner import popularity_matched
    assert popularity_matched(3, np.array([2.0, 3.0, 4.0])).tolist() == [False, True, False]
    assert popularity_matched(40, np.array([33.0, 36.0, 40.0, 46.0, 48.0])).tolist() == [False, True, True, True, False]
