import numpy as np
import pandas as pd
import pytest

from tripin_ai.training.triplets import other_taste_negative, survey_distance, survey_profiles


def _travelers(rows):
    cols = ["travel_id"] + [f"style_{i}" for i in range(1, 9)] + [f"motive_{i}" for i in range(1, 4)]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows], dtype=str)


def test_survey_distance_bounds():
    prof = survey_profiles(_travelers([
        ["same", 1, 4, 1, 4, 1, 1, 4, 4, "2", "7", None],
        ["twin", 1, 4, 1, 4, 1, 1, 4, 4, "7", "2", None],
        ["opposite", 7, 4, 7, 4, 7, 7, 4, 4, "8", None, None],
    ]))
    assert survey_distance(prof["same"], prof["twin"]) == 0.0      # 동기 순서는 상관없다
    assert survey_distance(prof["same"], prof["opposite"]) == pytest.approx(1.0)


def test_survey_distance_uses_service_styles_only():
    # 스타일 2·4·7·8은 서비스 온보딩에 없어 거리에 넣지 않는다
    prof = survey_profiles(_travelers([
        ["a", 4, 1, 4, 1, 4, 4, 1, 1, "1", None, None],
        ["b", 4, 7, 4, 7, 4, 4, 7, 7, "1", None, None],
    ]))
    assert survey_distance(prof["a"], prof["b"]) == 0.0


def test_other_taste_negative_picks_most_different_and_skips_visited():
    prof = survey_profiles(_travelers([
        ["me", 1, 4, 1, 4, 1, 1, 4, 4, "2", None, None],
        ["near", 2, 4, 1, 4, 1, 1, 4, 4, "2", None, None],
        ["far", 7, 4, 7, 4, 7, 7, 4, 4, "8", None, None],
    ]))
    liked = {"47130": (np.array(["near", "far", "far", "me"]), np.array([10, 20, 30, 40]))}
    rng = np.random.default_rng(0)
    # 30은 내가 이미 간 곳이라 빠지고, 가장 다른 여행자(far)의 20이 남는다
    assert other_taste_negative("me", 40, "47130", {40, 30}, liked, prof, rng) == 20
    assert other_taste_negative("me", 40, "11110", {40}, liked, prof, rng) is None
