import numpy as np
import pytest

from tripin_ai.encoder import from_base64, to_base64
from tripin_ai.templates import Attraction, TravelerSurvey, attraction_text, region_from_address, traveler_text


def test_attraction_v0_has_name_classes_region_without_description():
    item = Attraction(name="성산일출봉", class_names=["자연관광", "자연경관(산)", "산"], region_name="제주특별자치도 서귀포시",
                      description="약 5천 년 전 바닷속 화산 폭발로 생긴 봉우리")
    assert attraction_text(item, "v0") == "성산일출봉. 자연관광 > 자연경관(산) > 산. 제주특별자치도 서귀포시."


def test_attraction_v1_appends_description_prefix():
    item = Attraction(name="성산일출봉", description="  약 5천 년 전\n화산 폭발  " + "가" * 400)
    text = attraction_text(item, "v1")
    assert text.startswith("성산일출봉. 약 5천 년 전 화산 폭발 ")
    assert len(text) <= len("성산일출봉. ") + 300


def test_attraction_classes_are_deduplicated():
    item = Attraction(name="장터", class_names=["쇼핑", "시장", "시장"])
    assert attraction_text(item) == "장터. 쇼핑 > 시장."


def test_traveler_text_uses_poles_strength_and_skips_neutral():
    survey = TravelerSurvey(styles={1: 1, 3: 4, 5: 6, 6: 3, 2: 1, 8: 7}, motives=[2, 10, 7],
                            liked_regions=["강원도 강릉시", "강원도 강릉시", "제주특별자치도 제주시"])
    assert traveler_text(survey) == (
        "자연을 매우 선호, 체험 활동을 꽤 선호, 잘 알려지지 않은 곳을 약간 선호하는 여행자. "
        "여행에서 원하는 것은 휴식과 재충전, 새로운 경험. "
        "좋아하는 여행지는 강원도 강릉시, 제주특별자치도 제주시."
    )


def test_traveler_text_particle_follows_final_consonant():
    assert traveler_text(TravelerSurvey(styles={1: 7})) == "도시를 매우 선호하는 여행자."


def test_unknown_template_version_raises():
    with pytest.raises(ValueError):
        attraction_text(Attraction(name="x"), "v9")
    with pytest.raises(ValueError):
        traveler_text(TravelerSurvey(), "v9")


@pytest.mark.parametrize("address, expected", [
    ("서울특별시 종로구 북촌로 57 (가회동)", "서울특별시 종로구"),
    ("경기도 수원시 팔달구 창룡대로 103", "경기도 수원시 팔달구"),
    ("세종특별자치시 조치원읍 대첩로 1", "세종특별자치시"),
    ("", ""),
])
def test_region_from_address(address, expected):
    assert region_from_address(address) == expected


def test_base64_round_trip_is_float32_little_endian():
    vector = np.array([0.5, -1.25, 3.0], dtype=np.float32)
    encoded = to_base64(vector)
    assert encoded == "AAAAPwAAoL8AAEBA"
    assert np.array_equal(from_base64(encoded), vector)
