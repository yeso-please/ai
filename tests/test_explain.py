"""코스 소개·추천 이유 생성 (#7): 검증 규칙, 규칙 문장, 개인정보, 대체 흐름, API."""
import numpy as np
import pytest

from tripin_ai.explain.course_index import CourseExample
from tripin_ai.explain.fallback import personal_reason, rule_reason, rule_title
from tripin_ai.explain.llm import LLMUnavailable
from tripin_ai.explain.prompts import Place, build_user_prompt
from tripin_ai.explain.service import explain
from tripin_ai.explain.verify import check_text, verify

PLACES = [
    Place("1", "경포해변", ["자연관광", "자연경관(하천‧해양)", "해변. 해수욕장"], "강원특별자치도 강릉시",
          "길이 1.8km의 백사장", matched_features=["바다", "산책"], closest_liked_region="속초"),
    Place("2", "오죽헌", ["역사관광", "역사유적지", "생가"], "강원특별자치도 강릉시", "율곡 이이가 태어난 곳", order=2),
]
EXAMPLE = CourseExample("99", "속초 바다 여행", "힐링코스",
                        "백두대간의 허리에 우뚝 솟은 설악산이 든든하게 받쳐주고 시원한 동해가 펼쳐진다.",
                        ["설악산", "영금정"], ["", ""], 0.9)


class FakeLLM:
    model = "fake-llm"

    def __init__(self, response=None, error=None):
        self.response, self.error, self.prompts = response, error, []

    def generate_json(self, system, user):
        self.prompts.append(user)
        if self.error:
            raise LLMUnavailable(self.error)
        return self.response


GOOD = {"title": "바다와 역사를 잇는 강릉 하루", "intro": "푸른 바다와 오래된 이야기가 함께하는 강릉 여행이다. 해변을 걷고 오죽헌에서 쉬어 가자.",
        "reasons": [{"id": "1", "reason": "넓은 백사장을 따라 걷기 좋은 해변이다."},
                    {"id": "2", "reason": "율곡 이이가 태어난 곳을 둘러볼 수 있다."}]}


@pytest.mark.parametrize("text, problem", [
    ("설악산이 보이는 해변이다.", "FOREIGN_PLACE"),            # 예시 코스에만 있는 장소
    ("300년 된 소나무 숲이 있다.", "UNKNOWN_NUMBER"),          # 근거에 없는 숫자
    ("강릉 최고의 해변이다.", "BANNED_WORD"),
    ("백두대간의허리에우뚝솟은설악산 느낌.", "COPIED_EXAMPLE"),  # 예시 문장을 그대로
])
def test_check_text_catches_problems(text, problem):
    assert problem in check_text(text, 200, PLACES, [EXAMPLE])


def test_check_text_allows_facts_from_places_and_extra():
    assert check_text("길이 1.8km 백사장을 걷는 2일 코스.", 200, PLACES, [EXAMPLE], extra_facts="2일") == []


def test_verify_format_requires_every_place_once():
    bad = {**GOOD, "reasons": [{"id": "1", "reason": "좋다."}]}
    assert verify(bad, PLACES, [])["format"] == ["FORMAT"]
    assert verify({"title": "x"}, PLACES, [])["format"] == ["FORMAT"]


def test_llm_output_used_when_valid_and_personal_reason_kept_out_of_prompt():
    llm = FakeLLM(GOOD)
    result = explain("강원특별자치도 강릉시", 1, PLACES, None, None, llm)
    assert result.title_source == result.intro_source == "llm"
    assert result.reason_sources == {"1": "llm", "2": "llm"}
    assert result.personal_reasons == {"1": "좋아하신 속초 여행과 비슷한 분위기예요."}
    assert "속초" not in llm.prompts[0]          # 🔒 좋았던 여행지는 LLM에 보내지 않는다


def test_only_failing_parts_fall_back_to_rules():
    response = {**GOOD, "title": "강릉 최고의 여행", "reasons": [GOOD["reasons"][0], {"id": "2", "reason": "1450년에 지어졌다."}]}
    result = explain("강원특별자치도 강릉시", 1, PLACES, None, None, FakeLLM(response))
    assert result.title_source == "rule" and result.intro_source == "llm"
    assert result.reason_sources == {"1": "llm", "2": "rule"}


@pytest.mark.parametrize("llm", [None, FakeLLM(error="RATE_LIMITED")])
def test_no_llm_or_llm_error_gives_rule_sentences(llm):
    result = explain("강원특별자치도 강릉시", 2, PLACES, None, None, llm)
    assert result.title_source == result.intro_source == "rule"
    assert result.title == "강릉, 자연을 따라 걷는 2일"
    assert set(result.reason_sources.values()) == {"rule"}


def test_rule_sentences_use_correct_particles():
    assert rule_reason(PLACES[0]) == "바다·산책을 즐기기 좋은 해변. 해수욕장이에요."
    assert rule_reason(PLACES[1]) == "강릉의 생가예요."
    assert rule_title("서울특별시 종로구", 1, [PLACES[1]]) == "종로구, 역사를 따라 걷는 1일"
    assert personal_reason(PLACES[1]) is None


def test_prompt_contains_examples_and_facts():
    prompt = build_user_prompt("강원특별자치도 강릉시", 1, PLACES, [EXAMPLE])
    assert "속초 바다 여행" in prompt and "율곡 이이" in prompt and "features_matching_taste" in prompt


def test_api_explanations_with_fake_llm(monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as main
    from app.config import Settings
    from tests.test_api import FakeEncoder

    fake_llm = FakeLLM(GOOD)
    monkeypatch.setattr(main.state, "settings", Settings("fake", "mminilm-l12-v1", 1, 64))
    monkeypatch.setattr(main.state, "encoder", FakeEncoder())
    monkeypatch.setattr(main, "_load_encoder", lambda: None)
    monkeypatch.setattr(main, "_load_explainer", lambda: None)
    monkeypatch.setattr(main.state, "course_index", None)
    monkeypatch.setattr(main.state, "llm", fake_llm)
    body = {"requestId": "r1", "regionName": "강원특별자치도 강릉시", "days": 1, "places": [
        {"id": "1", "name": "경포해변", "lclsSystm1": "NA", "regionName": "강원특별자치도 강릉시",
         "description": "길이 1.8km의 백사장", "matchedFeatures": ["바다", "산책"], "closestLikedRegion": "속초"},
        {"id": "2", "name": "오죽헌", "lclsSystm1": "HS", "regionName": "강원특별자치도 강릉시",
         "description": "율곡 이이가 태어난 곳", "order": 2}]}
    with TestClient(main.app) as client:
        r = client.post("/explanations", json=body)
        assert r.status_code == 200
        data = r.json()
        assert data["aiGenerated"] is True and data["titleSource"] == "llm"
        assert [x["personalReason"] for x in data["reasons"]] == ["좋아하신 속초 여행과 비슷한 분위기예요.", None]
        assert "속초" not in fake_llm.prompts[0]
        assert client.post("/explanations", json={**body, "places": []}).status_code == 400


class JudgeLLM(FakeLLM):
    """첫 호출은 생성, 두 번째 호출은 사실 검증."""

    def __init__(self, response, unsupported):
        super().__init__(response)
        self.unsupported = unsupported

    def generate_json(self, system, user):
        self.prompts.append(user)
        return self.response if len(self.prompts) == 1 else {"unsupported": self.unsupported}


def test_claim_check_falls_back_intro_when_unsupported():
    result = explain("강원특별자치도 강릉시", 1, PLACES, None, None, JudgeLLM(GOOD, ["낙동강 물결"]), check_claims=True)
    assert result.intro_source == "rule" and result.title_source == "llm"
    assert result.problems["unsupported_claims"] == ["낙동강 물결"]
    ok = explain("강원특별자치도 강릉시", 1, PLACES, None, None, JudgeLLM(GOOD, []), check_claims=True)
    assert ok.intro_source == "llm"
