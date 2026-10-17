"""성향 맞춤 지역 문장 (backend 7-6). 실제 LLM 대신 정해진 답을 돌려주는 가짜를 쓴다."""
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import Settings
from tripin_ai.explain.llm import LLMUnavailable
from tripin_ai.explain.region_match import Landmark, RegionMatchInput, build_prompt, match_region, rule_text


class FakeLLM:
    model = "fake-llm"

    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.prompts = answer, error, []

    def generate_json(self, system, user):
        self.prompts.append(user)
        if self.error:
            raise LLMUnavailable(self.error)
        return self.answer


DATA = RegionMatchInput(
    persona_name="낭만적인 기록가", persona_summary="여행의 감정과 장면을 기억으로 남기는 타입",
    persona_likes=["골목", "카페", "바다"], region_name="강원특별자치도 강릉시", tagline="바다와 카페 사이, 느긋하게 머무는 곳",
    tags=["바다", "카페", "감성여행"], landmarks=[Landmark("경포해변", "솔숲 너머 고운 모래를 따라 걷는 바닷가"), Landmark("안목해변")])


def test_prompt_has_no_member_data_and_short_region():
    prompt = build_prompt(DATA)
    assert "강릉" in prompt and "강원특별자치도 강릉시" not in prompt
    assert "낭만적인 기록가" in prompt and "닉네임" not in prompt


def test_valid_llm_text_is_used():
    text = "바다를 따라 걷다 작은 카페에 머물며 장면을 기록하기 좋은 곳이에요. 경포해변의 솔숲 길을 천천히 걸어 보세요."
    result = match_region(DATA, FakeLLM({"text": text}))
    assert result.source == "llm" and result.text == text and result.generator_model == "fake-llm"


@pytest.mark.parametrize("answer,problem", [
    ({"text": "강릉 최고의 여행지예요."}, "BANNED_WORD"),
    ({"text": "1950년부터 이어진 바닷가예요."}, "UNKNOWN_NUMBER"),
    ({"text": "가" * 200}, "LENGTH"),
    ({"other": "x"}, "LENGTH"),
])
def test_invalid_text_falls_back_to_rule(answer, problem):
    result = match_region(DATA, FakeLLM(answer))
    assert result.source == "rule" and problem in result.problems
    assert result.text == rule_text(DATA) == "낭만적인 기록가에게 강릉의 경포해변, 안목해변을(를) 추천해요."


def test_llm_unavailable_or_missing():
    assert match_region(DATA, FakeLLM(error="RATE_LIMITED")).llm_error == "RATE_LIMITED"
    assert match_region(DATA, None).source == "rule"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main.state, "settings", Settings("fake", "mminilm-l12-v1", (1, 2), 64))
    monkeypatch.setattr(main, "_load_encoder", lambda: None)
    monkeypatch.setattr(main, "_load_explainer", lambda: None)
    with TestClient(main.app, raise_server_exceptions=False) as c:
        yield c


def test_api_region_match(client, monkeypatch):
    monkeypatch.setattr(main.state, "llm", None)
    r = client.post("/explanations/region-match", json={
        "requestId": "r", "personaName": "낭만적인 기록가", "personaSummary": "기록하는 타입", "personaLikes": ["골목"],
        "regionName": "강원특별자치도 강릉시", "tagline": None, "tags": ["바다"],
        "landmarks": [{"name": "경포해변", "oneLine": None, "category": None}]})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "rule" and body["promptVersion"] == "region-match-v1" and body["llmError"] == "NO_LLM"
    assert client.post("/explanations/region-match", json={"requestId": "r", "personaName": " ", "regionName": "강릉"}).status_code == 400
