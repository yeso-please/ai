"""한 줄 소개·태그 (yeso-please/backend#89). 실제 LLM 대신 정해진 답을 돌려주는 가짜를 쓴다."""
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import Settings
from tripin_ai.explain.llm import LLMUnavailable
from tripin_ai.summary.prompts import SummaryPlace, build_attraction_prompt
from tripin_ai.summary.service import NAME_CATEGORY, SOURCE_SUMMARY, summarize_attractions, summarize_region
from tripin_ai.summary.tags import DICTIONARY, MAX_TAGS, attraction_rule_tags, code_tags, merge_tags, region_rule_tags


class FakeLLM:
    model = "fake-llm"

    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.calls = answer, error, []

    def generate_json(self, system, user):
        self.calls.append((system, user))
        if self.error:
            raise LLMUnavailable(self.error)
        return self.answer


BEACH = SummaryPlace("1", "경포해변", region_name="강원특별자치도 강릉시",
                     description="고운 모래와 소나무 숲이 이어지는 해변으로 여름이면 해수욕객이 찾는다.",
                     codes=("NA", "NA02", "NA020900"))
NO_DESC = SummaryPlace("2", "솔바람다리", region_name="강원특별자치도 강릉시", codes=("VE", "VE01", "VE010300"))


def test_dictionary_has_21_unique_tags_and_rules_stay_inside():
    assert len(DICTIONARY) == 21 == len(set(DICTIONARY))
    from tripin_ai.summary.tags import CODE_TAGS, KEYWORD_TAGS
    assert {t for tags in CODE_TAGS.values() for t in tags} <= set(DICTIONARY)
    assert set(KEYWORD_TAGS) <= set(DICTIONARY)


def test_code_tags_prefers_most_specific_code():
    assert code_tags("HS", "HS01", "HS010600") == ["전통", "역사"]   # 민속마을
    assert code_tags("HS", "HS01", "HS010100") == ["역사"]           # 고궁은 대분류 규칙
    assert code_tags("NA", "NA02", None) == []                        # 중분류 규칙이 없으면 태그 없음
    assert code_tags(None, None, None) == []


def test_keyword_tags_only_for_tags_without_class_codes():
    # 박물관 설명에 "바다"가 나와도 바다 태그는 붙이지 않는다. 카페·야경은 키워드로 붙인다.
    tags = attraction_rule_tags("VE", "VE07", "VE070100", "해양박물관", "바다가 보이는 카페와 야경이 아름답다")
    assert tags[0] == "박물관" and "바다" not in tags and {"카페", "야경"} <= set(tags)


def test_region_rule_tags_use_share_and_count_market_separately():
    counts = {"NA020900": 12, "NA020800": 5, "HS010100": 4, "VE070100": 2, "SH040100": 300, "SH060200": 3}
    tags = region_rule_tags(counts)
    assert tags[0] == "바다"
    assert "시장" in tags            # 면세점(SH04)이 많아도 분모에 넣지 않고, 시장은 수로 본다
    assert "박물관" not in tags       # 2곳뿐이라 규칙 태그가 아니다


def test_merge_tags_accepts_mood_and_supported_keywords_only():
    tags = merge_tags(["바다"], ["#감성여행", "카페", "섬", "없는태그", "데이트"], "해변 앞 카페 거리")
    assert tags == ["바다", "감성여행", "카페", "데이트"]   # 섬은 근거 없음, 없는태그는 사전 밖
    assert len(merge_tags(["바다", "산", "숲", "섬", "꽃"], ["힐링"], "")) == MAX_TAGS
    assert "힐링" in merge_tags(["바다", "산", "숲", "섬"], ["힐링"], "")   # 분위기 한 자리는 남긴다


def test_prompt_marks_basis_and_hides_missing_description():
    prompt = build_attraction_prompt([BEACH, NO_DESC])
    assert '"basis": "SOURCE_SUMMARY"' in prompt and '"basis": "NAME_CATEGORY"' in prompt
    assert prompt.count('"description"') == 1


def test_summarize_attractions_keeps_valid_and_drops_invalid():
    llm = FakeLLM({"items": [
        {"id": "1", "oneLine": "솔숲 너머 고운 모래를 따라 걷는 바닷가", "tags": ["감성여행", "섬"]},
        {"id": "2", "oneLine": "1920년에 지어진 최고의 다리", "tags": ["데이트"]},
    ]})
    batch = summarize_attractions([BEACH, NO_DESC], llm)
    beach, bridge = batch.items
    assert beach.one_line and beach.source == "llm" and beach.basis == SOURCE_SUMMARY
    assert beach.tags == ["바다", "감성여행"]
    assert bridge.one_line is None and bridge.basis == NAME_CATEGORY
    assert {"BANNED_WORD", "UNKNOWN_NUMBER"} <= set(bridge.problems)
    assert batch.generator_model == "fake-llm"


def test_summarize_attractions_rejects_other_place_names():
    llm = FakeLLM({"items": [{"id": "1", "oneLine": "솔바람다리까지 이어지는 바닷길", "tags": []},
                             {"id": "2", "oneLine": "강물 위를 건너는 다리", "tags": []}]})
    beach, bridge = summarize_attractions([BEACH, NO_DESC], llm).items
    assert beach.one_line is None and "FOREIGN_PLACE" in beach.problems
    assert bridge.one_line == "강물 위를 건너는 다리"


def test_llm_failure_returns_rule_tags_without_sentence():
    batch = summarize_attractions([BEACH], FakeLLM(error="RATE_LIMITED"))
    assert batch.llm_error == "RATE_LIMITED"
    assert batch.items[0].one_line is None and batch.items[0].tags == ["바다"]
    assert summarize_attractions([BEACH], None).llm_error == "NO_LLM"


def test_region_tagline_rejects_place_names():
    counts = {"NA020900": 10}
    ok = summarize_region(counts, [BEACH], FakeLLM({"tagline": "파도 소리에 느긋해지는 바닷가", "tags": ["힐링"]}))
    assert ok.items[0].tagline and ok.items[0].tags == ["바다", "힐링"]
    bad = summarize_region(counts, [BEACH], FakeLLM({"tagline": "경포해변의 여름", "tags": ["힐링"]}))
    assert bad.items[0].tagline is None and bad.items[0].tags == ["바다"] and "PLACE_NAME" in bad.items[0].problems


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main.state, "settings", Settings("fake", "mminilm-l12-v1", (1, 2), 64))
    monkeypatch.setattr(main, "_load_encoder", lambda: None)
    monkeypatch.setattr(main, "_load_explainer", lambda: None)
    with TestClient(main.app, raise_server_exceptions=False) as c:
        yield c


def test_api_attraction_summaries(client, monkeypatch):
    monkeypatch.setattr(main.state, "llm", FakeLLM({"items": [
        {"id": "1", "oneLine": "솔숲 너머 고운 모래를 따라 걷는 바닷가", "tags": ["데이트"]}]}))
    r = client.post("/summaries/attractions", json={"requestId": "r1", "items": [
        {"id": "1", "name": "경포해변", "lclsSystm1": "NA", "lclsSystm2": "NA02", "lclsSystm3": "NA020900",
         "regionName": "강원특별자치도 강릉시", "description": BEACH.description}]})
    assert r.status_code == 200
    item = r.json()["items"][0]
    assert item == {"id": "1", "oneLine": "솔숲 너머 고운 모래를 따라 걷는 바닷가", "tags": ["바다", "데이트"],
                    "basis": "SOURCE_SUMMARY", "source": "llm", "problems": []}
    assert r.json()["promptVersion"] == "summary-v1"


def test_api_attraction_summaries_rejects_duplicate_ids_and_large_batches(client, monkeypatch):
    monkeypatch.setattr(main.state, "llm", None)
    item = {"id": "1", "name": "경포해변"}
    assert client.post("/summaries/attractions", json={"requestId": "r", "items": [item, item]}).status_code == 400
    many = [{"id": str(i), "name": f"장소{i}"} for i in range(11)]
    assert client.post("/summaries/attractions", json={"requestId": "r", "items": many}).status_code == 400


def test_api_region_summary_without_llm_returns_rule_tags(client, monkeypatch):
    monkeypatch.setattr(main.state, "llm", None)
    r = client.post("/summaries/region", json={"requestId": "r", "regionName": "강원특별자치도 강릉시",
                                               "classCounts": {"NA020900": 9, "HS010100": 1},
                                               "attractions": [{"id": "1", "name": "경포해변"}]})
    assert r.status_code == 200
    assert r.json()["tagline"] is None and r.json()["tags"] == ["바다"] and r.json()["llmError"] == "NO_LLM"
