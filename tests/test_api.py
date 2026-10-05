"""서버 계약 테스트. 실제 모델 대신 결정적인 가짜 인코더를 쓴다."""
import hashlib
import logging
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import Settings
from tripin_ai.encoder import from_base64

DIM = 384


class FakeEncoder:
    dimension = DIM

    def __init__(self):
        self.texts = []

    def encode(self, texts):
        self.texts += texts
        vectors = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).standard_normal(DIM)
            vectors.append(v / np.linalg.norm(v))
        return np.array(vectors, dtype="<f4")


@pytest.fixture
def client(monkeypatch):
    fake = FakeEncoder()
    monkeypatch.setattr(main.state, "settings", Settings("fake", "mminilm-l12-v1", (1, 2), 64))
    monkeypatch.setattr(main.state, "encoder", None)
    monkeypatch.setattr(main.state, "load_error", None)
    monkeypatch.setattr(main, "encoder_factory", lambda settings: fake)
    with TestClient(main.app, raise_server_exceptions=False) as c:
        for _ in range(100):   # 백그라운드 로딩 대기
            if main.state.encoder is not None:
                break
            time.sleep(0.01)
        c.fake = fake
        yield c


PROFILE = {"travelMbti": "ENFP", "scheduleDensity": "RELAXED", "experienceTags": ["바다", "산책"],
           "excludeTags": ["물놀이"],
           "likedTrips": [{"sigCd": "51150", "regionName": "강원특별자치도 강릉시", "tags": ["바다"], "note": "비밀 메모"}]}


def body(**overrides):
    return {"requestId": "123", "modelVersion": "mminilm-l12-v1", "templateVersion": 1, "profile": PROFILE, **overrides}


def test_embeddings_returns_float32_le_base64_with_dimension(client):
    r = client.post("/embeddings", json=body())
    assert r.status_code == 200
    data = r.json()
    assert set(data) == {"embeddingBase64", "dimension"} and data["dimension"] == DIM
    vec = from_base64(data["embeddingBase64"])
    assert vec.shape == (DIM,) and abs(np.linalg.norm(vec) - 1) < 1e-5


def test_same_profile_gives_same_vector_and_excludes_negatives(client):
    a = client.post("/embeddings", json=body()).json()["embeddingBase64"]
    reordered = {**PROFILE, "experienceTags": ["산책", "바다"]}
    b = client.post("/embeddings", json=body(profile=reordered)).json()["embeddingBase64"]
    assert a == b
    assert all("물놀이" not in t and "ENFP" not in t for t in client.fake.texts)


def test_version_mismatch_is_409(client):
    assert client.post("/embeddings", json=body(modelVersion="other")).status_code == 409
    assert client.post("/embeddings", json=body(templateVersion=3)).status_code == 409


def test_legacy_and_aihub_template_versions_are_accepted(client):
    legacy = client.post("/embeddings", json=body())
    aihub = client.post("/embeddings", json=body(
        templateVersion=2,
        profile={"travelStyles": {"1": 2, "3": 6, "5": 3, "6": 5},
                 "travelMotives": [2, 7], "likedRegions": ["강원 강릉시"]}))
    assert legacy.status_code == 200
    assert aihub.status_code == 200
    assert "자연을 꽤 선호" in client.fake.texts[-1]
    assert "새로운 경험" in client.fake.texts[-1]


def test_empty_profile_is_400(client):
    r = client.post("/embeddings", json=body(profile={"excludeTags": ["물놀이"]}))
    assert r.status_code == 400 and r.json()["detail"]["code"] == "EMPTY_PROFILE"


def test_malformed_request_is_400_without_echoing_input(client):
    bad = body(profile={"likedTrips": [{"note": "비밀 메모", "tags": "문자열이면 안 됨"}]})
    r = client.post("/embeddings", json=bad)
    assert r.status_code == 400
    assert "비밀 메모" not in r.text and "문자열이면" not in r.text


def test_model_loading_is_503(client, monkeypatch):
    monkeypatch.setattr(main.state, "encoder", None)
    assert client.post("/embeddings", json=body()).status_code == 503
    assert client.get("/health").json()["status"] == "LOADING"


def test_unexpected_error_is_500(client, monkeypatch):
    def boom(texts):
        raise RuntimeError("비밀 메모")
    monkeypatch.setattr(client.fake, "encode", boom)
    r = client.post("/embeddings", json=body())
    assert r.status_code == 500 and "비밀" not in r.text


def test_logs_do_not_contain_profile_text(client, caplog):
    with caplog.at_level(logging.INFO, logger="tripin.embedding"):
        client.post("/embeddings", json=body())
    assert "requestId=123" in caplog.text
    assert "비밀 메모" not in caplog.text and "바다" not in caplog.text


def test_health_reports_versions(client):
    h = client.get("/health").json()
    assert h == {"status": "UP", "modelVersion": "mminilm-l12-v1", "templateVersions": [1, 2], "dimension": DIM, "loadError": None}


ITEM = {"id": "126508", "name": "경복궁", "contentTypeId": "12", "lclsSystm1": "HS", "lclsSystm2": "HS01",
        "lclsSystm3": "HS010100", "regionName": "서울특별시 종로구", "description": "조선의 법궁"}


def batch(**overrides):
    return {"modelVersion": "mminilm-l12-v1", "templateVersion": 1, "items": [ITEM], **overrides}


def test_batch_matches_single_encoding_and_uses_class_names(client):
    r = client.post("/embeddings/batch", json=batch(items=[ITEM, {**ITEM, "id": "2", "name": "창덕궁"}]))
    assert r.status_code == 200
    data = r.json()
    assert data["dimension"] == DIM and [i["id"] for i in data["items"]] == ["126508", "2"]
    text = client.fake.texts[0]
    assert text.startswith("경복궁. 역사관광 > ") and "서울특별시 종로구" in text and text.endswith("조선의 법궁")
    single = client.post("/embeddings/batch", json=batch()).json()["items"][0]["embeddingBase64"]
    assert single == data["items"][0]["embeddingBase64"]


def test_batch_limits_and_validation(client):
    assert client.post("/embeddings/batch", json=batch(items=[])).status_code == 400
    assert client.post("/embeddings/batch", json=batch(items=[ITEM] * 65)).status_code == 400
    assert client.post("/embeddings/batch", json=batch(items=[{**ITEM, "name": " "}])).status_code == 400
    assert client.post("/embeddings/batch", json=batch(modelVersion="x")).status_code == 409


def test_unknown_class_code_is_ignored(client):
    r = client.post("/embeddings/batch", json=batch(items=[{**ITEM, "lclsSystm3": "ZZ999999"}]))
    assert r.status_code == 200


def test_batch_template_two_includes_tourapi_content_type(client):
    r = client.post("/embeddings/batch", json=batch(templateVersion=2, items=[{**ITEM, "contentTypeId": "14"}]))

    assert r.status_code == 200
    assert client.fake.texts[-1].startswith("경복궁. 문화시설.")
    assert "서울특별시 종로구" in client.fake.texts[-1]
    assert client.fake.texts[-1].endswith("조선의 법궁")


def test_batch_accepts_numeric_content_type_id_from_backend(client):
    # backend AttractionEmbeddingBatchRequest는 contentTypeId를 숫자로 직렬화한다.
    r = client.post("/embeddings/batch", json=batch(templateVersion=2, items=[{**ITEM, "contentTypeId": 14}]))

    assert r.status_code == 200
    assert client.fake.texts[-1].startswith("경복궁. 문화시설.")
