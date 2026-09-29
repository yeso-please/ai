# TriPin AI

회원 취향과 관광지를 같은 벡터 공간에 올리는 **임베딩 서버**와, AI Hub 실제 여행자 데이터로 하는 **취향 모델 평가·학습**을 둔다.

## 구성

| 폴더 | 내용 |
|---|---|
| `app/` | FastAPI 임베딩 서버 (`/embeddings`, `/embeddings/batch`, `/health`) — #1, #2 |
| `tripin_ai/` | 서버와 실험이 같이 쓰는 코어: 문장 템플릿, 인코더, 평가(`evaluation/`), 학습(`training/`) |
| `scripts/` | 데이터 수집·정리, 오프라인 임베딩, 평가, 파인튜닝 |
| `reports/` | 실험 결과 (집계만) |
| `docs/` | 데이터 카드(`data.md`), Colab 학습 방법(`colab.md`) |
| `data/reference/` | TourAPI 분류체계 코드표 (서버가 씀) |
| `data/raw`, `data/interim` | AI Hub·TourAPI 원본과 중간 산출물 — **커밋 금지**(AI Hub 재배포 금지, 레포는 공개) |

## 서버 실행

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[serve,dev]"
.venv/Scripts/python -m uvicorn app.main:app --port 8000
```

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `MODEL_NAME` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Hugging Face 모델 id 또는 파인튜닝한 모델 폴더 |
| `MODEL_VERSION` | `mminilm-l12-v1` | 백엔드 `embedding.model-version`과 같아야 한다 |
| `TEMPLATE_VERSION` | `1` | 백엔드 `embedding.template-version`과 같아야 한다 (`tripin_ai/templates.py`의 `TEMPLATE_SETS`) |
| `MAX_BATCH_ITEMS` | `64` | `/embeddings/batch` 최대 건수 |

모델 로딩(CPU 약 20초) 동안 요청은 503을 받는다(백엔드가 재시도). Docker: `docker build -t tripin-ai . && docker run -p 8000:8000 tripin-ai`

## 백엔드와의 계약

**`POST /embeddings`** — 회원 취향 벡터 (백엔드 `HttpEmbeddingClient`, 설정 `EMBEDDING_SERVICE_BASE_URL`)

```json
{
  "requestId": "123", "modelVersion": "mminilm-l12-v1", "templateVersion": 1,
  "profile": {
    "travelMbti": "ENFP", "scheduleDensity": "RELAXED",
    "experienceTags": ["바다", "산책"], "excludeTags": ["물놀이"],
    "likedTrips": [{"sigCd": "51150", "regionName": "강원특별자치도 강릉시", "tags": ["바다"], "note": "자유서술"}]
  }
}
```
→ `{"embeddingBase64": "...", "dimension": 384}`

- **문장은 ai가 만든다**(백엔드는 구조화된 값만 보낸다). 템플릿 1은 선호 경험·좋았던 여행지(지역 이름·태그·메모)를 쓰고, **제외 조건·MBTI·일정 밀도는 문장에 넣지 않는다.** 임베딩은 "싫다"를 이해하지 못해 제외 조건 단어 쪽으로 끌려간다(`reports/sanity_*`). 제외 조건은 백엔드가 필터로 처리한다.
- 템플릿 2(AI Hub 설문 형식: `travelStyles`, `travelMotives`, `likedRegions`)는 온보딩 전환(yeso-please/backend#67) 뒤에 쓴다.

**`POST /embeddings/batch`** — 관광지 벡터 (backend #54 배치)

```json
{"modelVersion": "mminilm-l12-v1", "templateVersion": 1,
 "items": [{"id": "126508", "name": "경복궁", "contentTypeId": "12",
            "lclsSystm1": "HS", "lclsSystm2": "HS01", "lclsSystm3": "HS010100",
            "regionName": "서울특별시 종로구", "description": "…"}]}
```
→ `{"dimension": 384, "items": [{"id": "126508", "embeddingBase64": "..."}]}` (최대 64건)

- 분류는 TourAPI KorService2 **새 분류체계 `lclsSystm1~3`**(옛 `cat1~3`은 대부분 비어 있다). `tags`는 받지만 쓰지 않는다(학습한 관광지 문장에 없음).

**공통**

| 항목 | 내용 |
|---|---|
| 벡터 형식 | L2 정규화한 **float32 리틀엔디언** 바이트를 base64로 |
| 차원 | 백엔드 `embedding.expected-dimension`(384)과 같아야 한다 |
| 버전 | 서버 `MODEL_VERSION`·`TEMPLATE_VERSION`과 요청이 다르면 **409**. 다른 모델·규칙의 벡터가 섞이지 않게 한다 |
| 상태 코드 | 200 / 400 형식 오류·빈 문장(재시도 안 함) / 409 버전 불일치(재시도 안 함) / 503 로딩 중(재시도) / 500 예상 못 한 오류(재시도) |
| 개인정보 | 프로필·메모·합성 문장은 로그와 오류 응답에 남기지 않는다. 형식 오류 응답도 입력값을 되돌려 주지 않는다 |

## 평가·학습 (요약)

- 데이터: AI Hub 국내 여행로그 2023(4개 권역) × TourAPI 매칭 → `docs/data.md`
- 평가: 인기 편향을 통제한 취향 평가 중심 (`scripts/evaluate.py`, `reports/eval_baseline_summary.md`)
- 학습: 실제 여행자의 (설문 → 만족한 관광지) 대조학습, Colab GPU (`scripts/finetune.py`, `docs/colab.md`)
- 결과: 인기를 맞춘 후보 비교에서 기성 모델 51.9% → 학습 54.8% (인기 51.2%) — `reports/finetune_v1_summary.md`

## 테스트

```bash
.venv/Scripts/python -m pytest
```
