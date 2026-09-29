"""임베딩 서버 (#1, #2). 백엔드 HttpEmbeddingClient와 관광지 임베딩 배치(yeso-please/backend#54)가 호출한다.

상태 코드 계약 (백엔드 재시도 판단)
  200 정상 / 400 요청 형식 오류·빈 문장(재시도 안 함) / 409 모델·템플릿 버전 불일치(재시도 안 함)
  503 모델 로딩 중(재시도) / 500 예상 못 한 오류(재시도)

🔒 회원 프로필(자유서술 메모 포함)과 합성 문장은 로그·오류 응답에 남기지 않는다.
   로그에는 requestId, 문장 길이, 건수, 처리 시간만 남긴다.

실행: uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from tripin_ai.classification import class_names_of
from tripin_ai.encoder import to_base64
from tripin_ai.templates import (TEMPLATE_SETS, Attraction, LikedTrip, ServiceProfile, TravelerSurvey,
                                 attraction_text, profile_text, traveler_text)

from .config import Settings, load_settings
from .schemas import (AttractionIn, BatchItemOut, BatchRequest, BatchResponse, EmbeddingRequest, EmbeddingResponse,
                      ProfileIn)

log = logging.getLogger("tripin.embedding")


class State:
    def __init__(self):
        self.settings: Settings = load_settings()
        self.encoder = None
        self.load_error: str | None = None


state = State()


def encoder_factory(settings: Settings):
    """테스트에서 가짜 인코더로 바꿔 끼운다."""
    from tripin_ai.encoder import Encoder
    return Encoder(settings.model_name, settings.model_version)


def _load_encoder() -> None:
    start = time.perf_counter()
    try:
        state.encoder = encoder_factory(state.settings)
        log.info("model loaded version=%s dim=%s in %.1fs", state.settings.model_version,
                 state.encoder.dimension, time.perf_counter() - start)
    except Exception as e:  # 로딩 실패는 /health로 드러나고 요청은 503을 받는다.
        state.load_error = type(e).__name__
        log.error("model load failed: %s", type(e).__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if state.settings.template_version not in TEMPLATE_SETS:
        raise RuntimeError(f"알 수 없는 TEMPLATE_VERSION: {state.settings.template_version}")
    # 모델 로딩은 수십 초 걸릴 수 있어 백그라운드로 한다. 그동안 요청은 503(백엔드가 재시도).
    threading.Thread(target=_load_encoder, daemon=True).start()
    yield


app = FastAPI(title="TriPin Embedding", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    # FastAPI 기본 응답(422)은 입력값을 그대로 되돌려 준다. 개인정보가 새지 않게 필드 위치만 알려 준다.
    fields = sorted({".".join(str(p) for p in err["loc"][1:]) for err in exc.errors()})
    return JSONResponse(status_code=400, content={"code": "INVALID_REQUEST", "fields": fields})


@app.exception_handler(Exception)
async def unexpected_error(_: Request, exc: Exception):
    log.error("unexpected error: %s", type(exc).__name__)
    return JSONResponse(status_code=500, content={"code": "INTERNAL_ERROR"})


def _ready_encoder():
    if state.encoder is None:
        raise HTTPException(status_code=503, detail={"code": "MODEL_LOADING" if state.load_error is None else "MODEL_UNAVAILABLE"})
    return state.encoder


def _check_versions(model_version: str, template_version: int) -> None:
    settings = state.settings
    if model_version != settings.model_version or template_version != settings.template_version:
        raise HTTPException(status_code=409, detail={
            "code": "VERSION_MISMATCH", "serverModelVersion": settings.model_version,
            "serverTemplateVersion": settings.template_version})


def compose_profile(profile: ProfileIn, template_version: int) -> str:
    profile_template, _ = TEMPLATE_SETS[template_version]
    if profile_template == "service-v1":
        return profile_text(ServiceProfile(
            experience_tags=profile.experience_tags, travel_mbti=profile.travel_mbti,
            schedule_density=profile.schedule_density, exclude_tags=profile.exclude_tags,
            liked_trips=[LikedTrip(t.region_name, t.tags, t.note) for t in profile.liked_trips]), profile_template)
    return traveler_text(TravelerSurvey(styles=profile.travel_styles, motives=profile.travel_motives,
                                        liked_regions=profile.liked_regions), profile_template)


def compose_attraction(item: AttractionIn, template_version: int) -> str:
    # tags는 받지만 문장에 넣지 않는다: 학습한 관광지 문장(AI Hub × TourAPI)에 태그가 없어 분포가 어긋난다.
    _, attraction_template = TEMPLATE_SETS[template_version]
    return attraction_text(Attraction(
        name=item.name, class_names=class_names_of(item.lcls_systm1, item.lcls_systm2, item.lcls_systm3),
        region_name=item.region_name, description=item.description), attraction_template)


@app.get("/health")
def health():
    encoder = state.encoder
    return {"status": "UP" if encoder else ("LOADING" if state.load_error is None else "DOWN"),
            "modelVersion": state.settings.model_version, "templateVersion": state.settings.template_version,
            "dimension": encoder.dimension if encoder else None, "loadError": state.load_error}


@app.post("/embeddings", response_model=EmbeddingResponse, response_model_by_alias=True)
def embed_profile(request: EmbeddingRequest):
    start = time.perf_counter()
    encoder = _ready_encoder()
    _check_versions(request.model_version, request.template_version)
    text = compose_profile(request.profile, request.template_version)
    if not text:
        raise HTTPException(status_code=400, detail={"code": "EMPTY_PROFILE"})
    vector = encoder.encode([text])[0]
    log.info("embedded requestId=%s chars=%d ms=%.0f", request.request_id, len(text), (time.perf_counter() - start) * 1000)
    return EmbeddingResponse(embedding_base64=to_base64(vector), dimension=int(vector.shape[0]))


@app.post("/embeddings/batch", response_model=BatchResponse, response_model_by_alias=True)
def embed_attractions(request: BatchRequest):
    start = time.perf_counter()
    encoder = _ready_encoder()
    _check_versions(request.model_version, request.template_version)
    if not request.items or len(request.items) > state.settings.max_batch_items:
        raise HTTPException(status_code=400, detail={"code": "INVALID_BATCH_SIZE", "max": state.settings.max_batch_items})
    if any(not item.name.strip() for item in request.items):
        raise HTTPException(status_code=400, detail={"code": "EMPTY_NAME"})
    texts = [compose_attraction(item, request.template_version) for item in request.items]
    vectors = encoder.encode(texts)
    log.info("embedded batch items=%d ms=%.0f", len(texts), (time.perf_counter() - start) * 1000)
    return BatchResponse(dimension=int(vectors.shape[1]),
                         items=[BatchItemOut(id=item.id, embedding_base64=to_base64(v)) for item, v in zip(request.items, vectors)])
