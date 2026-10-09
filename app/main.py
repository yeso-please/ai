"""임베딩 서버 (#1, #2). 백엔드 HttpEmbeddingClient와 관광지 임베딩 배치(yeso-please/backend#54)가 호출한다.

상태 코드 계약 (백엔드 재시도 판단)
  200 정상 / 400 요청 형식 오류·빈 문장(재시도 안 함) / 409 모델·템플릿 버전 불일치(재시도 안 함)
  503 모델 로딩 중(재시도) / 500 예상 못 한 오류(재시도)

🔒 회원 프로필(자유서술 메모 포함)과 합성 문장은 로그·오류 응답에 남기지 않는다.
   로그에는 requestId, 문장 길이, 건수, 처리 시간만 남긴다.

코스 소개·추천 이유(#7): POST /explanations. LLM(Gemini) 키가 없거나 실패·검증 실패면 규칙 문장으로 대체한다.
한 줄 소개·태그(backend#89): POST /summaries/attractions, /summaries/region. 검증에 걸린 문장은 null이고
   태그는 사전 값만 준다. 백엔드는 초안(DRAFT)으로 저장하고 사람이 승인한 것만 화면에 낸다.

실행: uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from tripin_ai.classification import class_names_of
from tripin_ai.encoder import to_base64
from tripin_ai.explain.course_index import CourseIndex
from tripin_ai.explain.llm import default_client
from tripin_ai.explain.prompts import Place
from tripin_ai.explain.service import explain
from tripin_ai.summary.prompts import SummaryPlace
from tripin_ai.summary.service import summarize_attractions, summarize_region
from tripin_ai.templates import (TEMPLATE_SETS, Attraction, LikedTrip, ServiceProfile, TravelerSurvey,
                                 attraction_text, profile_text, traveler_text)

from .config import Settings, load_settings
from .schemas import (AttractionIn, AttractionSummaryOut, AttractionSummaryRequest, AttractionSummaryResponse,
                      BatchItemOut, BatchRequest, BatchResponse, EmbeddingRequest, EmbeddingResponse,
                      ExplanationRequest, ExplanationResponse, ProfileIn, ReasonOut, RegionSummaryRequest,
                      RegionSummaryResponse, SummaryPlaceIn)

log = logging.getLogger("tripin.embedding")


class State:
    def __init__(self):
        self.settings: Settings = load_settings()
        self.encoder = None
        self.load_error: str | None = None
        self.course_index: CourseIndex | None = None
        self.llm = None


state = State()


def encoder_factory(settings: Settings):
    """테스트에서 가짜 인코더로 바꿔 끼운다."""
    from tripin_ai.encoder import Encoder
    return Encoder(settings.model_name, settings.model_version)


def course_index_path(settings: Settings) -> str:
    return os.environ.get("COURSE_INDEX_PATH", f"data/interim/course_index/{settings.model_version}.npz")


def _load_explainer() -> None:
    """코스 색인과 LLM은 없어도 서버가 뜬다(그때는 문체 예시 없이 생성하거나 규칙 문장)."""
    path = course_index_path(state.settings)
    if os.path.exists(path):
        index = CourseIndex(path)
        if index.model_version == state.settings.model_version:
            state.course_index = index
        else:
            log.warning("course index model version differs: %s", index.model_version)
    state.llm = default_client()
    log.info("explainer ready courses=%s llm=%s", len(state.course_index) if state.course_index else 0,
             getattr(state.llm, "model", None))


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
    unknown = sorted(set(state.settings.template_versions) - set(TEMPLATE_SETS))
    if not state.settings.template_versions or unknown:
        raise RuntimeError(f"알 수 없는 TEMPLATE_VERSIONS: {unknown}")
    # 모델 로딩은 수십 초 걸릴 수 있어 백그라운드로 한다. 그동안 요청은 503(백엔드가 재시도).
    threading.Thread(target=_load_encoder, daemon=True).start()
    _load_explainer()
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
    if model_version != settings.model_version or template_version not in settings.template_versions:
        raise HTTPException(status_code=409, detail={
            "code": "VERSION_MISMATCH", "serverModelVersion": settings.model_version,
            "serverTemplateVersions": list(settings.template_versions)})


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
        region_name=item.region_name, description=item.description,
        content_type_id=_content_type_id(item.content_type_id)), attraction_template)


def _content_type_id(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


@app.get("/health")
def health():
    encoder = state.encoder
    return {"status": "UP" if encoder else ("LOADING" if state.load_error is None else "DOWN"),
            "modelVersion": state.settings.model_version, "templateVersions": list(state.settings.template_versions),
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


@app.post("/explanations", response_model=ExplanationResponse, response_model_by_alias=True)
def explain_course(request: ExplanationRequest):
    start = time.perf_counter()
    encoder = _ready_encoder()
    if any(not p.name.strip() for p in request.places):
        raise HTTPException(status_code=400, detail={"code": "EMPTY_NAME"})
    places = [Place(id=p.id, name=p.name, class_names=class_names_of(p.lcls_systm1, p.lcls_systm2, p.lcls_systm3),
                    region_name=p.region_name, description=p.description, day=p.day, order=p.order,
                    matched_features=p.matched_features, closest_liked_region=p.closest_liked_region)
              for p in request.places]
    # 문체 예시 검색용 코스 벡터: 색인과 같은 관광지 문장(템플릿 v1)으로 만든다.
    vectors = None
    if state.course_index is not None:
        vectors = encoder.encode([attraction_text(Attraction(p.name, p.class_names, p.region_name, p.description), "v1")
                                  for p in places])
    result = explain(request.region_name, request.days, places, vectors, state.course_index, state.llm,
                     check_claims=os.environ.get("EXPLAIN_CLAIM_CHECK", "0") == "1")
    log.info("explained requestId=%s places=%d title=%s intro=%s llmError=%s ms=%.0f", request.request_id, len(places),
             result.title_source, result.intro_source, result.llm_error, (time.perf_counter() - start) * 1000)
    return ExplanationResponse(
        title=result.title, intro=result.intro,
        reasons=[ReasonOut(id=p.id, reason=result.reasons[p.id], source=result.reason_sources[p.id],
                           personal_reason=result.personal_reasons.get(p.id)) for p in places],
        title_source=result.title_source, intro_source=result.intro_source,
        ai_generated="llm" in {result.title_source, result.intro_source, *result.reason_sources.values()},
        prompt_version=result.prompt_version, generator_model=result.generator_model,
        example_course_ids=result.example_course_ids)


def _summary_place(p: SummaryPlaceIn) -> SummaryPlace:
    return SummaryPlace(id=p.id, name=p.name, class_names=class_names_of(p.lcls_systm1, p.lcls_systm2, p.lcls_systm3),
                        region_name=p.region_name, description=p.description,
                        codes=(p.lcls_systm1, p.lcls_systm2, p.lcls_systm3))


@app.post("/summaries/attractions", response_model=AttractionSummaryResponse, response_model_by_alias=True)
def summarize_attraction_batch(request: AttractionSummaryRequest):
    """임베딩 모델이 필요 없어 모델 로딩 중에도 받는다."""
    start = time.perf_counter()
    if any(not p.name.strip() for p in request.items) or len({p.id for p in request.items}) != len(request.items):
        raise HTTPException(status_code=400, detail={"code": "INVALID_ITEMS"})
    batch = summarize_attractions([_summary_place(p) for p in request.items], state.llm)
    log.info("summarized attractions requestId=%s items=%d llm=%d llmError=%s ms=%.0f", request.request_id,
             len(batch.items), sum(i.source == "llm" for i in batch.items), batch.llm_error,
             (time.perf_counter() - start) * 1000)
    return AttractionSummaryResponse(
        items=[AttractionSummaryOut(id=i.id, one_line=i.one_line, tags=i.tags, basis=i.basis, source=i.source,
                                    problems=i.problems) for i in batch.items],
        prompt_version=batch.prompt_version, generator_model=batch.generator_model, llm_error=batch.llm_error)


@app.post("/summaries/region", response_model=RegionSummaryResponse, response_model_by_alias=True)
def summarize_region_card(request: RegionSummaryRequest):
    start = time.perf_counter()
    if any(not p.name.strip() for p in request.attractions):
        raise HTTPException(status_code=400, detail={"code": "EMPTY_NAME"})
    batch = summarize_region(request.class_counts, [_summary_place(p) for p in request.attractions], state.llm)
    summary = batch.items[0]
    log.info("summarized region requestId=%s attractions=%d source=%s llmError=%s ms=%.0f", request.request_id,
             len(request.attractions), summary.source, batch.llm_error, (time.perf_counter() - start) * 1000)
    return RegionSummaryResponse(tagline=summary.tagline, tags=summary.tags, source=summary.source,
                                 problems=summary.problems, prompt_version=batch.prompt_version,
                                 generator_model=batch.generator_model, llm_error=batch.llm_error)
