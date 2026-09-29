"""요청·응답 모델. JSON 필드는 백엔드 계약대로 camelCase다."""
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore")


class LikedTripIn(CamelModel):
    sig_cd: str | None = None
    region_name: str = ""
    tags: list[str] = Field(default_factory=list)
    note: str = ""          # 🔒 자유서술. 로그·오류 응답에 절대 남기지 않는다.


class ProfileIn(CamelModel):
    # 템플릿 1 (현재 온보딩)
    travel_mbti: str = ""
    schedule_density: str = ""
    experience_tags: list[str] = Field(default_factory=list)
    exclude_tags: list[str] = Field(default_factory=list)
    liked_trips: list[LikedTripIn] = Field(default_factory=list)
    # 템플릿 2 (AI Hub 설문 형식, yeso-please/backend#67 결정 후)
    travel_styles: dict[int, int] = Field(default_factory=dict)   # 문항 번호 → 1~7
    travel_motives: list[int] = Field(default_factory=list)
    liked_regions: list[str] = Field(default_factory=list)


class EmbeddingRequest(CamelModel):
    request_id: str
    model_version: str
    template_version: int
    profile: ProfileIn


class EmbeddingResponse(CamelModel):
    embedding_base64: str
    dimension: int


class AttractionIn(CamelModel):
    id: str
    name: str
    content_type_id: str | None = None
    lcls_systm1: str | None = None     # TourAPI KorService2 새 분류체계 코드
    lcls_systm2: str | None = None
    lcls_systm3: str | None = None
    region_name: str = ""
    tags: list[str] = Field(default_factory=list)   # 받지만 현재 템플릿은 쓰지 않는다
    description: str = ""


class BatchRequest(CamelModel):
    model_version: str
    template_version: int
    items: list[AttractionIn]


class BatchItemOut(CamelModel):
    id: str
    embedding_base64: str


class BatchResponse(CamelModel):
    dimension: int
    items: list[BatchItemOut]
