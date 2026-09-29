"""환경변수 설정. 백엔드 `embedding.model-version`·템플릿 버전·차원과 맞춰야 한다."""
import os
from dataclasses import dataclass

from tripin_ai.encoder import DEFAULT_MODEL_NAME, DEFAULT_MODEL_VERSION
from tripin_ai.templates import TEMPLATE_SETS


@dataclass(frozen=True)
class Settings:
    model_name: str        # Hugging Face 모델 id 또는 파인튜닝한 모델 폴더 경로
    model_version: str     # 요청의 modelVersion과 같아야 한다
    template_versions: tuple[int, ...]  # 요청 templateVersion 허용 목록 (tripin_ai.templates.TEMPLATE_SETS)
    max_batch_items: int


def _template_versions() -> tuple[int, ...]:
    raw = os.environ.get("TEMPLATE_VERSIONS", "1,2")
    try:
        versions = tuple(dict.fromkeys(int(value.strip()) for value in raw.split(",") if value.strip()))
    except ValueError as exc:
        raise ValueError("TEMPLATE_VERSIONS는 쉼표로 구분한 정수 목록이어야 합니다") from exc
    if not versions:
        raise ValueError("TEMPLATE_VERSIONS는 하나 이상의 버전을 포함해야 합니다")
    unknown = sorted(set(versions) - set(TEMPLATE_SETS))
    if unknown:
        raise ValueError(f"지원하지 않는 TEMPLATE_VERSIONS: {unknown}")
    return versions


def load_settings() -> Settings:
    return Settings(
        model_name=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME),
        model_version=os.environ.get("MODEL_VERSION", DEFAULT_MODEL_VERSION),
        template_versions=_template_versions(),
        max_batch_items=int(os.environ.get("MAX_BATCH_ITEMS", "64")),
    )
