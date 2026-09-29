"""환경변수 설정. 백엔드 `embedding.model-version`·`template-version`·`expected-dimension`과 맞춰야 한다."""
import os
from dataclasses import dataclass

from tripin_ai.encoder import DEFAULT_MODEL_NAME, DEFAULT_MODEL_VERSION


@dataclass(frozen=True)
class Settings:
    model_name: str        # Hugging Face 모델 id 또는 파인튜닝한 모델 폴더 경로
    model_version: str     # 요청의 modelVersion과 같아야 한다
    template_version: int  # 요청의 templateVersion과 같아야 한다 (tripin_ai.templates.TEMPLATE_SETS)
    max_batch_items: int


def load_settings() -> Settings:
    return Settings(
        model_name=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME),
        model_version=os.environ.get("MODEL_VERSION", DEFAULT_MODEL_VERSION),
        template_version=int(os.environ.get("TEMPLATE_VERSION", "1")),
        max_batch_items=int(os.environ.get("MAX_BATCH_ITEMS", "64")),
    )
