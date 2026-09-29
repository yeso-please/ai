"""문장 → 벡터. 백엔드 계약: L2 정규화한 float32 리틀엔디언 바이트를 base64로 주고받는다."""
import base64

import numpy as np

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_MODEL_VERSION = "mminilm-l12-v1"


class Encoder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, model_version: str = DEFAULT_MODEL_VERSION):
        # 무거운 import는 실제로 모델을 쓸 때만 한다(템플릿만 쓰는 테스트가 가볍도록).
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.model_name = model_name
        self.model_version = model_version
        # sentence-transformers 6에서 이름이 바뀌었다.
        get_dimension = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dimension = get_dimension()

    def encode(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        vectors = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=show_progress)
        return vectors.astype("<f4")


def to_base64(vector: np.ndarray) -> str:
    return base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode("ascii")


def from_base64(value: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(value), dtype="<f4")
