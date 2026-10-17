"""문장 → 벡터. 백엔드 계약: L2 정규화한 float32 리틀엔디언 바이트를 base64로 주고받는다."""
import base64
import json
from pathlib import Path

import numpy as np

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_MODEL_VERSION = "mminilm-l12-v1"


class Encoder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, model_version: str = DEFAULT_MODEL_VERSION,
                 input_format: str = "auto"):
        # 무거운 import는 실제로 모델을 쓸 때만 한다(템플릿만 쓰는 테스트가 가볍도록).
        from sentence_transformers import SentenceTransformer

        info_path = Path(model_name) / "training_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
        if input_format == "auto":
            input_format = info.get("input_format", "e5" if "multilingual-e5" in model_name else "plain")
        if input_format not in ("plain", "e5"):
            raise ValueError("Unsupported model input format")
        if info.get("input_format") and info["input_format"] != input_format:
            raise ValueError("Input format differs from training")
        if input_format == "e5" and info.get("version") and info["version"] != model_version:
            raise ValueError("Model version differs from checkpoint")
        self.input_format = input_format
        self.model = SentenceTransformer(model_name, device="cpu")
        if input_format == "e5":
            self.model.max_seq_length = int(info.get("max_seq_length", 256))
        self.model_name = model_name
        self.model_version = model_version
        # sentence-transformers 6에서 이름이 바뀌었다.
        get_dimension = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dimension = get_dimension()
        if info.get("embedding_dimension") and info["embedding_dimension"] != self.dimension:
            raise ValueError("Embedding dimension differs from training")

    def encode(self, texts: list[str], batch_size: int = 64, show_progress: bool = False,
               role: str = "passage") -> np.ndarray:
        if role not in ("query", "passage"):
            raise ValueError("Unsupported embedding role")
        if self.input_format == "e5":
            texts = [f"{role}: " + text for text in texts]
        vectors = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=show_progress)
        return vectors.astype("<f4")


def to_base64(vector: np.ndarray) -> str:
    return base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode("ascii")


def from_base64(value: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(value), dtype="<f4")
