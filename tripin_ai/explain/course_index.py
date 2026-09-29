"""관광공사 추천코스 색인과 비슷한 코스 검색 (RAG의 검색 단계).

코스 벡터 = 코스 지점 관광지 문장 벡터의 평균(정규화). 우리 코스도 같은 방식으로 벡터를 만들어
코사인 유사도가 높은 관광공사 코스를 문체 예시로 가져온다. 예시는 문체 참고용이고 그대로 옮기지 않는다.

색인 파일(.npz)은 TourAPI 공개 데이터만 담는다(AI Hub 데이터 없음). scripts/build_course_index.py로 만든다.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CourseExample:
    course_id: str
    title: str
    theme: str
    overview: str
    stop_names: list[str]
    stop_descriptions: list[str]
    similarity: float


class CourseIndex:
    def __init__(self, path: str | Path):
        data = np.load(path, allow_pickle=True)
        self.vectors = data["vectors"].astype("<f4")
        self.course_ids = data["course_ids"].tolist()
        self.titles = data["titles"].tolist()
        self.themes = data["themes"].tolist()
        self.overviews = data["overviews"].tolist()
        self.stop_names = [s.split("|") for s in data["stop_names"].tolist()]
        self.stop_descriptions = [s.split("|") for s in data["stop_descriptions"].tolist()]
        self.model_version = str(data["model_version"])

    def __len__(self) -> int:
        return len(self.course_ids)

    def search(self, query: np.ndarray, k: int = 3, exclude: set[str] | None = None,
               require_overview: bool = True) -> list[CourseExample]:
        query = query / (np.linalg.norm(query) + 1e-12)
        order = np.argsort(-(self.vectors @ query))
        results = []
        for i in order:
            if exclude and self.course_ids[i] in exclude:
                continue
            if require_overview and not self.overviews[i].strip():
                continue
            results.append(CourseExample(self.course_ids[i], self.titles[i], self.themes[i], self.overviews[i],
                                         self.stop_names[i], self.stop_descriptions[i],
                                         float(self.vectors[i] @ query)))
            if len(results) == k:
                break
        return results


def course_vector(place_vectors: np.ndarray) -> np.ndarray:
    v = np.asarray(place_vectors, dtype="<f4").mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-12)
