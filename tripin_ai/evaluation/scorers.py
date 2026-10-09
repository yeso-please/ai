"""평가할 추천 방식. 모두 같은 인터페이스를 쓴다.

fit(train_visits)                   학습 묶음 방문으로 준비 (인기 계산, 행렬분해 등)
score(travel_id, items, history)    후보 관광지 인덱스들의 점수. 높을수록 추천.
                                    history는 웜 스타트에서 공개한 [(관광지 인덱스, 만족도)]. 콜드면 None.
                                    지원하지 않는 조건이면 None을 돌려준다(예: 콜드 스타트의 행렬분해).
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from .data import popularity


class Scorer:
    name = ""

    def fit(self, train_visits: pd.DataFrame, fold: int | None = None) -> None:
        pass

    def score(self, travel_id: str, items: np.ndarray, history=None) -> np.ndarray | None:
        raise NotImplementedError


class RandomScorer(Scorer):
    name = "무작위"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def score(self, travel_id, items, history=None):
        return self.rng.random(len(items))


class PopularityScorer(Scorer):
    """학습 묶음에서 많이 방문된 곳. 비개인화 기준선."""
    name = "인기"

    def __init__(self, n_attractions: int):
        self.n = n_attractions
        self.pop = np.zeros(n_attractions)

    def fit(self, train_visits, fold=None):
        self.pop = popularity(train_visits, self.n)

    def score(self, travel_id, items, history=None):
        return self.pop[items]


class TasteScorer(Scorer):
    """회원 벡터 · 관광지 벡터 (코사인, 둘 다 정규화돼 있음). 설문만 쓴다(콜드 스타트 가능)."""

    def __init__(self, name: str, attraction_vectors: np.ndarray, traveler_vectors: np.ndarray, traveler_ids: list[str]):
        self.name = name
        self.a_vec = attraction_vectors
        self.t_vec = traveler_vectors
        self.t_index = {t: i for i, t in enumerate(traveler_ids)}

    def score(self, travel_id, items, history=None):
        return self.a_vec[items] @ self.t_vec[self.t_index[travel_id]]


class PopularityFloorTasteScorer(Scorer):
    """인기를 순위가 아니라 품질 하한선으로만 쓴다: 방문 이력이 floor 이상인 곳을 먼저, 그 안에서는 취향순."""

    def __init__(self, taste: TasteScorer, n_attractions: int, floor: int = 1):
        self.name = f"인기 하한선(≥{floor}) + {taste.name}"
        self.taste, self.floor = taste, floor
        self.popularity = PopularityScorer(n_attractions)

    def fit(self, train_visits, fold=None):
        self.popularity.fit(train_visits, fold)

    def score(self, travel_id, items, history=None):
        # 취향 점수는 −1~1이라 +10이면 하한선을 넘은 곳이 항상 위에 온다.
        return 10.0 * (self.popularity.score(travel_id, items) >= self.floor) + self.taste.score(travel_id, items)


class ItemSimilarityScorer(Scorer):
    """웜 스타트용 콘텐츠 기반: 공개된 방문 중 만족(4점 이상)한 곳들과 관광지 벡터가 비슷한 정도."""

    def __init__(self, name: str, attraction_vectors: np.ndarray):
        self.name = name
        self.a_vec = attraction_vectors

    def score(self, travel_id, items, history=None):
        if not history:
            return None
        liked = [i for i, s in history if s >= 4] or [i for i, _ in history]
        return self.a_vec[items] @ self.a_vec[liked].mean(axis=0)


class PureSVDScorer(Scorer):
    """협업 필터링 기준선 PureSVD (여행자 × 관광지 방문 행렬의 절단 SVD).

    AI Hub 공식 모델은 Surprise의 SVD(평점 예측)였다. 여기서는 상위 추천 평가에 흔히 쓰는 PureSVD로 재현한다.
    새 여행자는 공개된 방문으로 접어 넣는다(fold-in): 점수 = r_u · V · Vᵀ.
    """
    name = "협업 필터링(PureSVD)"

    def __init__(self, n_attractions: int, factors: int = 64):
        self.n, self.factors = n_attractions, factors
        self.V = None

    def fit(self, train_visits, fold=None):
        users = {t: i for i, t in enumerate(train_visits.travel_id.unique())}
        rows = train_visits.travel_id.map(users).values
        matrix = csr_matrix((np.ones(len(rows)), (rows, train_visits.a_idx.astype(int).values)), shape=(len(users), self.n))
        _, _, vt = svds(matrix, k=min(self.factors, min(matrix.shape) - 1))
        self.V = vt.T   # (관광지, 요인)

    def score(self, travel_id, items, history=None):
        if not history:
            return None
        profile = self.V[[i for i, _ in history]].sum(axis=0)
        return self.V[items] @ profile


class FoldTasteScorer(Scorer):
    """겹마다 따로 학습한 모델의 벡터로 점수를 매긴다(평가 묶음 여행자는 그 겹의 학습에 쓰이지 않았다).

    vectors: 겹 번호 → (관광지 벡터 (전체 관광지 수 × 차원, 임베딩 안 한 곳은 0), {travel_id: 여행자 벡터})
    """

    def __init__(self, name: str, vectors: dict[int, tuple[np.ndarray, dict[str, np.ndarray]]]):
        self.name, self.vectors = name, vectors
        self.current = None

    def fit(self, train_visits, fold=None):
        self.current = self.vectors.get(fold)

    def score(self, travel_id, items, history=None):
        if self.current is None:
            return None
        a_vec, t_vec = self.current
        return a_vec[items] @ t_vec[travel_id]


class ShuffledSurveyScorer(Scorer):
    """대조군: 같은 겹의 **다른 여행자** 설문 벡터로 점수를 매긴다.

    관광지 벡터는 그대로이므로 "관광객이 갈 만한 곳" 같은 장소 자체의 매력은 그대로 남고, 개인 설문과의 대응만 깨진다.
    원래 점수와의 차이가 설문(개인화)에서 나온 몫이다. 자기 자신과 짝지어지지 않게 한 칸씩 밀어서 섞는다.
    """

    def __init__(self, tuned: FoldTasteScorer, seed: int = 0):
        self.name = f"{tuned.name} · 설문 섞음 (대조군)"
        self.tuned, self.seed = tuned, seed
        self.current = None

    def fit(self, train_visits, fold=None):
        vectors = self.tuned.vectors.get(fold)
        if vectors is None:
            self.current = None
            return
        a_vec, t_vec = vectors
        ids = sorted(t_vec)
        order = np.random.default_rng(self.seed + (fold or 0)).permutation(len(ids))
        donor = {ids[order[i]]: ids[order[(i + 1) % len(ids)]] for i in range(len(ids))}
        self.current = (a_vec, {t: t_vec[donor[t]] for t in ids})

    def score(self, travel_id, items, history=None):
        if self.current is None:
            return None
        a_vec, t_vec = self.current
        return a_vec[items] @ t_vec[travel_id]
