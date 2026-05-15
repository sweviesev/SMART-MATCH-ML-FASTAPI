from functools import lru_cache
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SmartMatchModel:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        return self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def score(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        query_embedding = self.encode([query])[0]
        candidate_embeddings = self.encode(candidates)
        similarities = candidate_embeddings @ query_embedding
        return [float(score) for score in similarities]


@lru_cache(maxsize=1)
def get_model() -> SmartMatchModel:
    return SmartMatchModel()


if __name__ == "__main__":
    get_model()