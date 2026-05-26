import os
# Limit threads to reduce memory footprint on Render free tier
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from functools import lru_cache
from typing import Iterable

import numpy as np
import torch

torch.set_num_threads(1)

from sentence_transformers import SentenceTransformer


MODEL_NAME = "./model_cache"


class SmartMatchModel:
    def __init__(self):
        # Force CPU to avoid CUDA initialization memory spike
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")

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