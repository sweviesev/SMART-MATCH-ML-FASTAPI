from fastapi import FastAPI, HTTPException

from app.model import get_model
from app.schemas import ScoreRequest, ScoreResponse


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload the model on startup so the first request doesn't timeout
    print("Loading ML model...")
    get_model()
    print("Model loaded successfully.")
    yield

app = FastAPI(
    title="CREATECH Smart Match ML API",
    description="Separate FastAPI service for semantic Smart Match scoring.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "createch-smart-match-ml",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
    }


@app.post("/predict", response_model=ScoreResponse)
def predict(payload: ScoreRequest):
    query = payload.query.strip()
    if len(query) < 3:
        raise HTTPException(status_code=400, detail="Query is too short.")

    model = get_model()
    raw_scores = model.score(query, [candidate.text for candidate in payload.candidates])
    return {
        "scores": [
            {"id": candidate.id, "score": score}
            for candidate, score in zip(payload.candidates, raw_scores)
        ]
    }
