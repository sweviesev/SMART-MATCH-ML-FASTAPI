from pydantic import BaseModel, Field


class CandidateInput(BaseModel):
    id: str
    text: str


class ScoreRequest(BaseModel):
    query: str
    candidates: list[CandidateInput] = Field(default_factory=list)


class CandidateScore(BaseModel):
    id: str
    score: float


class ScoreResponse(BaseModel):
    scores: list[CandidateScore]
