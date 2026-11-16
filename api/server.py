"""FastAPI 服务：健康检查、入库、问答、评估。"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from config import settings
from evaluation.runner import run as run_eval
from generation.answer import AnswerEngine
from ingest.pipeline import IngestPipeline

app = FastAPI(title="企业知识库 RAG 助手", version="1.0.0")
_engine = AnswerEngine()


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class IngestRequest(BaseModel):
    source: str = Field(description="文件或目录路径")


@app.get("/health")
def health():
    return {"status": "ok", "llm_mode": settings.llm_mode, "embed_model": settings.embed_model}


@app.post("/ingest")
def ingest(req: IngestRequest):
    result = IngestPipeline().ingest(Path(req.source))
    return result


@app.post("/ask")
def ask(req: AskRequest):
    answer = _engine.answer(req.question)
    return {
        "question": answer.question,
        "answer": answer.text,
        "refused": answer.refused,
        "confidence": answer.confidence,
        "citations": answer.citations,
    }


@app.post("/evaluate")
def evaluate():
    return run_eval()["summary"]
