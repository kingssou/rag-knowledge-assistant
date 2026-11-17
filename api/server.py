"""FastAPI 服务：健康检查、入库、问答、评估。"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import ROOT, settings
from evaluation.runner import run as run_eval
from generation.answer import AnswerEngine
from ingest.pipeline import IngestPipeline

app = FastAPI(title="企业知识库 RAG 助手", version="1.0.0")
_engine = AnswerEngine()
app.mount("/static", StaticFiles(directory=str(ROOT / "api" / "static")), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(ROOT / "api" / "static" / "index.html"))


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    history: bool = Field(default=False, description="是否启用多轮对话上下文")


class StructuredRequest(BaseModel):
    question: str = Field(min_length=1)


class IngestRequest(BaseModel):
    source: str = Field(description="文件或目录路径")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_mode": settings.llm_mode,
        "llm_model": settings.llm_model,
        "embed_model": settings.embed_model,
        "context_token_budget": settings.context_token_budget,
        "history_turns": settings.history_turns,
    }


@app.post("/ingest")
def ingest(req: IngestRequest):
    result = IngestPipeline().ingest(Path(req.source))
    return result


@app.post("/ask")
def ask(req: AskRequest):
    answer = _engine.answer(req.question, use_history=req.history)
    return {
        "question": answer.question,
        "answer": answer.text,
        "refused": answer.refused,
        "confidence": answer.confidence,
        "citations": answer.citations,
        "context_tokens": answer.context_tokens,
        "llm_mode": settings.llm_mode,
    }


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    """流式问答：SSE 风格逐块返回，模拟真实大模型打字效果。"""
    def generate():
        for chunk in _engine.stream(req.question):
            yield chunk
    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@app.post("/conversation/reset")
def reset_conversation():
    _engine.reset_conversation()
    return {"status": "cleared"}


@app.post("/structured")
def structured(req: StructuredRequest):
    """结构化输出演示：让大模型返回可解析的 JSON。"""
    from generation.prompts import build_messages

    hits, _, _ = _engine._prepare(req.question)
    if not hits:
        return {"answer": "根据现有资料无法回答该问题。", "confidence": 0.0, "sources": []}
    messages = build_messages(req.question, hits)
    if hasattr(_engine.llm, "structured"):
        result = _engine.llm.structured(messages)
    else:
        result = {"answer": _engine.llm.chat(messages), "sources": [h.source for h in hits]}
    return result


@app.post("/evaluate")
def evaluate():
    return run_eval()["summary"]
