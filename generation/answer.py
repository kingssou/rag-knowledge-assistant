"""带引用与拒答的答案生成。"""
from __future__ import annotations

import re
from dataclasses import dataclass

from config import Settings, settings
from generation.llm import LLM, build_llm
from generation.prompts import build_messages
from retrieval.hybrid import Hit, HybridRetriever

_CITE = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    question: str
    text: str
    citations: list[dict[str, str]]
    refused: bool
    confidence: float


class AnswerEngine:
    def __init__(self, cfg: Settings = settings, retriever: HybridRetriever | None = None, llm: LLM | None = None):
        self.cfg = cfg
        self.retriever = retriever or HybridRetriever(cfg)
        self.llm = llm or build_llm(cfg)

    def answer(self, question: str, hits: list[Hit] | None = None) -> Answer:
        hits = hits if hits is not None else self.retriever.retrieve(question)
        if not hits:
            return Answer(question, "根据现有资料无法回答该问题。", [], True, 0.0)

        confidence = hits[0].score
        if confidence < self.cfg.min_confidence:
            return Answer(question, "根据现有资料无法回答该问题。", [], True, confidence)

        raw = self.llm.chat(build_messages(question, hits[: self.cfg.rerank_top_k]))
        used = sorted({int(n) for n in _CITE.findall(raw)})
        citations = []
        for n in used:
            if 1 <= n <= len(hits):
                citations.append(
                    {
                        "index": n,
                        "source": hits[n - 1].source,
                        "chunk_id": hits[n - 1].chunk_id,
                        "score": round(hits[n - 1].score, 4),
                    }
                )
        refused = "根据现有资料无法回答" in raw
        return Answer(question, raw.strip(), citations, refused, round(confidence, 4))
