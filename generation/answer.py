"""带引用、上下文预算、多轮对话与流式输出的答案生成引擎。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from config import Settings, settings
from generation.context import Conversation, fit_context
from generation.llm import LLM, build_llm
from generation.prompts import build_messages
from retrieval.hybrid import Hit, HybridRetriever

_CITE = re.compile(r"\[(\d+)\]")
REFUSAL = "根据现有资料无法回答该问题。"


@dataclass
class Answer:
    question: str
    text: str
    citations: list[dict]
    refused: bool
    confidence: float
    context_tokens: int = 0


class AnswerEngine:
    def __init__(
        self,
        cfg: Settings = settings,
        retriever: HybridRetriever | None = None,
        llm: LLM | None = None,
    ):
        self.cfg = cfg
        self.retriever = retriever or HybridRetriever(cfg)
        self.llm = llm or build_llm(cfg)
        self.conversation = Conversation(max_turns=cfg.history_turns)

    def _prepare(self, question: str) -> tuple[list[Hit], int, float]:
        hits = self.retriever.retrieve(question)
        if not hits:
            return [], 0, 0.0
        confidence = hits[0].score
        if confidence < self.cfg.min_confidence:
            return [], 0, confidence
        # 上下文工程：在 token 预算内保留最相关的证据
        kept, used = fit_context(hits, self.cfg.context_token_budget)
        return kept, used, confidence

    def answer(self, question: str, use_history: bool = False) -> Answer:
        hits, used, confidence = self._prepare(question)
        if not hits:
            return Answer(question, REFUSAL, [], True, confidence)

        messages = build_messages(question, hits)
        if use_history:
            history = self.conversation.to_messages()
            messages = [messages[0], *history, messages[1]]

        raw = self.llm.chat(messages)
        text, citations = self._parse(raw, hits)
        if use_history:
            self.conversation.add(question, text)
        return Answer(question, text, citations, REFUSAL in text, round(confidence, 4), used)

    def stream(self, question: str) -> Iterable[str]:
        """流式问答：先做检索与拒答判断，再逐 token 返回生成结果。"""
        hits, used, confidence = self._prepare(question)
        if not hits:
            yield REFUSAL
            return
        yield from self.llm.stream(build_messages(question, hits))

    def _parse(self, raw: str, hits: list[Hit]) -> tuple[str, list[dict]]:
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
        return raw.strip(), citations

    def reset_conversation(self) -> None:
        self.conversation.clear()
