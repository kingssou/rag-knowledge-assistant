"""上下文工程：token 预算、上下文裁剪、多轮对话管理。

这是 RAG 落地的核心工程点之一：检索到的片段不能无脑塞进 prompt，
必须在有限 context window 内优先保留最相关的证据，同时控制历史轮数。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from retrieval.embedder import tokenize


def estimate_tokens(text: str) -> int:
    """token 估算（中文按字、英文按词近似）。

    不依赖 tiktoken，保证离线可用；生产环境应替换为对应模型的 tokenizer。
    """
    return len(tokenize(text))


def fit_context(hits: list, budget: int) -> tuple[list, int]:
    """在 token 预算内按相关度顺序保留检索片段。

    返回 (保留的 hits, 实际使用的 token 数)。分数高的优先保留，
    放不下就跳过，保证不会撑爆 context window。
    """
    kept, used = [], 0
    for hit in hits:
        cost = estimate_tokens(hit.text) + 8  # 8 为引用编号与来源的开销
        if used + cost > budget:
            continue
        kept.append(hit)
        used += cost
    return kept, used


@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class Conversation:
    """多轮对话状态，只保留最近 N 轮，避免历史无限增长。"""
    max_turns: int = 3
    turns: list[Turn] = field(default_factory=list)

    def add(self, question: str, answer: str) -> None:
        self.turns.append(Turn(question, answer))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def to_messages(self) -> list[dict[str, str]]:
        messages = []
        for turn in self.turns:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.answer})
        return messages

    def clear(self) -> None:
        self.turns.clear()
