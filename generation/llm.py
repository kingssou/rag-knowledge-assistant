"""LLM 适配器：mock / openai / ollama 三种模式统一接口。"""
from __future__ import annotations

import os
from typing import Protocol

from config import Settings


class LLM(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str: ...


class MockLLM:
    """离线可复现的假模型：从参考资料中挑出与问题最相关的句子，组装成带引用的答案。

    存在的意义是让整条链路（检索 → 生成 → 引用 → 评估）在没有 API Key 的情况下也能跑通，
    并且结果稳定，方便做回归测试。
    """

    def chat(self, messages: list[dict[str, str]]) -> str:
        from retrieval.embedder import tokenize

        user = next(m["content"] for m in messages if m["role"] == "user")
        context, question = _split_context_question(user)
        qtokens = set(tokenize(question))
        blocks = _parse_blocks(context)

        candidates = []
        for idx, text in blocks:
            sentences = [s.strip() for s in text.replace("\n", " ").split("。") if s.strip()]
            for sentence in sentences:
                stokens = set(tokenize(sentence))
                overlap = len(qtokens & stokens) / len(qtokens) if qtokens else 0
                if overlap >= 0.3:
                    candidates.append((overlap, idx, sentence))
        candidates.sort(reverse=True)

        if not candidates:
            return "根据现有资料无法回答该问题。"

        seen: set[str] = set()
        lines = []
        for overlap, idx, sentence in candidates[:3]:
            key = sentence[:20]
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"{sentence}。[{idx}]")
        return "\n".join(lines)


def _split_context_question(user: str) -> tuple[str, str]:
    context = user.split("【参考资料】\n", 1)[1].split("\n\n【问题】", 1)[0]
    question = user.split("【问题】\n", 1)[1].rsplit("\n\n请按系统规则回答。", 1)[0]
    return context, question


def _parse_blocks(context: str) -> list[tuple[int, str]]:
    blocks = []
    for raw in context.split("\n\n"):
        if not raw.startswith("["):
            continue
        idx = int(raw[1:].split("]", 1)[0])
        body = raw.split("]", 1)[1].strip()
        # 去掉 “来源：xxx” 元信息行，只保留正文
        body = "\n".join(
            line for line in body.split("\n") if not line.strip().startswith("来源：")
        ).strip()
        blocks.append((idx, body))
    return blocks


class OpenAILLM:
    def __init__(self, cfg: Settings):
        from openai import OpenAI

        self._client = OpenAI(base_url=cfg.openai_base_url, api_key=os.getenv("OPENAI_API_KEY", ""))
        self._model = cfg.llm_model

    def chat(self, messages: list[dict[str, str]]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, temperature=0.1
        )
        return resp.choices[0].message.content or ""


class OllamaLLM:
    """本地 Ollama，兼容 OpenAI 接口协议。"""

    def __init__(self, cfg: Settings):
        from openai import OpenAI

        self._client = OpenAI(base_url=f"{cfg.ollama_base_url}/v1", api_key="ollama")
        self._model = cfg.ollama_model

    def chat(self, messages: list[dict[str, str]]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, temperature=0.1
        )
        return resp.choices[0].message.content or ""


def build_llm(cfg: Settings):
    if cfg.llm_mode == "openai":
        return OpenAILLM(cfg)
    if cfg.llm_mode == "ollama":
        return OllamaLLM(cfg)
    if cfg.llm_mode != "mock":
        raise ValueError(f"未知 LLM_MODE：{cfg.llm_mode}")
    return MockLLM()
