"""Embedding 适配器。

默认 hash 模式完全离线、零依赖、结果可复现，适合演示与测试。
切换到真实模型只需要设置 EMBED_MODEL 环境变量。
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from config import Settings

_WORD = re.compile(r"[a-z0-9]+")
_HAN = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """中英文混合分词：英文词 + 中文单字 + 中文 bigram。"""
    low = text.lower()
    words = _WORD.findall(low)
    chars = _HAN.findall(low)
    bigrams = ["".join(pair) for pair in zip(chars, chars[1:])]
    return words + chars + bigrams


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """确定性哈希向量：无需下载模型，可复现，检索效果足以跑通全链路。"""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = tokenize(text)
        for tok in tokens:
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class SentenceTransformerEmbedder:
    """本地句向量模型，需要 pip install sentence-transformers。"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]


class OpenAIEmbedder:
    """OpenAI 兼容接口，可指向任意兼容服务。"""

    def __init__(self, base_url: str, model: str = "text-embedding-3-small", dim: int = 384):
        import os

        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=os.getenv("OPENAI_API_KEY", "unused"))
        self._model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts, dimensions=self.dim)
        return [item.embedding for item in resp.data]


def build_embedder(cfg: Settings):
    if cfg.embed_model == "sentence-transformers":
        return SentenceTransformerEmbedder(cfg.embed_model_name)
    if cfg.embed_model == "openai":
        return OpenAIEmbedder(cfg.openai_base_url)
    if cfg.embed_model != "hash":
        raise ValueError(f"未知 EMBED_MODEL：{cfg.embed_model}")
    return HashEmbedder(cfg.embed_dim)
