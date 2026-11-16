"""全局配置。默认使用可离线运行的 demo 模式，无需任何 API Key。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Settings:
    """所有配置项都可以通过环境变量覆盖，便于切换真实模型。"""

    data_dir: Path = ROOT / "data"
    sample_docs_dir: Path = ROOT / "data" / "sample_docs"
    index_dir: Path = ROOT / "data" / "chroma_index"
    eval_dir: Path = ROOT / "evaluation"

    # 分块与检索
    chunk_size: int = 250
    chunk_overlap: int = 40
    top_k: int = 8
    rerank_top_k: int = 3
    alpha: float = 0.5          # 混合检索中向量相似度的权重，1-BM25
    min_confidence: float = 0.22  # 低于该分数触发拒答

    #  Embedding：hash（离线可跑） | sentence-transformers | openai
    embed_model: str = "hash"
    embed_dim: int = 384
    embed_model_name: str = "all-MiniLM-L6-v2"

    #  LLM：mock（离线可跑） | openai | ollama
    llm_mode: str = "mock"
    llm_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.embed_model = os.getenv("EMBED_MODEL", s.embed_model)
        s.llm_mode = os.getenv("LLM_MODE", s.llm_mode)
        s.llm_model = os.getenv("LLM_MODEL", s.llm_model)
        s.openai_base_url = os.getenv("OPENAI_BASE_URL", s.openai_base_url)
        s.ollama_base_url = os.getenv("OLLAMA_BASE_URL", s.ollama_base_url)
        s.ollama_model = os.getenv("OLLAMA_MODEL", s.ollama_model)
        s.top_k = int(os.getenv("TOP_K", s.top_k))
        s.min_confidence = float(os.getenv("MIN_CONFIDENCE", s.min_confidence))
        s.data_dir.mkdir(parents=True, exist_ok=True)
        s.index_dir.mkdir(parents=True, exist_ok=True)
        return s


settings = Settings.from_env()
