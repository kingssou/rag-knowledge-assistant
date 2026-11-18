"""全局配置。默认使用可离线运行的 demo 模式，无需任何 API Key。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 常见模型服务的 Key 变量名，按顺序自动探测
_KEY_CANDIDATES = (
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "MOONSHOT_API_KEY",
    "ZHIPU_API_KEY",
    "ARK_API_KEY",
    "SILICONFLOW_API_KEY",
)


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """极简 .env 加载，不引入额外依赖；已存在的环境变量优先，不会被覆盖。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()


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

    #  LLM：mock（离线可跑） | openai | ollama | 任意 OpenAI 兼容服务（DeepSeek/通义/智谱等）
    llm_mode: str = "mock"
    llm_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"     # 从哪个环境变量读 Key
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"

    #  生成参数与上下文预算
    temperature: float = 0.1
    max_tokens: int = 600
    context_token_budget: int = 2400       # 注入 prompt 的上下文 token 上限
    history_turns: int = 3                 # 多轮对话保留的历史轮数
    stream: bool = False                   # 是否流式输出

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.embed_model = os.getenv("EMBED_MODEL", s.embed_model)
        s.llm_mode = os.getenv("LLM_MODE", s.llm_mode)
        s.llm_model = os.getenv("LLM_MODEL", s.llm_model)
        s.openai_base_url = os.getenv("OPENAI_BASE_URL", s.openai_base_url)
        s.api_key_env = os.getenv("API_KEY_ENV", s.api_key_env)
        s.ollama_base_url = os.getenv("OLLAMA_BASE_URL", s.ollama_base_url)
        s.ollama_model = os.getenv("OLLAMA_MODEL", s.ollama_model)
        s.top_k = int(os.getenv("TOP_K", s.top_k))
        s.min_confidence = float(os.getenv("MIN_CONFIDENCE", s.min_confidence))
        s.temperature = float(os.getenv("TEMPERATURE", s.temperature))
        s.max_tokens = int(os.getenv("MAX_TOKENS", s.max_tokens))
        s.context_token_budget = int(os.getenv("CONTEXT_TOKEN_BUDGET", s.context_token_budget))
        s.history_turns = int(os.getenv("HISTORY_TURNS", s.history_turns))
        s.data_dir.mkdir(parents=True, exist_ok=True)
        s.index_dir.mkdir(parents=True, exist_ok=True)
        return s

    def resolve_api_key(self) -> str:
        """按「api_key_env 指定名 → 常见 Key 变量名」的顺序找 Key。"""
        value = os.getenv(self.api_key_env, "").strip()
        if value:
            return value
        for name in _KEY_CANDIDATES:
            candidate = os.getenv(name, "").strip()
            if candidate:
                return candidate
        return ""


settings = Settings.from_env()
