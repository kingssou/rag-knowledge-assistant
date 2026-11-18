"""LLM 适配器层。

统一接口支持多种真实模型与离线 mock，并覆盖 AI 应用开发的核心技术点：
- 通用 OpenAI 兼容协议（可接 DeepSeek / 通义 / 智谱 / OpenAI / vLLM）
- 本地 Ollama
- 流式输出（streaming）
- 结构化 JSON 输出（structured output）
- 生成参数控制（temperature / max_tokens）
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Protocol

from config import Settings


class LLM(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str: ...

    def stream(self, messages: list[dict[str, str]]) -> Iterable[str]: ...


class _MockStream:
    """把整段回答按句切成流，模拟真实流式输出（保持与 chat 完全一致的内容）。"""

    def __init__(self, text: str):
        segments = text.split("。")
        self._parts = [s + "。" for s in segments[:-1] if s.strip()]
        if segments[-1].strip():
            self._parts.append(segments[-1])

    def __iter__(self):
        for part in self._parts:
            yield part


class MockLLM:
    """离线可复现的假模型，用于无 Key 环境与回归测试。

    注意：它不是大模型，只用于跑通链路与确定性测试。
    真实能力请切到 LLM_MODE=openai / ollama。
    """

    def __init__(self, temperature: float = 0.1):
        self.temperature = temperature

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

    def stream(self, messages: list[dict[str, str]]) -> Iterable[str]:
        return _MockStream(self.chat(messages))


class OpenAICompatibleLLM:
    """通用 OpenAI 兼容适配器。

    只要服务实现了 /chat/completions 就能接：OpenAI、DeepSeek、通义千问、
    智谱 GLM、Moonshot、vLLM、本地 Ollama 的 OpenAI 兼容端点等。
    切换模型只改环境变量，不改代码。
    """

    def __init__(self, cfg: Settings, base_url: str, model: str, api_key: str):
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        self._model = model
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens

    def _params(self, stream: bool) -> dict:
        return {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": stream,
        }

    def chat(self, messages: list[dict[str, str]]) -> str:
        resp = self._client.chat.completions.create(
            messages=messages, **self._params(False)
        )
        return resp.choices[0].message.content or ""

    def stream(self, messages: list[dict[str, str]]) -> Iterable[str]:
        for chunk in self._client.chat.completions.create(
            messages=messages, **self._params(True)
        ):
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def structured(self, messages: list[dict[str, str]], schema_name: str = "answer") -> dict:
        """结构化输出：要求模型返回 JSON，并做解析与容错。

        response_format=json_object 走原生 JSON 模式；若模型不支持则按文本解析。
        生产环境应改用 JSON Schema 约束（schema 参数）以保证字段稳定。
        """
        msgs = list(messages) + [
            {"role": "system", "content": f"只输出合法 JSON，顶层键为 {schema_name}，不要 Markdown 代码块。"}
        ]
        try:
            resp = self._client.chat.completions.create(
                messages=msgs,
                response_format={"type": "json_object"},
                **self._params(False),
            )
            raw = resp.choices[0].message.content or "{}"
        except Exception:
            raw = self.chat(msgs)
        return _safe_json(raw)


def _safe_json(raw: str) -> dict:
    """从可能含 Markdown 围栏或前后缀的文本中解析 JSON。"""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


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
        body = "\n".join(
            line for line in body.split("\n") if not line.strip().startswith("来源：")
        ).strip()
        blocks.append((idx, body))
    return blocks


def build_llm(cfg: Settings):
    if cfg.llm_mode == "mock":
        return MockLLM(cfg.temperature)
    if cfg.llm_mode == "openai":
        key = cfg.resolve_api_key()
        if not key:
            raise RuntimeError(
                "未找到 API Key。请在项目根目录的 .env 里写一行，例如：\n"
                "  DEEPSEEK_API_KEY=sk-xxxx\n"
                "或设置环境变量 LLM_MODE / OPENAI_BASE_URL / LLM_MODEL。\n"
                "参考 .env.example。"
            )
        return OpenAICompatibleLLM(cfg, cfg.openai_base_url, cfg.llm_model, key)
    if cfg.llm_mode == "ollama":
        return OpenAICompatibleLLM(
            cfg, f"{cfg.ollama_base_url}/v1", cfg.ollama_model, "ollama"
        )
    raise ValueError(f"未知 LLM_MODE：{cfg.llm_mode}")
