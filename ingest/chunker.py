"""带重叠的递归分块器。

优先在段落边界切分，段落过长时退化到句子边界，最后才按字符硬切。
每个分块保留来源文档名与序号，供后续引用溯源使用。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ingest.loader import Document


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: str
    start: int
    end: int


_SENTENCE_END = re.compile(r"(?<=[。！？!?；;\n])")


def split_into_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _merge(parts: list[str], size: int, overlap: int) -> list[tuple[str, int, int]]:
    """把片段合并为不超过 size 的窗口，窗口之间保留 overlap 个字符的重叠。"""
    windows: list[tuple[str, int, int]] = []
    buf: list[str] = []
    buf_len = 0
    start = 0
    for part in parts:
        added = len(part) + (1 if buf else 0)
        if buf and buf_len + added > size:
            text = "\n".join(buf)
            windows.append((text, start, start + len(text)))
            # 回退尾部若干字符作为重叠
            tail = text[-overlap:] if overlap else ""
            buf = [tail, part] if tail else [part]
            buf_len = len(tail) + (1 if tail else 0) + len(part)
            start = start + len(text) - len(tail)
        else:
            buf.append(part)
            buf_len += added
    if buf:
        text = "\n".join(buf)
        windows.append((text, start, start + len(text)))
    return windows


def chunk_document(doc: Document, size: int = 500, overlap: int = 80) -> list[Chunk]:
    if size <= 0:
        raise ValueError("size 必须为正数")
    overlap = max(0, min(overlap, size // 2))

    paragraphs = [p.strip() for p in doc.text.split("\n") if p.strip()]
    pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= size:
            pieces.append(para)
        else:
            pieces.extend(split_into_sentences(para))

    chunks = []
    for idx, (text, start, end) in enumerate(_merge(pieces, size, overlap)):
        chunks.append(
            Chunk(
                text=text,
                source=doc.source,
                chunk_id=f"{doc.source}::{idx}",
                start=start,
                end=end,
            )
        )
    return chunks


def chunk_documents(docs: list[Document], size: int = 500, overlap: int = 80) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, size, overlap))
    return out
