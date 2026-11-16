"""多格式文档加载器：txt / md / pdf / docx。

统一输出 Document(text, source, metadata)，屏蔽上游格式差异。
PDF 与 DOCX 解析器按需导入，缺依赖时给出明确提示而不是直接崩溃。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Document:
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


SUPPORTED = {".txt", ".md", ".markdown", ".pdf", ".docx"}


def load_one(path: Path) -> Document:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"不支持的文件类型 {suffix}：{path}")

    if suffix in {".txt", ".md", ".markdown"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("解析 PDF 需要安装 pypdf：pip install pypdf") from exc
        text = "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    else:
        try:
            from docx import Document as DocxDocument
        except ImportError as exc:
            raise RuntimeError("解析 DOCX 需要安装 python-docx：pip install python-docx") from exc
        doc = DocxDocument(str(path))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            parts.extend(" | ".join(cell.text for cell in row.cells) for row in table.rows)
        text = "\n".join(parts)

    return Document(
        text=clean_text(text),
        source=path.name,
        metadata={"suffix": suffix, "path": str(path), "size": path.stat().st_size},
    )


def clean_text(text: str) -> str:
    """清洗：统一换行、去多余空白、去页眉页脚常见的连续横线。"""
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    lines = [line for line in lines if line and set(line) != {"-"} and set(line) != {"_"}]
    return "\n".join(lines)


def load_dir(directory: Path) -> list[Document]:
    directory = Path(directory)
    docs = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            docs.append(load_one(path))
    return docs
