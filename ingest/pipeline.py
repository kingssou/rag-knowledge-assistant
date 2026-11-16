"""入库流程：加载 → 分块 → 向量化 → 写入 Chroma 索引。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from config import Settings, settings
from ingest.chunker import chunk_documents
from ingest.loader import load_dir, load_one
from retrieval.embedder import build_embedder


class IngestPipeline:
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg
        self.embedder = build_embedder(cfg)
        self._store = None

    def _collection(self):
        if self._store is None:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.cfg.index_dir))
            self._store = client.get_or_create_collection(
                name="knowledge_base", metadata={"hnsw:space": "cosine"}
            )
        return self._store

    def reset(self) -> None:
        import chromadb

        client = chromadb.PersistentClient(path=str(self.cfg.index_dir))
        try:
            client.delete_collection("knowledge_base")
        except Exception:
            pass
        self._store = None

    def ingest(self, source: str | Path) -> dict[str, Any]:
        """source 可以是单个文件或目录。"""
        path = Path(source)
        docs = [load_one(path)] if path.is_file() else load_dir(path)
        if not docs:
            raise ValueError(f"没有找到可解析的文档：{source}")

        chunks = chunk_documents(docs, self.cfg.chunk_size, self.cfg.chunk_overlap)
        vectors = self.embedder.embed([c.text for c in chunks])
        collection = self._collection()

        # 按来源清理旧版本，保证重复入库不产生幽灵分块
        sources = {d.source for d in docs}
        for src in sources:
            collection.delete(where={"source": src})

        collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "start": c.start, "end": c.end} for c in chunks],
            embeddings=vectors,
        )
        return {
            "documents": len(docs),
            "chunks": len(chunks),
            "sources": sorted(sources),
            "embed_model": self.cfg.embed_model,
        }
