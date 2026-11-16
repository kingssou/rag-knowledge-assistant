"""向量检索 + BM25 关键词检索的混合召回与重排序。

设计要点：
- 向量召回语义相近，BM25 召回关键词精确匹配，两者互补；
- 分数先分别归一化再加权，避免不同量纲直接相加；
- 重排序对答案中真正出现的查询词额外加权，缓解哈希向量精度有限的问题。
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from config import Settings, settings
from retrieval.embedder import Embedder, build_embedder, tokenize


@dataclass
class Hit:
    chunk_id: str
    text: str
    source: str
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0


class BM25:
    """轻量 BM25 实现，避免引入额外依赖。"""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [len(t) for t in self.tfs]
        self.avgdl = sum(self.lens) / len(self.lens) if self.lens else 1.0
        self.df: Counter[str] = Counter()
        for tf in self.tfs:
            self.df.update(tf.keys())
        self.n = len(docs)

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = [0.0] * self.n
        for term in q:
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self.tfs):
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
                out[i] += idf * f * (self.k1 + 1) / denom
        return out


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


class HybridRetriever:
    def __init__(self, cfg: Settings = settings, embedder: Embedder | None = None):
        self.cfg = cfg
        self.embedder = embedder or build_embedder(cfg)

    def _collection(self):
        import chromadb

        client = chromadb.PersistentClient(path=str(self.cfg.index_dir))
        return client.get_or_create_collection(
            name="knowledge_base", metadata={"hnsw:space": "cosine"}
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[Hit]:
        top_k = top_k or self.cfg.top_k
        collection = self._collection()
        total = collection.count()
        if total == 0:
            return []

        qvec = self.embedder.embed([query])[0]
        res = collection.query(query_embeddings=[qvec], n_results=min(top_k, total))
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        distances = res["distances"][0]
        vec_scores = [max(0.0, 1.0 - d) for d in distances]

        # BM25 在全库上算，保证不会因为向量召回漏掉关键词命中
        all_rows = collection.get(include=["documents", "metadatas"])
        all_ids, all_docs = all_rows["ids"], all_rows["documents"]
        bm = BM25(all_docs)
        bm_raw = bm.scores(query)
        bm_by_id = dict(zip(all_ids, bm_raw))
        bm_scores = [bm_by_id.get(i, 0.0) for i in ids]

        v_norm = _minmax(vec_scores)
        b_norm = _minmax(bm_scores)
        alpha = self.cfg.alpha

        hits = []
        for cid, text, meta, v, b, vn, bn in zip(
            ids, docs, metas, vec_scores, bm_scores, v_norm, b_norm
        ):
            hits.append(
                Hit(
                    chunk_id=cid,
                    text=text,
                    source=meta.get("source", "unknown"),
                    score=alpha * vn + (1 - alpha) * bn,
                    vector_score=v,
                    bm25_score=b,
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        return self.rerank(query, hits[:top_k])

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """轻量重排序：查询词在分块中的覆盖率作为额外信号。"""
        qtokens = set(tokenize(query))
        for hit in hits:
            tokens = set(tokenize(hit.text))
            coverage = len(qtokens & tokens) / len(qtokens) if qtokens else 0.0
            hit.score = 0.75 * hit.score + 0.25 * coverage
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: self.cfg.rerank_top_k]
