"""核心逻辑单测，不依赖外部服务。"""
from __future__ import annotations

import unittest

from evaluation import metrics
from generation.llm import MockLLM
from ingest.chunker import chunk_document
from ingest.loader import Document
from retrieval.embedder import HashEmbedder


class TestChunker(unittest.TestCase):
    def test_chunk_keeps_source_and_overlap(self):
        doc = Document(text="第一段内容。" * 40 + "\n第二段内容。" * 40, source="demo.md")
        chunks = chunk_document(doc, size=120, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.source == "demo.md" for c in chunks))
        self.assertTrue(all(len(c.text) <= 160 for c in chunks))
        self.assertEqual(chunks[0].chunk_id, "demo.md::0")


class TestEmbedder(unittest.TestCase):
    def test_hash_embedder_is_normalized_and_deterministic(self):
        emb = HashEmbedder(dim=64)
        a = emb.embed(["远程办公申请流程"])[0]
        b = emb.embed(["远程办公申请流程"])[0]
        self.assertEqual(a, b)
        self.assertAlmostEqual(sum(x * x for x in a), 1.0, places=5)


class TestMetrics(unittest.TestCase):
    def test_recall_and_mrr(self):
        self.assertEqual(metrics.recall_at_k({"a"}, ["b", "a"], 2), 1.0)
        self.assertEqual(metrics.mrr({"a"}, ["b", "a"]), 0.5)
        self.assertEqual(metrics.faithfulness("完全无关的内容", ["另一段文字"]), 0.0)
        self.assertEqual(metrics.refusal_accuracy(True, True), 1.0)


class TestMockLLM(unittest.TestCase):
    def test_mock_llm_cites_source(self):
        messages = [
            {"role": "system", "content": "规则"},
            {"role": "user", "content": (
                "【参考资料】\n[1] 来源：制度.md\n远程办公超过三天需要提前申请。\n\n"
                "【问题】\n远程办公超过几天要申请？\n\n请按系统规则回答。"
            )},
        ]
        out = MockLLM().chat(messages)
        self.assertIn("[1]", out)


if __name__ == "__main__":
    unittest.main()
