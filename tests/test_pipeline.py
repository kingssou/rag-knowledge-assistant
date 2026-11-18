"""核心逻辑单测，不依赖外部服务。"""
from __future__ import annotations

import unittest

from evaluation import metrics
from generation.context import Conversation, estimate_tokens, fit_context
from generation.llm import MockLLM, _safe_json
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

    def test_mock_stream_yields_chunks(self):
        messages = [
            {"role": "system", "content": "规则"},
            {"role": "user", "content": (
                "【参考资料】\n[1] 来源：制度.md\n远程办公超过三天需要提前申请。\n\n"
                "【问题】\n远程办公超过几天要申请？\n\n请按系统规则回答。"
            )},
        ]
        parts = list(MockLLM().stream(messages))
        self.assertGreater(len(parts), 0)
        self.assertEqual("".join(parts).replace(" ", ""),
                         MockLLM().chat(messages).replace(" ", ""))


class TestStructuredOutput(unittest.TestCase):
    def test_safe_json_parses_fenced_and_plain(self):
        self.assertEqual(_safe_json('```json\n{"answer": "ok"}\n```'), {"answer": "ok"})
        self.assertEqual(_safe_json('前缀 {"answer": 1} 后缀'), {"answer": 1})
        self.assertEqual(_safe_json('不是 json'), {})


class TestContextEngineering(unittest.TestCase):
    def test_fit_context_respects_budget(self):
        class Hit:
            def __init__(self, text):
                self.text = text
        hits = [Hit("长文本" * 20), Hit("短")]
        kept, used = fit_context(hits, budget=30)
        self.assertLessEqual(used, 30)
        self.assertTrue(all(h in kept for h in hits if estimate_tokens(h.text) + 8 <= 30))

    def test_conversation_keeps_last_n_turns(self):
        conv = Conversation(max_turns=2)
        for i in range(5):
            conv.add(f"问题{i}", f"回答{i}")
        self.assertEqual(len(conv.turns), 2)
        msgs = conv.to_messages()
        self.assertEqual(msgs[0]["content"], "问题3")
        self.assertEqual(msgs[-1]["content"], "回答4")


class TestConfig(unittest.TestCase):
    def test_resolve_api_key_prefers_named_env(self):
        import os
        from config import Settings

        os.environ["API_KEY_ENV"] = "MY_TEST_KEY"
        os.environ["MY_TEST_KEY"] = "sk-named"
        os.environ["DEEPSEEK_API_KEY"] = "sk-fallback"
        try:
            s = Settings.from_env()
            self.assertEqual(s.resolve_api_key(), "sk-named")
        finally:
            for k in ("API_KEY_ENV", "MY_TEST_KEY", "DEEPSEEK_API_KEY"):
                os.environ.pop(k, None)

    def test_resolve_api_key_falls_back_to_known_names(self):
        import os
        from config import Settings

        s = Settings()
        s.api_key_env = "NOT_EXIST_KEY_XYZ"
        os.environ["ZHIPU_API_KEY"] = "sk-zhipu"
        try:
            self.assertEqual(s.resolve_api_key(), "sk-zhipu")
        finally:
            os.environ.pop("ZHIPU_API_KEY", None)

    def test_build_llm_raises_when_key_missing(self):
        import os
        from config import Settings
        from generation.llm import build_llm

        for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY",
                     "MOONSHOT_API_KEY", "ZHIPU_API_KEY", "ARK_API_KEY", "SILICONFLOW_API_KEY"):
            os.environ.pop(name, None)
        cfg = Settings()
        cfg.llm_mode = "openai"
        with self.assertRaises(RuntimeError):
            build_llm(cfg)


if __name__ == "__main__":
    unittest.main()
