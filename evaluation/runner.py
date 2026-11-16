"""批量评估入口：读取评估集 → 逐条问答 → 计算指标 → 输出报告。"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from config import Settings, settings
from evaluation import metrics
from generation.answer import AnswerEngine


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def evaluate(engine: AnswerEngine, dataset: Path, top_k: int = 5) -> dict[str, Any]:
    rows = load_dataset(dataset)
    records = []
    totals = {k: [] for k in ("recall", "precision", "mrr", "faithfulness", "relevancy", "refusal")}

    for row in rows:
        answer = engine.answer(row["question"])
        hits = engine.retriever.retrieve(row["question"])
        retrieved = [h.source for h in hits[:top_k]]
        relevant = set(row.get("relevant_sources", []))
        contexts = [h.text for h in hits]

        scores = {
            "recall": metrics.recall_at_k(relevant, retrieved, top_k),
            "precision": metrics.precision_at_k(relevant, retrieved, top_k),
            "mrr": metrics.mrr(relevant, retrieved),
            "faithfulness": metrics.faithfulness(answer.text, contexts),
            "relevancy": metrics.answer_relevancy(answer.text, row.get("reference", "")),
            "refusal": metrics.refusal_accuracy(answer.refused, row.get("should_refuse", False)),
        }
        for k, v in scores.items():
            totals[k].append(v)
        records.append({
            "id": row.get("id"),
            "question": row["question"],
            "answer": answer.text,
            "refused": answer.refused,
            "confidence": answer.confidence,
            "citations": answer.citations,
            "retrieved_sources": retrieved,
            "scores": {k: round(v, 4) for k, v in scores.items()},
        })

    summary = {k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in totals.items()}
    summary["sample_count"] = len(rows)
    return {"generated_at": datetime.now().isoformat(timespec="seconds"), "summary": summary, "records": records}


def run(cfg: Settings = settings, dataset: Path | None = None, output: Path | None = None) -> dict[str, Any]:
    engine = AnswerEngine(cfg)
    dataset = dataset or cfg.eval_dir / "qa_dataset.jsonl"
    report = evaluate(engine, dataset)

    output = output or cfg.eval_dir / "last_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[评估完成] 样本数 {report['summary']['sample_count']}")
    for key, value in report["summary"].items():
        if key != "sample_count":
            print(f"  {key:>14}: {value}")
    print(f"报告已写入 {output}")
    return report
