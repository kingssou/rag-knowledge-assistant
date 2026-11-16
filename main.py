"""命令行入口：init / ask / eval / serve。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import settings
from generation.answer import AnswerEngine
from ingest.pipeline import IngestPipeline


def cmd_init(args: argparse.Namespace) -> None:
    pipeline = IngestPipeline()
    if args.reset:
        pipeline.reset()
    result = pipeline.ingest(args.source or settings.sample_docs_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_ask(args: argparse.Namespace) -> None:
    engine = AnswerEngine()
    answer = engine.answer(args.question)
    print(f"\n问题：{answer.question}")
    print(f"回答：{answer.text}")
    print(f"拒答：{answer.refused}   置信度：{answer.confidence}")
    if answer.citations:
        print("引用：")
        for c in answer.citations:
            print(f"  [{c['index']}] {c['source']}  score={c['score']}")


def cmd_eval(args: argparse.Namespace) -> None:
    from evaluation.runner import run

    run(dataset=Path(args.dataset) if args.dataset else None,
        output=Path(args.output) if args.output else None)


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("api.server:app", host=args.host, port=args.port, reload=args.reload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="企业知识库 RAG 问答系统")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="解析文档并建立向量索引")
    p_init.add_argument("--source", help="文件或目录，默认使用示例文档")
    p_init.add_argument("--reset", action="store_true", help="清空旧索引后重建")
    p_init.set_defaults(func=cmd_init)

    p_ask = sub.add_parser("ask", help="提问")
    p_ask.add_argument("question", help="问题内容")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="按评估集跑指标")
    p_eval.add_argument("--dataset", help="评估集 jsonl 路径")
    p_eval.add_argument("--output", help="报告输出路径")
    p_eval.set_defaults(func=cmd_eval)

    p_serve = sub.add_parser("serve", help="启动 HTTP 服务")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
