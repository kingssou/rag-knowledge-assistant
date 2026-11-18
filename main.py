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


def cmd_check(args: argparse.Namespace) -> None:
    """自检：确认当前用的是什么模型、Key 有没有读到、接口通不通。"""
    print(f"LLM_MODE        : {settings.llm_mode}")
    print(f"LLM_MODEL       : {settings.llm_model}")
    print(f"OPENAI_BASE_URL : {settings.openai_base_url}")
    print(f"EMBED_MODEL     : {settings.embed_model}")

    if settings.llm_mode == "mock":
        print("\n当前是离线 mock 模式：不调用任何大模型，回答由规则抽取拼接而成。")
        print("接真实模型：复制 .env.example 为 .env，填好 Key，把 LLM_MODE 改成 openai，再跑一次 check。")
        return

    try:
        import openai  # noqa: F401
    except ImportError:
        print("\n缺少 openai SDK，请先执行：")
        print("  pip install openai")
        return

    key = settings.resolve_api_key()
    print(f"API Key         : {'已读取（' + key[:6] + '…）' if key else '未找到'}")
    if not key:
        print("\n没有读到 Key。在项目根目录建一个 .env，写一行比如 DEEPSEEK_API_KEY=sk-xxx。")
        return

    from generation.llm import build_llm

    print("\n正在发起一次真实请求…")
    try:
        llm = build_llm(settings)
        reply = llm.chat([{"role": "user", "content": "只回复两个字：收到"}])
        print(f"连接成功，模型返回：{reply.strip()[:80]}")
    except Exception as exc:  # noqa: BLE001
        print(f"连接失败：{type(exc).__name__}: {exc}")
        print("常见原因：Key 无效、余额不足、base_url 写错、网络需要代理。")


QUICK_START = """\
企业知识库 RAG 问答系统

需要指定一个子命令，常用五条：

  python main.py init --reset        读取 data/sample_docs 并建立向量索引
  python main.py ask "问题内容"       命令行提问
  python main.py eval                跑评估集，输出 6 项指标
  python main.py serve               启动 Web 服务，浏览器打开 http://127.0.0.1:8000/
  python main.py check               检查模型配置、Key 与接口连通性

首次使用请先执行 init，否则 ask 和 serve 没有可检索的数据。
接真实大模型：复制 .env.example 为 .env，填入 Key，把 LLM_MODE 改为 openai。
（在 PyCharm 里直接点 Run 不会带参数，请在下方 Terminal 中执行上述命令，
 或在 Run/Debug Configurations 的 Script parameters 中填写子命令。）
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="企业知识库 RAG 问答系统",
        epilog=QUICK_START,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

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

    p_check = sub.add_parser("check", help="检查模型配置与连通性")
    p_check.set_defaults(func=cmd_check)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not getattr(args, "command", None):
        print(QUICK_START)
        return
    args.func(args)


if __name__ == "__main__":
    main()
