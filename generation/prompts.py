"""Prompt 模板。约束模型只能依据给定资料作答，并显式标注引用编号。"""
from __future__ import annotations

SYSTEM_PROMPT = """你是一个企业知识库助手。严格遵守以下规则：
1. 只能依据【参考资料】回答，不得使用外部知识或自行推测。
2. 每个事实性句子末尾必须标注引用编号，如 [1]、[2]。
3. 如果参考资料不足以回答，直接回答“根据现有资料无法回答该问题”，不要编造。
4. 回答简洁，优先分条陈述，不要复述问题。
"""

USER_TEMPLATE = """【参考资料】
{context}

【问题】
{question}

请按系统规则回答。"""


def build_context(hits: list) -> str:
    lines = []
    for i, hit in enumerate(hits, 1):
        lines.append(f"[{i}] 来源：{hit.source}\n{hit.text}")
    return "\n\n".join(lines)


def build_messages(question: str, hits: list) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            context=build_context(hits), question=question
        )},
    ]
