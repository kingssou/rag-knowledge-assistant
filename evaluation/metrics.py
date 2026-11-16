"""RAG 评估指标（轻量可复现实现）。

注意：这是工程上可落地的简化版本，不是 RAGAS 的完整实现。
面试时要能说清楚每个指标的定义、计算方式和它的局限。
"""
from __future__ import annotations

from retrieval.embedder import tokenize


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Recall@k：前 k 个召回结果中命中相关文档的比例。"""
    if not relevant:
        return 0.0
    return len(relevant & set(retrieved[:k])) / len(relevant)


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """前 k 个结果中相关来源的占比。

    以「去重后的来源」为分母，避免单文档多分块时 Precision 被人为压低。
    """
    sources = set(retrieved[:k])
    if not sources:
        return 0.0
    return len(relevant & sources) / len(sources)


def mrr(relevant: set[str], retrieved: list[str]) -> float:
    """平均倒数排名：第一个命中相关文档的位置的倒数。"""
    for i, item in enumerate(retrieved, 1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def faithfulness(answer: str, contexts: list[str]) -> float:
    """忠实度近似：答案中的词有多少能在引用上下文里找到。

    局限：只做词级覆盖，不能识别语义等价或否定语义。
    """
    atoks = set(tokenize(answer)) - {"的", "了", "和", "是", "在", "，", "。"}
    ctoks = set()
    for ctx in contexts:
        ctoks |= set(tokenize(ctx))
    if not atoks:
        return 0.0
    return len(atoks & ctoks) / len(atoks)


def answer_relevancy(answer: str, reference: str) -> float:
    """答案相关性近似：生成答案与参考答案的词级 F1。"""
    a = set(tokenize(answer))
    r = set(tokenize(reference))
    if not a or not r:
        return 0.0
    overlap = len(a & r)
    precision = overlap / len(a)
    recall = overlap / len(r)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def refusal_accuracy(refused: bool, should_refuse: bool) -> float:
    """拒答判断正确率。"""
    return 1.0 if refused == should_refuse else 0.0
