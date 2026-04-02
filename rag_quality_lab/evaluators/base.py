"""Evaluator base module.

Metrics are pure functions in metrics.py.
LLM judge is a stateful class in llm_judge.py.
This module provides the public re-exports.
"""
from rag_quality_lab.evaluators.llm_judge import JudgeScore, LLMJudge
from rag_quality_lab.evaluators.metrics import (
    compute_all_metrics,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "hit_rate_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "compute_all_metrics",
    "JudgeScore",
    "LLMJudge",
]
