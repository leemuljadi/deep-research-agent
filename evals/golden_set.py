"""A small labelled golden set for evaluating the deep-research agent.

Edit / extend these to reflect your own corpus. Accuracy is scored by keyword
coverage; faithfulness by LLM-as-judge against retrieved sources.
"""
from __future__ import annotations

from evals.eval_harness import EvalSample

GOLDEN_SET: list[EvalSample] = [
    EvalSample(
        question="What are the key features of the Deep-Research Agent project?",
        expected_keywords=["agent", "rag", "evaluation", "pgvector", "langgraph"],
    ),
    EvalSample(
        question="Which retrieval technologies does the system use?",
        expected_keywords=["pgvector", "hybrid", "vector"],
    ),
    EvalSample(
        question="How does the system keep answers grounded in sources?",
        expected_keywords=["sources", "citation", "grounded"],
    ),
]
