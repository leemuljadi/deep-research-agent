from __future__ import annotations

import unittest
from contextlib import nullcontext
from unittest.mock import patch

from pydantic import ValidationError

from src import graph
from src.agents import synthesizer
from src.llm import Usage
from src.schemas import ResearchPlan, ResearchReport, Source, SubQuestionResult


class GraphShapeTests(unittest.TestCase):
    def test_researcher_emits_named_sub_question_result(self) -> None:
        source = Source(title="Source", snippet="Evidence")
        usage = Usage(total_tokens=3, cost_usd=0.01)

        with (
            patch.object(
                graph,
                "research_sub_question",
                return_value=(["Finding"], [source], usage),
            ),
            patch.object(graph, "trace", return_value=nullcontext(object())),
            patch.object(graph, "record_cost"),
        ):
            update = graph._researcher(
                {"sub_question": "Question", "otel_ctx": None}
            )

        self.assertEqual(update["usage_log"], [usage])
        self.assertEqual(
            update["sub_results"],
            [
                SubQuestionResult(
                    sub_question="Question",
                    findings=["Finding"],
                    sources=[source],
                )
            ],
        )

    def test_sub_question_result_rejects_invalid_shape(self) -> None:
        with self.assertRaises(ValidationError):
            SubQuestionResult(
                sub_question="Question",
                findings=["Finding"],
            )

    def test_synthesizer_consumes_named_results_without_report_shape_change(self) -> None:
        plan = ResearchPlan(objective="Objective", sub_questions=["Question"])
        result = SubQuestionResult(
            sub_question="Question",
            findings=["Finding one", "Finding two"],
            sources=[Source(title="Input source", snippet="Input evidence")],
        )
        response = (
            '{"summary":"Summary","findings":["Final finding"],'
            '"sources":[{"title":"Output source","url":null,'
            '"snippet":"Output evidence","score":0.8}],"confidence":0.9}'
        )
        usage = Usage(total_tokens=7, cost_usd=0.02)

        with patch.object(
            synthesizer,
            "chat_with_usage",
            return_value=(response, usage),
        ) as chat:
            report, returned_usage = synthesizer.synthesize(plan, [result])

        self.assertEqual(returned_usage, usage)
        self.assertEqual(
            report.model_dump(),
            {
                "summary": "Summary",
                "findings": ["Final finding"],
                "sources": [
                    {
                        "title": "Output source",
                        "url": None,
                        "snippet": "Output evidence",
                        "score": 0.8,
                    }
                ],
                "confidence": 0.9,
                "cost_usd": 0.0,
                "latency_seconds": 0.0,
                "total_tokens": 0,
            },
        )
        user_prompt = chat.call_args.args[0][1]["content"]
        self.assertIn("## Question\n- Finding one\n- Finding two", user_prompt)

    def test_graph_synthesizer_forwards_named_results_and_folds_usage(self) -> None:
        plan = ResearchPlan(objective="Objective", sub_questions=["Question"])
        results = [
            SubQuestionResult(
                sub_question="Question",
                findings=["Finding"],
                sources=[],
            )
        ]
        report = ResearchReport(
            summary="Summary",
            findings=["Finding"],
            sources=[],
            confidence=0.5,
        )
        prior_usage = Usage(total_tokens=3, cost_usd=0.01)
        synthesis_usage = Usage(total_tokens=5, cost_usd=0.02)

        with (
            patch.object(
                graph,
                "synthesize",
                return_value=(report, synthesis_usage),
            ) as synthesize_call,
            patch.object(graph, "trace", return_value=nullcontext(object())),
            patch.object(graph, "record_cost"),
        ):
            update = graph._synthesizer(
                {
                    "plan": plan,
                    "sub_results": results,
                    "usage_log": [prior_usage],
                    "otel_ctx": None,
                }
            )

        synthesize_call.assert_called_once_with(plan, results)
        self.assertIs(update["report"], report)
        self.assertEqual(report.total_tokens, 8)
        self.assertEqual(report.cost_usd, 0.03)
        self.assertEqual(update["usage_log"], [synthesis_usage])

    def test_build_graph_compiles_expected_shape(self) -> None:
        """LangGraph runtime boundary (AD-12 floor guard): the compiled graph must
        construct and expose the fixed node set — a 1.x signature change in
        StateGraph/add_conditional_edges/Send breaks here, not at request time."""
        compiled = graph.build_graph()
        self.assertIn("planner", compiled.nodes)
        self.assertIn("_researcher", compiled.nodes)
        self.assertIn("synthesizer", compiled.nodes)


if __name__ == "__main__":
    unittest.main()
