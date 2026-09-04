from __future__ import annotations

import unittest
from contextlib import nullcontext
from unittest.mock import patch

from pydantic import ValidationError

from src import graph
from src.agents import synthesizer
from src.llm import Usage
from src.schemas import (
    PlannerReflection,
    ResearchPlan,
    ResearchReport,
    Source,
    SubQuestionResult,
    SynthesisReview,
)


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
        self.assertIn("planner_reflection", compiled.nodes)
        self.assertIn("synthesis_review", compiled.nodes)
        edges = {(edge.source, edge.target) for edge in compiled.get_graph().edges}
        self.assertIn(("_researcher", "planner_reflection"), edges)
        self.assertIn(("synthesizer", "synthesis_review"), edges)
        self.assertIn(("synthesis_review", "synthesizer"), edges)
        self.assertIn(("synthesis_review", "__end__"), edges)
        self.assertNotIn(("synthesizer", "__end__"), edges)

    def test_loop_off_is_single_pass_without_reflection_calls(self) -> None:
        plan_candidate = ResearchPlan(objective="Objective", sub_questions=["Question"])
        report = ResearchReport(
            summary="Summary", findings=["Finding"], sources=[], confidence=0.8
        )
        state = graph.initial_state("Question", [], reflection_enabled=False)
        with (
            patch.object(graph, "plan", return_value=(plan_candidate, Usage(total_tokens=1))) as planner,
            patch.object(
                graph,
                "research_sub_question",
                return_value=(["Finding"], [], Usage(total_tokens=2)),
            ) as researcher,
            patch.object(
                graph, "synthesize", return_value=(report, Usage(total_tokens=3))
            ) as synth,
            patch.object(
                graph, "chat_with_usage", side_effect=AssertionError("reflection call")
            ),
            patch.object(graph, "trace", return_value=nullcontext(object())),
            patch.object(graph, "record_cost"),
        ):
            completed, gate = graph.run_research_state(state)

        self.assertIsNone(gate)
        planner.assert_called_once()
        researcher.assert_called_once()
        synth.assert_called_once()
        self.assertEqual(completed["report"].total_tokens, 6)

    def test_planner_equal_score_retains_earlier_candidate(self) -> None:
        state = graph.initial_state(
            "Question",
            [],
            reflection_enabled=True,
            planner_reflection_max_iterations=1,
        )
        state["plan"] = ResearchPlan(objective="Objective", sub_questions=["Original"])
        state["sub_results"] = [
            SubQuestionResult(sub_question="Original", findings=["Strong"], sources=[])
        ]
        first = PlannerReflection(
            score=0.8,
            needs_more_research=True,
            additional_sub_questions=["Novel"],
            rationale="Gap",
        )
        graph._merge_update(state, graph._apply_planner_reflection(state, first))
        state["sub_results"].append(
            SubQuestionResult(sub_question="Novel", findings=["Weak"], sources=[])
        )
        tied = PlannerReflection(
            score=0.8,
            needs_more_research=True,
            additional_sub_questions=["Another"],
            rationale="No improvement",
        )

        update = graph._apply_planner_reflection(state, tied)

        self.assertTrue(update["planner_complete"])
        self.assertEqual(update["plan"].sub_questions, ["Original"])
        self.assertEqual(update["best_sub_results"], state["best_sub_results"])

    def test_planner_repeated_question_stops_without_increment(self) -> None:
        state = graph.initial_state(
            "Question",
            [],
            reflection_enabled=True,
            planner_reflection_max_iterations=2,
        )
        state["plan"] = ResearchPlan(objective="Objective", sub_questions=["Original"])
        state["sub_results"] = [
            SubQuestionResult(sub_question="Original", findings=["Finding"], sources=[])
        ]
        repeated = PlannerReflection(
            score=0.7,
            needs_more_research=True,
            additional_sub_questions=["  original  "],
            rationale="Repeated",
        )

        update = graph._apply_planner_reflection(state, repeated)

        self.assertTrue(update["planner_complete"])
        self.assertEqual(state["planner_iterations"], 0)
        self.assertEqual(update["pending_sub_questions"], [])

    def test_planner_replan_runs_one_novel_round_then_stops_at_cap(self) -> None:
        initial_plan = ResearchPlan(objective="Objective", sub_questions=["Original"])
        report = ResearchReport(
            summary="Summary", findings=["Original", "Novel"], sources=[], confidence=0.8
        )
        reflections = [
            (
                '{"score":0.5,"needs_more_research":true,'
                '"additional_sub_questions":["Novel"],"rationale":"Gap"}',
                Usage(total_tokens=1),
            ),
            (
                '{"score":0.9,"needs_more_research":true,'
                '"additional_sub_questions":["Blocked by cap"],"rationale":"One more gap"}',
                Usage(total_tokens=1),
            ),
        ]
        state = graph.initial_state(
            "Question",
            [],
            reflection_enabled=True,
            planner_reflection_max_iterations=1,
            synthesis_review_max_iterations=0,
        )
        with (
            patch.object(graph, "plan", return_value=(initial_plan, Usage(total_tokens=1))),
            patch.object(
                graph,
                "research_sub_question",
                side_effect=[
                    (["Original finding"], [], Usage(total_tokens=1)),
                    (["Novel finding"], [], Usage(total_tokens=1)),
                ],
            ) as researcher,
            patch.object(
                graph, "synthesize", return_value=(report, Usage(total_tokens=1))
            ),
            patch.object(graph, "chat_with_usage", side_effect=reflections),
            patch.object(graph, "trace", return_value=nullcontext(object())),
            patch.object(graph, "record_cost"),
        ):
            completed, gate = graph.run_research_state(state)

        self.assertIsNone(gate)
        self.assertEqual(
            [call.args[0] for call in researcher.call_args_list],
            ["Original", "Novel"],
        )
        self.assertEqual(completed["planner_iterations"], 1)
        self.assertTrue(completed["planner_complete"])
        self.assertEqual(
            completed["plan"].sub_questions,
            ["Original", "Novel"],
        )

    def test_worse_synthesis_revision_retains_first_report(self) -> None:
        state = graph.initial_state(
            "Question",
            [],
            reflection_enabled=True,
            synthesis_review_max_iterations=1,
        )
        first = ResearchReport(
            summary="First", findings=["Grounded"], sources=[], confidence=0.7
        )
        state["report"] = first
        first_review = SynthesisReview(
            score=0.8, needs_revision=True, feedback="Add detail"
        )
        graph._merge_update(
            state, graph._apply_synthesis_review(state, first_review)
        )
        state["report"] = ResearchReport(
            summary="Worse", findings=[], sources=[], confidence=0.9
        )
        worse_review = SynthesisReview(
            score=0.6, needs_revision=True, feedback="Try again"
        )

        update = graph._apply_synthesis_review(state, worse_review)

        self.assertTrue(update["synthesis_complete"])
        self.assertEqual(update["report"].summary, "First")
        self.assertEqual(update["best_report"].summary, "First")
        self.assertEqual(state["synthesis_iterations"], 1)

    def test_malformed_reflection_outputs_fail_validation(self) -> None:
        with self.assertRaises(ValidationError):
            PlannerReflection.model_validate_json('{"score": 2}')
        with self.assertRaises(ValidationError):
            SynthesisReview.model_validate_json('{"score": -1}')


class ReflectionConfigTests(unittest.TestCase):
    def test_reflection_env_parsers_keep_safe_defaults(self) -> None:
        from src.config import _bool_env, _non_negative_int_env

        self.assertTrue(_bool_env("not-a-bool", default=True))
        self.assertFalse(_bool_env("OFF", default=True))
        self.assertEqual(_non_negative_int_env("0", default=1), 0)
        self.assertEqual(_non_negative_int_env("-1", default=1), 1)
        self.assertEqual(_non_negative_int_env("invalid", default=1), 1)

    def test_initial_state_rejects_negative_iteration_caps(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            graph.initial_state(
                "Question",
                [],
                planner_reflection_max_iterations=-1,
            )


if __name__ == "__main__":
    unittest.main()
