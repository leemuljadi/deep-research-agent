from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src import llm


class RouterOrderingTests(unittest.TestCase):
    def _build_with(self, fallback: str | None):
        configured = SimpleNamespace(
            chat_model="primary-model",
            chat_model_fallback=fallback,
            embed_model="embed-model",
        )
        sentinel = object()
        with (
            patch.object(llm, "settings", configured),
            patch.object(llm, "Router", return_value=sentinel) as router,
        ):
            built = llm._build_router()
        self.assertIs(built, sentinel)
        return router.call_args.kwargs["model_list"]

    def test_primary_and_fallback_have_ordered_chat_priority(self) -> None:
        models = self._build_with("fallback-model")
        chat_deployments = [m for m in models if m["model_name"] == "chat"]
        embed_deployment = next(m for m in models if m["model_name"] == "embed")

        self.assertEqual(
            chat_deployments,
            [
                {
                    "model_name": "chat",
                    "litellm_params": {"model": "primary-model", "order": 1},
                },
                {
                    "model_name": "chat",
                    "litellm_params": {"model": "fallback-model", "order": 2},
                },
            ],
        )
        self.assertEqual(
            embed_deployment,
            {
                "model_name": "embed",
                "litellm_params": {"model": "embed-model"},
            },
        )

    def test_primary_only_has_one_ordered_chat_deployment(self) -> None:
        models = self._build_with(None)
        chat_deployments = [m for m in models if m["model_name"] == "chat"]

        self.assertEqual(
            chat_deployments,
            [
                {
                    "model_name": "chat",
                    "litellm_params": {"model": "primary-model", "order": 1},
                }
            ],
        )

    @staticmethod
    def _real_router() -> llm.Router:
        return llm.Router(
            model_list=[
                {
                    "model_name": "chat",
                    "litellm_params": {"model": "openai/primary", "order": 1},
                },
                {
                    "model_name": "chat",
                    "litellm_params": {"model": "openai/fallback", "order": 2},
                },
            ],
            num_retries=0,
            allowed_fails=0,
            cooldown_time=0,
        )

    @staticmethod
    def _response(model: str, content: str):
        return llm.litellm.ModelResponse(
            model=model,
            choices=[{"message": {"role": "assistant", "content": content}}],
        )

    def test_real_router_uses_primary_without_fallback(self) -> None:
        router = self._real_router()
        attempts = []

        def complete(**kwargs):
            attempts.append(kwargs["model"])
            return self._response(kwargs["model"], "primary ok")

        with patch.object(llm.litellm, "completion", side_effect=complete):
            response = router.completion(
                model="chat",
                messages=[{"role": "user", "content": "test"}],
            )

        self.assertEqual(attempts, ["openai/primary"])
        self.assertEqual(response.choices[0].message.content, "primary ok")

    def test_real_router_falls_back_after_primary_failure(self) -> None:
        router = self._real_router()
        attempts = []

        def complete(**kwargs):
            model = kwargs["model"]
            attempts.append(model)
            if model == "openai/primary":
                raise llm.litellm.InternalServerError(
                    message="primary unavailable",
                    model=model,
                    llm_provider="openai",
                )
            return self._response(model, "fallback ok")

        with patch.object(llm.litellm, "completion", side_effect=complete):
            response = router.completion(
                model="chat",
                messages=[{"role": "user", "content": "test"}],
            )

        self.assertEqual(attempts, ["openai/primary", "openai/fallback"])
        self.assertEqual(response.choices[0].message.content, "fallback ok")


if __name__ == "__main__":
    unittest.main()
