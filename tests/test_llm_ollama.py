from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from xlerobot_agent.llm import AgentLLMRouter, AgentModelSuite, ModelConfig


class _FakeHttpResponse:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class OllamaRouterTests(unittest.TestCase):
    def test_complete_json_messages_uses_ollama_generate_endpoint_and_images(self) -> None:
        router = AgentLLMRouter(
            AgentModelSuite(
                planner=ModelConfig(provider="ollama", model="gemma4:26b"),
                critic=ModelConfig(provider="mock", model="mock"),
                coder=ModelConfig(provider="mock", model="mock"),
            )
        )
        seen_request: dict[str, object] = {}

        def _fake_urlopen(request, timeout=0):
            seen_request["url"] = request.full_url
            seen_request["headers"] = dict(request.header_items())
            seen_request["payload"] = json.loads(request.data.decode("utf-8"))
            return _FakeHttpResponse(
                json.dumps(
                    {
                        "model": "gemma4:26b",
                        "response": json.dumps(
                            {
                                "decision_type": "finish",
                                "selected_frontier_id": None,
                                "selected_return_waypoint_id": None,
                                "frontier_ids_to_store": [],
                                "exploration_complete": True,
                                "reasoning_summary": "No remaining reachable frontiers.",
                                "semantic_updates": [],
                            }
                        ),
                    }
                )
            )

        messages = [
            {"role": "system", "content": "Return JSON only."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Choose the next exploration action."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,aGVsbG8=",
                        },
                    },
                ],
            },
        ]
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            parsed, trace = router.complete_json_messages(
                config=router.model_suite.planner,
                messages=messages,
            )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["decision_type"], "finish")
        self.assertEqual(trace.provider, "ollama")
        self.assertEqual(seen_request["url"], "http://localhost:11434/api/generate")
        payload = seen_request["payload"]
        assert isinstance(payload, dict)
        self.assertEqual(payload["model"], "gemma4:26b")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["format"], "json")
        self.assertEqual(payload["images"], ["aGVsbG8="])
        self.assertIn("SYSTEM:", payload["prompt"])
        self.assertIn("USER:", payload["prompt"])


if __name__ == "__main__":
    unittest.main()
