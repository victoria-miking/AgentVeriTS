import json
import unittest
from unittest.mock import patch

import httpx
import jsonschema
from openai import OpenAI, BadRequestError

from agentverits.reasoning.provider import OpenAIReasoningClient, ResponseError, OFFICIAL_BASE_URL, validate_strict_schema
from agentverits.reasoning.candidate_assessment import global_schema
from agentverits.reasoning.agentic_verification import evidence_schema

SCHEMA = {"type": "object", "additionalProperties": False, "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


def response_body(response_id="resp_1", **updates):
    body = {"id": response_id, "object": "response", "created_at": 1, "model": "gpt-5.6-sol", "status": "completed",
            "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": '{"ok":true}', "annotations": []}]}]}
    body.update(updates)
    return body


class ProviderTest(unittest.TestCase):
    def client(self, handler, **kwargs):
        sdk = OpenAI(api_key="test-key", base_url=OFFICIAL_BASE_URL,
                     http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)
        self.addCleanup(sdk.close)
        return OpenAIReasoningClient(client=sdk)

    def complete(self, client, **kwargs):
        return client.complete(instructions="per-turn rules", text="new evidence only", schema=SCHEMA, schema_name="test", **kwargs)

    def test_official_sdk_serializes_incremental_response_chain(self):
        calls = []
        def handler(request):
            self.assertEqual(str(request.url), OFFICIAL_BASE_URL + "/responses")
            calls.append(json.loads(request.content))
            return httpx.Response(200, json=response_body(f"resp_{len(calls)}"))
        client = self.client(handler)
        first = self.complete(client)
        second = self.complete(client, previous_response_id=first.response_id)
        self.assertEqual(second.response_id, "resp_2")
        self.assertNotIn("previous_response_id", calls[0])
        self.assertEqual(calls[1]["previous_response_id"], "resp_1")
        self.assertTrue(calls[1]["store"])
        self.assertEqual(calls[1]["instructions"], "per-turn rules")
        self.assertEqual(calls[1]["reasoning"]["context"], "all_turns")
        self.assertEqual(len(calls[1]["input"]), 1)
        self.assertEqual(client.call_log[-1]["previous_response_id"], "resp_1")

    def test_environment_cannot_redirect_official_client(self):
        with patch.dict("os.environ", {"OPENAI_BASE_URL": "https://relay.invalid/v1", "OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI") as factory:
                OpenAIReasoningClient()
                self.assertEqual(factory.call_args.kwargs["base_url"], OFFICIAL_BASE_URL)

    def test_incomplete_and_refused_replies_fail_without_restart(self):
        for body in [response_body(status="incomplete", incomplete_details={"reason": "max_output_tokens"}),
                     response_body(output=[{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                                            "content": [{"type": "refusal", "refusal": "no"}]}]),
                     response_body(id="")]:
            with self.subTest(body=body):
                calls = []
                def handler(request):
                    calls.append(request)
                    return httpx.Response(200, json=body)
                with self.assertRaises(ResponseError):
                    self.complete(self.client(handler), previous_response_id="resp_old")
                self.assertEqual(len(calls), 1)

    def test_missing_previous_response_does_not_reset_chain(self):
        calls = []
        def handler(request):
            calls.append(json.loads(request.content))
            return httpx.Response(400, json={"error": {"message": "previous response not found", "type": "invalid_request_error"}})
        client = self.client(handler)
        with self.assertRaises(BadRequestError):
            self.complete(client, previous_response_id="resp_expired")
        self.assertEqual(len(calls), 1)
        self.assertEqual(client.call_log[-1]["status"], "request_failed")

    def test_transient_retry_preserves_parent_and_input(self):
        calls = []
        def handler(request):
            calls.append(json.loads(request.content))
            if len(calls) == 1:
                return httpx.Response(429, headers={"retry-after-ms": "1"}, json={"error": {"message": "retry", "type": "rate_limit_error"}})
            return httpx.Response(200, json=response_body())
        self.complete(self.client(handler, max_retries=1), previous_response_id="resp_previous")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])

    def test_all_paper_schemas_are_closed_and_valid(self):
        for schema in [global_schema(), evidence_schema(), evidence_schema(allow_tools=False)]:
            validate_strict_schema(schema)
            jsonschema.Draft202012Validator.check_schema(schema)
        with self.assertRaises(ValueError):
            validate_strict_schema({"type": "object"})

    def test_store_false_is_rejected(self):
        with self.assertRaises(ValueError):
            OpenAIReasoningClient(store=False)


if __name__ == "__main__":
    unittest.main()
