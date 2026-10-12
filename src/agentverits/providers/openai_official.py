"""Official OpenAI Responses API transport.

The endpoint is pinned to api.openai.com and a server response chain is used
for every assessment/verification session. No relay headers or replayed local
conversation are accepted.
"""

from __future__ import annotations
import base64, json, mimetypes, os, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from ..contracts import ModelStepResult, ModelUsage


@dataclass(frozen=True)
class OfficialResponsesConfig:
    model: str = "gpt-5.6-sol"
    api_key: str | None = None
    reasoning_effort: str = "high"
    reasoning_context: str | None = None
    stream_responses: bool = True
    max_output_tokens: int = 24000
    request_timeout_seconds: float = 240.0
    request_retries: int = 6
    request_retry_base_seconds: float = 5.0


class OfficialResponsesClient:
    def __init__(
        self, *, config: OfficialResponsesConfig | None = None, client: Any = None
    ):
        self.config = config or OfficialResponsesConfig()
        if client is None:
            from openai import OpenAI

            key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError(
                    "OPENAI_API_KEY is required for the official endpoint"
                )
            # Explicitly omit OPENAI_BASE_URL and all relay-specific headers.
            client = OpenAI(
                api_key=key,
                base_url="https://api.openai.com/v1",
                timeout=self.config.request_timeout_seconds,
                max_retries=0,
            )
        self.client = client
        self.attempt_count = 0
        self.failed_attempt_count = 0
        self.attempt_observer = None

    @property
    def model_name(self):
        return self.config.model

    def step(
        self,
        *,
        instructions: str,
        input_message: dict[str, Any],
        previous_response_id: str | None,
        response_schema: Mapping[str, Any],
        prompt_cache_key: str | None = None,
        context_compact_threshold: int | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> ModelStepResult:
        kwargs = {
            "model": self.config.model,
            "instructions": instructions,
            "input": [self._convert(input_message)],
            "store": True,
            "parallel_tool_calls": False,
            "max_output_tokens": self.config.max_output_tokens,
            "reasoning": {"effort": self.config.reasoning_effort},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "agentverits_decision",
                    "strict": True,
                    "schema": dict(response_schema),
                }
            },
        }
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
        context = self.config.reasoning_context or (
            "all_turns" if self.config.model.startswith("gpt-5.6") else None
        )
        if context:
            kwargs["reasoning"]["context"] = context
        if prompt_cache_key:
            kwargs["prompt_cache_key"] = str(prompt_cache_key)[:64]
        if context_compact_threshold:
            kwargs["context_management"] = [
                {
                    "type": "compaction",
                    "compact_threshold": int(context_compact_threshold),
                }
            ]
        if metadata:
            kwargs["metadata"] = {str(k): str(v) for k, v in metadata.items()}
        raw = self._request(kwargs)
        if not isinstance(raw, dict) or raw.get("status") != "completed":
            raise RuntimeError("official Responses API response was not completed")
        rid = str(raw.get("id") or "")
        if not rid:
            raise RuntimeError("official Responses API returned no response id")
        output = self._output_text(raw)
        if not output:
            raise ValueError("official Responses API returned no structured output")
        parsed = json.loads(output)
        if not isinstance(parsed, dict):
            raise ValueError(
                "official Responses API structured output must be a JSON object"
            )
        return ModelStepResult(rid, parsed, self._usage(raw.get("usage")), raw)

    def _request(self, kwargs):
        last = None
        for attempt in range(max(1, self.config.request_retries + 1)):
            started = time.perf_counter()
            notice = {"attempt": attempt + 1, "status": "started"}
            if self.attempt_observer:
                self.attempt_observer(notice)
            try:
                self.attempt_count += 1
                if not self.config.stream_responses:
                    raw = self._dump(self.client.responses.create(**kwargs))
                    if self.attempt_observer:
                        self.attempt_observer(
                            {
                                **notice,
                                "status": "returned",
                                "seconds": time.perf_counter() - started,
                            }
                        )
                    return raw
                complete = None
                parts = []
                events = 0
                first_event = None
                first_delta = None
                stream = self.client.responses.create(**kwargs, stream=True)
                try:
                    for event in stream:
                        events += 1
                        if first_event is None:
                            first_event = time.perf_counter() - started
                        data = self._dump(event)
                        typ = str(data.get("type", ""))
                        if typ == "response.output_text.delta":
                            if first_delta is None:
                                first_delta = time.perf_counter() - started
                            parts.append(str(data.get("delta", "")))
                        if typ == "response.incomplete":
                            raise ValueError(
                                "official response incomplete; inspect output/context token budget before resume"
                            )
                        if typ in {"response.failed", "error"}:
                            raise RuntimeError(
                                "official response stream terminated without a completed response"
                            )
                        if typ == "response.completed":
                            complete = self._dump(data.get("response") or data)
                finally:
                    if hasattr(stream, "close"):
                        stream.close()
                if complete is None:
                    raise RuntimeError("stream ended before response.completed")
                if not complete.get("output_text"):
                    complete["output_text"] = "".join(parts)
                complete["_stream_diagnostics"] = {
                    "event_count": events,
                    "time_to_first_event_seconds": first_event,
                    "time_to_first_text_delta_seconds": first_delta,
                    "total_stream_seconds": time.perf_counter() - started,
                }
                if self.attempt_observer:
                    self.attempt_observer(
                        {
                            **notice,
                            "status": "returned",
                            "seconds": time.perf_counter() - started,
                        }
                    )
                return complete
            except Exception as exc:
                self.failed_attempt_count += 1
                last = exc
                status = getattr(exc, "status_code", None)
                if self.attempt_observer:
                    self.attempt_observer(
                        {
                            **notice,
                            "status": "failed",
                            "http_status": status,
                            "error_type": type(exc).__name__,
                            "seconds": time.perf_counter() - started,
                        }
                    )
                if attempt >= self.config.request_retries:
                    raise
                if status is not None and status not in {
                    408,
                    409,
                    429,
                    500,
                    502,
                    503,
                    504,
                }:
                    raise
                if status is None and not isinstance(
                    exc, (RuntimeError, ConnectionError, TimeoutError)
                ):
                    # SDK transport exceptions have no HTTP status.
                    if type(exc).__name__ not in {
                        "APIConnectionError",
                        "APITimeoutError",
                    }:
                        raise
                time.sleep(
                    min(300.0, self.config.request_retry_base_seconds * (2**attempt))
                )
        raise last

    @staticmethod
    def _output_text(raw):
        if raw.get("output_text"):
            return str(raw["output_text"])
        return "".join(
            str(part.get("text", ""))
            for item in raw.get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        )

    @staticmethod
    def _dump(value):
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return dict(getattr(value, "__dict__", {}) or {})

    @staticmethod
    def _usage(raw):
        raw = raw or {}
        inp = raw.get("input_tokens_details") or {}
        out = raw.get("output_tokens_details") or {}
        return ModelUsage(
            input_tokens=int(raw.get("input_tokens") or 0),
            cached_tokens=int(inp.get("cached_tokens") or 0),
            output_tokens=int(raw.get("output_tokens") or 0),
            reasoning_tokens=int(out.get("reasoning_tokens") or 0),
            total_tokens=int(raw.get("total_tokens") or 0),
        )

    @staticmethod
    def _convert(message):
        content = []
        for part in message.get("content", []):
            if part.get("type") == "text":
                content.append(
                    {"type": "input_text", "text": str(part.get("text", ""))}
                )
            elif part.get("type") == "media":
                uri = str(part.get("uri", ""))
                path = Path(uri)
                if (
                    not uri.startswith(("data:", "https://", "http://"))
                    and path.is_file()
                ):
                    mime = (
                        part.get("media_type")
                        or mimetypes.guess_type(path.name)[0]
                        or "image/png"
                    )
                    uri = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
                content.append(
                    {
                        "type": "input_image",
                        "image_url": uri,
                        "detail": part.get("detail", "high"),
                    }
                )
            else:
                raise ValueError(f"unsupported input part: {part.get('type')}")
        return {"role": message.get("role", "user"), "content": content}
