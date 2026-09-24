from __future__ import annotations

import base64
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

OFFICIAL_BASE_URL = "https://api.openai.com/v1"


@dataclass
class ModelReply:
    response_id: str
    payload: dict[str, Any]
    raw: dict[str, Any]


class ResponseError(RuntimeError):
    """A failed/incomplete/refused response must never silently start a fresh chain."""

    def __init__(self, message: str, response_id: str | None = None) -> None:
        super().__init__(message)
        self.response_id = response_id


def validate_strict_schema(schema: Mapping[str, Any]) -> None:
    """Catch unsupported open objects before incurring an API request."""
    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            properties = node.get("properties", {})
            if node.get("additionalProperties") is not False or set(node.get("required", [])) != set(properties):
                raise ValueError("strict schemas require closed objects and every property in required")
            for child in properties.values():
                walk(child)
        walk(node.get("items"))
        for name in ("anyOf", "allOf", "oneOf"):
            for child in node.get(name, []):
                walk(child)
    walk(dict(schema))


class OpenAIReasoningClient:
    """Official Responses API; incremental input and stored response continuity.

    OPENAI_BASE_URL is intentionally ignored. Inject a client only for testing.
    Tools use the paper's structured action/observation protocol, not Chat Completions.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-sol",
        reasoning_effort: str = "medium",
        max_output_tokens: int = 5000,
        store: bool = True,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        if not store:
            raise ValueError("previous_response_id continuity requires store=True")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = int(max_output_tokens)
        self.call_log: list[dict[str, Any]] = []
        self._image_cache: OrderedDict[tuple[str, int, int], str] = OrderedDict()
        if client is None:
            from openai import OpenAI
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=OFFICIAL_BASE_URL,
                timeout=timeout_seconds,
                max_retries=max_retries,
            )
        self.client = client

    def _image_url(self, path: str | Path) -> str:
        p = Path(path).resolve()
        stat = p.stat()
        key = (str(p), stat.st_mtime_ns, stat.st_size)
        if key not in self._image_cache:
            mime = "image/jpeg" if p.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
            self._image_cache[key] = f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode('ascii')}"
            if len(self._image_cache) > 8:
                self._image_cache.popitem(last=False)
        self._image_cache.move_to_end(key)
        return self._image_cache[key]

    def complete(
        self,
        *,
        instructions: str,
        text: str,
        schema: Mapping[str, Any],
        schema_name: str,
        images: Sequence[str | Path] = (),
        previous_response_id: str | None = None,
    ) -> ModelReply:
        validate_strict_schema(schema)
        content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
        content.extend({"type": "input_image", "image_url": self._image_url(p), "detail": "high"} for p in images)
        kwargs: dict[str, Any] = {
            "model": self.model,
            # Instructions are request-scoped; repeat them on every continuation.
            "instructions": instructions,
            "input": [{"role": "user", "content": content}],
            "store": True,
            "truncation": "disabled",
            "max_output_tokens": self.max_output_tokens,
            "reasoning": {"effort": self.reasoning_effort},
            "text": {"verbosity": "low", "format": {
                "type": "json_schema", "name": schema_name, "strict": True, "schema": dict(schema),
            }},
        }
        if self.model.startswith("gpt-5.6"):
            kwargs["reasoning"]["context"] = "all_turns"
        if previous_response_id is not None:
            if not previous_response_id.strip():
                raise ValueError("previous_response_id must not be empty")
            kwargs["previous_response_id"] = previous_response_id
        record: dict[str, Any] = {
            "schema_name": schema_name, "model": self.model,
            "previous_response_id": previous_response_id, "status": "requesting",
        }
        self.call_log.append(record)
        try:
            # The SDK retries transient connection/rate/server errors with the same payload.
            # Never drop previous_response_id or reconstruct a different conversation on failure.
            response = self.client.responses.create(**kwargs)
        except Exception as exc:
            record.update(status="request_failed", error_type=type(exc).__name__)
            raise
        raw = response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
        response_id = raw.get("id")
        record.update(response_id=response_id, status=raw.get("status"), usage=raw.get("usage"), reasoning=raw.get("reasoning"))
        if not isinstance(response_id, str) or not response_id:
            raise ResponseError("OpenAI response is missing its id; cannot continue the response chain")
        if raw.get("status") != "completed" or raw.get("error"):
            raise ResponseError(
                f"Response {response_id} was not completed: {raw.get('incomplete_details') or raw.get('error') or raw.get('status')}",
                response_id,
            )
        parts: list[str] = []
        for item in raw.get("output", []):
            for part in item.get("content", []) if item.get("type") == "message" else []:
                if part.get("type") == "refusal":
                    raise ResponseError(f"Response {response_id} refused the request", response_id)
                if part.get("type") == "output_text":
                    parts.append(part["text"])
        output_text = getattr(response, "output_text", None) or raw.get("output_text") or "".join(parts)
        try:
            payload = json.loads(output_text)
        except (ValueError, TypeError) as exc:
            raise ResponseError(f"Response {response_id} did not contain valid structured JSON", response_id) from exc
        if not isinstance(payload, dict):
            raise ResponseError(f"Response {response_id} must contain a JSON object", response_id)
        return ModelReply(response_id=response_id, payload=payload, raw=raw)
