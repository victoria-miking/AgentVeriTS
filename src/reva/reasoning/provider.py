from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass
class ModelReply:
    response_id: str
    payload: dict[str, Any]
    raw: dict[str, Any]


class OpenAIReasoningClient:
    """Small Responses API adapter with image input, structured output and continuity."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-sol",
        reasoning_effort: str = "medium",
        max_output_tokens: int = 5000,
        store: bool = True,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = int(max_output_tokens)
        self.store = bool(store)
        if not self.store:
            raise ValueError(
                "This public REVA client uses previous_response_id for continuous signal-level "
                "reasoning and therefore requires store=True. Implement explicit replay before "
                "disabling stored response state."
            )
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL") or None)
        self.client = client

    @staticmethod
    def _image_url(path: str | Path) -> str:
        p = Path(path)
        mime = "image/jpeg" if p.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode('ascii')}"

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
        content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
        for image in images:
            content.append({"type": "input_image", "image_url": self._image_url(image), "detail": "high"})
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": [{"role": "user", "content": content}],
            "store": self.store,
            "parallel_tool_calls": False,
            "max_output_tokens": self.max_output_tokens,
            "reasoning": {"effort": self.reasoning_effort},
            "text": {
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": dict(schema),
                },
            },
        }
        if self.model.startswith("gpt-5.6"):
            kwargs["reasoning"]["context"] = "all_turns"
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
        response = self.client.responses.create(**kwargs)
        raw = response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
        output_text = getattr(response, "output_text", None) or raw.get("output_text")
        if not output_text:
            raise RuntimeError("model returned no structured output")
        payload = json.loads(output_text)
        response_id = str(getattr(response, "id", None) or raw.get("id") or "")
        return ModelReply(response_id=response_id, payload=payload, raw=raw)
