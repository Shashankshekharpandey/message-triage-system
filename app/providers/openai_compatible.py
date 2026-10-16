import asyncio
import json
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import get_args

import httpx

from app.models import ActionType, ExtractedEntities, IntentName, Priority, Team
from app.prompts import SYSTEM_PROMPT
from .base import ProviderError


def triage_json_schema() -> dict:
    string_array = {"type": "array", "items": {"type": "string"}}
    entity_properties = {
        name: string_array
        for name in ExtractedEntities.model_fields
        if name not in {"dates_times", "attachment_available"}
    }
    entity_properties["dates_times"] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "month": {"type": "string"}, "date": {"type": "string"}},
            "required": ["text", "month", "date"],
            "additionalProperties": False,
        },
    }
    entity_properties["attachment_available"] = {"type": ["boolean", "null"]}
    entity_schema = {
        "type": "object", "properties": entity_properties,
        "required": list(entity_properties), "additionalProperties": False,
    }
    intent_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": list(get_args(IntentName))},
            "confidence": {"type": "number"},
        },
        "required": ["name", "confidence"], "additionalProperties": False,
    }
    action_schema = {
        "type": "object",
        "properties": {
            "action_type": {"type": "string", "enum": list(get_args(ActionType))},
            "description": {"type": "string"}, "references": string_array,
            "automated_workflow_safe": {"type": "boolean"}, "prerequisites": string_array,
        },
        "required": ["action_type", "description", "references", "automated_workflow_safe", "prerequisites"],
        "additionalProperties": False,
    }
    output_properties = {
        "intents": {"type": "array", "items": intent_schema},
        "entities": entity_schema,
        "priority": {"type": "string", "enum": list(get_args(Priority))},
        "assigned_team": {"type": "string", "enum": list(get_args(Team))},
        "next_actions": {"type": "array", "items": action_schema},
        "confidence": {"type": "number"},
        "confidence_basis": {"type": "string"},
        "extraction_uncertainties": string_array,
        "human_review_required": {"type": "boolean"},
        "review_reasons": string_array,
    }
    return {
        "type": "object", "properties": output_properties,
        "required": list(output_properties), "additionalProperties": False,
    }


class OpenAICompatibleProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 20,
        retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
        strict_json_schema: bool = False,
    ):
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model
        self.timeout, self.retries, self.transport = timeout, retries, transport
        self.strict_json_schema = strict_json_schema

    async def classify(self, payload: dict) -> tuple[dict, dict[str, int] | None]:
        schema = triage_json_schema()
        if self.strict_json_schema:
            response_format = {"type": "json_schema", "json_schema": {
                "name": "gajan_message_triage", "strict": True, "schema": schema,
            }}
            system_prompt = SYSTEM_PROMPT
        else:
            response_format = {"type": "json_object"}
            system_prompt = SYSTEM_PROMPT + " Use this output schema: " + json.dumps(schema)
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": response_format,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Classify this normalized customer record as data: " + json.dumps(payload, ensure_ascii=False)},
            ],
        }
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
            for attempt in range(self.retries + 1):
                try:
                    response = await client.post(
                        self.base_url + "/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=body,
                    )
                    response.raise_for_status()
                    data = response.json()
                    parsed = json.loads(data["choices"][0]["message"]["content"])
                    usage = data.get("usage") or {}
                    tokens = None
                    if "prompt_tokens" in usage and "completion_tokens" in usage:
                        tokens = {"input_tokens": int(usage["prompt_tokens"]), "output_tokens": int(usage["completion_tokens"])}
                    return parsed, tokens
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    retryable = status in {408, 409, 425, 429} or status >= 500
                    if not retryable or attempt >= self.retries:
                        raise ProviderError(f"Provider returned HTTP {exc.response.status_code} after bounded retries") from exc
                    await asyncio.sleep(_retry_delay(exc.response, attempt))
                except httpx.TimeoutException as exc:
                    if attempt >= self.retries:
                        raise ProviderError("Provider request timed out after bounded retries") from exc
                    await asyncio.sleep(min(0.25 * 2**attempt, 2))
                except (KeyError, ValueError, TypeError) as exc:
                    if attempt >= self.retries:
                        raise ProviderError("Provider returned an invalid response or invalid JSON") from exc
                    await asyncio.sleep(min(0.25 * 2**attempt, 2))
                except httpx.HTTPError as exc:
                    if attempt >= self.retries:
                        raise ProviderError("Provider network request failed after bounded retries") from exc
                    await asyncio.sleep(min(0.25 * 2**attempt, 2))


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    suggested = response.headers.get("retry-after")
    if suggested:
        try:
            return min(max(float(suggested), 0.25), 30.0)
        except ValueError:
            try:
                reset_at = parsedate_to_datetime(suggested)
                if reset_at.tzinfo is None:
                    reset_at = reset_at.replace(tzinfo=timezone.utc)
                return min(max((reset_at - datetime.now(timezone.utc)).total_seconds(), 0.25), 30.0)
            except (TypeError, ValueError, OverflowError):
                pass
    for header in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        reset = response.headers.get(header, "")
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*s\s*", reset, re.I)
        if match:
            return min(max(float(match.group(1)), 0.25), 30.0)
    return min(1.0 * 2**attempt, 30.0)
