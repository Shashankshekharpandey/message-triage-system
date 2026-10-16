import asyncio
import os
import time
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from app.models import TriageResult
from app.providers import LLMProvider, OpenAICompatibleProvider, ProviderError
from app.rules import deterministic_draft, make_result, normalized

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class MissingCredentialsProvider:
    def __init__(self, message: str):
        self.message = message

    async def classify(self, payload: dict) -> tuple[dict, dict | None]:
        raise ProviderError(self.message)


def configured_provider() -> LLMProvider | None:
    if os.getenv("TRIAGE_MODE", "mock").lower() != "llm": return None
    provider_name = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not provider_name:
        provider_name = "groq" if os.getenv("GROQ_API_KEY") else "openai"
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
    retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
    if provider_name == "groq":
        key = os.getenv("GROQ_API_KEY", "")
        model = os.getenv("GROQ_MODEL", "").strip()
        if not key:
            return MissingCredentialsProvider("Configuration error: GROQ_API_KEY is missing; add it to .env or the process environment")
        if not model:
            return MissingCredentialsProvider("Configuration error: GROQ_MODEL is missing; set the Groq model ID in .env")
        return OpenAICompatibleProvider(os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"), key, model, timeout, retries, strict_json_schema=True)
    if provider_name == "openai":
        key = os.getenv("LLM_API_KEY", "")
        if not key: return MissingCredentialsProvider("Configuration error: LLM_API_KEY is missing; add it to .env or the process environment")
        return OpenAICompatibleProvider(os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"), key,
            os.getenv("LLM_MODEL", "gpt-4o-mini"), timeout, retries)
    return MissingCredentialsProvider(f"Configuration error: unsupported LLM_PROVIDER '{provider_name}'")


async def triage_one(raw: Any, provider: LLMProvider | None = None, confidence_threshold: float | None = None) -> TriageResult:
    started = time.perf_counter()
    threshold = confidence_threshold if confidence_threshold is not None else float(os.getenv("TRIAGE_CONFIDENCE_THRESHOLD", ".65"))
    if not isinstance(raw, dict): raw = {"message": raw}
    try:
        msg, text = normalized(raw)
        draft = deterministic_draft(msg, text)
    except Exception:
        return make_result(raw, duration=(time.perf_counter()-started)*1000, confidence_threshold=threshold)
    # Handle clear mandatory-review cases deterministically so rate limits cannot delay safety routing.
    names = {i["name"] for i in draft["intents"]}
    fast = (not text or "unsubscribe" in names or bool(names & {"urgent_travel_disruption", "medical_or_health_question"})
        or bool(__import__("re").search(r"ignore.{0,100}(instructions?|system|admin)|override.{0,100}(system|policy)", text, __import__("re").I))
        or bool(__import__("re").search(r"\b(other one|that one|which one)\b|\b(work came up|something came up)\b", text, __import__("re").I)))
    if provider is None or fast:
        return make_result(raw, draft=draft, duration=(time.perf_counter()-started)*1000, confidence_threshold=threshold)
    try:
        result, usage = await provider.classify({"message_id": msg.message_id, "brand": msg.brand, "channel": msg.channel,
            "message": text, "timestamp": msg.timestamp, "conversation_id": msg.conversation_id, "metadata": msg.metadata})
        return make_result(raw, mode="llm", draft=result, duration=(time.perf_counter()-started)*1000, tokens=usage, confidence_threshold=threshold)
    except ProviderError as exc:
        detail = str(exc)
        message = detail if detail.startswith(("Configuration error:", "Provider returned HTTP ", "Provider request timed out", "Provider returned an invalid response", "Provider network request failed")) else "Provider request failed after bounded retries"
        return make_result(raw, mode="error", duration=(time.perf_counter()-started)*1000,
            error=message, confidence_threshold=threshold)
    except Exception:
        return make_result(raw, mode="error", duration=(time.perf_counter()-started)*1000,
            error="Provider request failed after bounded retries", confidence_threshold=threshold)


async def triage_many(items: list[dict], provider: LLMProvider | None = None, concurrency: int | None = None) -> list[TriageResult]:
    limit = max(1, concurrency or int(os.getenv("TRIAGE_MAX_CONCURRENCY", "8")))
    sem = asyncio.Semaphore(limit)
    async def run(item: dict) -> TriageResult:
        async with sem: return await triage_one(item, provider)
    return list(await asyncio.gather(*(run(item) for item in items)))
