from typing import Protocol


class ProviderError(RuntimeError):
    pass


class LLMProvider(Protocol):
    async def classify(self, payload: dict) -> tuple[dict, dict[str, int] | None]: ...
