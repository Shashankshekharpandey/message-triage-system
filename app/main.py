import os

from fastapi import FastAPI, HTTPException, Request

from app.models import BatchRequest, BatchResponse, IncomingMessage, TriageResult
from app.triage import configured_provider, triage_many, triage_one

app = FastAPI(title="GAJAN Message Triage Service", version="1.0.0")
MAX_BODY_BYTES = 1_000_000


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/triage", response_model=TriageResult)
async def triage(payload: IncomingMessage, request: Request) -> TriageResult:
    if len(await request.body()) > MAX_BODY_BYTES:
        raise HTTPException(413, "Request body exceeds 1 MB")
    return await triage_one(payload.model_dump(), configured_provider())


@app.post("/triage/batch", response_model=BatchResponse)
async def batch(payload: BatchRequest, request: Request) -> BatchResponse:
    if len(await request.body()) > MAX_BODY_BYTES:
        raise HTTPException(413, "Request body exceeds 1 MB")
    messages = payload.messages
    limit = int(os.getenv("TRIAGE_MAX_BATCH_SIZE", "500"))
    if len(messages) > limit:
        raise HTTPException(413, f"Batch exceeds configured limit ({limit})")
    inputs = [item if isinstance(item, dict) else {"message": item} for item in messages]
    results = await triage_many(inputs, configured_provider())
    return BatchResponse(
        received=len(messages),
        processed=sum(result.status == "processed" for result in results),
        requiring_review=sum(result.human_review_required for result in results),
        failed=sum(result.status == "failed" for result in results),
        results=results,
    )
