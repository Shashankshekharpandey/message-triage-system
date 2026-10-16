# GAJAN Group Holdings Message Triage Service

A small FastAPI service and CLI that convert customer messages into structured triage: one or more intents, supported entity extraction, priority, team, suggested next actions, confidence, and explicit human-review reasons. It recommends work; it does not contact customers, inspect business systems, make bookings, issue refunds, or move money.

## Dataset availability

The supplied `messages.json` is preserved at the project root and contains 25 records with fields `id`, `brand`, `channel`, `received_at`, and `text`. The input model maps `id` to `message_id` and `received_at` to `timestamp`, retaining these values in output. The dataset includes a null-text record, an attempted instruction override, multi-intent requests, and incomplete context. No supplied message is overwritten.

## Architecture

`/triage` and `/triage/batch` plus `python -m app.cli` share one normalization and orchestration path. Pydantic validates inputs and outputs; deterministic rules provide key safety fast paths and a credential-free demo classifier. Optional OpenAI-compatible chat completions are behind a provider interface. Safety overrides run in Python after any model response. Batch work uses an async semaphore and keeps result order aligned with input order.

## Requirements and setup

Python 3.10+ (developed for a compatible modern Python; Python 3.12 recommended).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Environment variables are read from the process environment; the app does not automatically load `.env`. In PowerShell, set values for the current shell:

```powershell
$env:TRIAGE_MODE = "mock"
```

### Demo mode

Demo mode requires no key and uses deterministic phrase/pattern rules. It is **not a real LLM evaluation**; it can miss paraphrases and multilingual nuance, and its scores are heuristic rather than calibrated.

```powershell
$env:TRIAGE_MODE = "mock"
python -m app.cli --input messages.json --output output/triage_results.jsonl
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Health is `GET /health`.

### Groq LLM mode

Groq uses its OpenAI-compatible Chat Completions endpoint through the same replaceable provider adapter. The default Groq URL is `https://api.groq.com/openai/v1`; set `LLM_PROVIDER=groq`, `GROQ_API_KEY`, and `GROQ_MODEL`. Groq mode requests strict JSON Schema output. The example uses `openai/gpt-oss-20b`, which is listed by the configured account and supports Groq Structured Outputs. Choose a model enabled for your account. Pricing and account limits are not hardcoded.

```dotenv
TRIAGE_MODE=llm
LLM_PROVIDER=groq
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=openai/gpt-oss-20b
```

Save these values in `.env`, then start or restart the service:

```powershell
uvicorn app.main:app --reload
```

For the OpenAI provider instead, set `LLM_PROVIDER=openai`, `LLM_API_KEY`, `LLM_MODEL`, and optionally `LLM_BASE_URL`.

Requests have a configurable timeout and a bounded retry count with exponential backoff. A missing provider key or model setting returns a configuration error; obvious deterministic fast paths remain explicitly marked `mock`. Messages requiring a model return `status: failed` instead of being presented as model classifications when configuration, provider, or output validation fails. Test/demo and provider metrics are exposed per result when available.

## API

Single message (`message` and `text` are accepted):

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/triage -ContentType 'application/json' -Body '{"message_id":"demo-1","brand":"Vitalis Wellness","channel":"WhatsApp","message":"My order VW-48812 tracking has not changed for four days."}'
```

Example response shape (confidence value illustrative of heuristic output):

```json
{
  "message_id": "demo-1",
  "status": "processed",
  "brand": "Vitalis Wellness",
  "channel": "WhatsApp",
  "intents": [{"name": "order_tracking", "confidence": 0.78}],
  "entities": {"order_ids": ["VW-48812"], "lookup_required": ["order_system_lookup"]},
  "priority": "normal",
  "assigned_team": "order_support",
  "next_actions": [{"action_type":"check_tracking","description":"Check shipment status using the supplied order reference","references":["VW-48812"],"automated_workflow_safe":true,"prerequisites":[]}],
  "confidence": 0.78,
  "confidence_basis": "Deterministic phrase and pattern evidence; heuristic, not statistically calibrated",
  "extraction_uncertainties": [],
  "human_review_required": false,
  "review_reasons": [],
  "processing_mode": "mock",
  "processing_time_ms": null,
  "token_usage": null,
  "error": null
}
```

Batch request is `{"messages":[{...},{...}]}` at `POST /triage/batch`. Response contains `received`, `processed`, `requiring_review`, `failed`, and ordered `results`. Default batch limit is 500. CLI reads a JSON array, `{ "messages": [...] }`, one JSON object, or JSONL and writes one JSON result per output line:

```powershell
python -m app.cli --input messages.json --output output/triage_results.jsonl --concurrency 8
```

## Confidence and review

Scores represent evidence strength from the current classification path, not calibrated probability. The configurable initial review threshold is `TRIAGE_CONFIDENCE_THRESHOLD=0.65`; evaluate it against representative, human-labeled messages before production. Extraction uncertainty is reported separately where identified. Review is mandatory for suspected prompt injection, medical questions, refunds/payment/invoice requests, urgent travel disruption, missing context, empty text, low confidence, and failed validation/provider results. Python safety rules override model intent, priority, routing, and requested suppression of review.

The injection detector uses conservative deterministic patterns as defense in depth; it cannot find every attack. Model content is treated as untrusted data and validated through Pydantic enums and schemas. No customer message can authorize financial action. Model failure is a failed result. Logs and CLI summaries avoid message bodies and secrets. Invoice attachments are not downloaded or inspected; an attachment mentioned in text is marked unavailable.

## Scaling and cost assumptions

10,000 messages/day averages about `10,000 / 86,400 = 0.116` messages/second; bursts, not the average, determine capacity. Async HTTP and bounded concurrency help overlap provider I/O. Production should tune concurrency against model-provider rate limits, observed burst profiles, timeout rates, and payload sizes. Retries are capped and exponential; outages produce explicit partial failures. The API caps batch count at 500 and request bodies at 1 MB. Monitor latency percentiles, error rates, review rate, token usage, and provider throttling. Use redacted structured logs, trace IDs, and alerting. Add idempotency keys and durable deduplication before retrying queued jobs. A managed queue and worker pool are reasonable next steps if bursts, replay, or durability requirements exceed a synchronous API; current local service does not provide durable queueing.

Illustrative usage only: assume 800 input and 150 output tokens per message. Let `P_in` and `P_out` be configured provider prices per million tokens, and `C_other` allocated non-model cost per 1,000. Then:

`C_1000 = 1000 × 800 × P_in / 1,000,000 + 1000 × 150 × P_out / 1,000,000 + C_other`

For demonstration only, if hypothetical rates were `$1.00` input and `$4.00` output per million tokens, model spend would be `$0.80 + $0.60 = $1.40 per 1,000 messages`, before `C_other`. These are not current vendor prices. Replace them with verified contracted pricing and measured token usage. The estimate excludes retries, infrastructure/logging (`C_other` is left as a deployment-specific input), and human-review labor; the supplied run uses mock mode, so provider token usage is unavailable. Local processing time is recorded per result; provider token usage is recorded only when the provider returns it.

## Tests and known limitations

```powershell
python -m pytest -q
```

The supplied file is small and cannot establish production accuracy. The rule engine is intentionally small and English-heavy; it does not perform broad multilingual understanding or calibrated evaluation. Extracted dates preserve an unknown month only for recognized day phrasing. Business-system lookups, actual attachment retrieval, authentication, durable storage, rate limiting across replicas, and side-effecting integrations are not included. Next steps: annotate a representative evaluation set, measure per-intent precision/recall and calibration, validate model/price choices, add authentication and operational observability, and conduct privacy/security review.
