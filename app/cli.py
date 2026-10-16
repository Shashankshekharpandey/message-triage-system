import argparse
import asyncio
import json
from pathlib import Path

from app.triage import configured_provider, triage_many

def read_messages(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    try: data = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip(): continue
            try: records.append(json.loads(line))
            except json.JSONDecodeError: records.append({"_parse_error": f"Malformed JSONL record at line {line_number}"})
        return records
    if isinstance(data, list): return [x if isinstance(x, dict) else {"message": x} for x in data]
    if isinstance(data, dict) and isinstance(data.get("messages"), list): return [x if isinstance(x, dict) else {"message": x} for x in data["messages"]]
    return [data] if isinstance(data, dict) else [{"message": data}]

async def run(args) -> int:
    items = read_messages(Path(args.input))
    results = await triage_many(items, configured_provider(), args.concurrency)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for result in results: stream.write(result.model_dump_json() + "\n")
    failed = sum(x.status == "failed" for x in results)
    print(f"received={len(items)} processed={len(results)-failed} failed={failed} review={sum(x.human_review_required for x in results)} output={output}")
    return 1 if failed else 0

def main() -> None:
    parser = argparse.ArgumentParser(description="Triage a JSON or JSONL message dataset")
    parser.add_argument("--input", default="messages.json")
    parser.add_argument("--output", default="output/triage_results.jsonl")
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))

if __name__ == "__main__": main()
