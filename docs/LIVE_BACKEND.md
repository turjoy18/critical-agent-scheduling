# Free-tier live backend of record

**Cost policy:** chat/completions use free-tier APIs only. Paid budget is GPU-only (≤ US$50; see [`GPU_CLOUD.md`](GPU_CLOUD.md)).

## Default

| Knob | Value |
|------|--------|
| Resolution order | `LLM_BACKEND` → **gemini** → groq → openrouter |
| Gemini model | `gemini-flash-lite-latest` (OpenAI-compat) |
| Env | `GEMINI_API_KEY` from Google AI Studio |

`gemini-2.5-flash` works but hits free RPM quickly. Bare `gemini-2.5-flash-lite` returns 404 on the OpenAI-compat endpoint; use `gemini-flash-lite-latest`.

## Run

```bash
cp env.example .env   # set GEMINI_API_KEY
python -m scheduler.live_runner --backend gemini --runs 2 --workers 2
```

Artifacts: `scheduler/results/live_comparison.csv`, `live_traces.jsonl`.

## Interpretation

Live makespans are dominated by provider latency/jitter, not Couloir’s cache utility (no local KV store on the free API path). Treat live runs as **integration smoke**, not paper claims. Paper numbers come from the discrete-event sim (`scheduler.couloir_sweep` / `scheduler.experiment`).
