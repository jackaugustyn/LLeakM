# LLM Cracker — SSE Side-Channel Harness

API and client for streaming LLM token generation via Server-Sent Events (SSE), with side-channel metadata logging for analysis (token lengths, frame sizes, timestamps).

## Requirements

- Python 3.10+
- CUDA (optional; falls back to CPU)

## Installation

```bash
pip install fastapi uvicorn torch transformers requests
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face model ID |
| `LOG_DIR` | `./logs` | Directory for run logs |

## Usage

### 1. Start the server

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

The model is loaded at startup. First run may take a while (model download).

### 2. Run the client

```bash
python client.py
```

The client streams SSE from `/generate_sse` and prints frame statistics.

## API

### `GET /generate_sse`

Stream token generation as SSE events.

| Query param | Type | Default | Description |
|-------------|------|---------|-------------|
| `prompt` | string | required | User prompt (1–4000 chars) |
| `max_new_tokens` | int | 128 | Max tokens to generate (1–1024) |
| `temperature` | float | 0.0 | Sampling temperature (0–2) |
| `top_p` | float | 1.0 | Nucleus sampling (0.1–1) |

**Response:** `text/event-stream` — each token as `data: <token>\n\n`. End marker: `data: [DONE]\n\n`.

## Output

- **Client:** Prints number of frames and first 10 frame lengths (bytes).
- **Server logs:** One JSON file per run in `LOG_DIR`, e.g. `{run_id}.json`, with:
  - Run metadata (model, device, prompt length, etc.)
  - Per-step: `token_utf8_len`, `sse_frame_utf8_len`, `t_rel_ms`

## Project structure

```
llm_cracker/
├── app.py      # FastAPI server with SSE streaming
├── client.py   # Example SSE client
├── logs/       # Run logs (created at runtime)
└── README.md
```

## Programmatic client example

```python
from client import collect_frame_lengths, URL

frame_lens = collect_frame_lengths(
    url=URL,
    params={
        "prompt": "Hello, world!",
        "max_new_tokens": 50,
        "temperature": 0.0,
    },
)
print(f"Frames: {len(frame_lens)}")
```
