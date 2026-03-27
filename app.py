"""LLM SSE side-channel harness API."""

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import AsyncGenerator

import traceback

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
LOG_DIR = os.environ.get("LOG_DIR", "./logs")

os.makedirs(LOG_DIR, exist_ok=True)

# Module-level model references (populated at startup).
_tokenizer: AutoTokenizer | None = None
_model: torch.nn.Module | None = None


@dataclass
class StepMeta:
    """Metadata for a single token generation step."""

    step: int
    token_text: str          # Decoded token text emitted in this SSE step.
    token_utf8_len: int      # Token length in UTF-8 bytes.
    sse_frame_utf8_len: int  # Bytes in SSE frame (data: ...\\n\\n).
    t_rel_ms: int            # Relative timestamp in milliseconds.


@dataclass
class RunLog:
    """Run metadata and per-step details for side-channel analysis."""

    run_id: str
    prompt: str
    model_id: str
    device: str
    max_new_tokens: int
    temperature: float
    top_p: float
    prompt_len_tokens: int
    generated_tokens: int
    response_text: str
    response_utf8_len: int
    steps: list[StepMeta]


def _load_model() -> None:
    """Load tokenizer and model at startup."""
    global _tokenizer, _model
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map=None,
    ).to(DEVICE)
    _model.eval()


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle: load model on startup."""
    _load_model()
    yield


app = FastAPI(title="LLM SSE Side-Channel Harness", version="0.1", lifespan=_lifespan)


def _sse_frame(data_str: str) -> bytes:
    """Build minimal SSE frame: 'data: <content>\\n\\n'."""
    return f"data: {data_str}\n\n".encode("utf-8")


def _decode_single_token(tok_id: int) -> str:
    """Decode a single token without cleaning up tokenization spaces."""
    return _tokenizer.decode(
        [tok_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _build_input_text(prompt: str) -> str:
    """
    Build model-specific prompt text for chat/instruction models.

    - Qwen2.5: explicit `<|im_start|>...` format.
    - Other models: plain instruction-style prompt.
    """
    model_lower = MODEL_ID.lower()
    if "qwen" in model_lower:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return f"User: {prompt}\nAssistant:"


@torch.inference_mode()
def _generate_one_by_one(
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: int | None,
) -> list[int]:
    """
    Generate tokens one-by-one without HF streamer.

    Provides 1:1 mapping of token to SSE frame for side-channel analysis.
    """
    past_key_values = None
    generated: list[int] = []

    for _ in range(max_new_tokens):
        out = _model(
            input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = out.logits[:, -1, :]
        past_key_values = out.past_key_values

        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)

            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                mask = cum > top_p
                mask[..., 0] = False  # Always keep at least one token.
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_in_sorted = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
                next_id = sorted_idx.gather(-1, next_in_sorted.unsqueeze(-1)).squeeze(-1)
            else:
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

        generated.append(int(next_id.item()))
        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=-1)

        if eos_token_id is not None and int(next_id.item()) == int(eos_token_id):
            break

    return generated


@app.get("/generate_sse")
async def generate_sse(
    prompt: str = Query(..., min_length=1, max_length=4000),
    max_new_tokens: int = Query(128, ge=1, le=1024),
    temperature: float = Query(0.0, ge=0.0, le=2.0),
    top_p: float = Query(1.0, ge=0.1, le=1.0),
) -> StreamingResponse:
    """
    Stream SSE with side-channel metadata.

    Returns "data: <fragment>\\n\\n" per token. Logs token_utf8_len
    and sse_frame_utf8_len for analysis.
    """
    run_id = str(uuid.uuid4())
    t0 = time.time()

    try:
        input_text = _build_input_text(prompt)
        enc = _tokenizer(input_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        prompt_len_tokens = int(input_ids.shape[-1])
        eos_id = _tokenizer.eos_token_id

        loop = asyncio.get_running_loop()
        gen_ids = await loop.run_in_executor(
            None,
            _generate_one_by_one,
            input_ids,
            max_new_tokens,
            temperature,
            top_p,
            eos_id,
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    steps_meta: list[StepMeta] = []

    async def event_stream() -> AsyncGenerator[bytes, None]:
        try:
            async for chunk in _emit_event_stream(
                gen_ids, t0, run_id, steps_meta, prompt_len_tokens
            ):
                yield chunk
        except Exception:
            traceback.print_exc()
            raise

    async def _emit_event_stream(
        gen_ids: list[int],
        t0: float,
        run_id: str,
        steps_meta: list[StepMeta],
        prompt_len_tokens: int,
    ) -> AsyncGenerator[bytes, None]:
        emitted_tokens: list[str] = []
        for i, tok_id in enumerate(gen_ids):
            tok_str = _decode_single_token(tok_id)
            emitted_tokens.append(tok_str)
            tok_utf8_len = len(tok_str.encode("utf-8"))

            frame = _sse_frame(tok_str)
            frame_len = len(frame)

            t_rel_ms = int((time.time() - t0) * 1000)
            steps_meta.append(
                StepMeta(
                    step=i,
                    token_text=tok_str,
                    token_utf8_len=tok_utf8_len,
                    sse_frame_utf8_len=frame_len,
                    t_rel_ms=t_rel_ms,
                )
            )

            yield frame
            await asyncio.sleep(0)  # Yield control to push frames to clients.

        yield _sse_frame("[DONE]")

        response_text = "".join(emitted_tokens)

        run_log = RunLog(
            run_id=run_id,
            prompt=prompt,
            model_id=MODEL_ID,
            device=DEVICE,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prompt_len_tokens=prompt_len_tokens,
            generated_tokens=len(gen_ids),
            response_text=response_text,
            response_utf8_len=len(response_text.encode("utf-8")),
            steps=steps_meta,
        )
        log_path = os.path.join(LOG_DIR, f"{run_id}.json")
        log_data = {
            **{k: v for k, v in asdict(run_log).items() if k != "steps"},
            "steps": [asdict(s) for s in steps_meta],
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # Disable nginx buffering when proxied.
        "X-Run-Id": run_id,
    }
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=headers,
    )
