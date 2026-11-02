"""
Rekonstrukcja odpowiedzi LLM na podstawie kanału bocznego długości tokenów.

Adaptacja metody z artykułu:
  Weiss et al. "What Was Your Prompt? A Remote Keylogging Attack on AI Assistants"
  USENIX Security 2024

Integruje się z:
  - client.collect_frame_lengths() — długości ramek SSE
  - logi JSON z app.py (LOG_DIR) — token_utf8_len z kroków generowania
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

# SSE frame prefix "data: " = 6 bajtów, suffix "\n\n" = 2 bajty
SSE_FRAME_OVERHEAD = 8

# Modele fine-tunowane z paperu (T5)
FIRST_SENTENCES_MODEL = "royweiss1/T5_FirstSentences"
MIDDLE_SENTENCES_MODEL = "royweiss1/T5_MiddleSentences"

LOG_DIR = os.environ.get("LOG_DIR", "./logs")
MIN_SEGMENT_TOKENS = 10
DEFAULT_MAX_LENGTH = 80
_MODEL_CACHE: dict[str, tuple] = {}


def frame_lengths_to_token_lengths(frame_lengths: list[int]) -> list[int]:
    """
    Konwertuje długości ramek SSE na długości tokenów (bajty UTF-8).

    Format ramki: "data: <token>\\n\\n" → token_len = frame_len - 8
    Ostatnia ramka (client przerywa po niej) to zazwyczaj [DONE] — jest pomijana.
    """
    if not frame_lengths:
        return []

    # Client dodaje ramkę przed break, więc ostatni element to [DONE]
    frames = frame_lengths[:-1]

    token_lengths = []
    for fl in frames:
        if fl < SSE_FRAME_OVERHEAD:
            continue
        tok_len = fl - SSE_FRAME_OVERHEAD
        token_lengths.append(tok_len)

    return token_lengths


def load_token_lengths_from_log(run_id: str, log_dir: str = LOG_DIR) -> list[int]:
    """
    Wczytuje token_utf8_len z pliku logu dla danego run_id.
    """
    log_path = Path(log_dir) / f"{run_id}.json"
    if not log_path.exists():
        raise FileNotFoundError(f"Log nie istnieje: {log_path}")
    with open(log_path, encoding="utf-8") as f:
        data = json.load(f)
    steps = data.get("steps", [])
    return [s["token_utf8_len"] for s in steps]


def heuristic_sentences_from_lengths(lengths: list[int]) -> list[list[int]]:
    """
    Heurystyczny podział sekwencji długości tokenów na zdania.

    Implementuje opis z papera:
    1) split po tokenach długości 1 (interpunkcja),
    2) segmenty <10 tokenów są łączone z kolejnym segmentem,
    3) edge-case dla sekwencji listy (3,1,1) przenoszony na początek
       kolejnego segmentu.
    """
    if not lengths:
        return []

    # Krok 1: split po tokenach interpunkcyjnych (ti == 1)
    rough_segments: list[list[int]] = []
    current: list[int] = []
    for tok_len in lengths:
        current.append(tok_len)
        if tok_len == 1:
            rough_segments.append(current)
            current = []
    if current:
        rough_segments.append(current)

    if not rough_segments:
        return []

    # Krok 2: edge-case list ":\n\n1." => (3,1,1), przenosimy na kolejne zdanie.
    for idx in range(len(rough_segments) - 1):
        seg = rough_segments[idx]
        if len(seg) >= 3 and seg[-3:] == [3, 1, 1]:
            rough_segments[idx] = seg[:-3]
            rough_segments[idx + 1] = [3, 1, 1] + rough_segments[idx + 1]

    rough_segments = [seg for seg in rough_segments if seg]

    # Krok 3: merge segmentów krótszych niż 10 tokenów z następnym.
    merged: list[list[int]] = []
    i = 0
    while i < len(rough_segments):
        segment = rough_segments[i].copy()
        while len(segment) < MIN_SEGMENT_TOKENS and i + 1 < len(rough_segments):
            i += 1
            segment.extend(rough_segments[i])
        merged.append(segment)
        i += 1

    return merged


def make_input_from_lengths(lengths: list[int]) -> str:
    """
    Formatuje sekwencję długości tokenów jako wejście dla modelu T5.

    Zgodne z GPT_Keylogger _encoding_lengths: " _3 _2 _5".
    """
    parts = [" _" + str(x) for x in lengths]
    return "Translate the Special Tokens to English. \nSpecial Tokens:" + "".join(parts)


def make_input_with_context(lengths: list[int], context: str) -> str:
    """Formatuje wejście z kontekstem poprzedniego zdania."""
    parts = [" _" + str(x) for x in lengths]
    tokens_str = "".join(parts)
    return f"Translate the Special Tokens to English, given the context. \nContext: {context} \nSpecial Tokens:{tokens_str}"


@dataclass
class ReconstructionResult:
    """Wynik rekonstrukcji — pierwsze zdanie i ewentualne kolejne."""

    first_sentence: str
    full_text: str
    sentence_count: int
    available_segment_count: int
    top_candidates: list[str]


def seed_reconstructor(seed: int) -> None:
    """Seed every RNG used by the stochastic reconstruction decoder.

    Calling this at the start of each reconstruction makes it possible to use
    common random numbers across experimental conditions.  CUDA generators are
    seeded as well when CUDA is available.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except ImportError:
        pass

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _generate_with_compat_fallback(model, **gen_kwargs):
    """
    Kompatybilne wywołanie generate dla nowych wersji transformers.

    W transformers>=5 Group Beam Search został wydzielony do custom_generate
    i bez trust_remote_code rzuca ValueError. W takim przypadku przechodzimy
    na standardowy beam search (bez num_beam_groups / diversity_penalty).
    """
    try:
        return model.generate(**gen_kwargs)
    except ValueError as exc:
        msg = str(exc)
        if "Group Beam Search requires `trust_remote_code=True`" not in msg:
            raise

        fallback_kwargs = dict(gen_kwargs)
        fallback_kwargs.pop("num_beam_groups", None)
        fallback_kwargs.pop("diversity_penalty", None)
        return model.generate(**fallback_kwargs)


def _choose_device() -> str:
    forced = os.environ.get("LLEAKM_RECONSTRUCT_DEVICE", "").strip().lower()
    try:
        import torch

        if forced == "cpu":
            return "cpu"
        if forced == "cuda" or (not forced and getattr(torch, "cuda", None) and torch.cuda.is_available()):
            if getattr(torch, "cuda", None) and torch.cuda.is_available():
                return "cuda"
    except ImportError:
        pass
    return "cpu"


def _load_model_bundle(model_name: str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    cached = _MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = _choose_device()
    model = model.to(device)
    model.eval()
    bundle = (model, tokenizer, device)
    _MODEL_CACHE[model_name] = bundle
    return bundle


def _prepare_input(tokenizer, device: str, text: str, max_length: int):
    inputs = tokenizer(
        [text],
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in inputs.items()}


def _sample_and_rank(
    model,
    tokenizer,
    device: str,
    input_text: str,
    *,
    samples: int,
    max_length: int = DEFAULT_MAX_LENGTH,
    candidate_audit: list | None = None,
) -> list[tuple[str, float]]:
    """
    Sample texts and rank processed transition scores through the first EOS.

    The first generated EOS is included, while every later forced PAD is
    excluded. With no generated EOS, all generation steps contribute. These
    are sampling-processed scores, not unconstrained model likelihoods.
    A single candidate retains the historical no-score generation path.
    ``candidate_audit`` optionally receives one JSON-compatible call record;
    observing the candidates does not draw random numbers or change sampling.
    """
    import math
    import operator
    import torch

    inputs = _prepare_input(tokenizer, device, input_text, max_length=max_length)
    best_per_text: dict[str, float] = {}
    best_indices: dict[str, int] = {}
    n_samples = max(1, samples)
    audited_candidates = []

    def json_score(value):
        value = float(value)
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "-Infinity" if value < 0 else "Infinity"
        return value

    # generate() uses generation_config when it exists. An explicit None
    # there disables EOS; do not replace it with a tokenizer's EOS token.
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        eos_value = getattr(generation_config, "eos_token_id", None)
    elif getattr(model, "config", None) is not None:
        eos_value = getattr(model.config, "eos_token_id", None)
    else:
        eos_value = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_value, torch.Tensor):
        eos_value = eos_value.tolist()
    eos_values = [] if eos_value is None else (eos_value if isinstance(eos_value, (list, tuple)) else [eos_value])
    eos_ids = list(dict.fromkeys(operator.index(value) for value in eos_values))

    with torch.inference_mode():
        gen_kwargs = dict(
            max_length=max_length,
            return_dict_in_generate=True,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            temperature=1.0,
            num_return_sequences=n_samples,
        )
        if n_samples > 1:
            gen_kwargs["output_scores"] = True
        outputs = _generate_with_compat_fallback(model, **inputs, **gen_kwargs)
        sequences = outputs.sequences
        transition = None
        if sequences.ndim != 2 or sequences.shape[0] != n_samples:
            raise ValueError("Generated candidate dimensions differ from the request")
        if n_samples > 1:
            if not getattr(outputs, "scores", None):
                raise RuntimeError("Multi-candidate ranking requires generation scores")
            # A score computation failure must stop the experiment. Silently
            # assigning zero would substitute a different selection rule.
            transition = model.compute_transition_scores(
                sequences,
                outputs.scores,
                normalize_logits=True,
            )
            if (transition.ndim != 2 or transition.shape[0] != sequences.shape[0]
                    or transition.shape[1] != len(outputs.scores)
                    or not 0 < transition.shape[1] < sequences.shape[1]):
                raise ValueError("Transition scores cannot align with generated candidates")
            decoder_prefix_length = sequences.shape[1] - transition.shape[1]
        else:
            # This function sends only encoder inputs to the T5 model. Its
            # generated sequence starts with one decoder-start token.
            decoder_prefix_length = 1
        for i in range(sequences.shape[0]):
            text = tokenizer.decode(sequences[i], skip_special_tokens=True).strip()
            generated_ids = sequences[i, decoder_prefix_length:].tolist()
            first_eos = next((j for j, token in enumerate(generated_ids) if token in eos_ids), None)
            if transition is None:
                confidence = 0.0
                scored_count = 0
            else:
                scored_count = first_eos + 1 if first_eos is not None else len(generated_ids)
                # Slice: multiplying masked -inf values by zero produces NaN.
                prefix_scores = transition[i, :scored_count]
                if not torch.isfinite(prefix_scores).all().item():
                    raise ValueError(f"Nonfinite transition score before or at EOS in candidate {i}")
                confidence = float(prefix_scores.sum().item())
                if not math.isfinite(confidence):
                    raise ValueError(f"Nonfinite EOS-bounded confidence in candidate {i}")
            if candidate_audit is not None:
                audited_candidates.append(dict(
                    candidate_index=i,
                    sequence_token_ids=sequences[i].tolist(),
                    generated_token_ids=generated_ids,
                    text=text,
                    transition_scores=None if transition is None else [json_score(x) for x in transition[i].tolist()],
                    first_eos_generated_offset=first_eos,
                    scored_token_count=scored_count,
                    confidence=confidence,
                    legacy_full_step_confidence=None if transition is None else json_score(transition[i].sum().item()),
                    eligible=bool(text),
                ))
            if text:
                existing = best_per_text.get(text)
                if existing is None or confidence > existing:
                    best_per_text[text] = confidence
                    best_indices[text] = i

    ranked = sorted(best_per_text.items(), key=lambda item: item[1], reverse=True)
    if candidate_audit is not None:
        ranked_indices = [best_indices[text] for text, _ in ranked]
        candidate_audit.append(dict(
            schema="eos-bounded-candidate-audit/1",
            selector="processed-score-sum-through-first-eos",
            samples_requested=n_samples,
            decoder_prefix_length=decoder_prefix_length,
            eos_token_ids=eos_ids,
            score_dtype=None if transition is None else str(transition.dtype),
            candidates=audited_candidates,
            ranked_candidate_indices=ranked_indices,
            ranked_texts=[text for text, _ in ranked],
            selected_candidate_index=ranked_indices[0] if ranked_indices else None,
        ))
    return ranked


def reconstruct(
    token_lengths: list[int],
    num_first_candidates: int = 5,
    max_sentences: int | None = 5,
    samples_per_segment: int = 10,
    seed: int | None = None,
) -> ReconstructionResult:
    """
    Rekonstruuje odpowiedź LLM z sekwencji długości tokenów (bajty UTF-8).

    Używa metody Weiss: heurystyka podziału na zdania, T5_FirstSentences
    dla pierwszego zdania, T5_MiddleSentences dla kolejnych z kontekstem.

    Args:
        token_lengths: Lista długości tokenów w bajtach (np. z frame_lengths).
        num_first_candidates: Ile kandydatów pierwszego zdania zwrócić.
        max_sentences: Maksymalna liczba rekonstruowanych segmentów. Wartość
            ``None`` albo liczba niedodatnia oznacza wszystkie segmenty.
        seed: Opcjonalny seed dekodera. Ten sam seed można zastosować do
            wszystkich warunków dla danej próbki (common random numbers).

    Returns:
        ReconstructionResult z najlepszym pierwszym zdaniem i pełnym tekstem.
    """
    sentence_lengths = heuristic_sentences_from_lengths(token_lengths)
    if not sentence_lengths:
        return ReconstructionResult(
            first_sentence="",
            full_text="",
            sentence_count=0,
            available_segment_count=0,
            top_candidates=[],
        )

    segment_limit = len(sentence_lengths)
    if max_sentences is not None and max_sentences > 0:
        segment_limit = min(segment_limit, max_sentences)

    first_model, first_tokenizer, first_device = _load_model_bundle(FIRST_SENTENCES_MODEL)
    middle_model, middle_tokenizer, middle_device = _load_model_bundle(MIDDLE_SENTENCES_MODEL)

    # Loading a checkpoint can consume RNG state. Seed after loading so a
    # resumed run's first record matches the same record with warm caches.
    if seed is not None:
        seed_reconstructor(seed)

    # Rekonstrukcja pierwszego segmentu
    first_input = make_input_from_lengths(sentence_lengths[0])
    first_ranked = _sample_and_rank(
        first_model,
        first_tokenizer,
        first_device,
        first_input,
        samples=max(num_first_candidates, samples_per_segment),
    )
    candidates = [text for text, _ in first_ranked]
    best_first = candidates[0] if candidates else ""
    top_candidates = candidates[: max(1, num_first_candidates)]

    full_parts = [best_first]
    context = best_first

    # Rekonstrukcja kolejnych zdań (z kontekstem)
    for sent_lens in sentence_lengths[1:segment_limit]:
        next_sentence = _generate_middle_sentence(
            sent_lens,
            context,
            middle_model,
            middle_tokenizer,
            middle_device,
            samples=samples_per_segment,
        )
        if next_sentence:
            full_parts.append(next_sentence)
            context = next_sentence
        else:
            break

    return ReconstructionResult(
        first_sentence=best_first,
        full_text=_concat_segments(full_parts),
        sentence_count=len(full_parts),
        available_segment_count=len(sentence_lengths),
        top_candidates=top_candidates,
    )


def _concat_segments(segments: list[str]) -> str:
    parts = [s.strip() for s in segments if s and s.strip()]
    if not parts:
        return ""
    text = " ".join(parts)
    return re.sub(r"\s+([,.;:!?])", r"\1", text)


def _generate_middle_sentence(
    lengths: list[int],
    context: str,
    model,
    tokenizer,
    device: str,
    *,
    samples: int,
) -> str:
    """Wywołuje model middle sentences z poprawnymi argumentami."""
    input_str = make_input_with_context(lengths, context)
    ranked = _sample_and_rank(
        model,
        tokenizer,
        device,
        input_str,
        samples=samples,
        max_length=DEFAULT_MAX_LENGTH,
    )
    return ranked[0][0] if ranked else ""


def reconstruct_from_frame_lengths(frame_lengths: list[int], **kwargs) -> ReconstructionResult:
    """
    Rekonstruuje odpowiedź z długości ramek SSE (wyjście client.collect_frame_lengths).
    """
    token_lengths = frame_lengths_to_token_lengths(frame_lengths)
    return reconstruct(token_lengths, **kwargs)


def main() -> None:
    """
    Demo: łączy client.collect_frame_lengths z rekonstrukcją Weiss.

    Użycie:
      python weiss_reconstruction.py           -- z żywego strumienia SSE
      python weiss_reconstruction.py <run_id>  -- z pliku logu (LOG_DIR)
    """
    import sys

    if len(sys.argv) >= 2:
        run_id = sys.argv[1]
        print(f"Wczytywanie z logu: {run_id}")
        token_lens = load_token_lengths_from_log(run_id)
        print(f"Otrzymano {len(token_lens)} tokenów z pliku logu.")
    else:
        from client import collect_frame_lengths, DEFAULT_PARAMS, URL

        print("Pobieranie strumienia SSE z harnessa...")
        frame_lens = collect_frame_lengths(URL, DEFAULT_PARAMS)
        print(f"Otrzymano {len(frame_lens)} ramek.")
        token_lens = frame_lengths_to_token_lengths(frame_lens)

    print(f"Długości tokenów (pierwsze 20): {token_lens[:20]}")

    if not token_lens:
        print("Brak tokenów do rekonstrukcji.")
        return

    print("\nRekonstrukcja (metoda Weiss)...")
    result = reconstruct(
        token_lens,
        num_first_candidates=3,
        max_sentences=3,
        samples_per_segment=10,
    )
    print(f"\nPierwsze zdanie (najlepsze): {result.first_sentence}")
    print(f"\nPełny tekst ({result.sentence_count} zdań): {result.full_text}")
    print("\nTop 3 kandydatów pierwszego zdania:")
    for i, c in enumerate(result.top_candidates, 1):
        print(f"  {i}. {c}")


if __name__ == "__main__":
    main()
