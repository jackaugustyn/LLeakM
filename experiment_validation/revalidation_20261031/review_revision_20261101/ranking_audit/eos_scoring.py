"""Audit-only EOS-aware score; the historical reconstructor is not modified.

Scores are the normalized, processed generation distributions returned by
compute_transition_scores, not the unprocessed model likelihoods. The first EOS
is included. Forced tokens after it are excluded even when their score is -inf.
"""
from __future__ import annotations

import torch


def scores_through_eos(sequences, transition_scores, eos_token_id):
    """Return (sums, valid_counts, first_EOS_generated_offsets).

The transition columns align with the suffix of ``sequences``: this handles T5's
one-token decoder start without treating the start/PAD as generated evidence.
PAD before EOS is deliberately retained. With no EOS the whole generated suffix
is retained. Nonfinite scores before/at EOS remain nonfinite; we do not silently
turn an invalid in-sequence score into a valid probability.
"""
    if sequences.ndim != 2 or transition_scores.ndim != 2:
        raise ValueError("Sequences and transition scores must be matrices")
    if sequences.shape[0] != transition_scores.shape[0]:
        raise ValueError("Candidate count differs")
    steps = transition_scores.shape[1]
    if not steps or sequences.shape[1] < steps:
        raise ValueError("Transition scores cannot align with sequences")
    ids = [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)
    if not ids:
        raise ValueError("At least one EOS token ID is required")
    generated = sequences[:, sequences.shape[1] - steps:]
    sums, counts, offsets = [], [], []
    for i, tokens in enumerate(generated):
        eos_positions = [j for j, token in enumerate(tokens.tolist()) if token in ids]
        first = eos_positions[0] if eos_positions else None
        count = first + 1 if first is not None else steps
        # Slice rather than multiply by a mask: 0 * (-inf) would produce NaN.
        sums.append(transition_scores[i, :count].sum())
        counts.append(count)
        offsets.append(first)
    return torch.stack(sums), counts, offsets
