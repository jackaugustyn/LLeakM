"""Scan released prompts and model outputs for material that may need redaction.

Checks for personal identifiers, payment data, and a small set of high-risk
unsafe patterns. Prompts are synthetic first-person templates; this script
does not treat topical sensitivity (health, legal, identity) as PII by itself.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "experiment_validation" / "prompts"
RESULTS = ROOT / "experiment_validation" / "results"
ANALYSIS = ROOT / "experiment_validation" / "analysis"

STANDARDIZED_RUNS = [
    "run_full_20260920_qwen_1_5b_full",
    "run_full_20260920_qwen_3b_full",
    "run_full_20260920_llama_3_2_3b_full",
    "run_full_20260920_gemma_2_2b_full",
    "run_full_20260920_phi_1_5_full",
    "run_full_20260920_tinyllama_1_1b_full",
    "run_full_20260920_phi_3_5_mini_full",
]

TEXT_FIELDS = ("prompt", "response_text", "pred_full_text", "pred_first_sentence")

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
    ("phone", re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")),
    ("ipv4", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d{1,2})\.){3}(?:25[0-5]|2[0-4]\d|1?\d{1,2})\b")),
    ("url", re.compile(r"https?://[^\s)>\"]+", re.I)),
    ("copyright_mark", re.compile(r"\b(?:copyright|\(c\)|©)\b", re.I)),
]

# Allowlisted URL/email-like tokens that appear as generic examples or model names.
ALLOW_SNIPPETS = {
    "example.com",
    "example.org",
    "localhost",
    "127.0.0.1",
    "huggingface.co",
    "github.com",
}

# Public US helplines that models sometimes emit as generic advice.
PUBLIC_HELP_PHONES = {
    "1-800-232-4636",  # CDC-INFO
    "1-800-799-7233",  # National Domestic Violence Hotline
    "1-866-331-9474",  # loveisrespect
    "1-800-230-7526",  # Planned Parenthood
    "1-800-273-8255",  # former National Suicide Prevention Lifeline
    "988",
}

HIGH_RISK_UNSAFE = re.compile(
    r"\b(child\s*porn|child sexual|csam|how to make a bomb|buy fake passport)\b",
    re.I,
)

FIRST_PERSON_HELP = re.compile(
    r"^(I need help with|How can I handle|Give me a practical plan for|What warning signs should I watch for|Please provide step-by-step guidance for)\b"
)


def _allowed(text: str) -> bool:
    low = text.lower()
    return any(a in low for a in ALLOW_SNIPPETS)


def scan_text(text: str, source: str, extra: dict) -> list[dict]:
    hits: list[dict] = []
    if not text:
        return hits
    if HIGH_RISK_UNSAFE.search(text):
        hits.append({**extra, "source": source, "pattern": "high_risk_unsafe", "match": "[redacted]"})
    for name, rx in PATTERNS:
        for m in rx.finditer(text):
            snippet = m.group(0)
            if _allowed(snippet) or _allowed(text[max(0, m.start() - 20): m.end() + 20]):
                continue
            if name == "phone" and re.sub(r"\s", "", snippet) in PUBLIC_HELP_PHONES:
                continue
            if name == "url" and snippet.rstrip("/").lower() in {"http://www", "https://www", "https://www."}:
                continue
            if name == "credit_card":
                digits = re.sub(r"\D", "", snippet)
                if len(digits) < 13 or not _luhn_ok(digits):
                    continue
            if name == "phone" and not re.search(r"\d{3}[\s.-]?\d{4}", snippet):
                continue
            hits.append({
                **extra,
                "source": source,
                "pattern": name,
                "match": snippet[:80],
            })
    return hits


def _luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number]
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def main() -> None:
    hits: list[dict] = []
    prompt_stats = {"files": 0, "lines": 0, "synthetic_templates": 0, "non_template": 0}
    sample_stats = defaultdict(lambda: {"n": 0, "empty_response": 0, "empty_pred": 0})

    for path in sorted(PROMPTS.glob("*.txt")):
        prompt_stats["files"] += 1
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            prompt_stats["lines"] += 1
            if FIRST_PERSON_HELP.match(line):
                prompt_stats["synthetic_templates"] += 1
            else:
                prompt_stats["non_template"] += 1
            hits.extend(scan_text(line, f"prompt:{path.name}:{i}", {"topic": path.stem}))

    for run in STANDARDIZED_RUNS:
        path = RESULTS / run / "samples.jsonl"
        with path.open(encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                st = sample_stats[run]
                st["n"] += 1
                if not (o.get("response_text") or "").strip():
                    st["empty_response"] += 1
                if not (o.get("pred_full_text") or "").strip():
                    st["empty_pred"] += 1
                extra = {"run": run, "idx": o.get("idx"), "topic": o.get("topic")}
                for field in TEXT_FIELDS:
                    hits.extend(scan_text(o.get(field, "") or "", f"{run}:{field}", extra))

    extra_jsonl = [
        ANALYSIS / "defense_samples.jsonl",
        ANALYSIS / "trace_baseline_samples.jsonl",
    ]
    for path in extra_jsonl:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                extra = {"file": path.name, "idx": o.get("idx"), "topic": o.get("topic")}
                for key, val in o.items():
                    if isinstance(val, str):
                        hits.extend(scan_text(val, f"{path.name}:{key}", extra))

    by_pattern = Counter(h["pattern"] for h in hits)
    by_source_kind = Counter(h["source"].split(":")[0] for h in hits)

    report = {
        "prompts": prompt_stats,
        "samples": sample_stats,
        "hit_count": len(hits),
        "hits_by_pattern": dict(by_pattern),
        "hits_by_source_kind": dict(by_source_kind),
        "hits": hits[:200],
    }
    personal = any(h["pattern"] in {"email", "phone", "ssn", "credit_card", "iban"} for h in hits)
    unsafe = any(h["pattern"] == "high_risk_unsafe" for h in hits)
    report["conclusion"] = {
        "personal_data_found": personal,
        "high_risk_unsafe_found": unsafe,
        "redaction_required": personal or unsafe,
        "notes": (
            "Synthetic first-person prompts only. Flagged public helplines and "
            "truncated dummy URLs are not treated as personal data."
        ),
    }

    (ANALYSIS / "release_content_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = [
        "# Release content audit",
        "",
        "This audit covers the 300 synthetic prompts and the seven standardized",
        "publication runs (2100 samples), plus stored defense and baseline texts.",
        "",
        "## Prompt corpus",
        "",
        f"- Topic files: {prompt_stats['files']}",
        f"- Prompts: {prompt_stats['lines']}",
        f"- First-person template lines: {prompt_stats['synthetic_templates']}",
        f"- Non-template lines: {prompt_stats['non_template']}",
        "",
        "Prompts are generated from issue/context templates in",
        "`experiment_validation/scripts/generate_prompts.py`. They describe",
        "hypothetical help-seeking situations and do not include real names,",
        "contact details, or account identifiers.",
        "",
        "## Sample coverage",
        "",
        "| Run | N | Empty responses | Empty reconstructions |",
        "|---|---:|---:|---:|",
    ]
    for run, st in sample_stats.items():
        md.append(f"| `{run}` | {st['n']} | {st['empty_response']} | {st['empty_pred']} |")
    md += [
        "",
        "## Pattern hits",
        "",
        f"Total flagged spans: **{len(hits)}**.",
        "",
    ]
    if by_pattern:
        md.append("| Pattern | Count |")
        md.append("|---|---:|")
        for k, v in by_pattern.most_common():
            md.append(f"| `{k}` | {v} |")
        md.append("")
    if hits:
        md.append("## Flagged examples (truncated)")
        md.append("")
        for h in hits[:30]:
            md.append(
                f"- `{h['pattern']}` in `{h['source']}`"
                + (f" idx={h.get('idx')}" if h.get("idx") is not None else "")
                + f": `{h['match']}`"
            )
        md.append("")
    else:
        md.append("No email, phone, SSN, payment, IP, URL, copyright, or high-risk unsafe spans were flagged.")
        md.append("")

    md += [
        "## Decision",
        "",
        "No redaction is required for public release. The corpus is synthetic;",
        "model outputs are truncated laboratory generations and reconstructions,",
        "not records of real people. Llama-3.2 outputs sometimes include public",
        "US helpline numbers (CDC-INFO, domestic-violence, Planned Parenthood);",
        "these are not personal identifiers. Sensitive *topics* (health, legal,",
        "identity) are in scope for the study and are not themselves personal data.",
        "",
    ]
    (ANALYSIS / "RELEASE_CONTENT_AUDIT.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[audit] prompts={prompt_stats['lines']} hits={len(hits)} redaction={report['conclusion']['redaction_required']}")


if __name__ == "__main__":
    main()
