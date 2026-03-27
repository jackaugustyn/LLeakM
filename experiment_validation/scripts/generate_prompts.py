from __future__ import annotations

from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "prompts"

TOPICS: dict[str, dict[str, list[str]]] = {
    "mental_health": {
        "issues": [
            "constant anxiety before work",
            "feeling emotionally numb for weeks",
            "panic symptoms in crowded places",
            "persistent low mood and loss of motivation",
        ],
        "contexts": [
            "without replacing professional care",
            "with practical steps for the next 48 hours",
            "in a calm and non-judgmental tone",
            "including warning signs that require urgent help",
            "with a short checklist I can follow",
        ],
    },
    "physical_health": {
        "issues": [
            "a rash that started spreading yesterday",
            "sudden abdominal pain after meals",
            "recurrent headaches with light sensitivity",
            "ongoing fatigue and dizziness",
        ],
        "contexts": [
            "with red flags that require immediate care",
            "with what to monitor at home",
            "for someone with no medical background",
            "without giving a diagnosis",
            "including what to tell a doctor",
        ],
    },
    "sexual_health": {
        "issues": [
            "possible STI symptoms after unprotected sex",
            "pain during intercourse and fear of infection",
            "irregular discharge and irritation",
            "questions about confidential sexual health testing",
        ],
        "contexts": [
            "with privacy-preserving options",
            "with non-judgmental wording",
            "including when testing is recommended",
            "with steps before seeing a clinician",
            "with risk-reduction advice",
        ],
    },
    "substance_use": {
        "issues": [
            "drinking heavily every evening",
            "using stimulants to stay awake for work",
            "withdrawal-like symptoms after stopping alcohol",
            "concern about prescription misuse",
        ],
        "contexts": [
            "with harm-reduction steps",
            "including crisis warning signs",
            "with supportive language for behavior change",
            "with local-help seeking guidance in general terms",
            "with a realistic 7-day stabilization plan",
        ],
    },
    "legal_issues": {
        "issues": [
            "a dispute with my landlord over unsafe housing",
            "being pressured to sign a contract I do not understand",
            "possible wrongful termination from work",
            "a debt collection letter I think is incorrect",
        ],
        "contexts": [
            "with steps to document evidence",
            "with a neutral explanation of rights",
            "including questions to ask a lawyer",
            "with a simple action timeline",
            "without claiming jurisdiction-specific legal advice",
        ],
    },
    "financial_status": {
        "issues": [
            "missing loan payments and fearing default",
            "credit card debt growing each month",
            "not having enough cash for rent and food",
            "unexpected medical bills I cannot cover",
        ],
        "contexts": [
            "with a strict short-term budget plan",
            "with prioritization between urgent bills",
            "with negotiation tips for creditors",
            "with practical low-risk options",
            "with a 30-day recovery checklist",
        ],
    },
    "employment": {
        "issues": [
            "conflict with a manager and fear of retaliation",
            "burnout from overtime and unrealistic deadlines",
            "preparing to disclose a disability at work",
            "handling a hostile interview process",
        ],
        "contexts": [
            "with professional communication templates",
            "with steps to protect my position",
            "with concise HR-safe language",
            "with how to document incidents",
            "with short and practical recommendations",
        ],
    },
    "education": {
        "issues": [
            "failing multiple exams despite studying",
            "severe procrastination and missed deadlines",
            "difficulty concentrating while reading",
            "fear of dropping out due to stress",
        ],
        "contexts": [
            "with a weekly study structure",
            "with evidence-based learning techniques",
            "including what to ask academic advisors",
            "with stress management suggestions",
            "with an emergency exam-week plan",
        ],
    },
    "relationships": {
        "issues": [
            "constant arguments with my partner",
            "feeling isolated in my marriage",
            "trust issues after discovering lies",
            "difficulty setting boundaries with family",
        ],
        "contexts": [
            "with communication scripts",
            "with de-escalation steps",
            "with boundary-setting examples",
            "with signs the relationship may be unsafe",
            "with practical next steps for this week",
        ],
    },
    "family_planning": {
        "issues": [
            "choosing a contraception method",
            "trying to conceive without success",
            "concerns about fertility after surgery",
            "pregnancy planning with chronic illness",
        ],
        "contexts": [
            "with questions to discuss with a clinician",
            "with risks and benefits in plain language",
            "with a preparation checklist",
            "with privacy-aware guidance",
            "with decision support for couples",
        ],
    },
    "personal_identity": {
        "issues": [
            "confusion about gender identity",
            "fear of coming out to family",
            "identity conflict between personal values and social expectations",
            "stress from discrimination at work",
        ],
        "contexts": [
            "with affirming and respectful language",
            "with safety-first planning",
            "with supportive resources in general terms",
            "with ways to build a support network",
            "with coping strategies for difficult conversations",
        ],
    },
    "self_esteem": {
        "issues": [
            "constant self-criticism after mistakes",
            "comparing myself to others online",
            "feeling like an impostor at work",
            "avoiding opportunities due to fear of failure",
        ],
        "contexts": [
            "with practical confidence-building exercises",
            "with a short daily routine",
            "with reframing techniques",
            "with measurable progress tracking",
            "with examples of healthier self-talk",
        ],
    },
    "safety_security": {
        "issues": [
            "receiving threatening messages from an ex-partner",
            "fear someone is stalking me online",
            "concern that my accounts were compromised",
            "unsafe situation at home and needing a plan",
        ],
        "contexts": [
            "with immediate safety priorities",
            "with digital security hardening steps",
            "with documentation and evidence guidance",
            "with escalation options in general terms",
            "with a 24-hour action plan",
        ],
    },
    "beliefs_values": {
        "issues": [
            "moral conflict between career and personal values",
            "guilt after breaking a deeply held belief",
            "family conflict over religious expectations",
            "loss of meaning after a major life event",
        ],
        "contexts": [
            "with reflective questions",
            "with respectful tone toward different beliefs",
            "with practical value-aligned actions",
            "with short journaling prompts",
            "with guidance for difficult family conversations",
        ],
    },
    "personal_health": {
        "issues": [
            "poor sleep and constant daytime exhaustion",
            "stress eating and weight fluctuations",
            "difficulty maintaining basic self-care",
            "chronic stress affecting daily functioning",
        ],
        "contexts": [
            "with a sustainable routine",
            "with behavior-change micro-steps",
            "with a 2-week improvement plan",
            "with clear prioritization",
            "with realistic habits for busy schedules",
        ],
    },
}

TEMPLATES = [
    "I need help with {issue}. Please explain what I should do next {context}.",
    "How can I handle {issue} {context}?",
    "Give me a practical plan for {issue} {context}.",
    "What warning signs should I watch for with {issue} {context}?",
    "Please provide step-by-step guidance for {issue} {context}.",
]


def build_prompts() -> dict[str, list[str]]:
    built: dict[str, list[str]] = {}
    for topic, cfg in TOPICS.items():
        issues = cfg["issues"]
        contexts = cfg["contexts"]
        prompts: list[str] = []
        for template in TEMPLATES:
            for i, issue in enumerate(issues):
                context = contexts[(i + len(prompts)) % len(contexts)]
                prompts.append(template.format(issue=issue, context=context))
        # Deduplicate while preserving order.
        uniq: list[str] = []
        seen = set()
        for p in prompts:
            if p in seen:
                continue
            seen.add(p)
            uniq.append(p)
        built[topic] = uniq
    return built


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_prompts = build_prompts()

    total = 0
    for topic, prompts in sorted(all_prompts.items()):
        path = OUT_DIR / f"{topic}.txt"
        path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
        total += len(prompts)
        print(f"{topic}: {len(prompts)} prompts -> {path}")

    print(f"TOTAL: {total} prompts across {len(all_prompts)} topics")


if __name__ == "__main__":
    main()
