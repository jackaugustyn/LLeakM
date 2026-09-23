# Release content audit

This audit covers the 300 synthetic prompts and the seven standardized
publication runs (2100 samples), plus stored defense and baseline texts.

## Prompt corpus

- Topic files: 15
- Prompts: 300
- First-person template lines: 300
- Non-template lines: 0

Prompts are generated from issue/context templates in
`experiment_validation/scripts/generate_prompts.py`. They describe
hypothetical help-seeking situations and do not include real names,
contact details, or account identifiers.

## Sample coverage

| Run | N | Empty responses | Empty reconstructions |
|---|---:|---:|---:|
| `run_20260325_104518_qwen_1_5b` | 300 | 0 | 0 |
| `run_20260327_124817_qwen_3b_final_resume` | 300 | 0 | 0 |
| `run_20260414_073030_llama_3_2_3b` | 300 | 0 | 0 |
| `run_20260415_080034_gemma_2_2b_it` | 300 | 0 | 0 |
| `run_20260329_085714_phi_1_5` | 300 | 0 | 0 |
| `run_20260325_153010_tinyllama_1_1b` | 300 | 0 | 0 |
| `run_20260414_111644_phi_3_5_mini` | 300 | 0 | 0 |

## Pattern hits

Total flagged spans: **0**.

No email, phone, SSN, payment, IP, URL, copyright, or high-risk unsafe spans were flagged.

## Decision

No redaction is required for public release. The corpus is synthetic;
model outputs are truncated laboratory generations and reconstructions,
not records of real people. Llama-3.2 outputs sometimes include public
US helpline numbers (CDC-INFO, domestic-violence, Planned Parenthood);
these are not personal identifiers. Sensitive *topics* (health, legal,
identity) are in scope for the study and are not themselves personal data.
