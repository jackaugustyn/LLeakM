# Validation Report

Generated: 2026-09-23T14:11:58

## Global Summary

- samples: 300
- complete responses (EOS before cap): 77 (25.67%)
- phi_mean: 0.5170
- ed_norm_mean: 0.8428
- rouge1_precision_mean: 0.2329
- rougeL_precision_mean: 0.1151
- ASR (phi > 0.5): 59.67%

## Threshold Metrics [%]

- ED_eq_0: 0.00
- ED_le_0_1: 0.00
- R1_eq_1: 0.00
- R1_ge_0_9: 0.00
- RL_eq_1: 0.00
- RL_ge_0_9: 0.00
- phi_eq_1: 0.00
- phi_gt_0_9: 0.00
- ASR_phi_gt_0_5: 59.67

## EOS-Complete Responses Only

- samples: 77
- phi_mean: 0.5090
- ed_norm_mean: 0.8388
- rouge1_precision_mean: 0.2344
- rougeL_precision_mean: 0.1208
- ASR (phi > 0.5): 53.25%

## Per-Topic

| topic | count | phi_mean | ed_norm_mean | R1_prec_mean | RL_prec_mean | ASR(phi>0.5) |
|---|---:|---:|---:|---:|---:|---:|
| beliefs_values | 20 | 0.5204 | 0.8423 | 0.2492 | 0.1209 | 65.00% |
| education | 20 | 0.5245 | 0.8569 | 0.2248 | 0.1066 | 65.00% |
| employment | 20 | 0.5442 | 0.8451 | 0.2431 | 0.1173 | 70.00% |
| family_planning | 20 | 0.4893 | 0.8538 | 0.2159 | 0.1124 | 40.00% |
| financial_status | 20 | 0.4778 | 0.8408 | 0.2136 | 0.1069 | 25.00% |
| legal_issues | 20 | 0.5446 | 0.8278 | 0.2428 | 0.1240 | 75.00% |
| mental_health | 20 | 0.5315 | 0.8379 | 0.2338 | 0.1152 | 70.00% |
| personal_health | 20 | 0.4880 | 0.8482 | 0.2229 | 0.1072 | 55.00% |
| personal_identity | 20 | 0.5571 | 0.8201 | 0.2615 | 0.1297 | 90.00% |
| physical_health | 20 | 0.5036 | 0.8458 | 0.2247 | 0.1162 | 55.00% |
| relationships | 20 | 0.5299 | 0.8495 | 0.2390 | 0.1117 | 65.00% |
| safety_security | 20 | 0.5116 | 0.8433 | 0.2221 | 0.1100 | 60.00% |
| self_esteem | 20 | 0.5018 | 0.8446 | 0.2366 | 0.1124 | 45.00% |
| sexual_health | 20 | 0.5194 | 0.8351 | 0.2362 | 0.1191 | 50.00% |
| substance_use | 20 | 0.5115 | 0.8505 | 0.2267 | 0.1162 | 65.00% |
