# Tool-Choice Evaluation Results

- Date: 2026-10-21
- Mode: Live OpenAI Responses API (`gpt-5.6-terra`)
- Question bank: 16 bilingual cases (8 Traditional Chinese, 8 English)

## First live run

- Score: **14/16 (87.5%)**
- Miss T01: an exam-score guarantee request returned no tool instead of `transfer_to_human`.
- Miss T02: an after-hours home-visit request selected `get_center_info` instead of `transfer_to_human`.

## Correction

The routing prompt was made explicit: unsupported Beacon-related promises and exceptional services—such as guaranteed scores, home visits, or after-hours tutors—must use `transfer_to_human`, and must not be answered from general knowledge or routed to `get_center_info`.

## Final live run

- Score: **16/16 (100.0%)**
- In-data lookup: 4/4
- Booking calculation: 4/4
- Missing-data or human-transfer requests: 4/4
- Out-of-scope requests (no tool): 4/4

The final run exited successfully. The automated project suite also passed: **16 tests and 8 subtests**.
