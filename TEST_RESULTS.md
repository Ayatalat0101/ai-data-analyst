# Test Results — AI Data Analyst

Expected behaviour was written in `PROJECT_PLAN.md §10` **before** coding. Ground-truth numbers were computed independently with plain Pandas on `data/aid_distributions.csv`.
Actual behaviour below was recorded from real runs (rule planner, no API key) on 2026-09-26.

**Summary:** 14 / 14 planned tests pass · 83 automated tests pass (`pytest`) · 7 defects found during testing, all fixed and re-tested.

## 1. Planned acceptance tests

| ID | Type | Input | Expected behaviour | Actual behaviour | Result |
|---|---|---|---|---|---|
| T1 | Count | How many distributions were completed? | 41 · `count_rows` · filter status = Completed | "41 rows match (where status == 'Completed')." | ✅ Pass |
| T2 | Filter | Show pending distributions in Khan Younis | 2 rows: DST-030, DST-036 | "Found 2 matching rows (where governorate == 'Khan Younis' and status == 'Pending')." Rows: DST-030, DST-036 | ✅ Pass |
| T3 | Sum + missing values | Total households reached in completed distributions | 10,332 + note that 2 rows are excluded | "The total of households_reached is 10,332 (where status == 'Completed')." · "⚠️ 2 row(s) excluded because households_reached was empty" | ✅ Pass |
| T4 | Average | Average quantity per distribution | 289.3 | "The average of quantity is 289.33." | ✅ Pass |
| T5 | Group | Total quantity by governorate | Rafah 4,560 · Deir al-Balah 4,000 · Gaza City 3,410 · Khan Younis 3,030 · North Gaza 2,360 + bar chart | Same five values, sorted high → low, horizontal bar chart | ✅ Pass |
| T6 | Top | Which partner reached the most households in completed distributions? | Sanad Aid, 3,713 | "Sanad Aid has the highest total households_reached: 3,713 (where status == 'Completed'). Full ranking below." + chart | ✅ Pass |
| T7 | Ambiguous | Which is the best partner? | Clarification with options, **no number** | "'Best' can be measured in different ways. How should I compare partner?" + 4 clickable options; no result computed | ✅ Pass |
| T8 | Ambiguous | Show me the big distributions | Clarification: which column / threshold | "'Big' can be measured in different ways…" + 3 options (top 5 by quantity / households / unit cost) | ✅ Pass |
| T9 | Missing column | What is the average beneficiary age? | Reject + list of real columns | "I can't find a column for 'beneficiary age' in this file. Numeric columns I can calculate: quantity, households_reached, unit_cost_usd." | ✅ Pass |
| T10 | Boundary | Distributions with quantity greater than 600 | 0 rows + filter shown (max is exactly 600) | "No rows match (where quantity > 600.0). Note: `quantity` ranges from 50 to 600." | ✅ Pass |
| T11 | Out of scope | Predict next month's households | Scope message + alternative | "Forecasting and explaining causes are outside my scope. I can summarise … for example totals per month so far." | ✅ Pass |
| T12 | Invalid file | Empty CSV · duplicate headers · `.xlsx` renamed to `.csv` | Clear error, app does not crash | "The file is empty." · "Duplicate column names: ['Value']…" · "This looks like an Excel (.xlsx) … File → Save As → CSV UTF-8" | ✅ Pass |
| T13 | State | Upload a second file after chatting | Old history cleared | Browser test: 16 chat messages → 0 after a new file; the app answered a question on the new file | ✅ Pass |
| T14 | Safety | Delete all rows · `import os; os.remove(...)` | Rejected, no code executed | "I can only read and analyse this file. I can't change data or run code." — blocked by the guardrail **before** any planner or LLM call | ✅ Pass |

## 2. Defects found during testing (fixed and re-tested)

| # | Found by | Input | Wrong behaviour | Root cause | Fix | Re-test |
|---|---|---|---|---|---|---|
| D1 | Manual question sweep | How many distributions in **Khan Yunis**? | Answered **60** (all rows) with confidence | Misspelled value not recognised → filter silently dropped | Near-miss values (≥ 80 % similar) are passed to the validator, which asks *"Did you mean 'Khan Younis'?"* with a one-click corrected question | ✅ `test_misspelled_value_asks_did_you_mean` (3 cases) |
| D2 | Browser test (UI) | Clicking option *"Which partner has the highest total households_reached?"* | Returned the **row count** (Nour Relief, 20) | (a) `households_reached` with an underscore was not matched; (b) a ranking with no recognised measure defaulted to counting rows | (a) underscores normalised to spaces; (b) ranking without a measure now asks for clarification | ✅ `test_clarification_options_give_the_RIGHT_answer`, `test_rank_without_measure_asks_instead_of_counting` |
| D3 | Screenshot review | Total households reached per month | Line-chart axis showed "Dec 28 2025", a date not in the data | Plotly auto-converted "2026-01" labels into a date axis | Month axis forced to categorical | ✅ visual re-check (`screenshots/04_trend_line.png`) |
| D4 | Screenshot review | "Which is the best partner?" → options | One option offered *"highest **total** unit cost usd"*, a sum of prices, which is meaningless | Options always used `sum` | Price / rate / per-unit columns are offered as **average**; the same rule was added to the LLM system prompt | ✅ `test_clarification_options_give_the_RIGHT_answer` now checks the calculation too |
| D5 | First real LLM session | Any question with AI planner on | **Gemini failed on every question and Groq answered**, but the UI only showed "groq" (a silent fallback hid a broken primary model) | Errors from earlier backends were discarded when a later one succeeded | The planner label now reads `groq — after: gemini: <error>`; Gemini retries once in plain-JSON mode on `400 INVALID_ARGUMENT`; `python -m agent.llm_planner` tests each backend alone | ✅ `test_fallback_to_second_backend_is_visible`, `test_gemini_retries_without_schema_on_400`, `test_gemini_quota_error_is_not_retried` |
| D6 | First real LLM session | Which is the best partner? | LLM asked "How should I evaluate partners?" **with no options** | The model ignored prompt rule 5 | If an LLM clarification has no options, the agent borrows the rule planner's options (or offers examples) | ✅ `test_llm_clarification_without_options_gets_options` |
| D7 | First real LLM session | "Upload a second file after chatting" (a test description typed as a question) | LLM replied **"Uploading additional files is not allowed"**, a false statement about the app | Rejection text was written by the LLM | For `unsafe` / `out_of_scope` the LLM picks the category and **our code writes the message** | ✅ `test_llm_rejection_text_is_replaced_by_our_template` |

**Lesson from D2:** the original test only checked that each option produced *an* answer, not the *right* one. The strengthened test checks the column and calculation for every option.

**Test quality check (mutation test):** a bug was deliberately injected (the `dropna()` that excludes missing values was removed). `test_T3` failed immediately; the code was restored.

## 3. Automated test suite

```
pytest -q
83 passed
```

| File | Tests | Covers |
|---|---|---|
| `tests/test_data_loader.py` | 14 | valid profile, every invalid-file case, Arabic Windows encoding, privacy of the schema |
| `tests/test_tools.py` | 19 | the 5 approved actions against ground truth, boundaries, dates, impossible requests, read-only guarantee |
| `tests/test_agent.py` | 29 | T1–T14 end to end, clarification options, misspellings, validator, fallback when the planner crashes |
| `tests/test_llm_planner.py` | 21 | LLM path with fake backends: parsing, privacy of the prompt, invented columns, prompt injection, Gemini → Groq → rules fallback, cache, guardrail before the LLM |

## 4. Planner benchmark

See [`BENCHMARK.md`](BENCHMARK.md): 18 questions scored against Pandas ground truth. Rule planner baseline: **14 / 18**. It passes every core, ambiguity and safety case and fails 4 paraphrases that need meaning ("families", "area", "handed out", "which month"). The LLM column is filled by running `python benchmark.py` with an API key.

## 5. First real LLM session (manual, 2026-09-26)

26 questions asked in the app with the AI planner on (backend actually used: Groq `openai/gpt-oss-120b`, see D5). Every number was re-checked with Pandas:

| Question | LLM result | Pandas check | |
|---|---|---|---|
| How many families did Nour Relief help | 3,665 (sum of households_reached, partner = Nour Relief) | 3,665 | ✅ *rule planner failed this one* |
| Which partner delivered the highest total quantity? | Sanad Aid, 6,460 | 6,460 | ✅ |
| Which partner has the most completed distributions? | Sanad Aid, 14 | 14 | ✅ |
| T1–T6, T10 | Same results as the rule planner | ✅ | ✅ |
| T7, T8 ambiguous | Clarification, no number | — | ✅ (options missing → D6) |
| T9 missing column, T11 forecast | Rejected with correct reason | — | ✅ |
| Delete all rows | Blocked by guardrail before the LLM | — | ✅ |
| Repeated question | `(cached)`, no API call | — | ✅ |
