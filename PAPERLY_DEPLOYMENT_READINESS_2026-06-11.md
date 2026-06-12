# Paperly Deployment Readiness Note - 2026-06-11

## Current Signal

Recent tests show a clear trend: smaller/shorter IGCSE QP-MS pairs are now
syncing cleanly and cheaply when the MS is uploaded first.

The strongest recent evidence is from 0607 Paper 2 pairs:

```text
0607_s18_ms_22:
- Native MS table extraction: 19 rows
- Gemini MS calls: 0
- Estimated MS cost: INR 0
- Status: ok

0607_s18_qp_22:
- QP pages: 8
- Saved MS anchor expected IDs: 19
- Final QP rows: 19
- Exact final QP/MS match after local skeleton: 19/19
- Gemini calls: 8 Flash-Lite page calls
- Estimated QP cost: INR 0.5692
- Status: ok

0607_s21_ms_23:
- Native MS table extraction: 22 rows
- Gemini MS calls: 0
- Estimated MS cost: INR 0
- Status: ok

0607_s21_qp_23:
- QP pages: 8
- Saved MS anchor expected IDs: 22
- Final QP rows: 22
- Exact final QP/MS match after local skeleton: 22/22
- Gemini calls: 8 Flash-Lite page calls
- Estimated QP cost: INR 0.5941
- Status: ok
```

## What This Proves

- The MS-first workflow is the right production direction.
- Native MS table extraction is stable and free for readable Cambridge MS tables.
- Saved MS IDs are successfully acting as the QP numbering contract.
- Local QP skeleton can override noisy Gemini page-level numbering and produce
  clean final QP/MS parity.
- Flash is not required for these normal/smaller papers.
- QP cost can stay well below INR 1 for 8-page papers.

## Important Interpretation

Gemini page extraction may still show temporary pre-merge issues such as:

```text
SlicerExit missing=['11.a'] extras=['11']
```

This is not automatically a final failure. The current architecture applies the
local-MS skeleton after Gemini. If the final logs show:

```text
local_exact=expected/expected
raw_missing=0
extras=[]
Assembly complete with expected row count
```

then the final review payload is clean.

## Remaining Edge Cases

The system is not claiming zero human review for every Cambridge paper. The
remaining risky cases are mostly:

- long 20-page 0580 Paper 4 files with dense grouped subparts
- QP rows where several subparts are grouped into one extracted text block
- diagrams or tables that confuse visual region detection
- cases where targeted rescue repeatedly finds the same grouped row instead of
  a true missing split

For these cases, the backend now emits conservative split hints instead of
forcing automatic rigid repairs.

## Deployment Position

The current pipeline is ready for controlled deployment if the production rule
is:

```text
Upload MS first -> save/review MS -> upload matching QP -> review only real QA cards -> save paired paper
```

Use the QA dashboard as a verification layer, not as a sign that every warning
is a fatal extraction failure.

## Production Guardrails

- Keep `GEMINI_QP_DEEP_ANCHOR_FLASH_FIRST=false`.
- Keep `GEMINI_QP_TARGETED_RESCUE_FLASH_FIRST=false`.
- Keep Gemini Files fallback disabled unless explicitly needed.
- Do not repeatedly run targeted rescue after it returns the same extracted IDs
  and recovers 0 rows.
- Treat clean final exact matches as deployable even if Gemini's pre-skeleton
  page output had temporary missing/extra IDs.

## Cross-Repo Sync Check - 2026-06-11

Checked the three active runtime pieces:

- Python engine:
  - `POST /api/extract`
  - `POST /api/extract/rescue-missing`
  - cost ledger and cache policy still attached to extraction/rescue routes
- Node backend:
  - frontend-facing routes remain under `/api/v1/internal`
  - `/process-page`, `/rescue-missing`, `/save-batch`, `/counts`,
    `/qa-dashboard`, and `/qa-dashboard/repair` are wired
  - QA cron still runs daily at midnight and can be forced manually from the
    dashboard
- Frontend dashboard:
  - `apiHandler.js` points to the same Node routes
  - upload review carousel still uses targeted rescue only for missing QP IDs
  - QA dashboard now includes a plain-English situation guide
  - production-facing dashboard wording no longer uses "intern"

Verification run:

```text
frontend npm run build: passed
node cron/qaAgent require: passed
node internalRoutes require: passed
python py_compile for extraction modules: passed
```

The Vite build produced only the normal large-bundle warning. That is not a
pipeline blocker.

## QA Dashboard Simplification - 2026-06-11

Updated the QA dashboard presentation without changing the underlying QA
rules, extraction pipeline, save pipeline, cron job, or repair actions.

What changed:

- the top card is now the primary workflow surface
- blocking/high/medium/low issues are summarized as one action queue
- each action says what is wrong, why it matters, and what to do next
- the "what does this mean" guide is hidden behind a help button
- raw technical evidence is still available, but collapsed by default
- the repeated detailed "Fix Priority" section is no longer shown inside the
  expanded technical evidence area
- production-facing dashboard wording avoids the word "intern"

Verification run:

```text
frontend npm run build: passed
frontend/backend exact "intern" word scan: clean
```

This keeps the QA checks strict while making the reviewer workflow simpler:

```text
Read decision -> fix first action -> rescan -> open technical evidence only if
exact IDs or repair buttons are needed.
```
