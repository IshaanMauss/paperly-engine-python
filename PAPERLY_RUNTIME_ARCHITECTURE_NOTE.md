# Paperly Runtime Architecture Note

Date: 2026-06-02

## Goal

Keep Gemini extraction controlled from one place so uploads do not accidentally
multiply API calls, retries, latency, and cost.

## Runtime Contract

```text
Frontend
  -> sends uploads/review/save requests to Node only

Node Backend
  -> owns DB, cache, pairing, QA, save, and request deduplication
  -> forwards extraction requests to Python only

Python Engine
  -> the only runtime that calls Gemini
  -> owns Gemini retries, parsing, fallback, diagrams, metadata, and extraction quality
```

## Changes Made

1. Added `services/gemini_runtime.py`.
   - Provides one process-wide Gemini limiter.
   - Default: `GEMINI_GLOBAL_MAX_CONCURRENCY=3`.
   - Default: `GEMINI_MIN_SECONDS_BETWEEN_CALLS=1.25`.
   - Can be tuned without code changes.

2. Added a PDF document-level queue in `api/extract_router.py`.
   - Default: `PAPERLY_MAX_CONCURRENT_PDF_EXTRACTIONS=2`.
   - This prevents five interns from starting five full QP render/extraction jobs
     at the same time. Extra uploads wait in-process instead of crashing memory
     or creating a Gemini burst.
   - Keep Python deployed as one Uvicorn process for Phase 2. Multiple Python
     workers each get their own in-memory limiter; use a Redis/DB queue before
     scaling Python horizontally.

3. Routed Python Gemini calls through the shared limiter:
   - `services/gemini_slicer.py`
   - `services/gemini_pdf_service.py`
   - `services/diagram_validator.py`
   - `services/pix2text_ocr.py`

3. Removed the unused direct Python client from frontend `apiHandler.js`.
   - Frontend now clearly uses Node as the extraction gateway.

4. Kept Node `pythonEngine.js` request deduplication and cache behavior intact.
   - Same PDF upload in progress waits for the existing request.
   - Normal uploads use cache.
   - Redo extraction bypasses cache intentionally.

5. QP extraction default is direct slicer mode.
   - Default: `GEMINI_QP_ENGINE=slicer`.
   - Whole-document mode remains available with `GEMINI_QP_ENGINE=pdf`.
   - Reason: whole-document QP can return malformed JSON; if it fails and then
     falls back to slicer, one upload pays for two extraction paths.

6. Partial page failures are no longer returned as successful extractions.
   - If Gemini fails on any page after retries, `gemini_slicer` raises a
     pipeline error instead of returning only the pages that succeeded.
   - This prevents half papers from reaching the review/save UI as if they were
     complete.

7. PDF extraction timeout is configurable.
   - Default: `PAPERLY_PDF_EXTRACTION_TIMEOUT_SECONDS=600`.
   - Reason: free-tier pacing can make a 20-page PDF take longer than 300s.

## Why This Should Help

- Prevents hidden Gemini bursts from multiple Python modules.
- Reduces `503`, `429`, and retry multiplication risk.
- Keeps API usage easier to debug.
- Keeps frontend/backend from becoming accidental AI callers.

## Tuning

Set this in the Python engine environment:

```text
GEMINI_GLOBAL_MAX_CONCURRENCY=3
GEMINI_MIN_SECONDS_BETWEEN_CALLS=1.25
PAPERLY_MAX_CONCURRENT_PDF_EXTRACTIONS=2
```

Recommended values:

- `3`: balanced speed and stability.
- `2`: safer if Gemini returns frequent `503`/`429`.
- `1`: maximum stability during debugging, slower.

Gemini pacing:

- `12.5`: safe for the observed free-tier `5 requests/minute` limit.
- `0`: no pacing; use only with a paid quota/key that can handle bursts.
- `6`: around 10 requests/minute if the key allows it.

QP engine:

```text
GEMINI_QP_ENGINE=slicer
```

Recommended values:

- `slicer`: default for speed/cost and direct single-pass page extraction.
- `pdf`: optional audit mode when document-global numbering is more important
  than speed and cost.

## Rollback

If this setup causes unexpected slowdown:

1. Set `GEMINI_GLOBAL_MAX_CONCURRENCY=5` temporarily and restart Python.
2. If still problematic, remove `run_gemini_async` / `run_gemini_sync` wrappers from:
   - `services/gemini_slicer.py`
   - `services/gemini_pdf_service.py`
   - `services/diagram_validator.py`
   - `services/pix2text_ocr.py`
3. Delete `services/gemini_runtime.py`.

Do not move Gemini calls back into frontend or Node. That would make cost and
retry behavior harder to control.

## 2026-06-04 Architecture Clarification

The stable runtime pieces are:

- Node remains the gateway.
- Python remains the only Gemini caller.
- PDF extraction jobs are queued in Python.
- Gemini calls are rate-limited through `services/gemini_runtime.py`.
- MS extraction should stay local-first.
- QP diagrams are cropped locally after vision regions are returned.

The unstable/experimental piece is QP canonical numbering.

Recent rollback:

- The text-only QP/MS teacher repair layer was removed.
- No teacher memory or learned examples should be used in production.
- QP numbering should not depend on a second LLM guessing repairs from damaged
  text rows.

Current QP routing idea:

```text
Saved MS IDs -> local page hints -> page-local QP Gemini prompt

Simple page-local IDs -> Flash Lite
Deep/nested page-local IDs -> Flash-first if enabled
Backend reconciliation -> deterministic only
```

Important rule:

Do not inject the full saved-MS ID list into every page-level Gemini call. That
lets the model choose IDs from other pages and causes jumps.

Current env controls:

```env
GEMINI_QP_MS_ANCHOR_FLASH_FIRST=false
GEMINI_QP_DEEP_ANCHOR_FLASH_FIRST=true
GEMINI_QP_DEEP_ANCHOR_MIN_DEPTH=3
```

If QP numbering still fails under the current hybrid mode, the remaining
production choices are:

- use Flash-only QP for deadline reliability, or
- build a local QP skeleton first and use Gemini only to fill content/diagrams,
  or
- evaluate an external layout/OCR provider for QP structure.

## 2026-06-11 Runtime Status

The local QP skeleton path is now proving useful in real runs.

Latest clean examples:

```text
0607_s18_qp_22:
  pages=8
  expected_ids=19
  final_questions=19
  local skeleton exact=19/19
  estimated_qp_cost=INR 0.5692

0607_s21_qp_23:
  pages=8
  expected_ids=22
  final_questions=22
  local skeleton exact=22/22
  estimated_qp_cost=INR 0.5941
```

MS for both pairs used native table extraction with zero Gemini calls.

Operational reading:

- Pre-skeleton Gemini `missing`/`extras` are diagnostic, not final truth.
- Final truth is after local skeleton + MS reconciler.
- A clean deployment signal is:
  - final row count equals saved MS expected count
  - no duplicate IDs
  - `raw_missing=0`
  - `extras=[]`
  - metadata synced to the final paper key

Keep the runtime defaults cost-safe:

```env
GEMINI_QP_ENGINE=slicer
GEMINI_QP_DEEP_ANCHOR_FLASH_FIRST=false
GEMINI_QP_TARGETED_RESCUE_FLASH_FIRST=false
GEMINI_ALLOW_FILES_FALLBACK=false
PAPERLY_MS_TABLE_FIRST=true
```

## 2026-06-20 IB Runtime Note

IB extraction should be treated as a separate runtime branch from IGCSE.

The tested IB QP was reasonably cheap in page-level Gemini mode:

```text
17 rendered pages
about 18 extracted rows
estimated cost about INR 1.08
```

The tested IB MS was not acceptable under the current generic MS Gemini path:

```text
31 rendered pages
41 extracted rows
estimated cost about INR 10.87
front matter was saved as fake MS question rows
```

Runtime implication:

```text
Do not send IB MS cover/copyright/instruction pages to Gemini.
Do not initialize sequence tracking before SECTION A or the first real answer.
```

Future IB route should:

- classify IB before IGCSE numbering logic runs
- skip front matter locally
- use local readable text/table structure first for IB MS
- send only real answer pages or uncertain blocks to Gemini
- avoid IGCSE-style exact leaf parity as the only QA signal

Full board-specific notes live in `PAPERLY_IB_BOARD_EXTRACTION_NOTES.md`.
