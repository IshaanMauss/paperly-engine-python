# Paperly Cost Checkpoint - 2026-06-03

## Why This Checkpoint Exists

The ingestion pipeline is currently working well for:

- local MS table extraction
- MS-first workflow
- MS anchoring for QP numbering
- QP diagram attachment
- QA dashboard issue carousel

The remaining production risk is cost/concurrency when several interns upload PDFs using the same Gemini API key.

## Change Added In This Checkpoint

1. Per-PDF Gemini cost ledger
   - Logs model name, Gemini call count, failed attempts, token counts, and estimated INR.
   - Shows a compact summary in `[API] PDF extraction response summary`.
   - Also logs cost summary before failure if extraction fails.

2. QP model plan is now env-controlled
   - Default plan: `gemini-2.5-flash-lite`, `gemini-2.5-flash-lite`, then `gemini-2.5-flash`.
   - Purpose: try the cheaper model first, then rescue with Flash if the page fails.
   - Fatal billing/spend-cap errors stop immediately; malformed JSON, parse errors, 503s, and other page-level failures continue to the next planned attempt.
   - MS local table extraction is unchanged.

3. Gemini Files fallback cost is counted if fallback is ever enabled.

## Rollback To Previous Model Behavior

If QP quality drops, set:

```env
GEMINI_QP_LITE_FIRST=false
GEMINI_MODEL_NAME=gemini-2.5-flash
GEMINI_MAX_RETRIES=3
```

This returns QP slicer calls to the earlier Flash-only behavior.

## Cost-Control Production Defaults To Test

```env
GEMINI_QP_LITE_FIRST=true
GEMINI_QP_PRIMARY_MODEL=gemini-2.5-flash-lite
GEMINI_QP_PRIMARY_RETRIES=2
GEMINI_QP_RESCUE_MODEL=gemini-2.5-flash
GEMINI_QP_RESCUE_RETRIES=0
GEMINI_GLOBAL_MAX_CONCURRENCY=3
GEMINI_MIN_SECONDS_BETWEEN_CALLS=1.25
PAPERLY_MAX_CONCURRENT_PDF_EXTRACTIONS=2
GEMINI_ALLOW_FILES_FALLBACK=false
```

## Test Gate Before Keeping This

Keep the change only if the same 5-10 known QP/MS pairs still pass:

- no ghost numbering
- no duplicate canonical IDs
- QP/MS alignment is clean or only hierarchy-covered
- diagrams attach to the correct question row
- cost summary is meaningfully lower than Flash-only

If any of these regress, use the rollback env above.

## MS-Guided QP Numbering Checkpoint

Added after the `igcse_0580_s20_43` failure where the saved MS had 48 IDs but
QP Gemini returned only 40 rows.

What changed:

- Node fetches saved MS canonical IDs before QP extraction.
- Node sends those IDs to Python as `extra_metadata.expected_canonical_ids`.
- Node extraction cache/dedup keys include this metadata, so an old unanchored
  QP extraction cannot mask an anchored redo.
- Python includes `extra_metadata` in the persistent cache key.
- Gemini slicer injects the saved MS ID list into each QP page prompt as a
  numbering whitelist/order guide.

Rollback for this checkpoint:

1. Remove the `extraMetadata` argument and cache-key changes from
   `paperly-backend-node/services/pythonEngine.js`.
2. Remove the `MSAnchorPreflight` block from
   `paperly-backend-node/controllers/ingestionController.js`.
3. Remove `extra_metadata` from `api/extract_router.py`.
4. Remove the `extra_metadata` parameter threading in
   `services/gemini_pdf_service.py`.
5. Remove the `SAVED MARKING-SCHEME NUMBERING ANCHOR` prompt block from
   `services/gemini_slicer.py`.

## Lite Page-Number Repair Checkpoint

Added after `igcse_0580_m21_42` showed Lite confusing Cambridge printed page
numbers with question roots.

Observed failure:

- Lite emitted IDs such as `8(a)(i)`, `9(ii)`, `11(c)(i)`, `13(b)`, and
  `15(b)` because those matched printed page numbers.
- The old poisoning guard detected the page-number leap but repaired using the
  stale tracker, causing cascades like `8(a)(i) -> 5.a.i`.

What changed:

- Python now logs whether saved-MS anchoring is active:
  `saved_ms_anchor=on/off`.
- When a QP page-number leap is detected and saved MS IDs are available, the
  slicer repairs to the next unused saved-MS ID with the same suffix and a root
  at or after the current tracker.
- Example mappings:
  - `5(c)` with tracker `3` -> `3.c`
  - `8(a)(i)` with tracker `5` -> `6.a.i`
  - `9(ii)` with tracker `6` -> `6.a.ii`
  - `11(c)(i)` with tracker `7` -> `7.c.i`
  - `13(b)` with tracker `8` -> `8.b`
  - `15(b)` with tracker `9` -> `9.b`

Rollback:

- Remove `_rebuild_model_with_anchor_id`, `_expected_anchor_parts`, and
  `_next_expected_anchor_for_suffix` from `services/gemini_slicer.py`.
- Remove the `expected_anchor_parts` config log and the
  `MSAnchorPageNumberRepair` branch inside the digit-flush rejection path.

## Local-First QP Skeleton Checkpoint

Added after post-repair alone still allowed first-page failures such as:

- printed Cambridge page `2`
- actual QP root `1`
- Lite output `2(a)` instead of `1(a)`

What changed:

- Before Gemini QP slicing, PyMuPDF now reads the PDF text locally and builds
  per-page hints:
  - printed Cambridge page number
  - likely visible/active question root
  - saved-MS IDs expected for that root/page
  - orphan subpart markers seen near the top of the page
- The per-page Gemini prompt receives only that page's local skeleton.
- Example hints from `igcse_0580_m21_42`:
  - page 2: printed page `2`, local root `1`, expected `1.a, 1.b, 1.c, 1.d`
  - page 8: printed page `8`, local root `6`, expected `6.a.i ... 6.c.ii`
  - page 13: printed page `13`, local root `8`, expected `8.a.i, 8.a.ii, 8.b`

This keeps Lite as the primary model while giving it a local numbering skeleton
before it sees the image, reducing page-number hallucinations without extra API
cost.

Rollback:

- Remove `_build_local_qp_page_hints` from `services/gemini_pdf_service.py`.
- Remove the `fallback_metadata["local_qp_page_hints"]` assignment in
  `_extract_via_gemini_slicer`.
- Remove the `LOCAL PDF TEXT SKELETON FOR THIS PAGE` prompt block from
  `services/gemini_slicer.py`.

## Local Skeleton Veto + MS Leaf Placeholder Checkpoint

Added after `igcse_0580_m21_42` still returned `49` QP rows while the saved MS
had `52` unique leaf IDs.

Observed failure:

- Lite emitted printed page numbers as question roots even when the jump looked
  plausible, e.g. printed page `9` produced `9(ii)` while the local skeleton
  knew the page belonged to question `6`.
- The old page-number guard only rejected large jumps, so `8 -> 9` slipped
  through.
- If Gemini grouped/skipped a visible child row, QP stayed below the MS leaf
  count and the dashboard only reported missing IDs.

What changed:

- `services/gemini_slicer.py` now treats a printed-page root as poison whenever
  the local QP skeleton expects a different active root, even if the root is
  only a `+1` step.
- If suffix repair fails, it can fall back to the next unused expected ID on
  that local page, e.g. bad `12(c)(iii)` can become expected `12.b.iii`.
- `services/gemini_pdf_service.py` now adds review-only QP placeholder rows for
  missing saved-MS leaf IDs only when the local skeleton confirms that expected
  ID belongs to the paper/page.
- Placeholder rows do not invent question text. They force the intern to open
  the PDF and paste/split the exact text before approval.

Local simulation:

- Input: `49` QP rows with missing `6.a.ii`, `6.a.iii`, `12.b.iii`
- Output: `52` QP rows with exactly those three review placeholders added

Rollback:

- Remove `_next_expected_anchor_for_page` and the local-skeleton root-conflict
  branch from `services/gemini_slicer.py`.
- Remove `_add_missing_qp_expected_leaf_stubs` and its call before
  `_add_missing_qp_root_stubs` in `services/gemini_pdf_service.py`.

## Targeted Missing-ID Rescue Checkpoint

Added after fake review placeholders created duplicate/extra QP rows and full
redo was too expensive for cases like only `20.c` and `20.d` missing.

What changed:

- `_add_missing_qp_expected_leaf_stubs` is now disabled. The pipeline no longer
  invents placeholder question rows for missing MS leaves.
- Missing IDs remain visible in the dashboard QA report.
- The review screen now shows `Rescue Missing IDs` for QP uploads when missing
  canonical IDs exist.
- The rescue action sends the original PDF plus missing IDs to a new targeted
  endpoint:
  - frontend: `rescueMissingQuestions`
  - Node: `POST /api/v1/internal/rescue-missing`
  - Python: `POST /api/extract/rescue-missing`
- Python renders pages locally, selects only likely pages from local QP skeleton
  and PDF text, then calls Gemini only for those pages.
- Only exact recovered missing IDs are merged back into the current review
  payload. No placeholders and no duplicate canonical IDs are added.
- Rescue logs:
  - missing IDs requested
  - selected PDF pages
  - extracted rows from targeted pages
  - exact recovered IDs
  - compact Gemini cost summary

Cost intent:

- Full redo: can call Gemini for every QP page again.
- Targeted rescue: usually calls Gemini for 1-3 likely pages.

Rollback:

- Remove `rescue_missing_qp_questions` and `/api/extract/rescue-missing`.
- Remove `rescueMissingQuestionsWithPython` and `/api/v1/internal/rescue-missing`.
- Remove the Dashboard `Rescue Missing IDs` button and merge handler.

## Dashboard Repair Assistant Checkpoint

Added after QP rows such as `24` (shared stem) and `24.a` (child subpart) were
stored separately, producing a real row-count mismatch even though QP/MS IDs
looked mostly aligned.

What changed:

- The upload issue carousel now has a contextual `Suggested fix` area.
- The dashboard detects parent-stem split candidates when:
  - the current upload has more rows than the paired counterpart, and
  - a parent row like `24` exists with child rows like `24.a`, `24.b`.
- New zero-cost repair primitive:
  - `Merge Stem Into Children`
  - Copies parent `question_latex` into every child row.
  - Copies parent diagrams into child rows without duplicates.
  - Removes the standalone parent row.
  - Marks affected child rows `needs_review=true` with a warning.
- Existing repair primitives are surfaced in the same carousel:
  - targeted rescue for missing IDs
  - delete row for duplicates/extras/parent stems
  - open row for missing-diagram manual paste

Cost intent:

- These repairs are local React state changes only.
- They do not call Gemini and do not slow extraction.

Rollback:

- Remove `parent_stem_split` issue creation from `buildUploadIssueCards`.
- Remove `handleMergeParentStemIntoChildren`.
- Remove carousel `actions` rendering.

## 2026-06-04 QP Numbering Cost/Accuracy Update

This checkpoint supersedes the earlier assumption that saved-MS anchoring alone
would make Lite QP extraction production-stable.

### What Was Rolled Back

The text-only "teacher/student" QP repair layer was removed.

Reason:

- It did not see the original page image.
- It could only reason from already-damaged extracted text rows.
- When the base QP order was poisoned, it could not reliably fix the structure.
- It added complexity and possible bad memory without guaranteeing correctness.

Removed items:

- `services/ms_anchor_teacher.py`
- `repair_qp_numbering_with_teacher(...)` call from `services/gemini_pdf_service.py`
- teacher memory file `data/anchor_repair_memory.jsonl`
- teacher env knobs from `.env.example`

### What Replaced The Full-List Anchor Prompt

The earlier QP prompt injected the full saved-MS ID list into every page.

Observed problem:

- Lite sometimes selected IDs from the wrong part of the paper.
- A page could emit IDs far ahead of the real visible page.
- This caused jumps, duplicates, and missing IDs.

Current replacement:

- Gemini receives only the saved-MS IDs that local PDF text believes belong to
  the current rendered page.
- Full saved-MS IDs remain available to backend reconciliation after extraction.
- The page prompt is now page-local, not whole-paper-global.

### Current Cost Strategy

```text
MS:
Use local table extraction. No Gemini in the common case.

QP simple pages:
Use Gemini Flash Lite first.

QP deep/nested pages:
Optionally use Gemini Flash first when page-local MS IDs contain deep IDs.

QP missing rows:
Use targeted missing-ID rescue instead of full redo.
```

Environment controls:

```env
GEMINI_QP_LITE_FIRST=true
GEMINI_QP_PRIMARY_MODEL=gemini-2.5-flash-lite
GEMINI_QP_PRIMARY_RETRIES=2
GEMINI_QP_RESCUE_MODEL=gemini-2.5-flash
GEMINI_QP_RESCUE_RETRIES=0

GEMINI_QP_MS_ANCHOR_FLASH_FIRST=false
GEMINI_QP_DEEP_ANCHOR_FLASH_FIRST=false
GEMINI_QP_DEEP_ANCHOR_MIN_DEPTH=3

GEMINI_QP_TARGETED_RESCUE_FLASH_FIRST=false
PAPERLY_RESCUE_MAX_PAGES=4
```

### Current Reality

This lowers cost, but it has not yet solved QP numbering for every paper.

Known latest failure:

```text
QP: 34 items / 33 unique IDs
MS: 37 IDs
Missing IDs: 15.c, 16, 4.a.i, 4.a.ii, 4.b
Duplicate ID: 7
```

Therefore the current Lite-first hybrid mode is not final production quality
until QP numbering is consistently clean across the known test set.

### Final Cost/Accuracy Options

1. Flash-only QP

- Most likely to improve numbering immediately.
- Highest cost.
- Easiest production fallback if deadline matters more than cost.

2. Lite simple pages + Flash deep pages

- Current experimental compromise.
- Cost lower than Flash-only.
- Needs better local page-to-MS-ID mapping before it can be trusted.

3. Local QP skeleton first, Gemini content second

- Best long-term cost architecture.
- Build canonical rows locally from PDF text/MS IDs.
- Use Gemini mainly for filling text/diagrams, not deciding numbering.
- More engineering work, but most scalable.

Current implementation:

```env
PAPERLY_QP_LOCAL_SKELETON_FIRST=true
PAPERLY_QP_LOCAL_SKELETON_MIN_COVERAGE=0.70
GEMINI_QP_RESCUE_RETRIES=0
GEMINI_QP_DEEP_ANCHOR_FLASH_FIRST=false
GEMINI_QP_TARGETED_RESCUE_FLASH_FIRST=false
```

The local skeleton is applied only when it finds enough saved-MS IDs in native
PDF text. Otherwise the pipeline falls back to the Gemini extraction result.

Cost impact:

- Native skeleton costs nothing.
- Gemini calls still run for QP pages so diagrams/math-friendly text can be
  reused, but normal production now defaults to Flash-off.
- Targeted rescue now tries native PDF text first, then Lite-first page rescue.
  Flash is an explicit emergency setting, not a hidden fallback.
- The next possible cost reduction is to skip Gemini text extraction on papers
  where the local skeleton is perfect and call vision only for diagram pages.

4. External layout/OCR provider

- Test Mathpix, Azure Document Intelligence, Google Document AI, AWS Textract,
  PaddleOCR/layout models, or other document parsers on the same PDF set.
- Keep only if it gives stable row labels and reading order at lower cost than
  Flash.

### Decision Gate

Before calling any mode "production default", run the same known QP/MS pairs and
require:

- zero duplicate canonical IDs
- no missing exact QP/MS IDs except accepted hierarchy-covered differences
- no page-number jumps
- diagrams attached to the correct rows
- cost logged per PDF

If these fail, do not keep adding small guards blindly. Switch strategy.

## 2026-06-11 Deployment Readiness Update

Recent 0607 Paper 2 tests show the cost/accuracy direction is now working for
small and normal-length papers.

Observed results:

- `0607_s18_ms_22`: 19 MS rows, native MS table path, 0 Gemini calls, INR 0.
- `0607_s18_qp_22`: 8 QP pages, 19 expected MS IDs, final 19/19 exact QP/MS
  match, INR 0.5692.
- `0607_s21_ms_23`: 22 MS rows, native MS table path, 0 Gemini calls, INR 0.
- `0607_s21_qp_23`: 8 QP pages, 22 expected MS IDs, final 22/22 exact QP/MS
  match, INR 0.5941.

Interpretation:

- Gemini page-level output may still show temporary `missing`/`extras` before
  the local-MS skeleton is applied.
- The final payload is what matters. If `QPLocalSkeleton` reports
  `local_exact=expected/expected`, `LeafStubGate` reports `raw_missing=0`, and
  final row count equals saved MS ID count, the extraction is clean.
- This confirms that MS-first + local QP skeleton + Flash-Lite page extraction
  is deployable for controlled production use.

Current deployment recommendation:

- Keep MS extraction local/free.
- Keep QP on Flash-Lite by default.
- Keep Flash and Gemini Files fallback opt-in only.
- Use targeted rescue only once per missing-ID set; if it repeatedly recovers
  0, switch to manual split/clean review rather than paying again.

## 2026-06-20 IB Cost Note

IB board testing showed a different cost pattern from IGCSE.

Tested IB QP:

```text
Mathematics_analysis_and_approaches_paper_1_TZ2_HL.pdf
17 pages
about 18 rows
estimated cost about INR 1.08
```

This is acceptable, but the row structure needs IB-specific continuation and
grouping rules.

Tested IB MS:

```text
Mathematics_analysis_and_approaches_paper_1_TZ2_HL_markscheme.pdf
31 pages
41 rows
estimated cost about INR 10.87
```

This is not acceptable as a default because pages of examiner instructions were
sent to Gemini and saved as fake MS rows.

Cost conclusion:

```text
IB QP can probably stay low-cost with page-level Flash-Lite.
IB MS needs local front-matter skipping and local parsing before Gemini.
```

The next IB cost target should be:

- skip cover/copyright/instruction pages before model calls
- start extraction at `SECTION A` or the first real answer row
- parse local MS text/tables where possible
- reserve Gemini for uncertain answer blocks only
