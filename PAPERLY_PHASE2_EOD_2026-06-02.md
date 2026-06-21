# Paperly Phase 2 EOD Note - 2026-06-02

Generated at: 2026-06-02 16:07 IST

## Product Direction

Paperly is staying with a human-in-loop extraction model.

AI should do the heavy lifting:

- Extract QP/MS JSON structure.
- Preserve correct canonical numbering.
- Extract LaTeX/text and metadata.
- Attach diagram crops where possible.
- Surface only evidence-backed QA issues.

Humans should verify:

- Hallucination did not happen.
- Diagram/question boundaries are acceptable.
- Rare numbering issues are fixed.
- Final save is approved only after review.

## Current Pipeline Model

The current architecture is:

```text
Frontend upload
  -> Node backend orchestration
  -> Python extraction engine
  -> Gemini multimodal/page extraction
  -> Defensive Python reconciliation
  -> Human review UI
  -> MongoDB save
  -> QA dashboard audit
```

Working name:

```text
Hybrid Gemini Extraction + Defensive Reconciliation
```

This is not pure "single prompt magic" anymore. Gemini extracts, but Python and QA controls protect the database from malformed JSON, partial extraction, duplicate canonical IDs, bad metadata, and obvious QP/MS alignment problems.

## Major Problems We Found

1. QP numbering was fragile.
   - Gemini sometimes duplicated IDs.
   - It sometimes grouped parent questions differently from MS child subparts.
   - It sometimes returned malformed JSON in whole-document mode.

2. MS extraction was mostly strong, but row-spanned MS tables caused problems.
   - Example: one MS label such as `6(a)` spans multiple answer rows.
   - The pipeline previously risked turning those answer rows into fake `6(b)`, `6(c)`, etc.

3. Cost and speed were affected by retry storms.
   - Multiple Gemini paths could run for the same upload.
   - Free-tier/limited RPM caused 429/503 retries.
   - Node could time out while Python was still processing.

4. QA dashboard was too hard to understand.
   - It was technically correct but too dense.
   - Interns needed simple instructions: what is wrong, where to look, what to do.

## Important Fixes Added

### Extraction Runtime

- Added process-wide Gemini throttling in `services/gemini_runtime.py`.
- Routed Gemini calls through a shared limiter.
- Default QP path was moved back to slicer mode to avoid whole-document malformed JSON cost/time failures.
- Whole-document QP mode remains available by env override.

### Partial Extraction Protection

- Gemini slicer now refuses to return a partial paper when pages fail.
- Failed pages now raise an extraction error instead of silently returning half a paper.

### Node Timeout Protection

- Backend Python engine calls now use a longer native HTTP timeout.
- This avoids `UND_ERR_HEADERS_TIMEOUT` during long extraction jobs.

### Metadata Session Truth

- Session display can be full month text, e.g. `October/November`.
- Saved key logic still uses:
  - February/March -> `m`
  - May/June or June/July -> `s`
  - October/November -> `w`
- First-page PDF metadata is treated as the final truth over file name when available.

### MS Rowspan Protection

- MS continuation rows are merged under the correct parent label when Cambridge tables use row-spanned labels.
- This prevents answer rows from being renamed into fake subparts.

### Duplicate Save Protection

- Save flow now uses canonical upsert behavior for existing canonical identities.
- Re-clicking save after a failed batch should not create duplicate identity errors for already-saved rows.

### QA Dashboard

- QA report remains evidence-only.
- No speculative diagram-keyword warnings are used in the database QA.
- Added fresh scan behavior so deleted DB data does not appear from stale cache.
- Added clearer summary and repair guidance.
- Latest change: simplified the dashboard top-level view.

## QA Dashboard Simplification

New dashboard shape:

```text
Today's QA Decision
  -> Do Not Approve Yet / Human Review Needed / Looks Clean
  -> Blocking count
  -> Needs Human Check count
  -> Low-Risk Grouping count
  -> Next fixes in order
  -> Show Detailed Evidence button
```

Meaning:

- Blocking:
  Must fix before approving. Usually duplicate IDs, QP/MS mismatch, unknown IDs, or broken payload.

- Needs Human Check:
  Pipeline found a risky row. It may be correct, but a human must compare with PDF.

- Low-Risk Grouping:
  Usually not missing data. One side has a parent row and the other side has child rows.

Detailed evidence remains available, but it is no longer the first thing an intern sees.

## Important Operating Rules

1. MS-first upload is recommended, not mandatory.
   - MS gives a strong anchor for expected question structure.
   - QP can still be uploaded first, but MS-first is better for validation.

2. Do not trust counts alone.
   - QP 51 and MS 51 can still be wrong if IDs differ.
   - Exact canonical identity is what matters for RAG and linking.

3. Parent/child differences are not always failures.
   - Example: QP has `7`, MS has `7.a.i`.
   - If QP parent text contains all child parts, it can be acceptable.

4. RAG can tolerate small grouping differences only if the identity layer is clean.
   - It should not be asked to fix badly corrupted database structure at query time.

## Current Risk Areas

- QP extraction is still the harder side because visual question papers have diagrams, nested subparts, and preambles.
- Gemini API rate limits can slow extraction when many pages run concurrently.
- Diagram crop quality still needs ongoing verification for complex graph/grid pages.
- QA dashboard should stay simple at top level and detailed only on demand.
- MS tables can contain row-spanned labels. A Gemini mistake here is dangerous because QP anchoring trusts MS numbering.
  A narrow post-Gemini repair for shifted roman row-spans was tested, but it is now disabled by default because it could
  swallow real visible labels such as `1(a)(ii)` when the row was a short one-mark answer. It can only run with
  `GEMINI_MS_ROWSPAN_SHIFT_REPAIR=true`.
- Cambridge MS labels can be deeper than root + letter + roman. The question normalizer now preserves labels such as
  `1(a)(iv)(a)` as `1.a.iv.a` instead of truncating them to `1.a.iv`. This is important because MS is used as the QP anchor.
- Added a native PyMuPDF MS-table-first path for Cambridge mark schemes. When the PDF exposes a real
  `Question | Answer | Marks | Partial Marks` table, the engine extracts MS rows deterministically and skips Gemini for MS.
  This improves MS numbering/mark alignment and reduces Gemini cost/503 exposure. If needed, disable with
  `PAPERLY_MS_TABLE_FIRST=false`.
- Cost controls added after observing increased spend:
  - Gemini transient retries now default to 3 targeted page attempts. This is cheaper than failing one page and forcing
    a full 20-page redo, while true monthly-spend-cap errors are treated as fatal and are not retried.
  - Expensive double-pass Gemini Files fallback is now opt-in via `GEMINI_ALLOW_FILES_FALLBACK=true`.
    Normal production should fail visibly instead of silently paying for slicer + whole-PDF extraction on the same upload.

## Recommended Next Steps

1. Test the simplified QA dashboard on real Atlas data.
2. Upload MS first, then QP, for the next few production samples.
3. Check whether QA summary is understandable without reading detailed evidence.
4. Keep failed/odd papers as regression samples.
5. Build a small Phase 3 RAG validation layer only after Phase 2 ingestion is stable.

## June 3 Deployment Hardening

- IGCSE difficulty tagging was aligned to the official command-word/marks rule:
  - 1 mark or recall/simple command words -> `LOW`
  - 2 marks or standard work-out/calculate/sketch/core algebra -> `MEDIUM`
  - 3+ marks or show/explain/prove/derive/graph/histogram/rate-density style tasks -> `HIGH`
- Native MS table extraction remains the preferred MS path because it avoids Gemini calls when Cambridge tables are readable.
- QP diagram crops remain automatic and compressed through the crop pipeline.
- MS speculative auto-diagram warnings are suppressed because automatic MS diagram crops previously produced random or oversized page snippets.
  Manual paste stays available for rare MS diagrams.
- Frontend upload flow now guides interns toward the stable workflow:
  `Expecting MS -> MS saved, upload matching QP -> paired/next -> expecting MS again`.
- Upload issue cards now support a local "resolved" action so interns can work through real issues without losing the queue.

## Files Recently Touched

- Python engine:
  - `services/gemini_runtime.py`
  - `services/gemini_slicer.py`
  - `services/gemini_pdf_service.py`
  - `services/diagram_validator.py`
  - `services/pix2text_ocr.py`
  - `api/extract_router.py`
  - `utils/key_builder.py`

- Backend:
  - `services/pythonEngine.js`
  - `controllers/ingestionController.js`
  - `cron/qaAgent.js`
  - `routes/internalRoutes.js`

- Frontend:
  - `src/components/QADashboard.jsx`
  - `src/services/apiHandler.js`

## Bottom Line

We have moved from a fragile single-pass experiment toward a production ingestion architecture:

```text
Gemini extraction + deterministic reconciliation + human verification + evidence-only QA
```

The goal is still the same:

```text
High speed, controlled API cost, correct paper identity, correct numbering, correct human-review workflow.
```

## June 11 Addendum - Ready For Controlled Deployment

New evidence from 0607 Paper 2 uploads shows the current architecture is now
working well for smaller/normal IGCSE QP-MS pairs.

Clean examples:

- `0607_s18_ms_22`: 19 MS rows, native MS table extraction, zero Gemini cost.
- `0607_s18_qp_22`: 19 QP rows, final exact 19/19 QP/MS match, about INR 0.57.
- `0607_s21_ms_23`: 22 MS rows, native MS table extraction, zero Gemini cost.
- `0607_s21_qp_23`: 22 QP rows, final exact 22/22 QP/MS match, about INR 0.59.

The key learning is that raw Gemini page output can have temporary numbering
noise, but the local-MS skeleton can still produce a clean final payload.

Deployment stance:

- Ready for controlled GitHub/deployment testing.
- Keep MS-first workflow.
- Keep Flash disabled by default.
- Keep QA dashboard for real review cards, especially long 0580 Paper 4
  grouped-subpart cases.
- Treat repeated rescue failures as a signal for manual split/clean, not full
  redo loops.
