# Paperly Phase 2 Journey: Ingestion, QA, and Production Hardening

## Purpose

Paperly Phase 2 is the ingestion layer for exam papers.

The product goal is simple:

```text
Turn QP/MS PDFs into reliable structured data for future RAG and student-facing learning.
```

For every uploaded paper, Paperly needs:

- Correct document type: Question Paper or Marking Scheme.
- Correct paper identity: board, subject, session, year, paper number, unified paper key.
- Correct canonical question numbering.
- Correct QP/MS pairing.
- Correct diagram attachment to the right question.
- Correct extracted question/answer/marking text.
- A human verification flow that catches real issues without creating noise.

The human-in-the-loop role is intentional. The AI should do the heavy lifting, while humans verify hallucination, boundaries, rare numbering issues, and final approval.

## Where We Started

The earlier production approach used OCR and LLM stages, including Groq-style extraction and separate image/vision handling.

That approach worked, but it had clear limits:

- OCR struggled with complex IGCSE and IB layouts.
- QP/MS parsing was slow.
- Diagrams and text needed separate logic.
- Multi-stage extraction increased latency.
- Cost was lower than some later experiments, but speed and reliability were not enough for bulk board ingestion.

The next idea was to reduce stages and move toward a multimodal single-pass system.

## Multimodal Single-Pass Experiment

The goal of the Gemini slicer path was:

```text
Render page -> send page image to Gemini -> get JSON + diagram boxes -> crop locally
```

This was attractive because one model call could see both text and layout.

Benefits:

- Better than OCR on visual pages.
- Could understand diagrams and question boundaries.
- QP image extraction improved.
- Reduced dependence on separate OCR + LLM passes.

But real paper testing exposed serious issues.

## Problems Found During Testing

### 1. QP Numbering Drift

Gemini sometimes confused:

- Page numbers with question numbers.
- Repeated labels inside diagrams or examples.
- Mark brackets like `[3]` with question IDs.
- Backward roots such as `4` appearing again after the tracker had moved to `8`, `10`, or `13`.

This caused ghost numbering, duplicate canonical IDs, and incorrect QP/MS alignment.

### 2. Over-Repairing

Some defensive guards became too aggressive.

They repaired real labels into wrong labels, especially when:

- QP grouped subparts but MS split them.
- Gemini emitted a partial label.
- The code tried to infer too much from sequence alone.

The lesson was important:

```text
Repair logic must be defensive, not creative.
```

### 3. MS Extraction Did Not Need Gemini

At first, Marking Schemes were also sent through Gemini because Gemini could return JSON.

But real Cambridge MS PDFs showed a better truth:

```text
Most Cambridge MS files are structured tables:
Question | Answer | Marks | Partial Marks
```

Using Gemini for these tables was unnecessary cost and risk.

### 4. MS Row-Span and Deep Label Bugs

Cambridge MS labels can be deeper than simple formats.

Examples:

- `1(a)(iv)(a)` -> `1.a.iv.a`
- `1(a)(iv)(b)` -> `1.a.iv.b`

Earlier normalization truncated deep labels, causing duplicate IDs like `1.a.iv`.

Some row-span repair also became too aggressive and swallowed visible labels such as `1(a)(ii)`.

### 5. Image Extraction Problems

QP diagrams improved with Gemini plus local cropping, but MS diagrams were different.

MS diagrams often live inside the Answer cell of a table. A text parser can read the row label and marks, but not the drawing itself.

Earlier Gemini-based MS image extraction risked:

- Random page crops.
- Oversized snippets.
- Whole-page images.
- Higher cost.

### 6. QA Dashboard Was Too Hard To Understand

The QA dashboard was evidence-rich, but too complex for interns.

It showed real problems, but the wording did not always explain:

- Is this fatal?
- Is this only a parent/child grouping difference?
- Should I redo extraction or manually fix one row?
- Where do I click?

The dashboard needed to become a working tool, not just a report.

## Current Production Architecture

The current Phase 2 ingestion architecture is hybrid:

```text
Question Paper:
Gemini page extraction -> defensive reconciliation -> local diagram crop -> human review

Marking Scheme:
Local PDF table extraction -> local answer-cell diagram crop -> human review
```

This is no longer a pure multimodal single-pass system.

It is now:

```text
Hybrid Gemini Extraction + Local Deterministic MS Parsing + Defensive Reconciliation
```

That is better for production.

## Question Paper Flow

```text
Upload QP
-> render pages
-> Gemini extracts page JSON and diagram regions
-> sequence guards repair only clear structural issues
-> PyMuPDF fallback finds missed visual diagrams
-> local crop and JPEG compression
-> dashboard review
-> save after human approval
```

QP still uses Gemini because QP layout is visually complex.

QP needs help with:

- Diagrams.
- Graphs.
- Number lines.
- Split question stems.
- Nested subparts.
- Page continuation.

Cost is controlled by keeping cropping local and avoiding unnecessary whole-PDF fallback.

## Marking Scheme Flow

```text
Upload MS
-> PyMuPDF reads Cambridge tables locally
-> rows become structured MS entries
-> canonical IDs are normalized locally
-> answer-cell visual content is detected locally
-> only visual answer cells are cropped and compressed
-> dashboard review
-> save after human approval
```

This means MS extraction now often uses:

```text
0 Gemini calls
```

This is why MS became extremely fast.

The log confirms this path:

```text
[NativeMSTable] Extracted N MS row(s) from X Cambridge table page(s). Gemini MS call skipped.
```

## Why Local MS Extraction Works

Cambridge MS PDFs usually expose table structure in the PDF itself.

PyMuPDF can read:

- Table rows.
- Table columns.
- Question labels.
- Answer text.
- Marks.
- Partial marks.
- Cell coordinates.

Because cell coordinates are available, Paperly can also crop only the Answer cell when a diagram exists.

That solves the Venn diagram problem:

```text
Text parser sees blank/numbers.
Local visual detector sees actual drawing inside Answer cell.
Paperly crops that cell and attaches it as diagram_urls.
```

No Gemini is needed for this.

## Why This Was Not Implemented Earlier

Earlier, we were optimizing for one system that could handle everything.

That pushed the architecture toward:

```text
One multimodal model path for QP and MS
```

But after many real PDFs, the pattern became clear:

```text
QP is a vision-layout problem.
MS is mostly a table-structure problem.
```

Treating both the same was the mistake.

The current design respects the document type.

## Numbering Improvements

The pipeline now protects the three-level identity model:

```text
document_type
unified_paper_key
canonical_question_id
```

Important improvements:

- Deep canonical labels are preserved, e.g. `1.a.iv.a`.
- Backend save validation no longer truncates deep labels.
- Duplicate canonical IDs are blocked before save.
- QP/MS pairing checks exact IDs.
- Hierarchy-covered differences are separated from true missing IDs.

This matters for Phase 3 RAG because RAG should not be asked to repair corrupted database structure.

## Metadata Improvements

Paper identity is now normalized more carefully.

Session mapping:

```text
February/March -> m
May/June       -> s
October/November -> w
```

Display can show full session labels while storage keeps stable key codes.

Native MS extraction now derives missing metadata from the paper key:

- Subject code.
- Paper number.
- Session.
- Year.
- Tier for IGCSE 0580 where possible.

For IGCSE 0580:

```text
Paper 1/3 -> Core
Paper 2/4 -> Extended
```

## Diagram Improvements

### QP Diagrams

QP diagrams remain automatic.

Flow:

```text
Gemini region OR PyMuPDF fallback
-> local crop
-> JPEG compression
-> attach to diagram_urls
```

This keeps image attachment tied to the same question number.

### MS Diagrams

MS diagrams now use local answer-cell detection.

Flow:

```text
Find answer cell
-> detect real vector drawing inside cell
-> ignore table borders
-> crop exact cell
-> compress JPEG
-> attach to diagram_urls
```

This avoids random full-page MS crops and keeps cost at zero.

## Difficulty Tagging

IGCSE difficulty tagging now follows the agreed command-word and marks rule:

```text
LOW:
1 mark or simple recall/write/state/give/plot/list/label style tasks

MEDIUM:
2 marks or work out/calculate/describe/sketch/determine/construct/complete/core algebra

HIGH:
3+ marks or show/explain/prove/derive/analyse/graph/histogram/rate-density style tasks
```

For native MS rows, marks are the primary signal:

```text
1 mark -> LOW
2 marks -> MEDIUM
3+ marks -> HIGH
```

## QA Dashboard Improvements

The dashboard moved from raw warning dump toward guided verification.

Current direction:

- MS-first workflow banner.
- After MS save, dashboard asks for the matching QP.
- After QP save, it resets to the next paper.
- Problem carousel explains what is wrong.
- Each problem can open the row.
- Interns can mark a problem resolved locally after checking/fixing.
- Evidence-only QA avoids speculative false warnings.

Recommended human workflow:

```text
Upload MS first
-> review/save MS
-> upload matching QP
-> review images/numbering
-> save QP
-> QA dashboard checks paired data
```

## Cost and Speed Improvements

The biggest cost improvement is:

```text
MS extraction no longer needs Gemini for readable Cambridge tables.
```

That means:

- Faster MS extraction.
- Lower API cost.
- Fewer Gemini 503 failures.
- More stable MS numbering.
- Better QP anchoring when MS is uploaded first.

QP still uses Gemini because it genuinely needs visual understanding.

Cost control rules:

- Do not send MS to Gemini when local table extraction succeeds.
- Do not use whole-PDF fallback unless explicitly enabled.
- Crop images locally.
- Compress diagrams before returning them.
- Keep MS image extraction deterministic and local.

## Current Known Limitations

The system is much stronger, but not magic.

Known limitations:

- Some complex math text in MS partial marks can be lossy if the PDF text layer itself is poor.
- QP extraction can still need human review for rare boundary or numbering issues.
- IB PDFs may need separate board-specific hardening.
- Some scanned/image-only MS PDFs may not expose tables. Those may still need fallback handling.

The important change is that these are now visible and bounded problems.

## Current Production Truth

The architecture we should trust now is:

```text
MS:
Local table parser + local answer-cell crop

QP:
Gemini visual extraction + local crop + defensive repair

QA:
Evidence-only checks + human approval
```

This is the best balance found so far:

- Low cost.
- Good speed.
- Correct numbering.
- Better diagram attachment.
- Clearer human verification.
- Stronger Phase 3 RAG foundation.

## Deployment Notes

Required Python dependency:

```text
PyMuPDF / fitz
```

Optional environment controls:

```text
PAPERLY_MS_TABLE_FIRST=false
```

Disables native MS table extraction if emergency rollback is needed.

```text
GEMINI_ALLOW_FILES_FALLBACK=true
```

Allows expensive whole-PDF Gemini fallback. Keep disabled by default for cost control.

## Bottom Line

Phase 2 started as an attempt to make one multimodal extraction system do everything.

Real PDFs taught us a better architecture:

```text
Use Gemini only where vision is actually needed.
Use deterministic local parsing where the PDF already has structure.
Use humans for verification, not for doing the extraction work.
```

That is the current Paperly Phase 2 direction.

## 2026-06-04 Reality Update: QP Numbering Is Still The Blocker

After more production-style testing, the wording above needs one important
correction: the MS side is now much stronger, but QP canonical numbering is not
yet production-stable across all variants.

### What Is Working

MS extraction is now the strongest part of Phase 2:

- Most Cambridge MS PDFs can be parsed locally from table geometry.
- This avoids Gemini cost for MS in the common case.
- MS canonical IDs are usually clean and become the best available numbering
  authority.
- MS extraction is fast because it uses the PDF text/table layer instead of
  page-by-page vision calls.
- MS answer-cell image extraction can be local when the answer cell has vector
  or visual content.

QP diagram extraction is also much better than earlier attempts:

- Gemini gives visual regions.
- PyMuPDF crops locally.
- Oversized/text-only crop validation prevents many random full-page crops.
- Manual paste remains available for missing or wrong diagrams.

### What Is Not Yet Solved

QP numbering still fails on some papers.

Observed failure patterns:

- Lite reads Cambridge printed page numbers as question roots.
- One rendered page can contain many unrelated rows if the model sees multiple
  labels or if page context is confusing.
- Tracker state can become stale after a grouped parent stem.
- Sequence repair can create duplicates instead of restoring the real ID.
- MS IDs can be correct, but QP extraction may still produce missing, extra, or
  duplicate canonical IDs.
- Deep nested IDs such as `4.a.ii`, `5.c.i`, `7.b.ii.a` are especially risky.

The latest example showed:

```text
QP: 34 items / 33 unique IDs
MS: 37 IDs
Missing: 15.c, 16, 4.a.i, 4.a.ii, 4.b
Duplicate: 7
```

This means the system still cannot be called fully production-ready for QP
canonical numbering.

### Approaches Tried For Low Cost + High Accuracy

1. OCR + LLM multi-stage pipeline

- Lower cost than full vision in some cases.
- Too slow and brittle for complex IGCSE/IB layouts.
- Diagram handling was weak.

2. Gemini multimodal single-pass slicer

- Stronger for QP text + diagrams together.
- QP images improved.
- Cost increased because every QP page calls Gemini.
- Numbering drift appeared on printed page numbers and grouped subparts.

3. Defensive sequence guards

- Added digit flush, poisoning guard, backward-root guard, sequence guard, and
  page-number hallucination checks.
- Helped many obvious cases.
- Became risky when too much inference was layered on top of noisy extraction.

4. Local MS table-first extraction

- Successful direction.
- Fast.
- Very low/no Gemini cost for MS.
- Produces the best canonical numbering authority available.

5. MS-first workflow

- Correct product direction.
- Upload/save MS first, then QP can use saved MS IDs as the numbering contract.
- Reduces ambiguity, but does not automatically solve QP row splitting.

6. Full MS ID prompt injected into every QP page

- Failed on some papers.
- Giving every page the full paper ID list let the model pick IDs from other
  pages, causing wrong jumps.

7. Local QP skeleton hints

- Useful but imperfect.
- PyMuPDF text can identify likely roots/pages cheaply.
- It should guide pages, not delete rows or overrule visible QP labels.

8. Targeted missing-ID rescue

- Good cost control idea.
- Calls Gemini only on likely pages for missing IDs.
- Can recover some missing rows.
- Not enough when the base QP extraction order is already badly poisoned.

9. Text-only teacher/student repair

- Rolled back.
- It added complexity and did not reliably fix numbering.
- It could not safely understand visual row boundaries from text-only rows.
- Teacher memory was removed to avoid carrying bad repair examples forward.

10. Page-local MS anchor + deep Flash routing

- Current experimental direction.
- Gemini receives only the saved-MS IDs expected for that rendered page, not the
  full paper list.
- Simple page-local IDs use Lite.
- Deep/nested page-local IDs can use Flash-first.
- This reduces cost compared with Flash on every page, but latest tests still
  show QP numbering failures.

### Current Honest Architecture State

```text
MS:
Local-first table extraction works well and should remain.

QP:
Gemini visual extraction works for diagrams and many questions, but canonical
numbering is not yet robust enough for production without review.

QA Dashboard:
Useful for proving what is wrong, but the repair workload can still be too high
when QP numbering fails early.
```

### Final Options Left To Discuss

1. Flash-only QP for production

- Highest known QP accuracy among current Gemini options.
- Highest cost.
- May be acceptable only for final production uploads, not every test run.

2. Hybrid QP: Lite for simple pages, Flash for deep/nested/MS-risk pages

- Best cost/quality compromise if routing is fixed.
- Needs reliable page-local MS ID mapping.
- Still under test.

3. Local-first QP skeleton, then vision only for content/diagrams

- Most scalable long-term architecture.
- Harder engineering problem.
- Would use local PDF text to build the canonical row skeleton first, then use
  Gemini to fill text/diagram content into known slots.

Implementation checkpoint:

- Added `PAPERLY_QP_LOCAL_SKELETON_FIRST=true`.
- Python now builds QP rows from native PDF text using saved MS IDs as the
  canonical contract.
- If native skeleton coverage is strong enough, it becomes the QP row/numbering
  source.
- Gemini rows are still used for math-friendly text and diagrams when the exact
  canonical ID matches, or for nearest-row diagram transfer with review warning.
- If native skeleton coverage is weak, the pipeline keeps the existing Gemini
  result instead of forcing bad local rows.
- Verified on `0580_s25_qp_41` + `0580_s25_ms_41`: native skeleton found all
  `37/37` MS IDs with no missing IDs and no duplicates.

4. External document-layout/OCR engine for QP structure

- Examples to evaluate: Mathpix, Azure Document Intelligence, Google Document
  AI, AWS Textract, Unstructured, PaddleOCR/layout models.
- Must be tested on real Cambridge QP/MS PDFs.
- Could reduce Gemini dependence if it extracts stable reading order and labels.

5. Human-assisted MS-authoritative mapper

- Save MS IDs first.
- QP extraction produces content blocks.
- Intern sees unmatched blocks and assigns them to MS IDs with smart UI.
- Lower AI risk, but more human work.

The next decision should be made from these options, not by adding more small
guards to the current QP slicer until it becomes impossible to reason about.

## June 11 - Deployment Readiness Trend

After the local-first QP skeleton work, the strongest positive pattern is now
visible on shorter IGCSE pairs.

Recent 0607 Paper 2 tests:

- `0607_s18_ms_22`: native MS table extraction produced 19 rows, skipped Gemini
  entirely, and cost INR 0.
- `0607_s18_qp_22`: 8 QP pages, saved MS anchor expected 19 IDs, local skeleton
  produced a final exact 19/19 QP/MS match, QP cost about INR 0.57.
- `0607_s21_ms_23`: native MS table extraction produced 22 rows, skipped Gemini
  entirely, and cost INR 0.
- `0607_s21_qp_23`: 8 QP pages, saved MS anchor expected 22 IDs, local skeleton
  produced a final exact 22/22 QP/MS match, QP cost about INR 0.59.

This is important because Gemini's raw page output still sometimes has
temporary pre-merge noise, such as a parent row `11` instead of `11.a`. The
local-MS skeleton now absorbs that noise when the saved MS numbering is clear.

Practical conclusion:

```text
MS first -> local MS IDs -> local QP skeleton -> Gemini Lite text/diagram support
```

is the right production direction.

Remaining complex cases are mostly long 0580 Paper 4 files with grouped
subparts, dense tables, diagrams, and row splitting problems. Those should be
handled by QA repair hints and targeted review, not by making the extractor
rigid for every paper.

## June 17 - Retry/Availability Checkpoint

A new `0580_w23_qp_42` run failed with `502 Bad Gateway`, but the root cause was
not canonical numbering. The logs showed repeated Gemini provider `503
UNAVAILABLE` responses on multiple pages. The slicer correctly refused to
return a partial QP because several pages exhausted their retry plan.

Important finding:

- The global slicer retry setting still defaults to `GEMINI_MAX_RETRIES=3`.
- The newer cost-safe Lite-first QP path also has a separate
  `GEMINI_QP_PRIMARY_RETRIES` control.
- Its current default is `2`, with `GEMINI_QP_RESCUE_RETRIES=0`.
- Therefore a QP page can show a model plan like:

```text
q_model_plan=['gemini-2.5-flash-lite', 'gemini-2.5-flash-lite']
```

This means the old practical behaviour of three page attempts is no longer
guaranteed for QP Lite-first extraction unless the QP-specific retry controls
are set accordingly.

Operational conclusion:

- This was partly a provider availability event and partly a retry-plan tuning
  issue.
- It was not caused by MS anchor numbering, local QP skeleton, or the row repair
  button.
- The pipeline did the safe thing by rejecting partial output instead of saving
  a damaged paper.

Next scalable fix before code changes:

```env
GEMINI_QP_PRIMARY_RETRIES=3
GEMINI_QP_RESCUE_RETRIES=0
GEMINI_MAX_RETRIES=3
GEMINI_SLICER_MAX_CONCURRENCY=2
PAPERLY_MAX_CONCURRENT_PDF_EXTRACTIONS=1
```

If 503 storms continue after that, the next improvement should be failed-page
retry/recovery: keep successful page results in memory for that request, retry
only failed pages after a longer backoff window, then assemble the final paper
only when all expected pages have completed. Do not re-enable expensive
whole-document fallback as the default.

Implementation checkpoint:

- QP Lite primary retries were restored to 3 by default.
- Local `.env` now explicitly pins the low-cost stability settings.
- `extract_pages_batch` now performs one delayed failed-page-only retry pass
  before rejecting the paper.
- Successful page results are kept in memory during that request; only pages
  that returned `_page_failed` are retried.
- Whole-document Gemini fallback remains disabled for cost control.

## June 18 - Smarter Targeted Rescue For Grouped IDs

Targeted rescue was still weak for nested grouped rows. Example:

```text
missing: 4.a.iii.a
model/source row found: 4.a.iii
```

The old behaviour returned only a repair hint:

```text
split grouped source row manually
```

That was honest but not practical enough. The updated rescue path now creates a
split-review row when it finds a high-confidence grouped source row:

- the new row uses the exact missing canonical ID
- text is copied from the grouped source conservatively
- `needs_review=true`
- warnings explain that the row was created from a grouped source and must be
  verified/trimmed against the PDF
- diagrams are intentionally not copied from the grouped source

This is not counted as a clean model extraction. It is a practical review row
so the dashboard can fix missing nested IDs without repeated Gemini rescue
calls or full redo.

## June 20 - IB Board Extraction Deferred Track

IB PDFs were tested separately from the IGCSE pipeline using:

```text
Mathematics_analysis_and_approaches_paper_1_TZ2_HL.pdf
Mathematics_analysis_and_approaches_paper_1_TZ2_HL_markscheme.pdf
```

The result was clear: IB should not be treated as an IGCSE variant.

Observed IB QP behaviour:

- correct IB upload rendered 17 pages
- returned about 18 QP rows
- estimated cost was about INR 1.08
- main issues were continuation pages, grouped parent questions, and over-split
  subparts

Observed IB MS behaviour:

- rendered 31 pages
- returned 41 rows
- estimated cost was about INR 10.87
- front matter/examiner instructions were incorrectly saved as MS rows such as
  `5`, `6`, `7`, `8`, `9`, and `10`
- sequence tracking was poisoned before the real `SECTION A` answers began

Conclusion:

```text
IGCSE:
Keep MS-first, native MS tables, local QP skeleton, Gemini Lite QP support.

IB:
Create a separate board-specific route before production use.
```

Required IB-specific work is documented in:

```text
PAPERLY_IB_BOARD_EXTRACTION_NOTES.md
```

Key future fixes:

- detect `board=IB` before IGCSE key/sequence assumptions
- skip IB MS front matter until `SECTION A` or first real answer row
- ignore IB QP cover/copyright/continued-only pages
- prefer grouped parent QP rows unless child boundaries are visibly proven
- use IB-specific QA instead of exact IGCSE leaf parity
- shorten persistent cache filenames for long IB source filenames
