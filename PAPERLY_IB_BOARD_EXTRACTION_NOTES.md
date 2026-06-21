# Paperly IB Board Extraction Notes

Date: 2026-06-20

Status: Deferred board-specific hardening note. Do not treat these findings as
IGCSE rules.

## Why This Note Exists

Paperly Phase 2 is currently strongest for Cambridge IGCSE flows, especially:

- MS-first upload
- native Cambridge MS table extraction
- saved MS canonical IDs as the QP numbering contract
- local QP skeleton plus Gemini Flash-Lite page extraction
- human QA before save

IB papers behave differently. The tested IB pair exposed several problems that
should be handled by an IB-specific extraction route, not by adding more IGCSE
guards.

Tested files:

```text
Question Paper:
Mathematics_analysis_and_approaches_paper_1_TZ2_HL.pdf

Marking Scheme:
Mathematics_analysis_and_approaches_paper_1_TZ2_HL_markscheme.pdf
```

Board reminder:

```text
This is IB, not IGCSE.
```

## Tested IB QP Behaviour

The same QP was first uploaded in a way that made the pipeline treat it as
IGCSE. That caused the wrong key path and poor row splitting.

Observed first-run issue:

```text
board=IGCSE
IB QP was pushed into IGCSE key/numbering assumptions
paper key generation failed or remained empty
Gemini produced too many QP rows
```

When uploaded as IB, the QP run was much better:

```text
board=IB
pages rendered: 17
rows returned: 18
estimated Gemini cost: about INR 1.08
Gemini failures: 0
```

This shows IB QP can be cheap enough with the current page-level Gemini path.
The problem is not raw cost on QP. The problem is board-specific structure and
row boundaries.

## IB QP Structure Differences

IB QP pages include:

- cover pages
- copyright/license pages
- instruction pages
- answer-space continuation pages
- pages that say a question is continued but contain no new question text
- large blank working areas
- long parent questions with multiple subparts
- section-level separation such as Section A / Section B

The tested QP had this visible structure:

```text
Page 1: copyright/license style page
Page 2: cover/instructions
Page 3: actual Question 1 with graph and parts
Page 4: Question 1 continuation/answer lines, not a new row
Page 5: Question 2
Page 6: Question 3 with diagram and parts
Later pages: grouped IB questions, often best stored as parent/grouped rows
```

Important lesson:

```text
IB QP should not be forced into IGCSE-style leaf splitting unless the row
boundary is visible and supported by the PDF text/layout.
```

## IB QP Problems Observed

### 1. Continuation Pages Became Rows

Rows such as:

```text
1 continued
blank_page
```

should not be saved as question rows. They should be ignored or treated as
continuation context for the nearest real question.

### 2. Duplicate/Grouped Question Labels Were Over-Repaired

One observed pattern:

```text
raw rows around Q3:
3(a)
3(a)
3(a)
```

The sequence guard then repaired these into:

```text
3.a
3.b
3.c
```

But the text showed they were not necessarily clean Q3(a), Q3(b), Q3(c) rows.
They were grouped/continued content. This is a classic case where IGCSE
sequence repair can become harmful for IB.

Rule for later:

```text
For IB, do not relabel duplicate subpart text into sibling IDs unless the
visible label and local page context prove that sibling.
```

### 3. Grouped Parent Questions Are Often The Safer Storage Unit

In IB QP, many questions are naturally written as:

```text
Question 1 with parts (a), (b), (c)
Question 2 with parts (a), (b)
Question 6 with multiple subparts
```

For RAG and future generation, it may be safer to store the complete parent
question with internal subpart text, then optionally add child metadata later.

Possible IB QP storage strategy:

```text
canonical_question_id = 1
question_latex contains all visible 1(a), 1(b), 1(c)
subparts = structured child spans when confidently available
```

This is different from IGCSE, where QP/MS exact leaf parity is often a stronger
requirement.

## Tested IB MS Behaviour

The IB MS run was much more problematic than the QP run.

Observed run:

```text
board=IB
file type=Marking Scheme
pages rendered: 31
rows returned: 41
model used: gemini-2.5-flash
estimated Gemini cost: about INR 10.87
```

This is too expensive and not clean enough as a production default.

## IB MS Structure Differences

The tested MS had:

```text
Page 1: cover
Page 2: copyright
Pages 3-6: Instructions to Examiners / general marking guidance
Page 7 onward: real markscheme starts at SECTION A / Question 1
```

The current extraction allowed front matter to become fake marking scheme rows.

Bad rows observed:

```text
5
6
7
8
9
10
```

These were not real question IDs. They came from front-matter numbering and
instruction sections such as:

- marking notes
- alternative methods/forms
- accuracy guidance
- no calculator rules
- crossed-out work instructions

## IB MS Problems Observed

### 1. Front Matter Was Saved As Questions

The first six extracted MS rows were instruction pages, not answers.

Production rule needed:

```text
For IB MS, ignore all pages before SECTION A or before the first real answer row
such as "1. (a)".
```

### 2. Sequence Guard Was Poisoned

Because front matter created fake roots like 5, 6, 7, 8, 9, 10, the tracker
then thought the paper had already reached Question 10.

When real Q1/Q2/Q3 rows appeared, the guard treated them as backward jumps and
rejected or distorted them.

Rule for later:

```text
IB MS sequence tracking must start only after the real markscheme section starts.
Do not let examiner-instruction pages initialize canonical question tracking.
```

### 3. IB MS Cost Was Too High

The tested MS used 31 Gemini page calls and cost about INR 10.87.

Cost issue:

```text
IB MS front matter pages should not be sent to Gemini.
IB MS text is often locally readable and should be parsed locally when possible.
```

Immediate cost improvement expected:

- Skip cover/copyright/instruction pages before Gemini.
- Try local text/table parsing for IB MS answers.
- Send only uncertain answer pages/blocks to Gemini.

This should reduce both cost and wrong rows.

### 4. Cache Write Had A Long-Path Failure

The IB MS cache write failed because the persistent cache filename became too
long.

Observed pattern:

```text
.extraction_cache\v3_math_text_guard_<long hash>_Mathematics_analysis_and_approaches_paper_1_TZ2_HL_markscheme.pdf_.tmp
```

Future fix:

```text
Use short hashed cache filenames for long source PDFs.
Do not include the full original PDF filename in the cache path.
```

## Key Generation Problems

The QP key generation failed in one IB run with:

```text
'NoneType' object has no attribute 'lower'
```

The MS produced:

```text
ib_mathaa_hl_s25_ms_p1
```

The QP should produce a matching identity, likely:

```text
ib_mathaa_hl_s25_qp_p1
unified: ib_mathaa_hl_s25_p1
```

The filename also contains:

```text
TZ2
HL
paper_1
Mathematics analysis and approaches
```

Future IB key builder should infer from both filename and cover:

- board: IB
- subject family: Mathematics
- course: Analysis and Approaches
- level: HL
- paper: 1
- timezone: TZ2 when present
- session/year: from cover and filename where possible
- document type: QP or MS

Do not allow a missing field to crash key generation.

## What Should Be Extracted For IB

### QP Fields

For IB QP, the extractor should capture:

- board = IB
- programme/course, e.g. Mathematics: analysis and approaches
- level = HL or SL
- paper number
- timezone/zone if visible or in filename
- session/year
- section, e.g. Section A / Section B
- canonical question root
- visible subpart labels
- full parent stem
- exact child text when confidently separated
- diagrams/graphs attached only when visually connected
- marks where visible
- calculator status from cover
- review warnings for continuation/grouping uncertainty

### MS Fields

For IB MS, the extractor should capture:

- real question number only after markscheme section begins
- subpart labels
- final answer
- method marks
- answer marks
- reasoning/alternative method notes
- total marks per row where available
- section
- examiner notes only when attached to a real answer row
- diagrams only when part of the answer, not from front matter

Front matter should be stored separately only if needed as document metadata,
not as question rows.

## Future IB Extraction Architecture

Recommended IB-specific route:

```text
1. Detect board=IB before extraction.
2. Build IB document metadata from filename + cover page.
3. For QP:
   - skip cover/copyright/instruction pages
   - detect real question starts
   - treat continuation pages as parent context
   - prefer grouped parent rows unless child split is clear
   - use Gemini mainly for difficult layout/diagram text
4. For MS:
   - skip front matter until SECTION A / first real answer row
   - reset sequence tracker at first real answer row
   - parse local text/table structure first
   - send only uncertain answer blocks/pages to Gemini
5. Run IB-specific QA:
   - not the same as IGCSE exact QP/MS leaf parity
   - flag front matter rows as fatal if saved
   - flag continuation-only QP rows as review/delete
```

## QA Rules For IB

IB QA should not copy IGCSE exact-leaf parity blindly.

IB QA should check:

- no front matter saved as question/MS rows
- no `blank_page` rows
- no `continued`-only rows
- no examiner instruction rows inside MS answers
- real QP roots appear in order
- real MS roots appear in order after SECTION A
- grouped QP parent rows are allowed when MS splits subparts
- child rows require visible child text evidence
- diagrams are tied to the right parent or subpart
- key fields match between QP and MS

## Cost/Time Reality

Observed IB QP:

```text
17 pages
18 rows returned
cost about INR 1.08
quality: structurally usable but needs IB continuation/grouping cleanup
```

Observed IB MS:

```text
31 pages
41 rows returned
cost about INR 10.87
quality: not acceptable because front matter became fake rows
```

Expected after IB-specific fixes:

- QP cost can likely remain low if page-level Flash-Lite remains enough.
- MS cost should reduce sharply by skipping front matter and using local parsing.
- Accuracy should improve more from board-specific filtering than from stronger
  models.

## Decision For Now

Do not block IGCSE Phase 2 deployment readiness on IB.

Store IB as a separate Phase 2+ hardening track:

```text
IGCSE path: continue current MS-first/local-skeleton architecture.
IB path: build board-specific extraction and QA rules before production uploads.
```

## Implementation Checklist For Later

- Add `board == "IB"` routing before IGCSE key/numbering assumptions.
- Fix IB key generation to never crash on missing fields.
- Add IB QP cover/front-matter/continued-page filter.
- Add IB MS front-matter filter until `SECTION A` or first real answer row.
- Reset IB MS sequence tracker only after first real answer row.
- Disable aggressive IGCSE page-number/backward-root repairs for IB.
- Prefer grouped parent QP rows unless child boundaries are proven.
- Add IB-specific QA issue cards.
- Shorten persistent cache filenames.
- Re-test the same QP/MS pair and compare:
  - row count
  - fake front-matter rows
  - key pairing
  - cost
  - diagrams
  - review warnings

