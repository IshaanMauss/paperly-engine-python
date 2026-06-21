"""
Script to rewrite gemini_pdf_service.py with the new Single-Pass Multimodal Concurrent Engine
"""

# Read the current file up to line 428
with open('services/gemini_pdf_service.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Keep only lines 1-428 (indices 0-427)
kept_lines = lines[:428]

# New content for SECTION 4 onwards - this is the complete new implementation
new_sections = '''

# ===========================================================================
# SECTION 4: Answer-blank sanitizer  (ORIGINAL LOGIC PRESERVED)
# ===========================================================================

def _sanitize_answer_blanks(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'(\\textunderscore){2,}', '', text)
    text = re.sub(r'\\underline\\{\\\\hspace\\{[^}]*\\}\\}', '', text)
    text = re.sub(r'\\\\dotfill', '', text)
    text = re.sub(r'\\s*\\[\\d+\\]\\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\\n{3,}', '\\n\\n', text)
    return text.strip()


# ===========================================================================
# SECTION 5: Normalisation helpers  (ORIGINAL LOGIC PRESERVED)
# ===========================================================================

_QUESTION_FIELD_ALIASES: dict[str, str] = {
    "question_text": "question_latex", "latex": "question_latex",
    "question_content": "question_latex", "text": "question_latex",
    "marking_scheme_latex": "official_marking_scheme_latex",
    "answer": "official_marking_scheme_latex", "mark_scheme": "official_marking_scheme_latex",
    "questionNumber": "question_latex", "question_number": "question_latex",
    "diagrams": "diagram_urls", "images": "diagram_urls",
    "templateable": "isTemplatizable", "is_templateable": "isTemplatizable",
    "is_templatizable": "isTemplatizable",
    "subject_code": "subjectCode", "subject": "subjectCode",
    "paper": "paperNumber", "paper_number": "paperNumber",
}

_METADATA_FIELD_ALIASES: dict[str, str] = {
    "subject_code": "subjectCode", "subject": "subjectCode",
    "paper": "paperNumber", "paper_number": "paperNumber",
}

_QUESTION_DEFAULTS: dict = {
    "document_type": "Question Paper", "curriculum": "", "program": None,
    "subjectCode": "", "tier": None, "paperNumber": 0, "session": None, "year": 0,
    "paper_reference_key": "", "unified_paper_key": "", "canonical_question_id": "",
    "parent_canonical_id": "",
    "question_number_metadata": QuestionNumberMetadata().model_dump(),
    "validation_status": "pending", "validation_warnings": [],
    "ref_code_base": "", "ref_code_full": "",
    "isTemplatizable": False, "variables": [], "question_latex": "",
    "question_id": "", "final_answer": "", "total_marks": 0, "method_steps": [],
    "official_marking_scheme_latex": None, "diagram_urls": [],
    "diagram_page_number": None, "diagram_y_range": [], "diagram_regions": [],
    "needs_review": False, "cognitive_demand": "MEDIUM", "difficulty_override": None,
}

_METADATA_DEFAULTS: dict = {
    "curriculum": "", "program": None, "subjectCode": "", "tier": None,
    "paperNumber": 0, "session": None, "year": 0, "paper_reference_key": "",
    "unified_paper_key": "", "validation_status": "pending", "validation_warnings": [],
    "ref_code_base": "", "ref_code_full": "",
}


def _normalize_tier(tier) -> str:
    if not tier or not isinstance(tier, str):
        return "N/A"
    t = tier.lower().strip()
    if "higher" in t or t == "hl": return "HL"
    if "standard" in t or t == "sl": return "SL"
    if "core" in t: return "Core"
    if "extended" in t: return "Extended"
    return "N/A"


def _remap_keys(raw: dict, alias_map: dict) -> dict:
    out = {}
    for k, v in raw.items():
        canonical = alias_map.get(k, k)
        if canonical not in out:
            out[canonical] = v
        else:
            out[k] = v
    return out


def _coerce_int(value, default: int = 0) -> int:
    if value is None: return default
    try: return int(value)
    except (ValueError, TypeError): return default


def _coerce_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool): return value
    if isinstance(value, str): return value.strip().lower() in ("true", "1", "yes")
    if isinstance(value, int): return bool(value)
    return default


def _coerce_list(value, default=None) -> list:
    if default is None: default = []
    if isinstance(value, list): return value
    if value is None: return default
    return [str(value)]


def _normalize_method_steps(raw_steps) -> list:
    if not isinstance(raw_steps, list): return []
    result = []
    for step in raw_steps:
        if isinstance(step, dict):
            result.append({
                "type": str(step.get("type", "")).strip(),
                "description": str(step.get("description", step.get("desc", ""))).strip(),
            })
        elif isinstance(step, str):
            result.append({"type": "note", "description": step.strip()})
    return result


def _normalize_metadata(
    raw: dict | None,
    filename: str,
    board: str,
    generated_key_override: str = "",
) -> dict:
    if not isinstance(raw, dict): raw = {}

    extracted_curr = str(raw.get("curriculum", "")).upper()
    if "INTERNATIONAL BACCALAUREATE" in extracted_curr or "IB" in extracted_curr:
        raw["curriculum"] = "IB"
    elif "CAMBRIDGE" in extracted_curr or "IGCSE" in extracted_curr:
        raw["curriculum"] = "IGCSE"

    if board and ("INTERNATIONAL BACCALAUREATE" in board.upper() or board.upper() == "IB"):
        board = "IB"
    elif board and ("CAMBRIDGE" in board.upper() or board.upper() == "IGCSE"):
        board = "IGCSE"

    raw = _remap_keys(raw, _METADATA_FIELD_ALIASES)
    result = dict(_METADATA_DEFAULTS)
    result.update({k: v for k, v in raw.items() if k in result})
    result["paperNumber"] = _coerce_int(result["paperNumber"], 0)
    result["year"]        = _coerce_int(result["year"], 0)
    result["tier"]        = _normalize_tier(result.get("tier"))

    generated_key = ""
    if board.upper() == "IGCSE":
        generated_key = _generate_igcse_paper_reference_key(filename)
        extracted_session = str(result.get("session", "")).lower().strip()
        if generated_key and extracted_session:
            real_season = None
            if any(x in extracted_session for x in ["mar", "feb"]): real_season = "m"
            elif any(x in extracted_session for x in ["may", "jun", "sum"]): real_season = "s"
            elif any(x in extracted_session for x in ["oct", "nov", "win"]): real_season = "w"
            if real_season:
                generated_key = re.sub(
                    r'^(igcse_\\d{4}_)[smw](\\d{2})',
                    fr'\\g<1>{real_season}\\g<2>',
                    generated_key,
                    flags=re.IGNORECASE,
                )
    else:
        generated_key = generated_key_override or result.get("paper_reference_key", "")

    result["paper_reference_key"] = generated_key or result.get("paper_reference_key", "")
    result["curriculum"] = board.upper()
    return result


def _normalize_question(
    raw: dict,
    fallback_metadata: dict,
    document_type: str,
    question_normalizer: QuestionNumberNormalizer,
) -> dict:
    if not isinstance(raw, dict):
        return dict(_QUESTION_DEFAULTS)

    raw = _remap_keys(raw, _QUESTION_FIELD_ALIASES)
    result = dict(_QUESTION_DEFAULTS)
    for k in result:
        if k in raw:
            result[k] = raw[k]

    q_id_raw = result.get("question_id") or result.get("question_latex") or ""
    if q_id_raw and fallback_metadata.get("paper_reference_key"):
        normalized_data = question_normalizer.normalize(
            raw_question_id=q_id_raw,
            paper_reference_key=fallback_metadata["paper_reference_key"],
        )
        result.update(normalized_data)
        result["question_number_metadata"] = QuestionNumberMetadata(
            **normalized_data["question_number_metadata"]
        ).model_dump()

    result["document_type"] = document_type
    result["tier"] = _normalize_tier(result.get("tier"))

    for meta_key in (
        "curriculum", "program", "subjectCode", "tier", "paperNumber", "session", "year",
        "paper_reference_key", "unified_paper_key", "validation_status", "validation_warnings",
        "ref_code_base", "ref_code_full",
    ):
        if not result.get(meta_key) and fallback_metadata.get(meta_key):
            result[meta_key] = fallback_metadata[meta_key]

    if fallback_metadata.get("curriculum"):
        result["curriculum"] = fallback_metadata["curriculum"]
    if not result.get("paper_reference_key") and fallback_metadata.get("paper_reference_key"):
        result["paper_reference_key"] = fallback_metadata["paper_reference_key"]

    if document_type.strip().lower() == "marking scheme":
        if not result.get("question_id"): result["question_id"] = result.get("question_latex", "")
        if not result.get("final_answer"): result["final_answer"] = ""
        result["total_marks"]  = _coerce_int(result.get("total_marks"), 0)
        result["method_steps"] = _normalize_method_steps(result.get("method_steps", []))

    result["paperNumber"]     = _coerce_int(result["paperNumber"], 0)
    result["year"]            = _coerce_int(result["year"], 0)
    result["isTemplatizable"] = _coerce_bool(result["isTemplatizable"], False)
    result["variables"]       = _coerce_list(result["variables"], [])

    raw_diagrams = _coerce_list(result["diagram_urls"], [])
    valid_urls = []
    has_diagram_indicator = False

    flattened: List[str] = []
    for item in raw_diagrams:
        if item is None: continue
        if isinstance(item, list):
            flattened.extend([str(si).strip() for si in item if si])
        else:
            flattened.append(str(item).strip())

    for item_str in flattened:
        if not item_str: continue
        if (
            item_str.startswith("http")
            or item_str.startswith("data:image")
            or item_str == "[NEEDS_CROP]"
        ):
            valid_urls.append(item_str)
        elif item_str not in ("[]", "null", "undefined"):
            has_diagram_indicator = True

    result["diagram_urls"] = valid_urls
    result["needs_review"] = _coerce_bool(result["needs_review"], False)

    _VALID_DEMANDS = {"LOW", "MEDIUM", "HIGH"}
    if str(result.get("cognitive_demand", "")).upper() not in _VALID_DEMANDS:
        result["cognitive_demand"] = "MEDIUM"
    else:
        result["cognitive_demand"] = str(result["cognitive_demand"]).upper()

    if result.get("difficulty_override") not in {"Easy", "Medium", "Hard", None}:
        result["difficulty_override"] = None

    if not result["diagram_urls"]:
        q_latex = (result.get("question_latex") or "").lower()
        if has_diagram_indicator or "diagram" in q_latex or "graph" in q_latex or "figure" in q_latex:
            result["diagram_urls"] = []

    if not isinstance(result["diagram_urls"], list):
        result["diagram_urls"] = []

    result["curriculum"]    = result["curriculum"] or ""
    result["subjectCode"]   = result["subjectCode"] or ""
    result["question_latex"] = _sanitize_answer_blanks(result.get("question_latex") or "")
    return result


def _normalize_response(
    parsed: dict,
    filename: str,
    document_type: str,
    board: str,
    generated_paper_reference_key: str = "",
    extra_metadata: dict = None,
) -> SlicedQuestionsResponse:
    meta_raw = parsed.get("metadata") or {}
    if extra_metadata:
        meta_raw.update(extra_metadata)
    meta_normalized = _normalize_metadata(meta_raw, filename, board, generated_paper_reference_key)

    question_normalizer = QuestionNumberNormalizer()
    if not meta_normalized.get("unified_paper_key") and meta_normalized.get("paper_reference_key"):
        meta_normalized["unified_paper_key"] = question_normalizer._generate_unified_paper_key(
            meta_normalized["paper_reference_key"]
        )

    questions_raw = parsed.get("questions_array") or []
    if not isinstance(questions_raw, list):
        questions_raw = []

    questions: List[ExtractedQuestion] = []
    qp_parent_ids: set = set()
    ms_parent_ids: set = set()

    for i, q in enumerate(questions_raw):
        try:
            normalized = _normalize_question(q, meta_normalized, document_type, question_normalizer)
            schema_fields = set(ExtractedQuestion.model_fields.keys())
            filtered = {k: v for k, v in normalized.items() if k in schema_fields}
            question_obj = ExtractedQuestion(**filtered)
            questions.append(question_obj)
            if question_obj.parent_canonical_id:
                if question_obj.document_type == "Question Paper":
                    qp_parent_ids.add(question_obj.parent_canonical_id)
                else:
                    ms_parent_ids.add(question_obj.parent_canonical_id)
        except Exception as exc:
            print(f"⚠️  [normalize] Skipping question {i}: {exc}")
            try:
                safe = dict(_QUESTION_DEFAULTS)
                safe.update({k: v for k, v in meta_normalized.items() if k in safe})
                safe["document_type"] = document_type
                safe["question_latex"] = str(q) if not isinstance(q, dict) else q.get("question_latex", "")
                safe["needs_review"] = True
                schema_fields = set(ExtractedQuestion.model_fields.keys())
                questions.append(ExtractedQuestion(**{k: v for k, v in safe.items() if k in schema_fields}))
            except Exception:
                pass

    # Sequence gap check
    validation_status = "ok"
    validation_warnings = []
    recommendation = "proceed"
    parent_ids_to_check = qp_parent_ids if document_type == "Question Paper" else ms_parent_ids

    if parent_ids_to_check:
        int_parents = [int(pid) for pid in parent_ids_to_check if str(pid).isdigit()]
        if int_parents:
            int_parents.sort()
            expected = set(range(min(int_parents), max(int_parents) + 1))
            missing  = expected - set(int_parents)
            if missing:
                validation_status = "warning"
                recommendation    = "review"
                validation_warnings.append(
                    f"Sequence gap detected in {document_type}. Missing parent questions: "
                    f"{', '.join(map(str, sorted(missing)))}"
                )

    meta_normalized["validation_status"]   = validation_status
    meta_normalized["validation_warnings"] = validation_warnings

    val_report = ValidationReport(
        status=validation_status,
        recommendation=recommendation,
        message=" | ".join(validation_warnings) if validation_warnings else "Sequence is continuous.",
        checks={"sequence_gaps": bool(validation_warnings)},
    )

    return SlicedQuestionsResponse(
        metadata=ExtractedPaperMetadata(**meta_normalized),
        questions_array=questions,
        validation_report=val_report,
    )


# ===========================================================================
# SECTION 6: JSON Parser  (ORIGINAL LOGIC PRESERVED)
# ===========================================================================

def _parse_json_payload(content: str) -> dict:
    if not content or not content.strip():
        return {"metadata": {}, "questions_array": []}

    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Smart single-backslash escape (preserves valid \\\\ and \\n)
    cleaned = re.sub(r'(?<!\\\\)\\\\(?!["\\\\n])', r'\\\\\\\\', cleaned)

    # Iterative auto-heal loop
    for attempt in range(10):
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            err_msg = str(e)
            if "Invalid \\\\escape" in err_msg or "Invalid \\\\u" in err_msg:
                pos = e.pos
                while pos > 0 and cleaned[pos] != '\\\\':
                    pos -= 1
                if cleaned[pos] == '\\\\':
                    cleaned = cleaned[:pos] + '\\\\\\\\' + cleaned[pos:]
                    continue
                else:
                    print(f"CRITICAL PARSE FAIL (Auto-Heal): {err_msg}")
                    break
            else:
                print(f"CRITICAL PARSE FAIL (Structure): {err_msg}")
                break

    return {"metadata": {}, "questions_array": []}


# ===========================================================================
# SECTION 7: Gemini client helpers  (ORIGINAL LOGIC PRESERVED)
# ===========================================================================

def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def _wait_for_file_ready(
    client: genai.Client, file_name: str, timeout_seconds: int = 240
):
    deadline = time.time() + timeout_seconds
    state_code_map = {0: "STATE_UNSPECIFIED", 1: "PROCESSING", 2: "ACTIVE", 3: "FAILED"}

    def _normalize_state(sv) -> str:
        if sv is None: return "UNKNOWN"
        name = getattr(sv, "name", None)
        if isinstance(name, str) and name: return name.upper()
        if isinstance(sv, int): return state_code_map.get(sv, str(sv))
        try: return state_code_map.get(int(sv), str(sv)).upper()
        except Exception: return str(sv).upper() or "UNKNOWN"

    last_state = "UNKNOWN"
    while time.time() < deadline:
        remote_file = client.files.get(name=file_name)
        last_state  = _normalize_state(getattr(remote_file, "state", None))
        if "ACTIVE" in last_state: return remote_file
        if "FAILED" in last_state:
            raise RuntimeError(f"Uploaded file FAILED: {last_state}")
        time.sleep(1.0)
    raise TimeoutError(f"File not ACTIVE before timeout. Last state: {last_state}")


def _generate_with_retry(
    client: genai.Client,
    model: str,
    contents: list,
    config: dict,
    retries: int = 3,
    delay: float = 5.0,
):
    last_exc = None
    for attempt in range(retries):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:
            last_exc = e
            err_str  = str(e)
            is_transient = any(
                code in err_str for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED")
            )
            if is_transient and attempt < retries - 1:
                wait = delay * (2 ** attempt)
                print(f"⚠️  [Gemini] Transient error, retry {attempt+1}/{retries} in {wait:.0f}s: {e}")
                time.sleep(wait)
                continue
            raise
    raise last_exc


_MODEL_PRIORITY: List[str] = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-1.5-flash",
]


def _pick_available_model(client: genai.Client, exclude: list = None) -> str:
    exclude_set = set(exclude or [])
    try:
        available = {m.name.replace("models/", "") for m in client.models.list()}
    except Exception as e:
        print(f"⚠️  [_pick_available_model] Could not fetch model list: {e}")
        available = set(_MODEL_PRIORITY)

    for m in _MODEL_PRIORITY:
        if m not in exclude_set and m in available:
            return m
    for m in _MODEL_PRIORITY:
        if m not in exclude_set:
            return m
    return _MODEL_PRIORITY[0]


# ===========================================================================
# SECTION 8: Native Diagram Cropping Pass
# ===========================================================================

async def _apply_diagram_regions_to_questions(
    questions_array: List[dict],
    pdf_base64: str,
    page_num: int,
) -> List[dict]:
    """
    For every question dict from a single page, check if it has diagram_regions.
    For each region with y_start_pct, y_end_pct, x_start_pct, x_end_pct:
      • Call crop_and_compress_diagram_async
      • Store resulting base64 JPEG in the question's diagram_urls array

    This runs ALL crops for all questions concurrently via asyncio.gather.
    """
    if not questions_array:
        return questions_array

    async def _process_question_regions(q: dict) -> dict:
        if not isinstance(q, dict):
            return q

        diagram_regions = q.get("diagram_regions", [])
        if not isinstance(diagram_regions, list) or not diagram_regions:
            return q

        # Initialize diagram_urls if not a list
        if not isinstance(q.get("diagram_urls"), list):
            q["diagram_urls"] = []

        # Collect all crop tasks for this question
        crop_tasks = []
        for region in diagram_regions:
            if not isinstance(region, dict):
                continue

            y_start = region.get("y_start_pct")
            y_end = region.get("y_end_pct")
            x_start = region.get("x_start_pct", 0.0)
            x_end = region.get("x_end_pct", 100.0)

            if y_start is None or y_end is None:
                continue

            try:
                y_start = float(y_start)
                y_end = float(y_end)
                x_start = float(x_start)
                x_end = float(x_end)
            except (ValueError, TypeError):
                continue

            if y_start >= y_end or y_start < 0 or y_end > 100:
                continue

            crop_tasks.append(
                crop_and_compress_diagram_async(
                    pdf_base64=pdf_base64,
                    page_num=page_num,
                    y_start_pct=y_start,
                    y_end_pct=y_end,
                    x_start_pct=x_start,
                    x_end_pct=x_end,
                )
            )

        if not crop_tasks:
            return q

        # Execute all crops concurrently
        cropped_b64_list = await asyncio.gather(*crop_tasks)

        for cropped_b64 in cropped_b64_list:
            if cropped_b64:
                q["diagram_urls"].append(f"data:image/jpeg;base64,{cropped_b64}")
                logger.info(
                    f"[Diagram Crop] ✅ Cropped diagram for question "
                    f"{q.get('question_id', '?')} (page={page_num})"
                )
            else:
                logger.warning(
                    f"[Diagram Crop] ⚠️  Crop returned None for question "
                    f"{q.get('question_id', '?')} (page={page_num})"
                )

        return q

    # Run all question region processing concurrently
    updated = await asyncio.gather(*[_process_question_regions(q) for q in questions_array])
    return list(updated)


# ===========================================================================
# SECTION 9: Public async entry-point with Single-Pass Multimodal Concurrent Engine
# ===========================================================================

async def extract_pdf_native_gemini(
    pdf_base64: str,
    document_type: str,
    filename: str,
    board: str = "IGCSE",
    page1_base64: str = None,
    use_cache: bool = False,
) -> SlicedQuestionsResponse:
    """
    Extract structured questions from a PDF using the Single-Pass Multimodal
    Concurrent Engine.

    For every rendered page:
      • Launch ONE Gemini call via extract_and_structure_page
      • Returns questions with native diagram_regions
      • All pages run concurrently via asyncio.gather

    Then apply native diagram cropping:
      • For each question with diagram_regions, crop to base64 JPEG
      • Store in diagram_urls array
    """
    print(f"🚀 [Single-Pass Engine] Starting extraction for {filename}…")

    # ── Step 1: Generate paper reference key ─────────────────────────────────
    paper_reference_key = ""
    extra_metadata: dict = {}

    if board.upper() == "IGCSE":
        paper_reference_key = _generate_igcse_paper_reference_key(filename)
        print(f"ℹ️  [Paper Key] IGCSE key: {paper_reference_key!r}")

    else:  # IB
        ib_metadata = {}
        if page1_base64:
            try:
                ib_metadata = (
                    await asyncio.to_thread(_extract_ib_metadata_from_page, page1_base64)
                ) or {}
            except Exception as ib_exc:
                logger.warning(f"[Paper Key] IB metadata call failed: {ib_exc}")

        # ref_code extraction from PDF bytes (local regex — free)
        normalized_b64 = (
            pdf_base64.strip().split(",", 1)[-1]
            if "," in pdf_base64
            else pdf_base64.strip()
        )
        pdf_bytes = base64.b64decode(normalized_b64)
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name
            ref_code, method = regex_extract_ref_code(tmp_path)
            if ref_code:
                session = ib_metadata.get("session", "")
                year    = ib_metadata.get("year", "")
                if not session or not year:
                    prefix = ref_code.session_prefix
                    if len(prefix) == 4:
                        year    = "20" + prefix[:2]
                        session = "may" if prefix[2:] == "25" else "november"
                paper_reference_key = build_paper_reference_key(
                    curriculum="ib",
                    subject=ib_metadata.get("subject_name", ""),
                    tier=ib_metadata.get("level", ""),
                    session=session,
                    year=year,
                    ref_code_base=ref_code.base,
                )
                extra_metadata = {
                    "ref_code_base": ref_code.base,
                    "ref_code_full": ref_code.raw,
                }
                print(f"ℹ️  [Paper Key] IB key: {paper_reference_key!r} via {method}")
        except Exception as ref_exc:
            logger.warning(f"[Paper Key] IB ref-code extraction failed: {ref_exc}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # ── Step 2: Render pages at high DPI for Gemini ──────────────────────────
    print("📄 [Single-Pass Engine] Rendering pages for Gemini extraction…")
    try:
        vision_pages = await pdf_base64_to_vision_pages_async(pdf_base64, dpi=300)
    except Exception as e:
        logger.error(f"[Single-Pass Engine] Page rendering failed: {e}")
        raise PipelineServiceError(
            stage="page_rendering",
            message="Failed to render PDF pages.",
            details={"provider": "pdf_processor", "reason": str(e)},
        )

    if not vision_pages:
        raise PipelineServiceError(
            stage="page_rendering",
            message="PDF produced zero pages.",
            details={"provider": "pdf_processor"},
        )

    print(f"📄 [Single-Pass Engine] {len(vision_pages)} page(s) rendered for processing.")

    # ── Step 3: Launch concurrent page extraction ────────────────────────────
    print("🔍 [Single-Pass Engine] Launching concurrent Gemini calls for all pages…")

    async def _extract_page_concurrent(page_b64: str, page_idx: int):
        """
        Single-pass extraction for one page via extract_and_structure_page.
        Returns list of question dicts with native diagram_regions.
        """
        try:
            questions = await asyncio.to_thread(
                extract_and_structure_page,
                page_b64,
                document_type,
                page_idx,
            )
            logger.debug(f"[Page {page_idx}] Extracted {len(questions) if questions else 0} question(s)")
            return questions or []
        except Exception as e:
            logger.warning(f"[Page {page_idx}] Extraction failed: {e}")
            return []

    page_results = await asyncio.gather(
        *[_extract_page_concurrent(page_b64, idx) for idx, page_b64 in enumerate(vision_pages)]
    )

    print(f"✅ [Single-Pass Engine] Page extraction complete.")

    # ── Step 4: Flatten all questions into single array ──────────────────────
    all_questions: List[dict] = []
    for page_idx, questions in enumerate(page_results):
        if isinstance(questions, list):
            for q in questions:
                if isinstance(q, dict):
                    q["diagram_page_number"] = page_idx + 1
                    all_questions.append(q)

    print(f"📊 [Single-Pass Engine] Total questions extracted: {len(all_questions)}")

    # ── Step 5: Apply native diagram cropping ────────────────────────────────
    print("🎨 [Diagram Cropping] Processing native diagram regions…")

    async def _crop_page_diagrams(page_idx: int, questions: List[dict]) -> List[dict]:
        """Crop all diagram regions for questions on a single page."""
        return await _apply_diagram_regions_to_questions(
            questions_array=questions,
            pdf_base64=pdf_base64,
            page_num=page_idx + 1,
        )

    # Group questions by page and crop concurrently
    crop_tasks = []
    for page_idx, questions in enumerate(page_results):
        if isinstance(questions, list) and questions:
            crop_tasks.append(_crop_page_diagrams(page_idx, questions))

    if crop_tasks:
        cropped_results = await asyncio.gather(*crop_tasks)
        # Flatten results back into single list
        all_questions = []
        for cropped_page_questions in cropped_results:
            if isinstance(cropped_page_questions, list):
                all_questions.extend(cropped_page_questions)

    print(f"✅ [Diagram Cropping] Complete. {len(all_questions)} question(s) finalized.")

    # ── Step 6: Build response metadata ──────────────────────────────────────
    parsed_dict = {
        "metadata": {
            "curriculum": board.upper(),
            "paper_reference_key": paper_reference_key,
        },
        "questions_array": all_questions,
    }
    if extra_metadata:
        parsed_dict["metadata"].update(extra_metadata)

    # ── Step 7: Normalize and return ─────────────────────────────────────────
    print("🔧 [Normalization] Normalizing response schema…")
    response = _normalize_response(
        parsed_dict,
        filename,
        document_type,
        board,
        generated_paper_reference_key=paper_reference_key,
        extra_metadata=extra_metadata if board.upper() != "IGCSE" else None,
    )

    print(f"🎯 [Single-Pass Engine] Complete. {len(response.questions_array)} final questions.")
    return response


__all__ = ["extract_pdf_native_gemini"]
'''

# Write the combined content
with open('services/gemini_pdf_service.py', 'w', encoding='utf-8') as f:
    f.writelines(kept_lines)
    f.write(new_sections)

print("✅ File rewritten successfully!")
print(f"Kept lines: {len(kept_lines)}")
print(f"File path: services/gemini_pdf_service.py")
