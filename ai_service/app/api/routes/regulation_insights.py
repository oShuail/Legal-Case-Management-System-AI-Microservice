from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from fastapi import APIRouter

from app.api.schemas.requests import (
    RegulationAmendmentImpactRequest,
    RegulationSummaryAnalysisRequest,
)
from app.api.schemas.responses import (
    RegulationAmendmentImpactResponse,
    RegulationCitation,
    RegulationInsightBullet,
    RegulationKeyDate,
    RegulationSummaryAnalysisResponse,
)
from app.config import settings
from app.core.llm_json import try_llm_json

router = APIRouter()

_OBLIGATION_PATTERNS = [
    r"\bmust\b",
    r"\bshall\b",
    r"\brequired\b",
    r"\bobligation\b",
    r"\bيلتزم\b",
    r"\bيجب\b",
    r"\bيتعين\b",
    r"\bيلزم\b",
]
_RISK_PATTERNS = [
    r"\bpenalty\b",
    r"\bfine\b",
    r"\bimprison\w*\b",
    r"\bsanction\w*\b",
    r"\bprohibited\b",
    r"\bforbidden\b",
    r"\bيعاقب\b",
    r"\bغرامة\b",
    r"\bالسجن\b",
    r"\bالحبس\b",
    r"\bيحظر\b",
    r"\bممنوع\b",
]
_AFFECTED_PATTERNS: list[tuple[str, list[str]]] = [
    ("أصحاب العمل والمنشآت", [r"\bemployer\b", r"\bcompany\b", r"\bمنشأة\b", r"\bصاحب العمل\b"]),
    ("العاملون والموظفون", [r"\bemployee\b", r"\bworker\b", r"\bعامل\b", r"\bموظف\b"]),
    ("الجهات التنظيمية والرقابية", [r"\bauthority\b", r"\bregulator\b", r"\bجهة\b", r"\bوزارة\b", r"\bهيئة\b"]),
    ("العملاء والمستفيدون", [r"\bcustomer\b", r"\bconsumer\b", r"\bعميل\b", r"\bمستفيد\b"]),
]


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _truncate(value: str, limit: int) -> str:
    return (value or "")[: max(1, limit)].strip()


def _split_sentences(text: str) -> list[str]:
    normalized = (text or "").strip()
    if not normalized:
        return []
    chunks = re.split(r"(?<=[\.\!\?؟؛;\n])\s+", normalized)
    output: list[str] = []
    for chunk in chunks:
        sentence = chunk.strip()
        if sentence:
            output.append(sentence)
    return output


def _severity_for_text(text: str) -> str:
    lowered = text.lower()
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in _RISK_PATTERNS):
        return "high"
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in _OBLIGATION_PATTERNS):
        return "medium"
    return "low"


def _dedupe_items(values: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = _normalize_text(value)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _extract_dates(text: str, limit: int = 8) -> list[RegulationKeyDate]:
    matches: list[str] = []
    patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        r"\b\d{1,2}-\d{1,2}-\d{4}\b",
    ]
    for pattern in patterns:
        matches.extend(re.findall(pattern, text))

    unique = _dedupe_items(matches, limit)
    output: list[RegulationKeyDate] = []
    for value in unique:
        output.append(
            RegulationKeyDate(
                label="تاريخ نظامي مهم",
                value=value,
                source="النص التنظيمي",
            )
        )
    return output


def _extract_citations(sentences: list[str], limit: int = 8) -> list[RegulationCitation]:
    scored: list[tuple[float, str]] = []
    for sentence in sentences:
        normalized = _normalize_text(sentence)
        if not normalized:
            continue
        score = 0.25
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in _OBLIGATION_PATTERNS):
            score += 0.35
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in _RISK_PATTERNS):
            score += 0.35
        score += min(0.05, len(normalized) / 20000)
        scored.append((min(score, 0.99), normalized[:400]))

    scored.sort(key=lambda item: item[0], reverse=True)
    citations: list[RegulationCitation] = []
    seen: set[str] = set()
    for relevance, snippet in scored:
        if snippet in seen:
            continue
        seen.add(snippet)
        citations.append(
            RegulationCitation(
                snippet=snippet,
                section_ref=None,
                relevance=round(relevance, 3),
            )
        )
        if len(citations) >= limit:
            break
    return citations


def _derive_bullet_title(sentence: str, fallback: str) -> str:
    """Extract a short meaningful title from an Arabic legal sentence.

    Strategy (in order):
    1. Numbered/ordinal prefix before a dash or colon (e.g. "١-", "أولاً:").
    2. Text before the first colon/comma, if ≤ 40 chars.
    3. First 5 Arabic words.
    4. ``fallback`` if nothing useful is found.
    """
    text = sentence.strip()

    # Numbered item prefix: "١-", "١ -", "أولاً:" etc.
    m = re.match(r"^([\u0660-\u06690-9]+\s*[-:–—]|[\u0600-\u06FF]{2,10}\s*[:-])\s*", text)
    if m:
        prefix = m.group(1).rstrip("-:–— ").strip()
        if 1 < len(prefix) <= 30:
            return prefix

    # Text before first colon (Arabic or Latin) or Arabic comma.
    for sep in (":", "：", "،"):
        idx = text.find(sep)
        if 0 < idx <= 45:
            candidate = text[:idx].strip()
            if candidate:
                return candidate

    # First 5 Arabic words.
    words = re.findall(r"[\u0600-\u06FF]+", text)
    if words:
        return " ".join(words[:5])

    return fallback


def _build_bullets(sentences: list[str], title: str, limit: int) -> list[RegulationInsightBullet]:
    output: list[RegulationInsightBullet] = []
    for sentence in _dedupe_items(sentences, limit):
        description = sentence[:450]
        output.append(
            RegulationInsightBullet(
                title=_derive_bullet_title(sentence, title),
                description=description,
                severity=_severity_for_text(description),
            )
        )
    return output


def _coerce_bullets(raw: Any, default_title: str, limit: int) -> list[RegulationInsightBullet]:
    if not isinstance(raw, list):
        return []
    output: list[RegulationInsightBullet] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or default_title).strip() or default_title
        description = str(item.get("description") or "").strip()
        if not description:
            continue
        severity = item.get("severity")
        output.append(
            RegulationInsightBullet(
                title=title,
                description=description[:450],
                severity=str(severity) if severity is not None else _severity_for_text(description),
            )
        )
        if len(output) >= limit:
            break
    return output


def _coerce_key_dates(raw: Any, limit: int) -> list[RegulationKeyDate]:
    if not isinstance(raw, list):
        return []
    output: list[RegulationKeyDate] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "تاريخ نظامي مهم").strip() or "تاريخ نظامي مهم"
        value = str(item.get("value") or "").strip()
        if not value:
            continue
        source = item.get("source")
        output.append(
            RegulationKeyDate(
                label=label[:120],
                value=value[:120],
                source=str(source)[:120] if source else None,
            )
        )
        if len(output) >= limit:
            break
    return output


def _coerce_citations(raw: Any, limit: int) -> list[RegulationCitation]:
    if not isinstance(raw, list):
        return []
    output: list[RegulationCitation] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        snippet = str(item.get("snippet") or "").strip()
        if not snippet:
            continue
        section_ref = item.get("section_ref")
        relevance = item.get("relevance")
        parsed_relevance: float | None = None
        if isinstance(relevance, (int, float)):
            parsed_relevance = float(relevance)
        output.append(
            RegulationCitation(
                snippet=snippet[:400],
                section_ref=str(section_ref)[:120] if section_ref else None,
                relevance=parsed_relevance,
            )
        )
        if len(output) >= limit:
            break
    return output


@router.post(
    "/regulations/summary-analysis", response_model=RegulationSummaryAnalysisResponse
)
async def regulation_summary_analysis(
    payload: RegulationSummaryAnalysisRequest,
) -> RegulationSummaryAnalysisResponse:
    language_code = (payload.language_code or "ar").lower()
    text = _normalize_text(payload.regulation_text)
    if not text:
        return RegulationSummaryAnalysisResponse(
            status="error",
            summary="",
            obligations=[],
            risk_flags=[],
            key_dates=[],
            citations=[],
            method="regulation_summary_analysis_v1",
            warnings=["regulation_text is required."],
            error_code="validation_error",
        )

    max_chars = max(500, min(payload.max_source_chars, settings.reg_insights_max_source_chars))
    working_text = _truncate(text, max_chars)
    sentences = _split_sentences(working_text)
    top_summary_sentences = sentences[:4] if sentences else [working_text[:300]]

    obligations_candidates = [
        sentence
        for sentence in sentences
        if any(
            re.search(pattern, sentence, flags=re.IGNORECASE)
            for pattern in _OBLIGATION_PATTERNS
        )
    ]
    risk_candidates = [
        sentence
        for sentence in sentences
        if any(
            re.search(pattern, sentence, flags=re.IGNORECASE)
            for pattern in _RISK_PATTERNS
        )
    ]

    fallback = {
        "summary": f"ملخص تنظيمي: {' '.join(top_summary_sentences)[:900]}",
        "obligations": [
            bullet.model_dump()
            for bullet in _build_bullets(obligations_candidates[:8], "التزام تنظيمي", 8)
        ],
        "risk_flags": [
            bullet.model_dump() for bullet in _build_bullets(risk_candidates[:8], "مؤشر مخاطر", 8)
        ],
        "key_dates": [item.model_dump() for item in _extract_dates(working_text)],
        "citations": [item.model_dump() for item in _extract_citations(sentences)],
    }

    warnings: list[str] = []
    method = "heuristic_structured_v1"

    llm_result, llm_error = await try_llm_json(
        system_prompt=(
            "أنت محلل قانوني متخصص. أرجع JSON فقط بالمفاتيح التالية:\n"
            "- summary: ملخص تنفيذي موجز للنظام (فقرة واحدة).\n"
            "- obligations: قائمة من البنود، كل بند له:\n"
            "    title: عنوان قصير ومحدد يصف الالتزام (3-6 كلمات، لا تستخدم 'التزام تنظيمي' عامًا).\n"
            "    description: وصف موجز للالتزام بجملة أو جملتين.\n"
            "- risk_flags: قائمة من البنود، كل بند له:\n"
            "    title: عنوان قصير ومحدد يصف المخاطرة (3-6 كلمات، لا تستخدم 'مؤشر مخاطر' عامًا).\n"
            "    description: وصف موجز للمخاطرة بجملة أو جملتين.\n"
            "- key_dates: قائمة من {label, value} للتواريخ والمدد الزمنية المذكورة.\n"
            "- citations: قائمة من {snippet} لأبرز المقتطفات النصية الداعمة.\n"
            "اجعل كل عنوان فريدًا ومحددًا بما يعكس مضمون البند تحديدًا."
        ),
        user_payload={
            "language_code": language_code,
            "regulation_title": payload.regulation_title,
            "source_metadata": payload.source_metadata or {},
            "regulation_text": working_text,
        },
        timeout_seconds=settings.llm_timeout_seconds,
    )

    if llm_result:
        method = "llm_structured_v1"
        generated = {
            "summary": str(llm_result.get("summary") or fallback["summary"]),
            "obligations": llm_result.get("obligations") or fallback["obligations"],
            "risk_flags": llm_result.get("risk_flags") or fallback["risk_flags"],
            "key_dates": llm_result.get("key_dates") or fallback["key_dates"],
            "citations": llm_result.get("citations") or fallback["citations"],
        }
    else:
        generated = fallback
        if llm_error and llm_error not in {"llm_disabled", "llm_not_configured"}:
            warnings.append(llm_error)

    obligations = _coerce_bullets(generated["obligations"], "التزام تنظيمي", 10)
    risk_flags = _coerce_bullets(generated["risk_flags"], "مؤشر مخاطر", 10)
    key_dates = _coerce_key_dates(generated["key_dates"], 10)
    citations = _coerce_citations(generated["citations"], 12)

    if not obligations:
        obligations = _build_bullets(sentences[:4], "التزام تنظيمي", 4)
    if not risk_flags:
        risk_flags = _build_bullets(risk_candidates[:4] or sentences[:2], "مؤشر مخاطر", 4)

    return RegulationSummaryAnalysisResponse(
        status="ok",
        summary=str(generated["summary"]),
        obligations=obligations,
        risk_flags=risk_flags,
        key_dates=key_dates,
        citations=citations,
        method=method,
        warnings=warnings,
        error_code=None,
    )


@router.post(
    "/regulations/amendment-impact", response_model=RegulationAmendmentImpactResponse
)
async def regulation_amendment_impact(
    payload: RegulationAmendmentImpactRequest,
) -> RegulationAmendmentImpactResponse:
    old_text = _normalize_text(payload.old_text)
    new_text = _normalize_text(payload.new_text)
    if not old_text or not new_text:
        return RegulationAmendmentImpactResponse(
            status="error",
            what_changed=[],
            legal_impact=[],
            affected_parties=[],
            citations=[],
            method="regulation_amendment_impact_v1",
            warnings=["old_text and new_text are required."],
            error_code="validation_error",
        )

    max_chars = max(500, min(payload.max_source_chars, settings.reg_impact_max_source_chars))
    old_working = _truncate(old_text, max_chars)
    new_working = _truncate(new_text, max_chars)

    old_sentences = _split_sentences(old_working)
    new_sentences = _split_sentences(new_working)
    old_set = set(old_sentences)
    new_set = set(new_sentences)

    added_sentences = _dedupe_items([s for s in new_sentences if s not in old_set], 10)
    removed_sentences = _dedupe_items([s for s in old_sentences if s not in new_set], 10)

    ratio = SequenceMatcher(None, old_working[:8000], new_working[:8000]).ratio()

    what_changed_fallback: list[RegulationInsightBullet] = []
    for sentence in added_sentences[:6]:
        what_changed_fallback.append(
            RegulationInsightBullet(
                title=f"إضافة: {_derive_bullet_title(sentence, 'نص مضاف')}",
                description=f"تمت إضافة نص: {sentence[:380]}",
                severity=_severity_for_text(sentence),
            )
        )
    for sentence in removed_sentences[:4]:
        what_changed_fallback.append(
            RegulationInsightBullet(
                title=f"حذف: {_derive_bullet_title(sentence, 'نص محذوف أو معدل')}",
                description=f"تم حذف/تعديل نص: {sentence[:380]}",
                severity=_severity_for_text(sentence),
            )
        )

    legal_impact_fallback: list[RegulationInsightBullet] = []
    impact_sentence_pool = added_sentences[:5] + removed_sentences[:3]
    if not impact_sentence_pool:
        impact_sentence_pool = new_sentences[:3]

    for sentence in impact_sentence_pool[:6]:
        legal_impact_fallback.append(
            RegulationInsightBullet(
                title=f"أثر على: {_derive_bullet_title(sentence, 'التطبيق النظامي')}",
                description=f"قد يؤثر هذا التغيير على التطبيق النظامي: {sentence[:360]}",
                severity=_severity_for_text(sentence),
            )
        )

    affected_parties_fallback: list[RegulationInsightBullet] = []
    combined_text = f"{old_working}\n{new_working}"
    for label, patterns in _AFFECTED_PATTERNS:
        if any(re.search(pattern, combined_text, flags=re.IGNORECASE) for pattern in patterns):
            affected_parties_fallback.append(
                RegulationInsightBullet(
                    title=label,
                    description=f"التعديل قد يؤثر على {label} وفق النصوص المعدلة.",
                    severity="medium",
                )
            )

    if not affected_parties_fallback:
        affected_parties_fallback.append(
            RegulationInsightBullet(
                title="الأطراف الخاضعة للنظام",
                description="يرجى مراجعة نطاق التطبيق لتحديد الأطراف المتأثرة بشكل أدق.",
                severity="low",
            )
        )

    citations_fallback = _extract_citations(added_sentences + removed_sentences + new_sentences, limit=12)

    fallback = {
        "what_changed": [item.model_dump() for item in what_changed_fallback],
        "legal_impact": [item.model_dump() for item in legal_impact_fallback],
        "affected_parties": [item.model_dump() for item in affected_parties_fallback],
        "citations": [item.model_dump() for item in citations_fallback],
    }

    warnings: list[str] = []
    if ratio > 0.985:
        warnings.append("التغييرات النصية طفيفة جداً بين النسختين.")

    method = "heuristic_structured_v1"
    llm_result, llm_error = await try_llm_json(
        system_prompt=(
            "أنت محلل تشريعي متخصص في مقارنة النصوص القانونية. أرجع JSON فقط بالمفاتيح التالية:\n"
            "- what_changed: قائمة من البنود، كل بند له:\n"
            "    title: عنوان قصير ومحدد يصف ما تغيّر (3-6 كلمات، مثل: 'إضافة شرط التسجيل').\n"
            "    description: وصف موجز للتغيير.\n"
            "- legal_impact: قائمة من البنود، كل بند له:\n"
            "    title: عنوان قصير يصف الأثر القانوني المحدد (3-6 كلمات، مثل: 'تشديد متطلبات الترخيص').\n"
            "    description: وصف موجز للأثر.\n"
            "- affected_parties: قائمة من البنود، كل بند له:\n"
            "    title: اسم الجهة أو الفئة المتأثرة تحديدًا (مثل: 'المحامون المرخصون').\n"
            "    description: كيف يؤثر عليهم التعديل.\n"
            "- citations: قائمة من {snippet} لأبرز المقتطفات النصية الداعمة.\n"
            "اجعل كل عنوان فريدًا ومحددًا بما يعكس مضمون البند، لا تستخدم عناوين عامة."
        ),
        user_payload={
            "regulation_title": payload.regulation_title,
            "from_version_label": payload.from_version_label,
            "to_version_label": payload.to_version_label,
            "language_code": payload.language_code,
            "diff_summary": payload.diff_summary or {},
            "old_text": old_working,
            "new_text": new_working,
        },
        timeout_seconds=settings.llm_timeout_seconds,
    )

    if llm_result:
        method = "llm_structured_v1"
        generated = {
            "what_changed": llm_result.get("what_changed") or fallback["what_changed"],
            "legal_impact": llm_result.get("legal_impact") or fallback["legal_impact"],
            "affected_parties": llm_result.get("affected_parties")
            or fallback["affected_parties"],
            "citations": llm_result.get("citations") or fallback["citations"],
        }
    else:
        generated = fallback
        if llm_error and llm_error not in {"llm_disabled", "llm_not_configured"}:
            warnings.append(llm_error)

    return RegulationAmendmentImpactResponse(
        status="ok",
        what_changed=_coerce_bullets(generated["what_changed"], "تغيير تشريعي", 12),
        legal_impact=_coerce_bullets(generated["legal_impact"], "أثر قانوني", 12),
        affected_parties=_coerce_bullets(generated["affected_parties"], "طرف متأثر", 10),
        citations=_coerce_citations(generated["citations"], 12),
        method=method,
        warnings=warnings,
        error_code=None,
    )
