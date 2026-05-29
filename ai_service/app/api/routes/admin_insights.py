"""
Admin AI intelligence endpoints.

Composes the deterministic risk signals the backend assembles (reusing its
command-center queries) into an explainable per-case risk score, an org-level
executive summary, and a review-prioritization ranking.

Design (matches regulation_insights.py):
- The deterministic baseline ALWAYS runs and is authoritative for scores,
  urgency, evidence, and recommended actions.
- Optional LLM enrichment (off by default via LLM_PROVIDER=heuristic) only adds
  a narrative `rationale`/`headline`/`bullets`. It never overrides scores.
- Every response carries `method`, `confidence`, and `warnings`.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas.requests import (
    CaseRiskProfileRequest,
    CaseRiskSignalInput,
    OrgIntelligenceSummaryRequest,
    ReviewPrioritizationRequest,
)
from app.api.schemas.responses import (
    CaseRiskEvidence,
    CaseRiskProfileResponse,
    CaseRiskRecommendedAction,
    OrgIntelligenceSummaryResponse,
    ReviewPrioritizationItem,
    ReviewPrioritizationResponse,
)
from app.config import settings
from app.core.llm_json import LLM_DISABLED_REASONS, try_llm_json

router = APIRouter()


# ── Deterministic case-risk scoring ───────────────────────────────────────────
# Additive model, clamped to 0..100. Inputs come pre-computed from the backend's
# command-center queries; this module owns the weights so they stay unit-testable.

_DOC_BUCKET_LABEL = "مخاطر في المستندات"


def _score_case(signals: CaseRiskSignalInput) -> tuple[int, list[CaseRiskEvidence], list[str]]:
    evidence: list[CaseRiskEvidence] = []
    fired: list[str] = []
    score = 0

    if signals.overdue_hearing:
        contribution = min(35, 20 + max(0, signals.days_overdue) * 2)
        score += contribution
        fired.append("overdue_hearing")
        evidence.append(
            CaseRiskEvidence(
                signal="overdue_hearing",
                label="جلسة متأخرة",
                severity="critical",
                contribution=contribution,
                detail=(
                    f"الجلسة متأخرة بـ {signals.days_overdue} يومًا"
                    if signals.days_overdue
                    else "الجلسة متأخرة"
                ),
            )
        )
    elif signals.hearing_this_week:
        score += 15
        fired.append("hearing_this_week")
        evidence.append(
            CaseRiskEvidence(
                signal="hearing_this_week",
                label="جلسة هذا الأسبوع",
                severity="high",
                contribution=15,
                detail="توجد جلسة خلال الأسبوع الحالي",
            )
        )

    if signals.stale:
        extra = max(0, signals.days_stale - signals.stale_threshold_days)
        contribution = min(20, 10 + extra // 3)
        score += contribution
        fired.append("stale")
        evidence.append(
            CaseRiskEvidence(
                signal="stale",
                label="بحاجة إلى تحديث",
                severity="medium",
                contribution=contribution,
                detail=f"آخر تحديث منذ {signals.days_stale} يومًا",
            )
        )

    if signals.unassigned:
        score += 15
        fired.append("unassigned")
        evidence.append(
            CaseRiskEvidence(
                signal="unassigned",
                label="قضية غير مُسندة",
                severity="high",
                contribution=15,
                detail="لا يوجد محامٍ مسؤول عن القضية",
            )
        )

    if signals.unverified_links > 0:
        contribution = min(15, signals.unverified_links * 3)
        score += contribution
        fired.append("unverified_links")
        evidence.append(
            CaseRiskEvidence(
                signal="unverified_links",
                label="روابط أنظمة غير مُراجَعة",
                severity="medium",
                contribution=contribution,
                detail=f"{signals.unverified_links} رابط بحاجة لمراجعة",
            )
        )

    if signals.recent_regulation_update:
        score += 10
        fired.append("recent_regulation_update")
        evidence.append(
            CaseRiskEvidence(
                signal="recent_regulation_update",
                label="تحديث نظامي حديث",
                severity="medium",
                contribution=10,
                detail="صدر تحديث نظامي مرتبط بالقضية مؤخرًا",
            )
        )

    if signals.failed_extraction or signals.document_risk:
        score += 10
        key = "failed_extraction" if signals.failed_extraction else "document_risk"
        fired.append(key)
        evidence.append(
            CaseRiskEvidence(
                signal=key,
                label=_DOC_BUCKET_LABEL,
                severity="high" if signals.failed_extraction else "medium",
                contribution=10,
                detail=(
                    "فشل استخراج محتوى أحد المستندات"
                    if signals.failed_extraction
                    else "توجد مؤشرات مخاطر في مستندات القضية"
                ),
            )
        )

    if signals.lawyer_overloaded:
        score += 10
        fired.append("lawyer_overloaded")
        evidence.append(
            CaseRiskEvidence(
                signal="lawyer_overloaded",
                label="عبء مرتفع على المحامي",
                severity="medium",
                contribution=10,
                detail="المحامي المسؤول لديه عبء قضايا مرتفع",
            )
        )

    score = max(0, min(100, score))
    # Highest-contribution signals first so the UI surfaces the strongest driver.
    evidence.sort(key=lambda e: e.contribution, reverse=True)
    return score, evidence, fired


def _urgency_for(score: int, overdue: bool) -> str:
    if score >= 70:
        return "critical"
    if score >= 45:
        urgency = "high"
    elif score >= 20:
        urgency = "medium"
    else:
        urgency = "low"
    # Hard override: an overdue hearing is never less than "high".
    if overdue and urgency in {"medium", "low"}:
        return "high"
    return urgency


def _confidence_for(signals: CaseRiskSignalInput, fired: list[str], ai_healthy: bool) -> str:
    if not ai_healthy:
        return "low"
    if not signals.has_activity and not signals.has_documents:
        return "low"
    if len(fired) >= 3:
        return "high"
    return "medium"


def _recommended_actions(signals: CaseRiskSignalInput) -> list[CaseRiskRecommendedAction]:
    actions: list[CaseRiskRecommendedAction] = []
    if signals.unassigned:
        actions.append(
            CaseRiskRecommendedAction(action="assign_owner", label="إسناد محامٍ للقضية", target="case")
        )
    if signals.overdue_hearing or signals.hearing_this_week:
        actions.append(
            CaseRiskRecommendedAction(action="schedule_hearing", label="مراجعة موعد الجلسة", target="case")
        )
    if signals.unverified_links > 0:
        actions.append(
            CaseRiskRecommendedAction(action="review_links", label="مراجعة روابط الأنظمة", target="linking")
        )
    if signals.failed_extraction or signals.document_risk:
        actions.append(
            CaseRiskRecommendedAction(action="review_documents", label="مراجعة المستندات", target="documents")
        )
    if signals.stale:
        actions.append(
            CaseRiskRecommendedAction(action="update_case", label="تحديث حالة القضية", target="case")
        )
    if signals.lawyer_overloaded:
        actions.append(
            CaseRiskRecommendedAction(action="rebalance_workload", label="إعادة توزيع العبء", target="lawyer")
        )
    return actions


@router.post("/admin/case-risk-profile", response_model=CaseRiskProfileResponse)
async def admin_case_risk_profile(payload: CaseRiskProfileRequest) -> CaseRiskProfileResponse:
    signals = payload.signals or CaseRiskSignalInput()
    score, evidence, fired = _score_case(signals)
    urgency = _urgency_for(score, signals.overdue_hearing)
    confidence = _confidence_for(signals, fired, payload.ai_healthy)
    actions = _recommended_actions(signals)

    warnings: list[str] = []
    method = "heuristic_risk_v1"
    rationale: str | None = None

    # Optional LLM narrative — score/urgency/evidence remain authoritative.
    llm_result, llm_error = await try_llm_json(
        system_prompt=(
            "أنت مساعد قانوني تنفيذي. لديك إشارات مخاطر محسوبة لقضية. "
            "أرجع JSON فقط بالمفتاح rationale: فقرة عربية قصيرة (جملتان كحد أقصى) "
            "تشرح سبب أهمية متابعة القضية الآن دون اختلاق وقائع جديدة."
        ),
        user_payload={
            "language_code": payload.language_code,
            "case_title": payload.title,
            "case_type": payload.case_type,
            "score": score,
            "urgency": urgency,
            "signals": fired,
            "case_summary": payload.case_summary,
        },
        timeout_seconds=settings.llm_timeout_seconds,
    )
    if llm_result:
        method = "llm_risk_v1"
        candidate = llm_result.get("rationale")
        if isinstance(candidate, str) and candidate.strip():
            rationale = candidate.strip()[:600]
    elif llm_error and llm_error not in LLM_DISABLED_REASONS:
        warnings.append(llm_error)

    return CaseRiskProfileResponse(
        status="ok",
        case_id=payload.case_id,
        score=score,
        urgency=urgency,
        confidence=confidence,
        signals=fired,
        evidence=evidence,
        recommended_actions=actions,
        rationale=rationale,
        method=method,
        warnings=warnings,
        error_code=None,
    )


@router.post("/admin/org-intelligence-summary", response_model=OrgIntelligenceSummaryResponse)
async def admin_org_intelligence_summary(
    payload: OrgIntelligenceSummaryRequest,
) -> OrgIntelligenceSummaryResponse:
    counts = payload.urgency_counts or {}
    critical = int(counts.get("critical", 0) or 0)
    high = int(counts.get("high", 0) or 0)
    medium = int(counts.get("medium", 0) or 0)
    low = int(counts.get("low", 0) or 0)

    aggregate_risk = {
        "critical": critical,
        "high": high,
        "medium": medium,
        "low": low,
        "total": payload.total_active_cases,
        "average_score": round(payload.average_score, 1),
    }
    workload_signals = {
        "overloaded_lawyers": payload.overloaded_lawyers,
        "unassigned_cases": payload.unassigned_cases,
        "document_risk_cases": payload.document_risk_cases,
        "regulation_impact_cases": payload.regulation_impact_cases,
    }

    # Deterministic headline + bullets (authoritative fallback). Calm, status
    # framing — this is an organization overview, not an alarm.
    priority = critical + high
    headline = (
        f"{payload.total_active_cases} قضية نشطة، {priority} منها بحاجة إلى متابعة."
        if priority
        else f"{payload.total_active_cases} قضية نشطة، والوضع مستقر حاليًا."
    )
    bullets: list[str] = []
    if critical:
        bullets.append(f"{critical} قضية ذات أولوية للمتابعة أولًا.")
    if payload.unassigned_cases:
        bullets.append(f"{payload.unassigned_cases} قضية بحاجة إلى إسناد محامٍ.")
    if payload.overloaded_lawyers:
        bullets.append(f"{payload.overloaded_lawyers} محامٍ لديه عبء قضايا مرتفع.")
    if payload.document_risk_cases:
        bullets.append(f"{payload.document_risk_cases} قضية بحاجة إلى مراجعة مستندات.")
    if payload.regulation_impact_cases:
        bullets.append(f"{payload.regulation_impact_cases} قضية متأثرة بتحديثات نظامية حديثة.")
    for case in payload.top_cases[:3]:
        label = case.case_number or case.title or f"#{case.case_id}"
        reason = f" — {case.top_reason}" if case.top_reason else ""
        bullets.append(f"الأعلى أولوية: {label} ({case.score}){reason}")
    if not bullets:
        bullets.append("وضع المنظمة مستقر حاليًا، لا توجد بنود بحاجة لمتابعة عاجلة.")

    warnings: list[str] = []
    method = "heuristic_org_summary_v1"
    confidence = "low" if not payload.ai_healthy else "medium"

    llm_result, llm_error = await try_llm_json(
        system_prompt=(
            "أنت مدير تنفيذي قانوني. لديك مؤشرات مخاطر مجمّعة لمنظمة. أرجع JSON فقط "
            "بالمفتاحين headline (جملة واحدة) و bullets (قائمة من 3-5 نقاط عربية موجزة) "
            "تلخّص أولويات الأدمن دون اختلاق أرقام جديدة."
        ),
        user_payload={
            "language_code": payload.language_code,
            "aggregate_risk": aggregate_risk,
            "workload_signals": workload_signals,
            "top_cases": [c.model_dump() for c in payload.top_cases[:5]],
        },
        timeout_seconds=settings.llm_timeout_seconds,
    )
    if llm_result:
        method = "llm_org_summary_v1"
        llm_headline = llm_result.get("headline")
        if isinstance(llm_headline, str) and llm_headline.strip():
            headline = llm_headline.strip()[:300]
        llm_bullets = llm_result.get("bullets")
        if isinstance(llm_bullets, list):
            coerced = [str(b).strip()[:300] for b in llm_bullets if str(b).strip()]
            if coerced:
                bullets = coerced[:6]
    elif llm_error and llm_error not in LLM_DISABLED_REASONS:
        warnings.append(llm_error)

    return OrgIntelligenceSummaryResponse(
        status="ok",
        headline=headline,
        bullets=bullets,
        aggregate_risk=aggregate_risk,
        workload_signals=workload_signals,
        confidence=confidence,
        method=method,
        warnings=warnings,
        error_code=None,
    )


@router.post("/admin/review-prioritization", response_model=ReviewPrioritizationResponse)
async def admin_review_prioritization(
    payload: ReviewPrioritizationRequest,
) -> ReviewPrioritizationResponse:
    ranked: list[ReviewPrioritizationItem] = []
    for item in payload.items:
        reasons: list[str] = []
        priority = 0.0

        priority += item.unverified_links * 2.0
        if item.unverified_links:
            reasons.append(f"{item.unverified_links} رابط غير مُراجَع")

        if item.max_link_score:
            priority += item.max_link_score * 10.0
            reasons.append("درجة تطابق مرتفعة")

        priority += item.evidence_count * 1.0
        priority += item.document_support * 1.0
        if item.document_support:
            reasons.append("مدعوم بمستندات")

        if item.recent_regulation_update:
            priority += 5.0
            reasons.append("تحديث نظامي حديث")

        if item.case_risk_score:
            priority += item.case_risk_score * 0.2
            if item.case_risk_score >= 45:
                reasons.append("قضية عالية الخطورة")

        ranked.append(
            ReviewPrioritizationItem(
                case_id=item.case_id,
                case_number=item.case_number,
                title=item.title,
                priority_score=round(priority, 2),
                unverified_links=item.unverified_links,
                reasons=reasons,
            )
        )

    ranked.sort(key=lambda r: r.priority_score, reverse=True)

    return ReviewPrioritizationResponse(
        status="ok",
        items=ranked,
        method="heuristic_review_priority_v1",
        confidence="high",
        warnings=[],
        error_code=None,
    )
