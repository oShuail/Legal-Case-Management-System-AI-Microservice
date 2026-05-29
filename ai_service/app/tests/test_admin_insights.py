"""
Contract + deterministic-scoring tests for the admin AI intelligence endpoints.

These run with EMBEDDINGS_PROVIDER=fake and LLM_PROVIDER=heuristic (defaults), so
they exercise the deterministic baseline and the LLM-unavailable fallback path.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ── case-risk-profile ─────────────────────────────────────────────────────────


def _profile(signals: dict, **kwargs):
    body = {"case_id": kwargs.pop("case_id", 1), "signals": signals, **kwargs}
    r = client.post("/admin/case-risk-profile", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_case_profile_contract_fields_present():
    data = _profile({})
    for key in ("score", "urgency", "confidence", "signals", "evidence", "method", "warnings"):
        assert key in data
    assert data["method"] == "heuristic_risk_v1"  # LLM off by default
    assert isinstance(data["warnings"], list)
    # llm_disabled is expected and must NOT surface as a warning.
    assert "llm_disabled" not in data["warnings"]


def test_case_profile_empty_signals_is_low_risk():
    data = _profile({})
    assert data["score"] == 0
    assert data["urgency"] == "low"


def test_overdue_hearing_forces_at_least_high():
    # Only an overdue hearing with a moderate score should still be >= high.
    data = _profile({"overdue_hearing": True, "days_overdue": 5})
    assert "overdue_hearing" in data["signals"]
    assert data["urgency"] in ("high", "critical")
    overdue_ev = next(e for e in data["evidence"] if e["signal"] == "overdue_hearing")
    assert overdue_ev["severity"] == "critical"
    assert overdue_ev["contribution"] > 0


def test_combined_signals_reach_critical_and_high_confidence():
    data = _profile(
        {
            "overdue_hearing": True,
            "days_overdue": 10,
            "unassigned": True,
            "stale": True,
            "days_stale": 30,
            "stale_threshold_days": 14,
            "unverified_links": 5,
        }
    )
    assert data["score"] >= 70
    assert data["urgency"] == "critical"
    assert data["confidence"] == "high"  # >= 3 signals fired + ai healthy
    # recommended actions should include assigning an owner + reviewing links.
    actions = {a["action"] for a in data["recommended_actions"]}
    assert "assign_owner" in actions
    assert "review_links" in actions


def test_unverified_links_contribution_is_capped():
    data = _profile({"unverified_links": 99})
    ev = next(e for e in data["evidence"] if e["signal"] == "unverified_links")
    assert ev["contribution"] == 15  # min(99*3, 15)


def test_failed_extraction_and_document_risk_counted():
    data = _profile({"failed_extraction": True})
    assert "failed_extraction" in data["signals"]
    data2 = _profile({"document_risk": True})
    assert "document_risk" in data2["signals"]


def test_recent_regulation_update_signal():
    data = _profile({"recent_regulation_update": True})
    assert "recent_regulation_update" in data["signals"]
    assert data["score"] == 10


def test_lawyer_overloaded_signal():
    data = _profile({"lawyer_overloaded": True})
    assert "lawyer_overloaded" in data["signals"]


def test_degraded_ai_health_lowers_confidence():
    data = _profile({"unassigned": True}, ai_healthy=False)
    assert data["confidence"] == "low"


# ── org-intelligence-summary ──────────────────────────────────────────────────


def test_org_summary_contract_and_aggregate():
    body = {
        "organization_id": 1,
        "total_active_cases": 12,
        "urgency_counts": {"critical": 2, "high": 3, "medium": 4, "low": 3},
        "average_score": 41.5,
        "unassigned_cases": 4,
        "overloaded_lawyers": 1,
        "document_risk_cases": 2,
        "regulation_impact_cases": 1,
        "top_cases": [
            {"case_id": 9, "case_number": "C-9", "score": 88, "urgency": "critical", "top_reason": "جلسة متأخرة"}
        ],
    }
    r = client.post("/admin/org-intelligence-summary", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    for key in ("headline", "bullets", "aggregate_risk", "workload_signals", "confidence", "method", "warnings"):
        assert key in data
    assert data["method"] == "heuristic_org_summary_v1"
    assert data["aggregate_risk"]["critical"] == 2
    assert data["aggregate_risk"]["total"] == 12
    assert data["workload_signals"]["unassigned_cases"] == 4
    assert len(data["bullets"]) >= 1
    assert "llm_disabled" not in data["warnings"]


def test_org_summary_empty_has_safe_bullet():
    body = {"organization_id": 1, "total_active_cases": 0, "urgency_counts": {}}
    r = client.post("/admin/org-intelligence-summary", json=body)
    assert r.status_code == 200
    data = r.json()
    assert len(data["bullets"]) >= 1


# ── review-prioritization ─────────────────────────────────────────────────────


def test_review_prioritization_ranks_by_priority():
    body = {
        "items": [
            {"case_id": 1, "unverified_links": 1, "max_link_score": 0.2},
            {"case_id": 2, "unverified_links": 8, "max_link_score": 0.9, "recent_regulation_update": True, "case_risk_score": 80},
            {"case_id": 3, "unverified_links": 3},
        ]
    }
    r = client.post("/admin/review-prioritization", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    for key in ("items", "method", "confidence", "warnings"):
        assert key in data
    ids = [it["case_id"] for it in data["items"]]
    assert ids[0] == 2  # highest priority first
    assert data["items"][0]["priority_score"] >= data["items"][1]["priority_score"]


def test_review_prioritization_empty():
    r = client.post("/admin/review-prioritization", json={"items": []})
    assert r.status_code == 200
    assert r.json()["items"] == []
