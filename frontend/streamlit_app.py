"""Reviewer console for the Aptino claim decision engine.

Layout (top to bottom): sidebar (API + legend) -> three tabs:
  1. Analyze a claim   - pick / build / paste / upload a case, run it, read the verdict,
                         then audit it in the reviewer panel (claim -> citation -> chunk -> verdict,
                         assumptions, unmodelled risks, agent workflow) and record a review
  2. How it works      - the pipeline, retrieval, verification, decision statuses
  3. Evaluation        - stored metrics + ablation + a live batch run against the API
"""
from __future__ import annotations

import html
import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Aptino Claim Decision Engine", layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------
# Static vocab: everything the UI says in plain English lives here.
# --------------------------------------------------------------------------

GREEN, AMBER, RED, GREY, BLUE = "#1a7f37", "#b7791f", "#cf222e", "#6e7781", "#0969da"

DECISIONS = {
    "ADMISSIBLE": (GREEN, "Admissible", "The policy covers this claim in full."),
    "ADMISSIBLE_WITH_LIMITS": (AMBER, "Admissible, with limits", "The treatment is covered, but some amounts are reduced by policy limits."),
    "PARTIALLY_ADMISSIBLE": (AMBER, "Partially admissible", "Part of the claim is covered; another part is not payable."),
    "NOT_ADMISSIBLE": (RED, "Not admissible", "The policy rules this claim out."),
    "NEEDS_REVIEW": (GREY, "Needs review", "The system is abstaining instead of guessing: key evidence is missing or the policy is unclear. A person must decide."),
}

DIMENSIONS = {
    "initial_waiting_period": ("Initial waiting period", "Had the policy been in force long enough?"),
    "named_disease_first_year_waiting_period": ("First-year wait for listed illnesses", "Is this a listed illness that needs a one-year wait?"),
    "pre_existing_disease_waiting_period": ("Pre-existing disease wait", "Has enough continuous cover built up for a pre-existing condition?"),
    "hospital_definition": ("Facility qualifies as a hospital", "Does the facility meet the policy's definition of a hospital?"),
    "domiciliary_treatment_conditions": ("Home-treatment conditions", "Were the conditions for treatment at home met?"),
    "day_care_less_than_24h": ("Short-stay (day-care) treatment", "Can a stay under 24 hours be covered?"),
    "pre_post_hospitalization_window": ("Pre / post hospitalization window", "Are the before/after expenses inside the allowed days?"),
    "cosmetic_exclusion": ("Cosmetic treatment exclusion", "Is the treatment cosmetic or aesthetic?"),
    "experimental_unproven_treatment": ("Experimental / unproven treatment", "Is the treatment experimental, and does the policy exclude it?"),
    "category_sub_limits": ("Expense-category caps", "Do any caps reduce what is payable?"),
    "unmodelled_policy_risk": ("Policy rules not modelled", "Does the policy name this diagnosis in a specific rule the system does not evaluate?"),
}

STATUSES = {
    "SUPPORTS_ADMISSIBLE": ("Passed", GREEN),
    "SUPPORTS_LIMIT": ("Limit applies", AMBER),
    "SUPPORTS_EXCLUSION": ("Blocks the claim", RED),
    "INSUFFICIENT_EVIDENCE": ("Cannot confirm", GREY),
    "NOT_APPLICABLE": ("No cap reached", GREEN),
}

VERDICTS = {"SUPPORTED": ("Supported", GREEN), "UNSUPPORTED": ("Unsupported", RED), "CONTRADICTED": ("Contradicted", RED)}

AGENTS = [
    ("CaseAnalysisAgent", "Case Analysis", "Reads the claim and decides which policy questions apply."),
    ("PolicyEvidenceAgent", "Policy Evidence", "Searches the policy: meaning-based + keyword search, merged, then re-ranked."),
    ("CoverageExclusionAgent", "Coverage & Exclusion", "Checks each question against the retrieved policy text."),
    ("DecisionAgent", "Decision", "Combines all checks into one verdict, with limits and confidence."),
    ("ValidationAgent", "Validation", "Verifies every citation is backed by text that was actually retrieved."),
]

EXPENSE_LABELS = {
    "room": "Room & nursing",
    "doctor_fees": "Doctor / surgeon fees",
    "medicines_diagnostics": "Medicines & diagnostics",
    "pre_hospitalization": "Pre-hospitalization",
    "post_hospitalization": "Post-hospitalization",
    "ambulance": "Ambulance",
}

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.card {border: 1px solid rgba(128,128,128,.28); border-radius: 12px; padding: 16px 18px; margin: 6px 0 14px 0;}
.card h4 {margin: 0 0 6px 0; font-size: 0.8rem; letter-spacing: .06em; text-transform: uppercase; opacity: .65;}
.verdict {border-left: 8px solid var(--c); border-radius: 12px; padding: 18px 22px; margin: 6px 0 14px 0;
          background: color-mix(in srgb, var(--c) 9%, transparent); border-top: 1px solid rgba(128,128,128,.2);
          border-right: 1px solid rgba(128,128,128,.2); border-bottom: 1px solid rgba(128,128,128,.2);}
.verdict .label {font-size: 2rem; font-weight: 700; color: var(--c); line-height: 1.15;}
.verdict .plain {font-size: 1.05rem; margin-top: 4px;}
.verdict .meta {opacity: .7; font-size: .85rem; margin-top: 8px;}
.pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600;
       color: var(--c); border: 1px solid var(--c); background: color-mix(in srgb, var(--c) 10%, transparent); margin: 0 6px 4px 0;}
.chip {display: inline-block; padding: 2px 10px; border-radius: 6px; font-size: .78rem; margin: 0 6px 4px 0;
       background: rgba(128,128,128,.16);}
.check {border: 1px solid rgba(128,128,128,.25); border-left: 6px solid var(--c); border-radius: 10px;
        padding: 10px 14px; margin: 8px 0;}
.check .t {font-weight: 650;}
.check .q {opacity: .65; font-size: .82rem;}
.check .s {margin-top: 4px;}
.quote {border-left: 4px solid rgba(128,128,128,.5); padding: 6px 12px; margin: 8px 0; font-size: .88rem;
        background: rgba(128,128,128,.08); border-radius: 0 8px 8px 0;}
.bar {height: 8px; border-radius: 6px; background: rgba(128,128,128,.22); overflow: hidden;}
.bar > div {height: 100%; background: var(--c);}
.step {display: flex; gap: 12px; align-items: flex-start; padding: 10px 0; border-bottom: 1px dashed rgba(128,128,128,.3);}
.step .n {min-width: 30px; height: 30px; border-radius: 50%; background: var(--c); color: #fff; font-weight: 700;
          display: flex; align-items: center; justify-content: center;}
.flow {display: flex; flex-wrap: wrap; align-items: stretch; gap: 8px; margin: 10px 0 4px 0;}
.flow .box {flex: 1 1 150px; border: 1px solid rgba(128,128,128,.35); border-radius: 10px; padding: 10px 12px; font-size: .86rem;}
.flow .box b {display: block; margin-bottom: 2px;}
.flow .arrow {align-self: center; opacity: .5; font-size: 1.3rem;}
.big {font-size: 1.6rem; font-weight: 700;}
.sub {opacity: .65; font-size: .82rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def esc(x) -> str:
    return html.escape(str(x))


def inr(x) -> str:
    return "n/a" if x is None else f"INR {x:,.0f}"


def pill(text: str, color: str) -> str:
    return f'<span class="pill" style="--c:{color}">{esc(text)}</span>'


@st.cache_data(show_spinner=False)
def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_case_sources() -> dict[str, dict]:
    sources: dict[str, dict] = {}
    for label, rel in (("public", "data/candidate_data/public_test_cases.json"), ("custom", "data/custom_cases/custom_test_cases.json")):
        p = REPO_ROOT / rel
        if p.exists():
            for c in load_json(str(p)):
                sources[f"[{label}] {c['case_id']} - {c.get('treatment', {}).get('diagnosis', '')}"] = c
    return sources


def load_metrics() -> dict | None:
    p = REPO_ROOT / "eval_results" / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def expected_for(case_id: str) -> str | None:
    metrics = load_metrics()
    if not metrics:
        return None
    return next((c["expected_decision"] for c in metrics["per_case"] if c["case_id"] == case_id), None)


def call_api(api_url: str, case: dict) -> dict:
    resp = requests.post(f"{api_url.rstrip('/')}/analyze", json=case, timeout=120)
    resp.raise_for_status()
    return resp.json()


def relevance(score: float | None) -> float:
    return 0.0 if score is None else 1 / (1 + math.exp(-score / 2))


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Backend")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL, label_visibility="collapsed")
    try:
        health = requests.get(f"{api_url.rstrip('/')}/health", timeout=5).json()
        st.markdown(pill("API online", GREEN), unsafe_allow_html=True)
        st.caption(f"{health.get('chunk_count', '?')} policy chunks indexed | rationale writer: "
                   f"{'LLM-assisted' if health.get('llm_provider') != 'offline' else 'offline template'}")
    except Exception:
        st.markdown(pill("API unreachable", RED), unsafe_allow_html=True)
        st.caption("Start the backend or fix the URL above.")

    st.markdown("### What the verdicts mean")
    for key, (color, label, plain) in DECISIONS.items():
        st.markdown(f'{pill(label, color)}<div class="sub" style="margin:-2px 0 8px 2px">{esc(plain)}</div>', unsafe_allow_html=True)

st.title("Aptino Claim Decision Engine")
st.caption("Checks a health-insurance claim against the policy wording, shows exactly which clauses drove the answer, "
           "and refuses to guess when the evidence is not there.")

tab_analyze, tab_how, tab_eval = st.tabs(["Analyze a claim", "How it works", "Evaluation"])

# ==========================================================================
# TAB 1: ANALYZE
# ==========================================================================

def build_case_form() -> dict | None:
    with st.form("build_case"):
        c1, c2, c3 = st.columns(3)
        case_id = c1.text_input("Case ID", "MY-001")
        policy_start = c2.date_input("Policy start date", date(2024, 1, 1))
        claim_date = c3.date_input("Claim date", date(2026, 6, 1))

        c1, c2, c3, c4 = st.columns(4)
        sum_insured = c1.number_input("Sum insured (INR)", 0, 100_000_000, 500_000, step=50_000)
        cont_months = c2.number_input("Continuous cover (months)", 0, 600, 24)
        prior_years = c3.number_input("Prior insurer cover (years)", 0, 40, 0)
        age = c4.number_input("Patient age", 0, 120, 40)

        c1, c2, c3 = st.columns(3)
        hospital = c1.text_input("Hospital name", "City Hospital")
        network = c2.checkbox("Network provider", True)
        reg = c3.selectbox("Hospital registration evidence", ["Not mentioned", "Confirmed registered", "Unresolved (unknown)"])

        c1, c2, c3, c4 = st.columns(4)
        ttype = c1.selectbox("Treatment type", ["inpatient", "day_care", "domiciliary"])
        hours = c2.number_input("Admission hours", 0, 2000, 96)
        diagnosis = c3.text_input("Diagnosis", "Acute appendicitis")
        procedure = c4.text_input("Procedure", "Appendectomy")

        c1, c2, c3, c4 = st.columns(4)
        pre_existing = c1.checkbox("Pre-existing condition")
        experimental = c2.checkbox("Experimental treatment")
        room_unavail = c3.selectbox("Hospital room unavailable? (home treatment)", ["Not stated", "Yes", "No"])
        cannot_move = c4.selectbox("Patient cannot be moved? (home treatment)", ["Not stated", "Yes", "No"])

        st.markdown("**Expenses (INR)**")
        cols = st.columns(6)
        exp = {k: cols[i].number_input(EXPENSE_LABELS[k], 0, 100_000_000, v, step=1000)
               for i, (k, v) in enumerate(zip(EXPENSE_LABELS, [20000, 30000, 60000, 0, 0, 0]))}

        c1, c2 = st.columns(2)
        docs = c1.multiselect("Documents supplied", ["claim_form", "discharge_summary", "itemized_bill", "doctor_prescription",
                                                      "prior_medical_records", "doctor_certificate", "pre_post_expense_records"],
                              default=["claim_form", "discharge_summary", "itemized_bill"])
        with c2:
            known_timing = st.checkbox("I know the pre/post-hospitalization timing")
            t1, t2 = st.columns(2)
            pre_days = t1.number_input("Pre days before admission", 0, 365, 15)
            post_days = t2.number_input("Post days after discharge", 0, 365, 30)
            history = st.checkbox("Prior insurer claim history received", True)

        submitted = st.form_submit_button("Use this case", type="primary")

    if not submitted:
        return st.session_state.get("built_case")

    tri = {"Not stated": None, "Yes": True, "No": False}
    case: dict = {
        "case_id": case_id, "policy_id": "USGIC-CSC-2017-2018",
        "policy_start_date": policy_start.isoformat(), "claim_date": claim_date.isoformat(),
        "sum_insured_inr": sum_insured, "continuous_coverage_months": cont_months,
        "prior_insurer_continuous_years": prior_years, "patient": {"age": age},
        "hospital": {"name": hospital, "network_provider": network},
        "treatment": {"type": ttype, "admission_hours": hours, "diagnosis": diagnosis, "procedure": procedure,
                      "pre_existing": pre_existing, "experimental": experimental},
        "expenses_inr": exp, "documents": docs,
        "task": "Determine whether this claim is admissible under the policy.",
    }
    if ttype == "domiciliary":
        case["treatment"]["hospital_room_unavailable"] = tri[room_unavail]
        case["treatment"]["patient_cannot_be_moved"] = tri[cannot_move]
    if reg != "Not mentioned":
        case["evidence_context"] = {"hospital_registered": True if reg.startswith("Confirmed") else None}
    if known_timing:
        case["expense_timing"] = {"pre_hospitalization_days_before_admission": pre_days,
                                  "post_hospitalization_days_after_discharge": post_days, "same_condition_confirmed": True}
    if prior_years > 0:
        case["prior_policy"] = {"insurer_type": "Indian individual health insurer", "continuous_years": prior_years,
                                "database_and_claim_history_received": history, "previous_sum_insured_inr": sum_insured}
    st.session_state["built_case"] = case
    return case


def case_card(case: dict) -> None:
    t, h, p = case.get("treatment", {}), case.get("hospital", {}), case.get("patient", {})
    exp = case.get("expenses_inr", {})
    flags = []
    if t.get("pre_existing"):
        flags.append(pill("pre-existing", AMBER))
    if t.get("experimental"):
        flags.append(pill("experimental", AMBER))
    if h.get("network_provider"):
        flags.append(pill("network hospital", BLUE))
    docs = "".join(f'<span class="chip">{esc(d)}</span>' for d in case.get("documents", []))
    c1, c2, c3 = st.columns([1.1, 1, 1])
    with c1:
        st.markdown(f'<div class="card"><h4>Treatment</h4><div class="big">{esc(t.get("diagnosis", "-"))}</div>'
                    f'<div>{esc(t.get("procedure", ""))}</div><div class="sub">{esc(t.get("type", ""))} | '
                    f'{esc(t.get("admission_hours", "?"))} h | patient age {esc(p.get("age", "?"))}</div>'
                    f'<div style="margin-top:8px">{"".join(flags)}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="card"><h4>Policy</h4><div class="big">{inr(case.get("sum_insured_inr"))}</div>'
                    f'<div class="sub">sum insured</div><div>Started {esc(case.get("policy_start_date", "?"))}</div>'
                    f'<div>Claimed {esc(case.get("claim_date", "?"))}</div>'
                    f'<div class="sub">{esc(case.get("continuous_coverage_months", 0))} months continuous cover, '
                    f'{esc(case.get("prior_insurer_continuous_years", 0))} prior-insurer years</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="card"><h4>Claim</h4><div class="big">{inr(sum(v for v in exp.values() if isinstance(v, (int, float))))}</div>'
                    f'<div class="sub">total claimed</div><div>{esc(h.get("name", "-"))}</div>'
                    f'<div style="margin-top:8px">{docs}</div></div>', unsafe_allow_html=True)
    if case.get("task"):
        st.markdown(f'<div class="quote"><b>Question for the system:</b> {esc(case["task"])}</div>', unsafe_allow_html=True)


def render_verdict(result: dict, case: dict) -> None:
    color, label, plain = DECISIONS.get(result["decision"], (GREY, result["decision"], ""))
    conf = result.get("confidence", 0)
    expected = expected_for(result.get("case_id", ""))
    match = ""
    if expected:
        ok = expected == result["decision"]
        match = pill("matches labeled expected outcome" if ok else f"differs from labeled outcome ({expected})", GREEN if ok else RED)
    st.markdown(
        f'<div class="verdict" style="--c:{color}"><div class="label">{esc(label)}</div>'
        f'<div class="plain">{esc(plain)}</div>'
        f'<div style="margin-top:12px;max-width:340px"><div class="sub">Confidence {conf:.0%}</div>'
        f'<div class="bar" style="--c:{color}"><div style="width:{conf * 100:.0f}%"></div></div></div>'
        f'<div class="meta">Case {esc(result.get("case_id"))} | validation {esc(result.get("validation", {}).get("status", "?"))} {match}</div></div>',
        unsafe_allow_html=True)


def render_money(result: dict) -> None:
    amt = result.get("amounts")
    if not amt:
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Claimed", inr(amt["claimed_total_inr"]))
    c2.metric("Reduced by policy limits", inr(amt["total_deductions_inr"]))
    payable = amt["estimated_payable_inr"]
    c3.metric("Estimated payable", inr(payable) if payable is not None else "Undecided",
              help="Estimate = claimed minus limit deductions. Optional benefits (daily cash, health check) are not modelled.")
    if amt.get("expense_breakdown_inr"):
        df = pd.DataFrame({"Category": [EXPENSE_LABELS.get(k, k) for k in amt["expense_breakdown_inr"]],
                           "INR": list(amt["expense_breakdown_inr"].values())}).set_index("Category")
        with st.expander("Where the claimed money goes"):
            st.bar_chart(df)
    for lim in result.get("applicable_limits", []):
        st.markdown(f'<div class="check" style="--c:{AMBER}"><div class="t">{esc(lim["description"])}</div>'
                    f'<div class="s">Reduces payable amount by <b>{inr(lim.get("deduction_inr"))}</b></div></div>', unsafe_allow_html=True)


def render_checks(result: dict) -> None:
    findings = result.get("findings")
    if not findings:  # older backend: fall back to plain sentences
        for f in result.get("key_findings", []):
            st.markdown(f"- {f}")
        return
    order = {"SUPPORTS_EXCLUSION": 0, "INSUFFICIENT_EVIDENCE": 1, "SUPPORTS_LIMIT": 2, "SUPPORTS_ADMISSIBLE": 3, "NOT_APPLICABLE": 4}
    st.caption("Each row is one question the policy asks about this claim. Blocking and uncertain checks are listed first.")
    for f in sorted(findings, key=lambda x: order.get(x["status"], 9)):
        title, question = DIMENSIONS.get(f["dimension"], (f["dimension"], ""))
        s_label, s_color = STATUSES.get(f["status"], (f["status"], GREY))
        refs = "".join(f'<span class="chip">{esc(c)}</span>' for c in f.get("chunk_ids", []))
        st.markdown(
            f'<div class="check" style="--c:{s_color}"><div style="float:right">{pill(s_label, s_color)}</div>'
            f'<div class="t">{esc(title)}</div><div class="q">{esc(question)}</div>'
            f'<div class="s">{esc(f["statement"])}</div>'
            + "".join(f'<div class="sub" style="margin-top:2px">Assumes: {esc(a)}</div>' for a in f.get("assumptions", []))
            + f'<div style="margin-top:4px">{refs}</div></div>', unsafe_allow_html=True)


def worst_verdict_by_chunk(result: dict) -> dict[str, str]:
    rank = {"SUPPORTED": 0, "UNSUPPORTED": 1, "CONTRADICTED": 2}
    out: dict[str, str] = {}
    for v in result.get("validation", {}).get("verifications", []):
        cid = v.get("chunk_id")
        if cid and rank[v["verdict"]] >= rank.get(out.get(cid, "SUPPORTED"), 0):
            out[cid] = v["verdict"]
    return out


def render_evidence(result: dict) -> None:
    citations = result.get("citations", [])
    if not citations:
        st.info("No policy citations were produced for this case.")
        return
    used_for: dict[str, list[str]] = {}
    for f in result.get("findings", []):
        for cid in f.get("chunk_ids", []):
            used_for.setdefault(cid, []).append(DIMENSIONS.get(f["dimension"], (f["dimension"],))[0])
    verdicts = worst_verdict_by_chunk(result)
    st.caption("Exact policy text the system relied on. Every citation is traceable to a page and chunk of the supplied PDF, "
               "and carries the verdict of the evidence check (does this chunk really say what the claim says?).")
    for c in citations:
        rel = relevance(c.get("rerank_score"))
        uses = ", ".join(sorted(set(used_for.get(c["chunk_id"], [])))) or "-"
        excerpt = c.get("excerpt")
        v = verdicts.get(c["chunk_id"])
        vpill = pill(*VERDICTS[v]) if v else ""
        st.markdown(
            f'<div class="card"><div style="float:right">{vpill}</div>'
            f'<div><b>{esc(c["section"])}</b> <span class="sub">| page {esc(c["page"])} | {esc(c["chunk_id"])}</span></div>'
            f'<div class="sub">Supports: {esc(c["claim"])} | used for: {esc(uses)}</div>'
            + (f'<div class="quote">{esc(excerpt)}</div>' if excerpt else "")
            + f'<div class="sub" style="max-width:300px">Relevance to the question {rel:.0%}'
              f'</div><div class="bar" style="--c:{BLUE};max-width:300px"><div style="width:{rel * 100:.0f}%"></div></div></div>',
            unsafe_allow_html=True)


def render_missing(result: dict, prominent: bool) -> None:
    missing = result.get("missing_evidence", [])
    if not missing:
        if prominent:
            st.info("Nothing is missing.")
        return
    if prominent:
        st.warning("The system is abstaining. To reach a decision it would need the items below.")
    for m in missing:
        title = DIMENSIONS.get(m["field"], (m["field"],))[0]
        st.markdown(f'<div class="check" style="--c:{GREY}"><div class="t">{esc(title)}</div>'
                    f'<div class="s">{esc(m["reason"])}</div></div>', unsafe_allow_html=True)


def render_workflow(result: dict) -> None:
    """The agent state machine: what each agent read, wrote, and how the shared state changed."""
    trace, handoffs = result.get("trace", []), result.get("handoffs", [])
    total = sum(t.get("elapsed_ms", 0) for t in trace) or 1
    attempts = sorted({t.get("attempt", 1) for t in trace})
    st.caption(f"Five specialised agents pass one typed state object down the line; total {total:,.0f} ms. "
               "Only actions, counts and timings are shown, never hidden reasoning.")
    retries = [h for h in handoffs if h["kind"] == "retry"]
    for h in retries:
        st.warning(f"Retry loop: {h['payload']}")

    used: set[int] = set()
    for attempt in attempts:
        label = "Attempt 1" if attempt == 1 else f"Attempt {attempt} (retry with widened retrieval)"
        st.markdown(f"**{label}**")
        prev: dict = {}
        for t in [x for x in trace if x.get("attempt", 1) == attempt]:
            name, does = next(((n, d) for k, n, d in AGENTS if k == t["agent"]), (t["agent"], ""))
            idx = next((i for i, n in enumerate(AGENTS, 1) if n[0] == t["agent"]), 0)
            snap = t.get("snapshot", {})
            changes = [f"{k} {prev.get(k, 0)} -> {v}" for k, v in snap.items() if v != prev.get(k, 0 if isinstance(v, int) else None)
                       and k not in ("decision", "validation")]
            for k in ("decision", "validation"):
                if snap.get(k) and snap.get(k) != prev.get(k):
                    changes.append(f"{k} = {snap[k]}")
            prev = snap
            reads = "".join(f'<span class="chip">{esc(r)}</span>' for r in t.get("reads", []))
            writes = "".join(f'<span class="chip" style="background:rgba(9,105,218,.16)">{esc(w)}</span>' for w in t.get("writes", []))
            extra = f' | {t["retrieval_count"]} evidence chunks' if t.get("retrieval_count") else ""
            share = t.get("elapsed_ms", 0) / total
            st.markdown(
                f'<div class="step" style="--c:{BLUE}"><div class="n">{idx}</div><div style="flex:1">'
                f'<div><b>{esc(name)}</b> <span class="sub">{esc(does)}</span></div>'
                f'<div class="sub">{esc(t["detail"])}</div>'
                f'<div class="sub" style="margin-top:4px">reads {reads} writes {writes}</div>'
                f'<div class="sub">state change: {esc("; ".join(changes) or "none")}</div>'
                f'<div class="sub">{t["elapsed_ms"]:,.1f} ms{esc(extra)}</div>'
                f'<div class="bar" style="--c:{BLUE};max-width:360px"><div style="width:{max(share * 100, 1):.0f}%"></div></div></div></div>',
                unsafe_allow_html=True)
            for i, h in enumerate(handoffs):
                if i not in used and h["from_agent"] == t["agent"]:
                    used.add(i)
                    color = AMBER if h["kind"] == "retry" else GREY
                    st.markdown(f'<div class="sub" style="margin:2px 0 6px 42px;color:{color}">&darr; hands off to <b>{esc(h["to_agent"])}</b>: {esc(h["payload"])}</div>',
                                unsafe_allow_html=True)
                    break


def _attention_items(result: dict) -> list[tuple[str, str, str]]:
    """(severity colour, title, detail) for everything a reviewer should look at first."""
    items: list[tuple[str, str, str]] = []
    ver = result.get("validation", {}).get("verifications", [])
    for v in ver:
        if v["verdict"] != "SUPPORTED":
            items.append((RED, f'{v["verdict"].title()} claim: {v["claim"]}', v["reason"]))
    for r in result.get("unmodelled_policy_risks", []):
        items.append((AMBER, "Policy rule the system does not model", r["statement"]))
    for m in result.get("missing_evidence", []):
        if m["field"] != "unmodelled_policy_risk":  # already shown as its own item above
            items.append((GREY, f'Missing evidence: {m["field"]}', m["reason"]))
    n_assume = len(result.get("assumptions", []))
    if n_assume:
        items.append((BLUE, f"{n_assume} assumption(s) behind this answer", "The system filled gaps with the assumptions listed below; confirm they hold for this claim."))
    attempts = max((t.get("attempt", 1) for t in result.get("trace", [])), default=1)
    if attempts > 1:
        items.append((BLUE, "Evidence was re-retrieved", "A first-pass citation failed verification; retrieval was widened and the answer recomputed."))
    return items


def render_reviewer_panel(result: dict, case: dict) -> None:
    ver = result.get("validation", {}).get("verifications", [])
    counts = result.get("validation", {}).get("counts", {})

    st.markdown("##### What needs your attention")
    items = _attention_items(result)
    hard = [i for i in items if i[0] in (RED, AMBER)]
    if not hard:
        st.markdown(f'<div class="check" style="--c:{GREEN}"><div class="t">All {len(ver)} claims verified against retrieved policy text</div>'
                    f'<div class="s">No unsupported or contradicted claim, and no unmodelled policy rule flagged.</div></div>', unsafe_allow_html=True)
    for color, title, detail in items:
        st.markdown(f'<div class="check" style="--c:{color}"><div class="t">{esc(title)}</div><div class="s">{esc(detail)}</div></div>', unsafe_allow_html=True)

    st.markdown("##### Claim audit: decision claim, citation, policy chunk, verdict")
    st.caption(f"{counts.get('SUPPORTED', 0)} supported, {counts.get('UNSUPPORTED', 0)} unsupported, {counts.get('CONTRADICTED', 0)} contradicted. "
               "Each row is one statement the system makes; the checks show exactly what was compared with the cited text.")
    order = {"CONTRADICTED": 0, "UNSUPPORTED": 1, "SUPPORTED": 2}
    for v in sorted(ver, key=lambda x: order[x["verdict"]]):
        label, color = VERDICTS[v["verdict"]]
        kind = {"policy_claim": "Policy claim", "limit_claim": "Limit / deduction", "decision_claim": "Decision"}[v["kind"]]
        title = DIMENSIONS.get(v["dimension"], (v["dimension"],))[0] if v["dimension"] != "decision" else "Final decision"
        src = f'{esc(v.get("section"))}, page {esc(v.get("page"))}, chunk {esc(v.get("chunk_id"))}' if v.get("chunk_id") else "derived from the findings above"
        with st.expander(f'{v["claim_id"]} | {label} | {kind}: {title}'):
            st.markdown(f'<div class="check" style="--c:{color}"><div style="float:right">{pill(label, color)}</div>'
                        f'<div class="t">{esc(v["claim"])}</div><div class="q">Source: {src}</div>'
                        f'<div class="s">{esc(v["reason"])}</div></div>', unsafe_allow_html=True)
            for c in v.get("checks", []):
                mark, mcolor = ("passed", GREEN) if c["passed"] else ("failed", RED)
                st.markdown(f'{pill(mark, mcolor)} {esc(c["name"])} <span class="sub">{esc(c.get("detail", ""))}</span>', unsafe_allow_html=True)
            if v.get("excerpt"):
                st.markdown(f'<div class="quote">{esc(v["excerpt"])}</div>', unsafe_allow_html=True)

    st.markdown("##### Assumptions the system made")
    assumptions = result.get("assumptions", [])
    if assumptions:
        st.table(pd.DataFrame([{"Check": DIMENSIONS.get(a["dimension"], (a["dimension"],))[0], "Assumption": a["assumption"]} for a in assumptions]))
    else:
        st.caption("None: every conclusion rests on stated facts and cited text.")

    st.markdown("##### Your decision")
    system_decision = result["decision"]
    with st.form(f"review_{result['case_id']}"):
        stance = st.radio("Your assessment", ["Agree with the system", "Override the decision", "Escalate: more information needed"], horizontal=True)
        override = st.selectbox("Decision you would give (used if you override)", list(DECISIONS), index=list(DECISIONS).index(system_decision))
        c1, c2 = st.columns([1, 2])
        reviewer = c1.text_input("Reviewer name (optional)")
        notes = c2.text_area("Notes / rationale", height=80)
        submitted = st.form_submit_button("Record review", type="primary")
    if submitted:
        record = {
            "case_id": result["case_id"], "system_decision": system_decision, "system_confidence": result["confidence"],
            "reviewer_assessment": stance,
            "reviewer_decision": system_decision if stance.startswith("Agree") else (override if stance.startswith("Override") else None),
            "reviewer": reviewer or None, "notes": notes,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "verification_counts": counts, "open_items": [{"title": t, "detail": d} for c, t, d in items if c in (RED, AMBER, GREY)],
            "assumptions": [a["assumption"] for a in assumptions],
            "estimated_payable_inr": (result.get("amounts") or {}).get("estimated_payable_inr"),
        }
        st.session_state.setdefault("reviews", []).append(record)
    reviews = [r for r in st.session_state.get("reviews", []) if r["case_id"] == result["case_id"]]
    if reviews:
        last = reviews[-1]
        st.success(f"Recorded: {last['reviewer_assessment']} at {last['recorded_at']}. "
                   "Reviews live in this browser session and are exported as JSON; the API itself is stateless.")
        st.download_button("Download review record (JSON)", json.dumps(last, indent=2), file_name=f"{last['case_id']}_review.json", mime="application/json")
    all_reviews = st.session_state.get("reviews", [])
    if len(all_reviews) > 1 or (all_reviews and not reviews):
        st.caption("Reviews recorded this session")
        st.dataframe(pd.DataFrame([{k: r[k] for k in ("case_id", "system_decision", "reviewer_assessment", "reviewer_decision", "reviewer", "recorded_at")}
                                   for r in all_reviews]), hide_index=True)


with tab_analyze:
    st.subheader("1. Choose a claim")
    mode = st.radio("Case source", ["Supplied case", "Build a case", "Paste JSON", "Upload JSON"], horizontal=True, label_visibility="collapsed")
    case: dict | None = None

    if mode == "Supplied case":
        sources = load_case_sources()
        if sources:
            case = sources[st.selectbox("Case", list(sources.keys()))]
        else:
            st.warning("No case files found under data/candidate_data or data/custom_cases.")
    elif mode == "Build a case":
        st.caption("Fill in the claim; unknown fields can be left as 'Not stated' so you can see the system abstain.")
        case = build_case_form()
    elif mode == "Paste JSON":
        raw = st.text_area("Claim case JSON", height=260)
        if raw.strip():
            try:
                case = json.loads(raw)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
    else:
        up = st.file_uploader("Claim case JSON file", type="json")
        if up is not None:
            try:
                case = json.loads(up.read())
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    if case:
        case_card(case)
        with st.expander("Raw case JSON"):
            st.json(case)

    st.subheader("2. Run the analysis")
    if st.button("Analyze claim", type="primary", disabled=case is None):
        with st.spinner("Five agents are working on it..."):
            try:
                st.session_state["result"] = call_api(api_url, case)
                st.session_state["result_case"] = case
            except requests.HTTPError as e:
                st.session_state.pop("result", None)
                st.error(f"The API rejected the request ({e.response.status_code}): {e.response.text[:400]}")
            except Exception as e:
                st.session_state.pop("result", None)
                st.error(f"Could not reach the API: {e}")

    result = st.session_state.get("result")
    if result and case is not None and result.get("case_id") == case.get("case_id"):
        st.subheader("3. The answer")
        render_verdict(result, case)
        render_money(result)
        if result["decision"] == "NEEDS_REVIEW":
            render_missing(result, prominent=True)
        if result.get("validation", {}).get("status") == "FAIL":
            st.error("Evidence verification failed for at least one claim; see the Reviewer panel.")

        n_attention = sum(1 for c, _, _ in _attention_items(result) if c in (RED, AMBER))
        t_rev, t_why, t_evi, t_miss, t_flow, t_raw = st.tabs(
            [f"Reviewer panel ({n_attention} to check)" if n_attention else "Reviewer panel", "Why: the checks", "Policy evidence",
             "Missing evidence", "Agent workflow", "Raw response"])
        with t_rev:
            render_reviewer_panel(result, case)
        with t_why:
            render_checks(result)
            with st.expander("Written rationale"):
                st.write(result.get("rationale", ""))
        with t_evi:
            render_evidence(result)
        with t_miss:
            if result.get("missing_evidence"):
                render_missing(result, prominent=False)
            else:
                st.info("Nothing flagged as missing.")
        with t_flow:
            render_workflow(result)
        with t_raw:
            st.download_button("Download result JSON", json.dumps(result, indent=2), file_name=f"{result['case_id']}_decision.json", mime="application/json")
            st.json(result)
    elif result:
        st.info("The selected case changed since the last run. Press 'Analyze claim' to update the answer.")

# ==========================================================================
# TAB 2: HOW IT WORKS
# ==========================================================================

with tab_how:
    st.subheader("The idea in one paragraph")
    st.write("A claims reviewer reads a claim, hunts for the policy clauses that matter, checks the facts against them, and writes a "
             "decision. This system does the same in five steps, keeps every step inspectable, and stops to ask for a human when the "
             "policy or the evidence does not support a safe answer.")

    st.subheader("The five agents")
    flow = '<div class="flow">' + '<div class="arrow">&rarr;</div>'.join(
        f'<div class="box"><b>{i}. {esc(name)}</b>{esc(does)}</div>' for i, (_, name, does) in enumerate(AGENTS, start=1)) + "</div>"
    st.markdown(flow, unsafe_allow_html=True)
    st.caption("Agents hand each other one typed state object, not free text. The Agent workflow tab shows what each agent read, wrote, and how the state changed. "
               "If Validation finds a claim it cannot verify, control loops back to Policy Evidence once with wider retrieval before the system gives up and abstains.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("How the policy is searched")
        st.markdown(
            "1. **Chunking:** the 17-page PDF is split along its own structure (one definition, one exclusion, one condition per chunk), "
            "keeping page and section for every chunk.\n"
            "2. **Two searches:** a meaning-based (dense) search and a keyword (BM25) search run side by side.\n"
            "3. **Fusion:** results are merged by rank (Reciprocal Rank Fusion), so neither score scale dominates.\n"
            "4. **Re-ranking:** a cross-encoder reads the question with each candidate and keeps only the best few.\n"
            "5. **Citations:** every finding points to the chunk it came from, with page and section.")
    with c2:
        st.subheader("Why you can trust an answer")
        st.markdown(
            "- **Numbers come from the retrieved text**, not from memory: waiting periods, percentages and day limits are read out of the cited clause.\n"
            "- **The AI never decides.** The verdict comes from deterministic rules over cited findings; a language model may only re-word the explanation.\n"
            "- **Every claim is verified** against the chunk it cites (see below). Anything unverified triggers a re-retrieval, then abstention.\n"
            "- **Abstention:** unresolved facts, missing policy support, or a policy rule the system does not model always produce *Needs review*, never a guess.")

    st.subheader("How a claim is verified")
    st.markdown('<div class="flow"><div class="box"><b>Decision claim</b>e.g. "excluded: 30-day wait"</div><div class="arrow">&rarr;</div>'
                '<div class="box"><b>Citation</b>page, section, chunk id</div><div class="arrow">&rarr;</div>'
                '<div class="box"><b>Retrieved policy chunk</b>the exact text</div><div class="arrow">&rarr;</div>'
                '<div class="box"><b>Evidence check</b>numbers, phrases, arithmetic</div><div class="arrow">&rarr;</div>'
                '<div class="box"><b>Verdict</b>supported / unsupported / contradicted</div></div>', unsafe_allow_html=True)
    st.markdown(
        "- **Supported:** the cited chunk contains what the claim asserts (for a limit: the same percentage, and the cap and deduction add up).\n"
        "- **Unsupported:** the chunk does not contain it, or it was never retrieved.\n"
        "- **Contradicted:** the chunk states a *different* figure, the arithmetic is wrong, or the final decision conflicts with its own findings.")

    st.subheader("The eleven checks")
    st.table(pd.DataFrame([{"Check": t, "Question": q} for t, q in DIMENSIONS.values()]))
    st.caption("The last check is a safety net: the policy has specific rules (dental, pregnancy, listed illnesses, ...) that the ten modelled checks do not evaluate. "
               "When a diagnosis names one of them, the system abstains and shows the clause instead of ignoring it.")

    st.subheader("Verdicts")
    st.table(pd.DataFrame([{"Verdict": label, "Meaning": plain} for _, label, plain in DECISIONS.values()]))

# ==========================================================================
# TAB 3: EVALUATION
# ==========================================================================

with tab_eval:
    metrics = load_metrics()
    st.subheader("Stored evaluation results")
    if not metrics:
        st.info("No eval_results/metrics.json found. Run `python -m aptino_claims.eval.run_eval`.")
    elif "decision" not in metrics:
        st.warning("eval_results/metrics.json is from an older evaluation format; re-run the evaluation.")
    else:
        dec, amt, ret, cit, ctl = metrics["decision"], metrics["amounts"], metrics["retrieval"], metrics["citations"], metrics["controls"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Decision accuracy", f"{dec['correct']}/{dec['n']}", help="Exact match against hand-derived expected outcomes.")
        c2.metric("Unsafe decisions", len(dec["unsafe_decisions"]), help="Confident (non-abstaining) decisions that are wrong.")
        c3.metric("Payable amounts exact", f"{amt['exact_match_rate']:.0%}" if amt["exact_match_rate"] is not None else "n/a", help=f"{amt['n']} cases with hand-derived deductions.")
        c4.metric("Reproducible", "yes" if metrics["reproducibility"]["identical"] else "NO", help="Two cold runs produced byte-identical responses.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retrieval recall@5", f"{ret['systems']['reranked']['recall@5']:.2f}")
        c2.metric("Gold-citation precision", f"{cit['gold_citation_precision']:.0%}")
        c3.metric("Claims verified", f"{cit['verifier_counts']['SUPPORTED']}/{cit['verifier_claims']}")
        c4.metric("Fault injection caught", f"{ctl['detected']}/{ctl['injected']}", help="Known-bad citations, numbers, arithmetic and decisions injected to test the verifier.")
        st.caption(f"{metrics['n_cases']} cases ({metrics['n_public']} supplied + {metrics['n_custom']} candidate-authored). "
                   f"Latency p50 {metrics['latency_ms']['p50']:.0f} ms, p95 {metrics['latency_ms']['p95']:.0f} ms on a cold cache. "
                   "Labels are the author's reading of the policy; see eval_results/report.md for what this evaluation does not prove.")

        st.markdown("##### Retrieval ablation: what does each stage add?")
        names = {"dense": "Dense only", "sparse": "BM25 only", "fused": "Dense + BM25 (RRF)", "reranked": "RRF + rerank (system)"}
        abl = pd.DataFrame([{"Retriever": names[s], **{m: v for m, v in ret["systems"][s].items()}} for s in names]).set_index("Retriever")
        st.dataframe(abl)
        st.bar_chart(abl[["recall@1", "recall@3", "mrr"]], stack=False)

        with st.expander("Confusion matrix and per-class scores"):
            classes = list(dec["confusion_matrix"])
            st.caption("Rows = expected, columns = predicted.")
            st.dataframe(pd.DataFrame(dec["confusion_matrix"]).T[classes])
            st.dataframe(pd.DataFrame(dec["per_class"]).T)
        with st.expander("Fault-injection results (how well does the verifier catch bad claims?)"):
            st.dataframe(pd.DataFrame([{"Fault": k.replace("_", " "), "Injected": v["injected"], "Detected": v["detected"], "Rate": v["detection_rate"]}
                                       for k, v in ctl["by_type"].items()]), hide_index=True)

        st.markdown("##### Per-case results")
        df = pd.DataFrame(metrics["per_case"])[["case_id", "expected_decision", "actual_decision", "decision_correct", "confidence",
                                                "expected_deduction_inr", "actual_deduction_inr", "claims_supported", "claims_total", "attempts"]]
        df.columns = ["Case", "Expected", "Actual", "Correct", "Confidence", "Expected deduction", "Actual deduction", "Verified", "Claims", "Attempts"]
        st.dataframe(df, hide_index=True)

    st.subheader("Run all cases live against the API")
    st.caption("Sends every supplied and custom case through the running backend and compares with the labeled outcomes.")
    if st.button("Run live evaluation"):
        cases = list(load_case_sources().values())
        bar, rows = st.progress(0.0), []
        for i, c in enumerate(cases, start=1):
            try:
                res = call_api(api_url, c)
                exp = expected_for(c["case_id"])
                rows.append({"Case": c["case_id"], "Expected": exp, "Actual": res["decision"], "Match": exp == res["decision"],
                             "Confidence": res["confidence"], "Payable (INR)": (res.get("amounts") or {}).get("estimated_payable_inr"),
                             "Claims verified": f"{res['validation'].get('counts', {}).get('SUPPORTED', '?')}/{sum(res['validation'].get('counts', {}).values()) or '?'}"})
            except Exception as e:
                rows.append({"Case": c["case_id"], "Expected": expected_for(c["case_id"]), "Actual": f"error: {e}", "Match": False,
                             "Confidence": None, "Payable (INR)": None, "Claims verified": None})
            bar.progress(i / len(cases))
        live = pd.DataFrame(rows)
        st.metric("Live decision accuracy", f"{live['Match'].mean():.0%}" if len(live) else "n/a")
        st.dataframe(live, hide_index=True)
