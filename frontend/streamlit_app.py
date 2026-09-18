"""Reviewer-facing frontend for the Aptino claim decision engine."""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Aptino Claim Decision Engine", layout="wide")

DECISION_STYLE = {
    "ADMISSIBLE": ("#1a7f37", "Admissible"),
    "ADMISSIBLE_WITH_LIMITS": ("#9a6700", "Admissible, with limits"),
    "PARTIALLY_ADMISSIBLE": ("#9a6700", "Partially admissible"),
    "NOT_ADMISSIBLE": ("#cf222e", "Not admissible"),
    "NEEDS_REVIEW": ("#57606a", "Needs review (abstained)"),
}


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_case_sources() -> dict[str, dict]:
    sources: dict[str, dict] = {}
    public_path = REPO_ROOT / "data" / "candidate_data" / "public_test_cases.json"
    if public_path.exists():
        for c in load_json(public_path):
            sources[f"[public] {c['case_id']}"] = c
    custom_path = REPO_ROOT / "data" / "custom_cases" / "custom_test_cases.json"
    if custom_path.exists():
        for c in load_json(custom_path):
            sources[f"[custom] {c['case_id']}"] = c
    return sources


def call_analyze(api_url: str, case: dict) -> dict:
    resp = requests.post(f"{api_url}/analyze", json=case, timeout=60)
    resp.raise_for_status()
    return resp.json()


st.title("Aptino Claim Decision Engine")
st.caption("Policy-aware multi-agent RAG claim decision engine — reviewer console")

with st.sidebar:
    st.subheader("Backend")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL)
    try:
        health = requests.get(f"{api_url}/health", timeout=5).json()
        st.success(f"API reachable — {health.get('chunk_count', '?')} policy chunks indexed, LLM: {health.get('llm_provider')}")
    except Exception:
        st.error("API not reachable at this URL. Start the backend or update the URL above.")

    st.subheader("Case input")
    mode = st.radio("Source", ["Pick a supplied case", "Paste JSON", "Upload JSON file"])

case: dict | None = None

if mode == "Pick a supplied case":
    sources = load_case_sources()
    if not sources:
        st.warning("No case files found under data/candidate_data or data/custom_cases.")
    else:
        key = st.selectbox("Case", list(sources.keys()))
        case = sources[key]
        st.json(case, expanded=False)
elif mode == "Paste JSON":
    raw = st.text_area("Claim case JSON", height=300)
    if raw.strip():
        try:
            case = json.loads(raw)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
else:
    uploaded = st.file_uploader("Claim case JSON file", type="json")
    if uploaded is not None:
        try:
            case = json.loads(uploaded.read())
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")

analyze_clicked = st.button("Analyze claim", type="primary", disabled=case is None)

if analyze_clicked and case is not None:
    with st.spinner("Running multi-agent analysis..."):
        try:
            result = call_analyze(api_url, case)
        except requests.HTTPError as e:
            st.error(f"API returned an error: {e.response.status_code} — {e.response.text}")
            result = None
        except Exception as e:
            st.error(f"Could not reach the API: {e}")
            result = None

    if result:
        decision = result.get("decision", "NEEDS_REVIEW")
        color, label = DECISION_STYLE.get(decision, ("#57606a", decision))

        if decision == "NEEDS_REVIEW":
            st.warning(f"⚠️ System is abstaining — **{label}**. A confident decision could not be made from the supplied policy evidence.")
        else:
            st.markdown(f"<h2 style='color:{color}'>{label}</h2>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Confidence", f"{result.get('confidence', 0):.0%}")
        col2.metric("Citations", len(result.get("citations", [])))
        col3.metric("Validation", result.get("validation", {}).get("status", "?"))

        st.subheader("Key findings")
        for f in result.get("key_findings", []):
            st.markdown(f"- {f}")

        limits = result.get("applicable_limits", [])
        if limits:
            st.subheader("Applicable limits / deductions")
            st.table(
                [
                    {
                        "Description": l["description"],
                        "Deduction (INR)": f"{l['deduction_inr']:,.0f}" if l.get("deduction_inr") is not None else "-",
                    }
                    for l in limits
                ]
            )

        missing = result.get("missing_evidence", [])
        if missing:
            st.subheader("Missing evidence")
            for m in missing:
                st.markdown(f"- **{m['field']}**: {m['reason']}")

        st.subheader("Policy evidence & citations")
        citations = result.get("citations", [])
        if citations:
            st.table(
                [
                    {
                        "Claim supported": c["claim"],
                        "Section": c["section"],
                        "Page": c["page"],
                        "Chunk ID": c["chunk_id"],
                        "Rerank score": round(c["rerank_score"], 2) if c.get("rerank_score") is not None else "-",
                    }
                    for c in citations
                ]
            )
        else:
            st.info("No citations were produced for this case.")

        validation = result.get("validation", {})
        if validation.get("unsupported_claims"):
            st.subheader("Validation issues")
            for u in validation["unsupported_claims"]:
                st.markdown(f"- {u}")

        st.subheader("Rationale")
        st.write(result.get("rationale", ""))

        with st.expander("Execution trace (agents, actions, timings — no hidden chain-of-thought)"):
            st.table(
                [
                    {
                        "Agent": t["agent"],
                        "Action": t["action"],
                        "Detail": t["detail"],
                        "Elapsed (ms)": t["elapsed_ms"],
                        "Retrieval count": t.get("retrieval_count") or "-",
                    }
                    for t in result.get("trace", [])
                ]
            )

        with st.expander("Raw response JSON"):
            st.json(result)
