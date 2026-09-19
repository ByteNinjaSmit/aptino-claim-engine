import json

from aptino_claims.agents import rules_extract as rx
from aptino_claims.agents import dimensions as dim
from aptino_claims.agents.state import EvidenceItem
from aptino_claims.config import settings

CHUNKS = [json.loads(l) for l in (settings.index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
COVER = "\n".join(c["text"] for c in CHUNKS if c["chunk_id"].startswith("what_we_cover"))


def test_every_category_limit_is_extracted_from_the_real_policy_text():
    assert rx.extract_percent_of_si(COVER, "Room") == 1.0
    assert rx.extract_percent_of_si(COVER, "fees") == 25.0
    assert rx.extract_percent_of_si(COVER, "Anesthesia") == 40.0      # "40% Sum Insured": no "of" (regression)
    assert rx.extract_percent_of_si(COVER, "package") == 75.0
    assert rx.extract_percent_of_si(COVER, "Domiciliary") == 20.0
    assert rx.extract_percent_of_si(COVER, "Ambulance") == 1.0
    assert rx.extract_flat_amount(COVER, "Ambulance") == 1000.0        # "lower of 1% and Rs 1000" (regression)


def _ev(cid, text, section="What We Exclude", label="What We Exclude - Item 7"):
    return EvidenceItem(chunk_id=cid, section=section, label=label, page_start=9, page_end=9, text=text)


def test_unmodelled_exclusion_is_flagged_but_not_decided():
    facts = {"diagnosis": "Dental caries", "procedure": "Tooth extraction", "_cited_chunk_ids": []}
    finding, _, missing = dim._unmodelled_eval(facts, [_ev("we-7", "7. Dental treatment or surgery of any kind.")])
    assert finding.status == "INSUFFICIENT_EVIDENCE" and finding.applicable
    assert finding.citations[0].chunk_id == "we-7" and missing


def test_clauses_already_used_by_a_modelled_check_are_not_reflagged():
    facts = {"diagnosis": "Cataract", "procedure": "Eye surgery", "_cited_chunk_ids": ["we-3"]}
    finding, _, _ = dim._unmodelled_eval(facts, [_ev("we-3", "3. ... i) Cataract ii) Hernia ...")])
    assert finding.status == "NOT_APPLICABLE"


def test_generic_words_do_not_trigger_the_unknown_dimension():
    facts = {"diagnosis": "Acute infection", "procedure": "Hospital treatment", "_cited_chunk_ids": []}
    body = "20. Treatment of following diseases: xi) Tonsillitis and Upper Respiratory Tract infection including Laryngitis"
    finding, _, _ = dim._unmodelled_eval(facts, [_ev("we-20", body)])
    assert finding.status == "NOT_APPLICABLE"


def test_package_cap_only_applies_when_a_package_was_agreed():
    base = {"treatment_type": "inpatient", "sum_insured": 100_000, "admission_hours": 288,
            "expenses": {"room": 12_000, "doctor_fees": 25_000, "medicines_diagnostics": 40_000}}
    evidence = [_ev("wc", COVER, section="What We Cover")]
    _, without, _ = dim._sublimits_eval(dict(base), evidence)
    assert not any("package" in l.description for l in without)
    _, with_pkg, _ = dim._sublimits_eval({**base, "package_charges_agreed": True}, evidence)
    assert any("package" in l.description for l in with_pkg)
