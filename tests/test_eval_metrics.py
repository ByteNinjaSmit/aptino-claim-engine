from aptino_claims.eval.gold import gold_chunks, requirements_for
from aptino_claims.eval.metrics import confusion_matrix, per_class_prf, percentile, recall_at_k, reciprocal_rank


def test_recall_and_mrr():
    reqs = {"a": {"c1"}, "b": {"c9", "c8"}}
    assert recall_at_k(["c1", "x", "c8"], reqs, 1) == 0.5
    assert recall_at_k(["c1", "x", "c8"], reqs, 3) == 1.0
    assert reciprocal_rank(["x", "y", "c8"], reqs) == 1 / 3
    assert reciprocal_rank(["x"], reqs) == 0.0


def test_confusion_and_prf():
    pairs = [("A", "A"), ("A", "B"), ("B", "B"), ("B", "B")]
    cm = confusion_matrix(pairs, ["A", "B"])
    assert cm["A"] == {"A": 1, "B": 1} and cm["B"] == {"A": 0, "B": 2}
    prf = per_class_prf(pairs, ["A", "B"])
    assert prf["A"]["precision"] == 1.0 and prf["A"]["recall"] == 0.5
    assert round(prf["B"]["precision"], 3) == 0.667 and prf["B"]["recall"] == 1.0


def test_percentile():
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10


def test_gold_requirements_follow_the_expenses_actually_claimed():
    facts = {"treatment_type": "inpatient", "expenses": {"room": 100, "doctor_fees": 0, "medicines_diagnostics": 5, "ambulance": 0}}
    assert set(requirements_for("category_sub_limits", facts)) == {"room", "medicines"}
    assert set(requirements_for("category_sub_limits", {"treatment_type": "domiciliary", "expenses": {}})) == {"domiciliary"}
    corpus = {"a": "Normal Room  expenses: 1%", "b": "unrelated"}
    assert gold_chunks({"room": [["normal room expenses"]]}, corpus) == {"room": {"a"}}
