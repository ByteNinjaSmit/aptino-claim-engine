from aptino_claims.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_favors_items_ranked_highly_in_both_lists():
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    sparse = [("b", 5.0), ("a", 3.0), ("d", 2.0)]
    fused = reciprocal_rank_fusion([dense, sparse])
    ids = [c for c, _ in fused]
    assert ids[0] in {"a", "b"}
    assert "d" in ids and "c" in ids


def test_rrf_respects_top_k():
    dense = [("a", 1.0), ("b", 0.9), ("c", 0.8)]
    fused = reciprocal_rank_fusion([dense], top_k=2)
    assert len(fused) == 2
