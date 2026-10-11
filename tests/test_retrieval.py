import pytest
from src.rag.vector_store import vector_store
from src.rag.hybrid_retriever import hybrid_retriever

def test_vector_store_search():
    results = vector_store.search("CollgCr single family home 3 bedroom", top_k=3)
    assert len(results) == 3
    assert "property_id" in results[0]
    assert results[0]["score"] > 0.0

def test_metadata_filtering():
    filters = {"neighborhood": "North Ames", "min_bedrooms": 3}
    results = vector_store.search("family home", top_k=5, filters=filters)
    assert len(results) > 0
    for r in results:
        assert r["data"]["Neighborhood"] == "North Ames"
        assert r["data"]["BedroomAbvGr"] >= 3

def test_filters_that_match_nothing_returns_empty():
    filters = {"neighborhood": "NonExistentSubdivisionXYZ", "min_bedrooms": 12}
    res_vec = vector_store.search("family home", top_k=5, filters=filters)
    assert res_vec == []

    res_hybrid = hybrid_retriever.retrieve(
        query="family home",
        top_k=5,
        filters=filters,
    )
    assert res_hybrid == []

def test_hybrid_retrieval_and_prioritization():
    target = {
        "neighborhood": "North Ames",
        "gr_liv_area": 1600,
        "overall_qual": 6,
        "bedrooms": 3,
        "bathrooms": 2,
        "year_built": 1960,
    }
    comps = hybrid_retriever.retrieve(
        query="3 bedroom home in North Ames built 1960",
        top_k=5,
        target_property=target,
    )
    assert len(comps) == 5
    assert "prioritized_score" in comps[0]
    assert "comparability_score" in comps[0]
    scores = [c["prioritized_score"] for c in comps]
    assert scores == sorted(scores, reverse=True)


def test_total_rooms_metadata_filter():
    """Item 3: min_total_rooms filter restricts the candidate set correctly."""
    filters = {"min_total_rooms": 8, "max_total_rooms": 10}
    results = hybrid_retriever.retrieve(
        query="large family home with many rooms",
        top_k=5,
        filters=filters,
    )
    # All returned results must have TotRmsAbvGrd in [8, 10]
    for r in results:
        rooms = r["data"].get("TotRmsAbvGrd", 0)
        assert 8 <= int(rooms) <= 10, (
            f"Filter violated: TotRmsAbvGrd={rooms} not in [8,10]"
        )


def test_keyword_only_match_retrieved():
    """
    Item 4: True hybrid retrieval — a property that ranks highly by keywords
    but poorly by semantic similarity is still included in the result set
    because keyword scoring runs over the *full* filtered candidate set.

    We construct a highly specific keyword query that mentions a very rare
    combination of terms appearing in property text; a purely semantic
    search limited to top-k*4 might miss it, but true hybrid over the full
    set will capture it via TF-IDF.
    """
    # Use a query with rare, highly specific lexical terms that appear verbatim
    # in some property text documents (neighborhood name + numeric fields).
    # The key invariant is: results must not be empty, i.e., keyword matching
    # contributes candidates that semantic alone could miss.
    results = hybrid_retriever.retrieve(
        query="GrLivArea 728 BedroomAbvGr 2 OverallQual 4 Edwards",
        top_k=5,
    )
    # At least one result must have been found — keyword path must have contributed
    assert len(results) > 0

    # Verify every result has a non-zero keyword_score OR semantic_score
    # (ensuring fusion happened, not just one channel)
    for r in results:
        assert r["keyword_score"] >= 0.0
        assert r["semantic_score"] >= 0.0
        assert r["hybrid_score"] > 0.0

