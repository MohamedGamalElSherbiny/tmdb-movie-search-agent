"""
Tests for fuzzy title matching -- confidence thresholds and ambiguity
handling, based on the exact cases from the assignment's examples.
"""
import pytest

from src.data_preprocessing import load_processed
from src.fuzzy_search import FuzzyIndex


@pytest.fixture(scope="module")
def fuzzy_index():
    df = load_processed()
    return FuzzyIndex(df)


def test_exact_match(fuzzy_index):
    r = fuzzy_index.find("Avatar")
    assert r["status"] == "matched"
    assert r["match"].title == "Avatar"
    assert r["match"].score == 100.0


def test_typo_still_matches(fuzzy_index):
    r = fuzzy_index.find("Avatr")
    assert r["status"] == "matched"
    assert r["match"].title == "Avatar"


def test_short_typo_via_margin_promotion(fuzzy_index):
    """'Intersteler' scores ~87 (below the 90 hard threshold) but is a
    clear winner over the runner-up -- should still resolve confidently."""
    r = fuzzy_index.find("Intersteler")
    assert r["status"] == "matched"
    assert r["match"].title == "Interstellar"


def test_partial_title_matches(fuzzy_index):
    r = fuzzy_index.find("the dark knight rises")
    assert r["status"] == "matched"
    assert r["match"].title == "The Dark Knight Rises"


def test_genuinely_ambiguous_title_is_not_guessed(fuzzy_index):
    """Three 'Lord of the Rings' films score identically -- must report
    ambiguous rather than silently picking one."""
    r = fuzzy_index.find("lord of the rings")
    assert r["status"] == "ambiguous"
    assert r["match"] is None
    assert len(r["candidates"]) >= 3


def test_nonsense_query_not_found(fuzzy_index):
    r = fuzzy_index.find("asdkjaskjd")
    assert r["status"] == "not_found"
    assert r["match"] is None


def test_empty_query_not_found(fuzzy_index):
    r = fuzzy_index.find("")
    assert r["status"] == "not_found"