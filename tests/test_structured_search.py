"""
Tests for deterministic structured search. No LLM involved -- these
verify the pandas logic directly against known values from the dataset.
"""
import pandas as pd
import pytest

from src.data_preprocessing import load_processed
from src.structured_search import Filter, StructuredSearchError, structured_search


@pytest.fixture(scope="module")
def df():
    return load_processed()


def test_top_10_by_revenue(df):
    result = structured_search(df, sort_by="revenue_clean", sort_desc=True, limit=10)
    titles = result["display"]["title"].tolist()
    assert titles[0] == "Avatar"
    assert titles[1] == "Titanic"
    assert len(titles) == 10


def test_count_after_2010(df):
    result = structured_search(df, filters=[Filter("release_year", "gt", 2010)], aggregate="count")
    assert result["mode"] == "count"
    assert result["count"] == 1221


def test_multi_condition_filter(df):
    result = structured_search(df, filters=[
        Filter("genre_list", "contains_any", ["Action"]),
        Filter("release_year", "gt", 2010),
        Filter("vote_average", "gt", 7.5),
    ])
    assert result["total_matching"] == 10
    assert "The Dark Knight Rises" in result["display"]["title"].values


def test_group_by_genre(df):
    result = structured_search(df, group_by="genre_list", sort_desc=True, limit=5)
    top_genre = result["display"].iloc[0]
    assert top_genre["genre_list"] == "Drama"
    assert top_genre["count"] == 2297


def test_unknown_field_raises(df):
    with pytest.raises(StructuredSearchError):
        structured_search(df, filters=[Filter("nonexistent_field", "gt", 5)])


def test_all_uids_preserves_sort_order_before_limit(df):
    """Regression test: state memory must reflect the SORTED order the
    user actually saw, not the unsorted filtered set (this was a real bug
    found during manual testing -- 'first' resolved to the wrong movie
    before this was fixed)."""
    result = structured_search(df, sort_by="vote_average", sort_desc=True, limit=3)
    displayed_first_uid = result["display"].iloc[0]["movie_uid"]
    assert result["all_uids"][0] == displayed_first_uid


def test_budget_filter_treats_zero_as_missing(df):
    """budget_clean should exclude movies with unknown (0) budget."""
    result = structured_search(df, filters=[Filter("budget_clean", "gt", 0)])
    assert result["total_matching"] < len(df)  # some movies have unknown budget
    assert (result["display"]["budget_clean"] > 0).all()