"""
Tests for data preprocessing -- verifying join integrity, JSON parsing,
and the zero-as-missing handling for budget/revenue/runtime.
"""
import pandas as pd
import pytest

from src.data_preprocessing import load_processed


@pytest.fixture(scope="module")
def df():
    return load_processed()


def test_expected_row_count(df):
    assert len(df) == 4803


def test_join_preserved_all_rows(df):
    assert df["cast"].notna().sum() == 4803


def test_genres_parsed_to_list(df):
    avatar = df[df["title"] == "Avatar"].iloc[0]
    assert "Action" in avatar["genre_list"]
    assert "Science Fiction" in avatar["genre_list"]


def test_director_extracted(df):
    avatar = df[df["title"] == "Avatar"].iloc[0]
    assert avatar["director"] == "James Cameron"


def test_zero_budget_treated_as_missing(df):
    zero_budget_movies = df[df["budget"] == 0]
    assert len(zero_budget_movies) > 0  # sanity: some do exist in raw data
    assert zero_budget_movies["budget_clean"].isna().all()


def test_release_year_parsed(df):
    avatar = df[df["title"] == "Avatar"].iloc[0]
    assert avatar["release_year"] == 2009


def test_movie_uid_is_unique(df):
    assert df["movie_uid"].is_unique
    assert df["movie_uid"].min() == 0
    assert df["movie_uid"].max() == len(df) - 1


def test_rag_document_contains_key_fields(df):
    avatar = df[df["title"] == "Avatar"].iloc[0]
    doc = avatar["rag_document"]
    assert "Avatar" in doc
    assert "James Cameron" in doc
    assert "Science Fiction" in doc
    # raw JSON should never leak into the document
    assert "{" not in doc and "}" not in doc