"""
Data preprocessing for the TMDB 5000 movie dataset.

Preprocessing decisions:
- Join movies.id <-> credits.movie_id (left join on movies; every credits
  row has a match in this dataset, verified during development).
- Parse JSON-encoded columns (genres, keywords, cast, crew,
  production_companies, production_countries, spoken_languages) into
  Python lists/dicts.
- Derive flat, query-friendly columns for structured search (genre_list,
  cast_list, director, release_year) rather than mutating the raw JSON
  columns — originals are preserved untouched.
- budget == 0 and revenue == 0 are treated as "unknown", not real zero
  values -> stored as NaN in *_clean columns.
- release_date is parsed to a real datetime; unparsable dates become NaT.
- overview/tagline missing values are filled with "" ONLY in the
  RAG-document-facing columns, never in the originals used for display.
- A stable positional id (movie_uid) is assigned after preprocessing so
  every tool can reference movies unambiguously.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MOVIES_CSV = DATA_DIR / "tmdb_5000_movies.csv"
CREDITS_CSV = DATA_DIR / "tmdb_5000_credits.csv"
PROCESSED_PARQUET = DATA_DIR / "processed_movies.parquet"


def _safe_json_load(value) -> list:
    if pd.isna(value) or value == "":
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _names(items: list, key: str = "name") -> list[str]:
    return [d.get(key) for d in items if isinstance(d, dict) and d.get(key)]


def _extract_director(crew: list) -> str | None:
    for member in crew:
        if isinstance(member, dict) and member.get("job") == "Director":
            return member.get("name")
    return None


def _extract_top_cast(cast: list, n: int = 8) -> list[str]:
    ordered = sorted(
        [c for c in cast if isinstance(c, dict)],
        key=lambda c: c.get("order", 999),
    )
    return [c.get("name") for c in ordered[:n] if c.get("name")]


def load_and_join() -> pd.DataFrame:
    movies = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)

    credits = credits.drop(columns=["title"]).rename(columns={"movie_id": "id"})
    df = movies.merge(credits, on="id", how="left", validate="one_to_one")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["genres_parsed"] = df["genres"].apply(_safe_json_load)
    df["keywords_parsed"] = df["keywords"].apply(_safe_json_load)
    df["cast_parsed"] = df["cast"].apply(_safe_json_load)
    df["crew_parsed"] = df["crew"].apply(_safe_json_load)
    df["production_companies_parsed"] = df["production_companies"].apply(_safe_json_load)
    df["production_countries_parsed"] = df["production_countries"].apply(_safe_json_load)
    df["spoken_languages_parsed"] = df["spoken_languages"].apply(_safe_json_load)

    df["genre_list"] = df["genres_parsed"].apply(_names)
    df["keyword_list"] = df["keywords_parsed"].apply(_names)
    df["cast_list"] = df["cast_parsed"].apply(_extract_top_cast)
    df["director"] = df["crew_parsed"].apply(_extract_director)
    df["production_company_list"] = df["production_companies_parsed"].apply(_names)
    df["production_country_list"] = df["production_countries_parsed"].apply(_names)
    df["spoken_language_list"] = df["spoken_languages_parsed"].apply(_names)

    df["release_date_parsed"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date_parsed"].dt.year

    df["budget_clean"] = df["budget"].where(df["budget"] > 0, other=pd.NA)
    df["revenue_clean"] = df["revenue"].where(df["revenue"] > 0, other=pd.NA)
    df["runtime_clean"] = df["runtime"].where(df["runtime"] > 0, other=pd.NA)

    df["overview_filled"] = df["overview"].fillna("")
    df["tagline_filled"] = df["tagline"].fillna("")
    df["vote_count"] = df["vote_count"].fillna(0).astype(int)

    df = df.reset_index(drop=True)
    df["movie_uid"] = df.index

    def _as_list(val):
        if val is None:
            return []
        if isinstance(val, float) and pd.isna(val):
            return []
        return list(val)

    def build_rag_document(row) -> str:
        parts = [f"Title: {row['title']}"]
        if row["tagline_filled"]:
            parts.append(f"Tagline: {row['tagline_filled']}")
        parts.append(f"Overview: {row['overview_filled']}")
        genres = _as_list(row["genre_list"])
        keywords = _as_list(row["keyword_list"])
        cast = _as_list(row["cast_list"])
        if genres:
            parts.append(f"Genres: {', '.join(genres)}")
        if keywords:
            parts.append(f"Keywords: {', '.join(keywords)}")
        if cast:
            parts.append(f"Cast: {', '.join(cast)}")
        director = row["director"]
        if director is not None and not (isinstance(director, float) and pd.isna(director)):
            parts.append(f"Director: {director}")
        return "\n".join(parts)

    df["rag_document"] = df.apply(build_rag_document, axis=1)
    return df


def load_processed(force_rebuild: bool = False) -> pd.DataFrame:
    """Load cached processed dataframe, or build + cache it if absent."""
    if not force_rebuild and PROCESSED_PARQUET.exists():
        return pd.read_parquet(PROCESSED_PARQUET)
    df = preprocess(load_and_join())
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PARQUET)
    return df


if __name__ == "__main__":
    d = load_processed(force_rebuild=True)
    print(d.shape)
    print(d[["title", "genre_list", "director", "release_year", "budget_clean"]].head(5))
    print("\n--- sample RAG doc ---")
    print(d.iloc[0]["rag_document"])