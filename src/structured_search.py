"""
Deterministic, data-backed structured search over the preprocessed movie
dataframe. No LLM involvement here -- every number returned is computed
directly with pandas. The LLM only ever calls this module with a parsed
filter spec; it never invents counts, aggregates, or records.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

LIST_FIELDS = {"genre_list", "keyword_list", "cast_list", "production_company_list"}

ALLOWED_FIELDS = {
    "vote_average", "vote_count", "runtime_clean", "release_year",
    "budget_clean", "revenue_clean", "popularity", "original_language",
    "director", "status", "genre_list", "keyword_list", "cast_list",
    "production_company_list", "title",
}


class StructuredSearchError(ValueError):
    pass


@dataclass
class Filter:
    field: str
    op: str  # eq, gt, gte, lt, lte, contains, contains_any
    value: Any


def _elems(lst) -> set:
    if lst is None:
        return set()
    if isinstance(lst, float) and pd.isna(lst):
        return set()
    return {str(x).lower() for x in lst}


def apply_filter(data: pd.DataFrame, f: Filter) -> pd.DataFrame:
    if f.field not in ALLOWED_FIELDS:
        raise StructuredSearchError(f"Unknown/unsupported filter field: {f.field}")
    if f.field not in data.columns:
        raise StructuredSearchError(f"Field not present in dataset: {f.field}")

    col = data[f.field]

    if f.field in LIST_FIELDS:
        needle = f.value if isinstance(f.value, list) else [f.value]
        needle_lower = {str(n).lower() for n in needle}
        if f.op == "contains_any":
            mask = col.apply(lambda lst: bool(needle_lower & _elems(lst)))
        else:
            mask = col.apply(lambda lst: needle_lower.issubset(_elems(lst)))
        return data[mask]

    if f.op == "eq":
        mask = col.astype(str).str.lower() == str(f.value).lower() if col.dtype == object else col == f.value
    elif f.op == "gt":
        mask = col > f.value
    elif f.op == "gte":
        mask = col >= f.value
    elif f.op == "lt":
        mask = col < f.value
    elif f.op == "lte":
        mask = col <= f.value
    elif f.op == "contains":
        mask = col.astype(str).str.contains(str(f.value), case=False, na=False)
    else:
        raise StructuredSearchError(f"Unsupported operator: {f.op}")

    return data[mask]


def structured_search(
    df: pd.DataFrame,
    filters: list[Filter] | None = None,
    sort_by: str | None = None,
    sort_desc: bool = True,
    limit: int | None = 20,
    aggregate: str | None = None,   # "count"
    group_by: str | None = None,
) -> dict:
    """Apply filters, then either count, group-by, or sort+limit records.

    Returns a dict describing what happened, always including the FULL
    filtered movie_uid list (for multi-turn memory) separately from the
    possibly-truncated display records.
    """
    filtered = df
    for f in (filters or []):
        filtered = apply_filter(filtered, f)

    if aggregate == "count":
        return {
            "mode": "count",
            "count": len(filtered),
            "all_uids": filtered["movie_uid"].tolist(),
        }

    if group_by:
        if group_by not in ALLOWED_FIELDS:
            raise StructuredSearchError(f"Unsupported group_by field: {group_by}")
        if group_by in LIST_FIELDS:
            exploded = filtered.explode(group_by)
            grouped = exploded.groupby(group_by).size().reset_index(name="count")
        else:
            grouped = filtered.groupby(group_by).size().reset_index(name="count")
        grouped = grouped.sort_values("count", ascending=not sort_desc)
        if limit:
            grouped = grouped.head(limit)
        return {
            "mode": "group_by",
            "display": grouped,
            "all_uids": filtered["movie_uid"].tolist(),
        }

    result = filtered
    if sort_by:
        if sort_by not in df.columns:
            raise StructuredSearchError(f"Unsupported sort field: {sort_by}")
        result = result.sort_values(sort_by, ascending=not sort_desc, na_position="last")

    result_full_sorted = result  # full sorted set, used for multi-turn memory
    if limit:
        result = result.head(limit)

    return {
        "mode": "records",
        "total_matching": len(filtered),
        "shown": len(result),
        "display": result,
        "all_uids": result_full_sorted["movie_uid"].tolist(),  # sorted order == displayed order
    }