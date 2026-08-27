"""
Fuzzy movie-title matching.

Approach: rapidfuzz's WRatio scored against a normalized version of
`title`/`original_title`. Three-tier confidence scheme:
  - >= HIGH_CONFIDENCE (90): confident single match.
  - Below that but with a clear score gap over the runner-up
    (>= MIN_SCORE_FOR_MARGIN_PROMOTION and margin >= CLEAR_WINNER_MARGIN):
    still treated as confident. This handles short-title / single-typo
    cases (e.g. "Intersteler" -> "Interstellar" scores ~87, below 90, but
    is clearly the only real candidate).
  - >= LOW_CONFIDENCE (70): ambiguous -- report candidates, don't guess.
  - Below LOW_CONFIDENCE: not found.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from rapidfuzz import fuzz, process

HIGH_CONFIDENCE = 90
LOW_CONFIDENCE = 70
CLEAR_WINNER_MARGIN = 8
MIN_SCORE_FOR_MARGIN_PROMOTION = 80


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


@dataclass
class FuzzyMatch:
    movie_uid: int
    title: str
    score: float


class FuzzyIndex:
    """Builds the normalized-title -> movie_uid lookup once, reused across queries."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.choices: dict[str, int] = {}
        for uid, title, orig_title in zip(df["movie_uid"], df["title"], df["original_title"]):
            for candidate in {title, orig_title}:
                if pd.isna(candidate):
                    continue
                norm = _normalize(str(candidate))
                if norm and norm not in self.choices:
                    self.choices[norm] = uid

    def find(self, query: str, top_k: int = 5) -> dict:
        if not query or not query.strip():
            return {"status": "not_found", "match": None, "candidates": []}

        norm_query = _normalize(query)
        results = process.extract(norm_query, self.choices.keys(), scorer=fuzz.WRatio, limit=top_k)

        candidates: list[FuzzyMatch] = []
        seen_uid = set()
        for matched_norm, score, _ in results:
            uid = self.choices[matched_norm]
            if uid in seen_uid:
                continue
            seen_uid.add(uid)
            real_title = self.df.loc[self.df["movie_uid"] == uid, "title"].iloc[0]
            candidates.append(FuzzyMatch(movie_uid=int(uid), title=str(real_title), score=score))

        if not candidates:
            return {"status": "not_found", "match": None, "candidates": []}

        top = candidates[0]
        runner_up_score = candidates[1].score if len(candidates) > 1 else 0

        if top.score >= HIGH_CONFIDENCE:
            tied = [c for c in candidates if c.score >= top.score - 1 and c.movie_uid != top.movie_uid]
            if tied:
                return {"status": "ambiguous", "match": None, "candidates": candidates}
            return {"status": "matched", "match": top, "candidates": candidates}

        if top.score >= MIN_SCORE_FOR_MARGIN_PROMOTION and (top.score - runner_up_score) >= CLEAR_WINNER_MARGIN:
            return {"status": "matched", "match": top, "candidates": candidates}

        if top.score >= LOW_CONFIDENCE:
            return {"status": "ambiguous", "match": None, "candidates": candidates}

        return {"status": "not_found", "match": None, "candidates": candidates}