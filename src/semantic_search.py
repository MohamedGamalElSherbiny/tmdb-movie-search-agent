"""
Semantic retrieval for RAG.

Embedding model: sentence-transformers/all-MiniLM-L6-v2 (384-dim,
downloaded once, cached locally). Cosine similarity via normalized dot
product against one embedding per movie's rag_document.

Document representation: one document per movie (title, tagline,
overview, genres, keywords, cast, director) -- not raw CSV rows, not
chunked (movie overviews are short; splitting them would fragment the
very context being matched against).

"Similar to X" queries: resolved by fuzzy-matching the title, then using
that movie's OWN embedding as the query vector, rather than embedding the
literal phrase "similar to X" (which performs poorly -- a bare movie name
carries little distributional signal on its own).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDINGS_PATH = DATA_DIR / "movie_embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"
LOW_CONFIDENCE_THRESHOLD = 0.3


class SemanticIndex:
    def __init__(self, df: pd.DataFrame, force_rebuild: bool = False):
        self.df = df
        self.model = SentenceTransformer(MODEL_NAME)

        if not force_rebuild and EMBEDDINGS_PATH.exists():
            embeddings = np.load(EMBEDDINGS_PATH)
        else:
            documents = df["rag_document"].tolist()
            embeddings = self.model.encode(documents, show_progress_bar=True, batch_size=64)
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            np.save(EMBEDDINGS_PATH, embeddings)

        self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def search(self, query: str = None, top_k: int = 8, seed_title_uid: int | None = None,
               allowed_uids: set[int] | None = None) -> dict:
        if seed_title_uid is not None:
            row_idx = self.df.index[self.df["movie_uid"] == seed_title_uid][0]
            query_vec = self.embeddings[row_idx]
        elif query:
            query_vec = self.model.encode([query])[0]
            query_vec = query_vec / np.linalg.norm(query_vec)
        else:
            raise ValueError("search() requires either query or seed_title_uid")

        sims = self.embeddings @ query_vec
        order = np.argsort(-sims)

        results = []
        for i in order:
            uid = int(self.df.iloc[i]["movie_uid"])
            if seed_title_uid is not None and uid == seed_title_uid:
                continue
            if allowed_uids is not None and uid not in allowed_uids:
                continue
            results.append({
                "movie_uid": uid,
                "title": self.df.iloc[i]["title"],
                "score": float(sims[i]),
            })
            if len(results) >= top_k:
                break

        low_confidence = bool(results) and results[0]["score"] < LOW_CONFIDENCE_THRESHOLD
        return {
            "status": "low_confidence" if low_confidence else ("ok" if results else "no_results"),
            "results": results,
        }