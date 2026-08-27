# Technical Design Document — TMDB Movie Discovery Agent

**Author:** Mohamed Gamal El Sherbiny
**Assignment:** SiliconExpert AI Engineer Technical Assignment

This document explains the engineering decisions behind the system, not how to use it (see `README.md` for setup/run instructions).

---

## 1. Data Decisions

**Dataset files used:** `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` from the TMDB 5000 Movie Dataset (Kaggle).

**Number of movies processed:** 4,803 (full dataset, no subsetting).

**Join strategy:** Left join of `movies` on `credits`, matching `movies.id == credits.movie_id`. `credits.title` is dropped before the join (duplicate of `movies.title`) and `movie_id` is renamed to `id` for the merge key. All 4,803 rows in `movies` have a matching `credits` row — verified via `df["cast"].notna().sum() == 4803` (also asserted in `tests/test_data_preprocessing.py`).

**JSON fields parsed:** `genres`, `keywords`, `cast`, `crew`, `production_companies`, `production_countries`, `spoken_languages`. Each is parsed via `json.loads` with a fallback to `[]` on malformed/missing input (`_safe_json_load`), so no row is dropped due to a parse failure.

**Derived flat columns** (added alongside, not replacing, the originals):
- `genre_list`, `keyword_list`, `production_company_list`, `production_country_list`, `spoken_language_list` — flat lists of `name` values, used for structured filtering.
- `cast_list` — top 8 actors by TMDB's own billing `order` field, names only (no character mapping — the raw `cast` JSON does contain per-actor character names, but we deliberately do not surface that in the flat structured-search field; see the RAG document rationale in Section 4 for why the agent is explicitly told not to imply character assignments).
- `director` — extracted from `crew` by filtering `job == "Director"`.

**Missing-value handling:**
- `release_date`: parsed via `pd.to_datetime(errors="coerce")`. 1 of 4,803 rows fails to parse and becomes `NaT` / `release_year = NaN`. This one row is preserved (not dropped) since other fields are still usable for structured/semantic search.
- `budget` / `revenue`: **0 is treated as "unknown," not a real value.** `budget_clean` / `revenue_clean` set to `NaN` wherever the raw value is 0. This matters materially: 1,037 of 4,803 movies (21.6%) have unknown budget, and 1,427 (29.7%) have unknown revenue. Treating these as real zeros would have corrupted every budget/revenue-based structured query (e.g. "top 10 by revenue" would have been dominated by legitimate $0-revenue indie films sorting incorrectly, or "budget above $100M" excluding nothing when it should exclude unknowns).
- `runtime`: same treatment — 37 movies have runtime 0, treated as unknown (`runtime_clean = NaN`).
- `overview` / `tagline`: missing values filled with `""` **only** in dedicated `overview_filled` / `tagline_filled` columns used for RAG document construction — the original `overview` / `tagline` columns are left as `NaN` for anywhere that distinguishing "actually empty" from "no data" matters.
- `vote_count`: filled with 0 where missing (rare) and cast to `int`.

**Fields used for structured search:** `vote_average`, `vote_count`, `runtime_clean`, `release_year`, `budget_clean`, `revenue_clean`, `popularity`, `original_language`, `director`, `status`, `title`, and the list fields `genre_list`, `keyword_list`, `cast_list`, `production_company_list`. Restricted to a field whitelist (`ALLOWED_FIELDS` in `structured_search.py`) so the LLM cannot pass an arbitrary/unsafe column name through to pandas.

**Fields used for fuzzy search:** `title` and `original_title`, normalized (lowercased, punctuation stripped) before matching. Both are indexed so non-English original titles still resolve.

**Fields used for semantic retrieval:** see Section 4 (RAG Pipeline) — a constructed document per movie, not raw columns.

**What's preserved for generation:** the full original dataframe (all raw + derived columns) is available to `movie_details`; nothing is dropped at preprocessing time. The RAG document (Section 4) is a *subset* view built specifically for embedding, not a replacement for the full record.

---

## 2. Agent Design

**How routing works:** the agent uses Gemini's native function-calling API (`gemini-3.5-flash-lite`). Five tools are declared as `types.FunctionDeclaration` schemas with typed parameters; the model is given the user's message plus conversation history and decides which tool(s) to call, with what arguments. My code executes the actual deterministic Python function, returns the JSON result to the model, and the model either calls another tool or produces final text. This loop runs up to 4 hops per turn.

**Which tools exist and their boundaries:**

| Tool | Boundary |
|---|---|
| `structured_search` | Filters/sorts/aggregates/group-by over the dataframe. The LLM supplies field/operator/value; pandas computes the actual result. The LLM never sees or invents a number that didn't come from this tool. |
| `fuzzy_movie_search` | Resolves a possibly-misspelled/partial title to a `movie_uid`, or reports ambiguous/not-found. Does not return movie details itself. |
| `semantic_search` | Retrieves top-k movies by embedding cosine similarity, either from a free-text query or from another movie's own embedding (`seed_title`, for "similar to X"). Returns titles + scores + documents, not an answer. |
| `movie_details` | Full structured record for exactly one movie, resolved either by title (fuzzy-matched) or by `reference: first/last` against the previous turn's result set. |
| `rag_answer` | Generates natural-language text grounded in the movies already retrieved by `semantic_search` (via `state.last_retrieved`). Does not retrieve on its own unless state is empty (fallback). |

**When multiple tools are invoked (real examples from testing):**
- *"How many movies have Christopher Nolan as director?"* → not directly tested with this exact phrasing, but the equivalent pattern was: `structured_search(filters=[director eq "Christopher Nolan"], aggregate="count")` — a single call, since `director` is a plain field, no fuzzy resolution needed unless the name itself is misspelled.
- *"Tell me about Intersteler"* → `fuzzy_movie_search("Intersteler")` → `movie_details(title="Interstellar")`. Two calls, sequential dependency (details tool needs the resolved title).
- *"I want a movie about someone trying to survive alone on another planet"* → `semantic_search(query=...)` → `movie_details(title="The Martian")`. Notably, the agent chose to fetch full structured details of the top semantic hit rather than call `rag_answer` — a valid alternative to the assignment's example flow (`semantic_search → RAG → grounded answer`), since a single strong match is often better served by a full record than a generated summary. See Section 6, example 5, for a case where it *did* use the RAG path.
- *"Find me a funny science-fiction movie from after 2010 that is under 2 hours"* → single `structured_search` call with filters `genre_list contains_any [Science Fiction, Comedy]`, `release_year > 2010`, `runtime_clean < 120`. The agent interpreted "funny" as a **structured genre tag** (Comedy) rather than a semantic re-ranking signal — see Section 3/6 for discussion of why this is a reasonable but not the only valid interpretation.

**Why this architecture was selected:** native tool-calling (rather than prompting the model to output free-form JSON and parsing it manually) gives a *typed* contract — the model can only call one of 5 declared functions with schema-validated arguments, which is what makes the "the system should not allow the LLM to invent movie metadata" requirement enforceable: every fact-bearing tool result is produced by deterministic Python, and the system instruction (Section 4) explicitly forbids the model from adding anything not present in a tool result.

---

## 3. Search System

### Structured Search

Implemented as a single `structured_search()` function (`src/structured_search.py`) taking a list of `Filter(field, op, value)` objects plus optional `sort_by`/`sort_desc`/`limit`/`aggregate`/`group_by`.

- **Filtering:** numeric comparisons (`gt`, `gte`, `lt`, `lte`, `eq`) applied directly via pandas boolean masks. List-valued fields (`genre_list`, `cast_list`, etc.) use `contains_any` (set-intersection non-empty) semantics rather than plain `eq`, since a movie can have multiple genres/cast members.
- **Sorting:** any allowed field, ascending or descending, `na_position="last"` so missing budget/revenue don't corrupt the ordering by sorting first.
- **Multi-condition queries:** filters are applied sequentially (each narrows the working dataframe); order doesn't affect the result set, only intermediate row counts logged for debugging.
- **Aggregation:** a dedicated `aggregate="count"` mode was added after testing revealed a bug (below) — counts are computed on the **full filtered set before any limit is applied**, never inferred from a truncated preview.
- **Group-by:** `explode()` + `groupby().size()` for list fields (e.g. genre counts), plain `groupby().size()` for scalar fields (e.g. movies per year).
- **Date ranges:** `release_year` (derived from parsed `release_date`) is used for year-based filtering rather than raw date strings.

**A real bug found during testing:** initially, `structured_search` only returned a `limit`-truncated dataframe, and "how many movies were released after 2010?" resolved by counting the *displayed* 20-row page (`row_count: 20`), not the true total. The agent actually noticed this was wrong mid-conversation and self-corrected by re-querying with a huge limit — clever, but fragile and not something to rely on. Fixed by adding an explicit `aggregate="count"` mode that always counts the full filtered set, with the tool's LLM-facing description explicitly instructing "if the user asks 'how many', set aggregate='count' — do not rely on row counts from a limited record list."

### Fuzzy Search

Implemented in `src/fuzzy_search.py` using **rapidfuzz**'s `WRatio` scorer (a composite of simple ratio, partial ratio, and token-sort ratio — chosen because it handles typos, partial titles, and reordered words in one metric, all three of which appear in the assignment's own examples).

**Normalization:** lowercase, strip all non-alphanumeric characters, collapse whitespace, before matching. Matched against both `title` and `original_title` so non-English original titles are still findable.

**Thresholds (0–100 score):**
- `>= 90` (`HIGH_CONFIDENCE`): confident match, provided there's no tie with another candidate at the same score.
- `>= 80` **and** beating the runner-up by `>= 8` points (`MIN_SCORE_FOR_MARGIN_PROMOTION` / `CLEAR_WINNER_MARGIN`): also treated as confident. This exists because short single-word queries with one typo (e.g. "Intersteler" → "Interstellar") cap out around 87 under WRatio — below the naive 90 threshold — but are unambiguously the intended movie once you look at the gap to the next-best candidate.
- `>= 70` (`LOW_CONFIDENCE`) and below the above: **ambiguous** — return all candidates, do not guess.
- `< 70`: **not found**.

**Ambiguity handling verified by test:** "lord of the rings" returns exactly 3 candidates all scoring 90.0 (Fellowship, Two Towers, Return of the King) — genuinely tied, correctly reported as ambiguous rather than picking one arbitrarily (`tests/test_fuzzy_search.py::test_genuinely_ambiguous_title_is_not_guessed`).

---

## 4. RAG Pipeline

**Document construction:** one text document per movie (`build_rag_document`), containing `Title`, `Tagline` (if present), `Overview`, `Genres`, `Keywords`, `Cast` (top 8 names), and `Director`. Deliberately excludes budget/revenue/vote numbers — those carry no semantic/distributional meaning for similarity matching and are handled by `structured_search` instead. Example (Avatar):

Title: Avatar
Tagline: Enter the World of Pandora.
Overview: In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization.
Genres: Action, Adventure, Fantasy, Science Fiction
Keywords: culture clash, future, space war, space colony, society, ...
Cast: Sam Worthington, Zoe Saldana, Sigourney Weaver, Stephen Lang, ...
Director: James Cameron


**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, CPU-friendly, no external API dependency/cost for embeddings). Encoding all 4,803 documents takes several minutes on CPU on first run; results are cached to `data/movie_embeddings.npy` so subsequent runs load in ~1 second.

**Vector database/index:** no external vector DB — a plain in-memory NumPy array with cosine similarity via normalized dot product (`SemanticIndex` in `src/semantic_search.py`). Justified by scale: 4,803 × 384-dim floats is ~7MB, trivially fits in memory, and a full brute-force similarity scan over that size is sub-second. Introducing Chroma/FAISS/Pinecone would add real dependency and operational weight without a measurable benefit at this dataset size.

**Retrieval strategy:** cosine similarity, brute-force top-k (`np.argsort`). Two query modes:
1. **Free-text query** — embeds the user's natural-language description directly.
2. **Seed-title mode** ("movies similar to X") — resolves `X` via fuzzy search first, then uses *that movie's own embedding* as the query vector, rather than embedding the literal phrase "similar to X." This was a deliberate design choice after testing showed embedding a bare movie name in isolation carries very little distributional signal (a short proper noun doesn't share vocabulary with anything).

**Top-k:** defaults to 8, overridable per call (`top_k` argument in the tool schema).

**Metadata filtering / hybrid search:** the current implementation does not do true structured-filter-then-semantic-rerank hybrid retrieval (the assignment's described "ideal" pattern). Instead, when the query mixes a structured constraint with a fuzzy semantic descriptor (e.g. "funny science-fiction... under 2 hours"), the router LLM tends to fold everything into a single `structured_search` call, treating adjective-like terms ("funny") as genre tags where a plausible one exists (Comedy). This is a legitimate interpretation but not the assignment's literal "filter then semantically re-rank within the filtered set" design — documented as a known limitation in Section 7.

**Reranking:** none — raw cosine similarity order is used directly.

**Context construction:** `rag_answer`'s prompt includes each retrieved movie's full document text plus its similarity score, formatted as a numbered list, followed by the user's question and a system prompt (below).

**Generation model:** Gemini (`gemini-3.5-flash-lite` for both routing and generation, to conserve free-tier quota — `gemini-3.6-flash`'s free tier is limited to 20 requests/day, discovered mid-development and worked around by switching models).

**Hallucination mitigation** — this required three iterations to get right, documented here because the iteration itself is informative:
1. **First attempt:** a `rag_answer`-only system prompt forbidding invented facts. Worked for `rag_answer` calls, but `movie_details` calls (a separate tool, not routed through `rag_answer`) were unconstrained — testing "Tell me about Interstellar" produced a fluent answer that included Kip Thorne's involvement, the "Gargantua" black hole name, and Hans Zimmer's score — none of which are in our dataset at all, pulled straight from the model's training knowledge.
2. **Second attempt:** added a global `AGENT_SYSTEM_INSTRUCTION` (not just the `rag_answer` prompt) telling the model to treat tool results as the only ground truth. This removed the production trivia, but a subtler leak remained: the model listed cast as `"Michael Caine as Professor John Brand"` — the flat `cast` field in our tool result contains actor *names only*, no character mapping, so any character attribution is invented, even though it happens to be factually correct for a well-known film.
3. **Final fix:** explicit rule in the system instruction: *"The 'cast' field is a flat list of ACTOR NAMES ONLY — it does NOT tell you which character each actor played. NEVER state or imply which character an actor played... unless a character name is literally present in the tool result."* Verified fixed by re-running the same query and confirming cast is now listed as plain names.

Also tested and confirmed: `rag_answer` correctly refuses to answer when given empty context ("I couldn't find a good match... unable to share details") and correctly identifies irrelevant retrieved context as irrelevant rather than forcing a connection (asked about a "war veteran seeking revenge" with `Minions`/`Frozen` as context — it explicitly said neither movie matches).

---

## 5. Multi-Turn Memory

State is held in an explicit `ConversationState` object (`src/agent.py`), not implicitly inferred from chat history text:

- `last_result_uids: list[int]` — the `movie_uid`s of the most recent result set, **in the exact order they were displayed to the user**. This is the mechanism behind resolving "the first one," "that one," etc.
- `last_filters: list[Filter]` — the filters behind `last_result_uids`, available for follow-up composition (not currently required by any tested example, but present for extension).
- `last_retrieved: list[dict]` — the most recent `semantic_search` results (with full documents attached), consumed by `rag_answer` for grounding.
- Full Gemini `contents` history is also carried turn-to-turn for language-level context (pronoun resolution, conversational tone) — but explicitly **not relied upon** for anything numeric or factual; that always comes from `ConversationState`.

**A real bug found and fixed here too:** `state.last_result_uids` was initially set from the filtered-but-unsorted dataframe, while the table shown to the user was sorted. So "tell me about the first one" after "only show ones rated above 7.5" resolved to whatever pandas' default row order put first — not what the user saw as first on screen. Fixed by capturing `all_uids` from the dataframe *after* sorting but *before* the `limit` truncation, so memory order matches displayed order exactly. Verified via a regression test (`test_all_uids_preserves_sort_order_before_limit`) and via the full 3-turn scenario run live in the Streamlit app (Section 6, example 6).

**What persists between turns:** `last_result_uids`, `last_filters`, `last_retrieved`, and the raw Gemini conversation history.
**What does not persist:** nothing is summarized/compressed across turns — history grows for the session's duration (acceptable at this scale; would need pruning for very long sessions, noted in Section 7).

---

## 6. Example Queries & Results

All outputs below are real, captured from actual runs against the live Streamlit app (not hypothetical).

**1. Aggregation — "What are the top 10 movies by revenue?"**
→ `structured_search(sort_by=revenue_clean, sort_desc=True, limit=10)`. Correct: Avatar ($2.79B), Titanic ($1.85B), The Avengers, Jurassic World, Furious 7, Avengers: Age of Ultron, Frozen, Iron Man 3, Minions, Captain America: Civil War.

**2. Group-by — "How many movies are there in each genre?"**
→ `structured_search(group_by=genre_list)`. Correct, all 20 genres: Drama 2,297, Comedy 1,722, Thriller 1,274, Action 1,154, ... down to TV Movie 8.

**3. Filtering with a field the schema initially omitted — "Which production companies have the most movies?"**
First attempt returned *"That information isn't available in the dataset"* — a false negative. The field (`production_company_list`) existed in the dataframe and was already whitelisted in `structured_search.py`, but the Gemini tool schema's field description string didn't list it, so the router LLM didn't know it could ask for it. Fixed by expanding the schema description. Re-run: correct top-10 (Warner Bros. 319, Universal Pictures 311, Paramount 285, ...).

**4. Fuzzy title lookup with a bug — "Tell me about Avatr"**
First attempt returned **"My Date with Drew" (2005)** — completely wrong. Root cause: `movie_details`'s dispatch function checked `args["reference"]` before `args["title"]`, and the model had (reasonably) passed both `reference="last"` and `title="Avatar"` in the same call; stale `reference` state from an unrelated prior query won out over the explicit, correct title. Fixed by reordering the priority so an explicit `title` argument always takes precedence over `reference`. Re-run: correctly returns Avatar (2009), full record, from a `fuzzy_movie_search` → `movie_details` two-step trace.

**5. Semantic search with grounded generation — "I'm looking for an emotional movie about friendship and overcoming difficult circumstances"**
This is the query that **does not work well** (required by the assignment). The literal phrase embeds poorly: top semantic hits were generic/unrelated titles ("Dysfunctional Friends," "Nothing"). In the live agent test, the model tried `semantic_search` twice with reworded queries, both weak, then fell back to a broad "top-rated Drama" `structured_search`, picked The Shawshank Redemption via `movie_details` (a thematically defensible answer), and then ran out of the 4-hop tool-call budget before producing final text — the turn ended with "I wasn't able to settle on an answer within the allotted tool-call budget." **Why it fails:** MiniLM-L6-v2 embeddings capture surface vocabulary and concrete plot elements well but are weaker on abstract emotional/thematic language ("overcoming difficult circumstances") that doesn't share vocabulary with how movie overviews are actually written. A larger/fine-tuned embedding model, or a query-rewriting step before embedding, would likely help. Documented here rather than special-cased/hidden.

**6. Hybrid search — "Find me a funny science-fiction movie from after 2010 that is under 2 hours"**
→ single `structured_search` call: `genre_list contains_any [Science Fiction, Comedy]`, `release_year > 2010`, `runtime_clean < 120`. Returned *The World's End* (2013, Comedy/Action/Sci-Fi, 109 min, dir. Edgar Wright). Correct and sensible, but notably interprets "funny" as a **genre filter**, not a semantic re-ranking signal within a structured-filtered set — a different (simpler) mechanism than the assignment's suggested hybrid pattern. Discussed further in Section 7.

**7. Multi-turn memory (3 connected turns, run live in the Streamlit UI):**
- *Turn 1:* "Show me science fiction movies from after 2010" → 20-of-140 shown, unsorted-by-rating list (Interstellar, Guardians of the Galaxy, Mad Max: Fury Road, ...).
- *Turn 2:* "Only show ones rated above 7.5" → correctly filters *within* Turn 1's SciFi/post-2010 result set (not a fresh unrelated query): 8 movies, Interstellar first.
- *Turn 3:* "Tell me more about the first one" → correctly resolves to **Interstellar** (rank 1 in Turn 2's displayed list), full grounded record via `movie_details`.

**8. Ambiguity handling — "Tell me about lord of the rings"**
→ `fuzzy_movie_search("lord of the rings")` returns `status: ambiguous` with 5 candidates (3 genuine LOTR films tied at score 90, plus 2 lower-scoring irrelevant titles). Agent correctly stops immediately (after a system-instruction fix — see below) and asks: *"There are multiple films in The Lord of the Rings series in the dataset. Did you mean [Fellowship], [Two Towers], or [Return of the King]?"* **A related bug found and fixed:** before adding an explicit "stop on ambiguous/not_found" rule to the system instruction, the model would additionally fire off irrelevant `structured_search` (treating "Lord of the Rings" as a literal genre) and `semantic_search` calls after already getting an ambiguous result — wasted calls that didn't change the final answer but burned quota. Fixed with an explicit rule 6 in `AGENT_SYSTEM_INSTRUCTION`.

**9. Not-found handling — "Tell me about asdkjaskjd"**
→ `fuzzy_movie_search` returns `not_found` (top candidate scores below 70), agent responds: *"I couldn't find any movie matching 'asdkjaskjd' in the dataset. Did you mean something else?"* — no hallucinated movie.

---

## 7. Conclusions

**Known limitations:**
- **Semantic retrieval weakness on abstract/emotional phrasing** (Section 6, example 5) — the most significant retrieval-quality gap found through testing.
- **Hybrid search is filter-first, not filter-then-semantic-rerank.** The current agent tends to fold adjective-like semantic descriptors into genre filters when a plausible genre tag exists, rather than performing the assignment's suggested "structured filter → semantically rerank within that subset" pattern. This works for "funny" → Comedy but wouldn't generalize to a descriptor with no corresponding genre (e.g. "melancholic").
- **Tool schema completeness is easy to get subtly wrong** — the production-company bug (example 3) wasn't a data or logic bug; the underlying capability existed and worked once exposed. This suggests the LLM-facing tool description strings are a real, non-obvious surface area that needs deliberate test coverage, not just the underlying Python functions.
- **Fixed 4-hop tool-call budget** can be exhausted by queries that need several retries (example 5) before ever reaching a final answer, rather than gracefully returning a lower-confidence answer sooner.
- **Free-tier LLM rate limits** (20 requests/day on `gemini-3.6-flash`) constrained how much live end-to-end testing was practical; `gemini-3.5-flash-lite` was used instead for its separate quota, which is untested at higher-quality reasoning tasks than what this project required.
- **No persistent conversation-history pruning** — a very long session would grow the Gemini `contents` payload unboundedly; not an issue at the scale tested (a handful of turns) but would need addressing for a production deployment.
- **Character-to-actor mapping is unavailable in structured search/RAG**, by design (the flat `cast_list` intentionally drops it to avoid the hallucination risk described in Section 4) — a real feature gap if a user specifically wants "who played X in movie Y," not just a display quirk.

**Data-quality issues:** ~21.6% of movies have unknown budget and ~29.7% have unknown revenue (zero-value rows correctly excluded rather than misrepresented as $0, but this means budget/revenue-based queries are systematically working with an incomplete subset of the catalog — worth surfacing to end users, which the current UI does not do explicitly).

**LLM limitations observed directly:** even with an explicit, iterated-on system instruction, the model initially added invented factual claims (production trivia, character-actor mappings) that required two rounds of testing and prompt-hardening to fully suppress — a reminder that "don't hallucinate" instructions are necessary but not sufficient without testing against the model's actual failure modes, and that grounding constraints need to be applied uniformly across *every* tool that presents movie facts, not just the RAG-labeled one.

**Scalability considerations:** the current brute-force NumPy cosine similarity search and full-dataframe pandas filtering are appropriate at 4,803 rows (sub-second) but would need a real vector index (FAISS/Chroma) and possibly a proper database (rather than an in-memory parquet-backed dataframe) at a materially larger catalog size — e.g. tens of thousands of movies or more.

**What I'd improve with another week:**
1. Query rewriting/expansion before semantic embedding, to address the abstract-phrasing weakness directly.
2. True hybrid retrieval: structured pre-filter, then semantic rerank within the filtered candidate set, rather than folding everything into one structured_search call.
3. A lightweight LLM-based reranking or confidence-scoring step after retrieval, surfaced in the UI (we already compute similarity scores and show them in the trace, but don't act on low-confidence results beyond a single threshold flag).
4. Automated tests around the agent's tool-schema completeness (e.g. asserting every `ALLOWED_FIELD` in `structured_search.py` also appears in the Gemini tool description) to catch the class of bug in example 3 before it reaches manual testing.
5. Deployment to a public URL with basic rate-limiting to protect API quota during review.