
### 5. Run the app

```bash
streamlit run app.py
```

The first run will preprocess the dataset and build the semantic embedding index
(cached to `data/` for subsequent runs — this can take several minutes on first
run, seconds after).

**Note (macOS):** if you hit an `OMP: Error #15` crash related to `libiomp5.dylib`,
this is a known local OpenMP conflict between certain ML libraries. `app.py`
already sets `KMP_DUPLICATE_LIB_OK=TRUE` to work around it; if running other
scripts directly, prefix the command with `KMP_DUPLICATE_LIB_OK=TRUE`.

## Running tests

```bash
pytest tests/ -v
```

Tests cover the deterministic components (data preprocessing, structured search,
fuzzy search) directly — no LLM calls are made in the test suite, so tests run
free of API cost/quota.

## Example queries to try

- "What are the top 10 movies by revenue?"
- "How many movies are there in each genre?"
- "Which production companies have the most movies?"
- "Tell me about Intersteler" (typo-tolerant fuzzy match)
- "I want a movie about someone trying to survive alone on another planet" (semantic search)
- "Find movies similar to Inception"
- "Show me science fiction movies from after 2010" → "Only show ones rated above 7.5" → "Tell me more about the first one" (multi-turn memory)

## Known limitations

See `DESIGN.md` section 7 (Conclusions) for a full discussion. Briefly:
- Semantic search quality varies by query phrasing — abstract emotional themes
  (e.g. "friendship and overcoming difficult circumstances") retrieve more weakly
  than concrete plot descriptions.
- The agent's tool-call budget (4 hops) can be exhausted on queries requiring
  multiple retry strategies.
- Uses Gemini's free tier, which has daily per-model rate limits.