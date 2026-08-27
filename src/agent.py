"""
Agent layer: routes a natural-language user turn to one or more tools via
Gemini function-calling, maintains multi-turn conversational state, and
assembles the final response.

Design:
  - Routing is done by the LLM via native tool-calling, not regex/keyword
    rules -- gives a typed interface: the model can only invoke the 5
    defined tools with validated arguments.
  - Tool functions are 100% deterministic/data-backed; the LLM decides
    WHICH tool and WHAT arguments, never computes results itself.
  - Multi-turn memory is an explicit state object (ConversationState), not
    "put everything in the prompt and hope" -- last_result_uids preserves
    the exact displayed order so "the first one" resolves correctly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from google import genai
from google.genai import types

from src.fuzzy_search import FuzzyIndex
from src.rag_answer import generate_rag_answer
from src.semantic_search import SemanticIndex
from src.structured_search import Filter, StructuredSearchError, structured_search


# ---------------------------------------------------------------------------
# JSON sanitization (NaN is not valid JSON; numpy scalar types aren't
# natively serializable) -- required before returning any tool result to
# the Gemini API.
# ---------------------------------------------------------------------------
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if math.isnan(obj) else float(obj)
    return obj


# ---------------------------------------------------------------------------
# Conversation state (multi-turn memory)
# ---------------------------------------------------------------------------
@dataclass
class ConversationState:
    last_result_uids: list[int] = field(default_factory=list)
    last_filters: list[Filter] = field(default_factory=list)
    last_retrieved: list[dict] = field(default_factory=list)  # for rag_answer grounding


# ---------------------------------------------------------------------------
# Tool schemas (Gemini function declarations)
# ---------------------------------------------------------------------------
def build_tools() -> types.Tool:
    structured_search_tool = types.FunctionDeclaration(
        name="structured_search",
        description=(
            "Deterministic filtering/sorting/aggregation over the movie dataset. "
            "Use for counts, top-N by a numeric field, filters on rating/runtime/"
            "year/budget/genre/cast/director, and group-by counts. IMPORTANT: if the "
            "user asks 'how many', set aggregate='count' -- do not rely on row counts "
            "from a limited record list."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {
                "filters": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "field": {"type": "STRING", "description": (
                                "One of: vote_average, vote_count, runtime_clean, "
                                "release_year, budget_clean, revenue_clean, popularity, "
                                "genre_list, cast_list, director"
                            )},
                            "op": {"type": "STRING", "enum": ["eq", "gt", "gte", "lt", "lte", "contains_any"]},
                            "value": {"type": "STRING"},
                        },
                        "required": ["field", "op", "value"],
                    },
                },
                "sort_by": {"type": "STRING"},
                "sort_desc": {"type": "BOOLEAN"},
                "limit": {"type": "INTEGER", "description": "max records to return, ignored if aggregate is set"},
                "aggregate": {"type": "STRING", "enum": ["count"], "description": "set to 'count' for 'how many' questions"},
                "group_by": {"type": "STRING", "description": "e.g. genre_list, release_year -- for 'movies per genre' style questions"},
            },
        },
    )

    fuzzy_search_tool = types.FunctionDeclaration(
        name="fuzzy_movie_search",
        description="Resolve a possibly-misspelled or partial movie title to an exact movie in the dataset.",
        parameters={
            "type": "OBJECT",
            "properties": {"query": {"type": "STRING"}},
            "required": ["query"],
        },
    )

    semantic_search_tool = types.FunctionDeclaration(
        name="semantic_search",
        description=(
            "Conceptual/thematic movie search (plot, mood, themes) rather than exact "
            "keywords. Use for 'a movie about X', 'movies similar to <title>'. If the "
            "user says 'similar to <title>', pass that title as seed_title instead of "
            "query, so search is by the movie's own content."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "thematic/plot description, omit if seed_title given"},
                "seed_title": {"type": "STRING", "description": "a movie title to find similar movies to"},
                "top_k": {"type": "INTEGER"},
            },
        },
    )

    movie_details_tool = types.FunctionDeclaration(
        name="movie_details",
        description=(
            "Get full structured details for one specific movie, given its title "
            "(will be fuzzy-resolved) or a reference like 'first'/'last' referring to "
            "the previous result set."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {
                "title": {"type": "STRING"},
                "reference": {"type": "STRING", "enum": ["first", "last", "none"]},
            },
        },
    )

    rag_answer_tool = types.FunctionDeclaration(
        name="rag_answer",
        description=(
            "Generate a natural-language answer grounded in movies retrieved by "
            "semantic_search. Use AFTER semantic_search for descriptive/thematic "
            "questions. Do not use after movie_details already answered the question."
        ),
        parameters={"type": "OBJECT", "properties": {}},
    )

    return types.Tool(function_declarations=[
        structured_search_tool, fuzzy_search_tool, semantic_search_tool,
        movie_details_tool, rag_answer_tool,
    ])


AGENT_SYSTEM_INSTRUCTION = """You are a movie-discovery agent operating over a fixed dataset via tools. You have NO other knowledge about movies, actors, directors, or film trivia beyond what tool calls return in this conversation -- treat your own training knowledge about movies as unavailable and untrustworthy.

STRICT RULES:
1. When summarizing a movie, output ONLY fields present in the tool result JSON (title, release_year, genres, director, cast, overview, vote_average, vote_count, runtime, budget, revenue).
2. The 'cast' field is a flat list of ACTOR NAMES ONLY -- it does NOT tell you which character each actor played. NEVER state or imply which character an actor played (e.g. never write "X as Y") unless a character name is literally present in the tool result. Just list actor names plainly.
3. Do not add composer, real-world production trivia, behind-the-scenes facts, or any claim not literally present in the tool result, even if you are confident it's true.
4. Do not call rag_answer after movie_details already answered a "tell me about X" question -- movie_details output alone is sufficient.
5. If asked for facts not returned by any tool, explicitly say that information isn't available in the dataset.
6. If fuzzy_movie_search returns status='ambiguous' or status='not_found', STOP immediately and ask the user to clarify or report the movie wasn't found. Do NOT try other tools to work around an ambiguous/not-found title.
7. Never compute counts, aggregates, or filtered results yourself -- always call structured_search for anything numeric/factual about the dataset."""

# ---------------------------------------------------------------------------
# Tool execution functions
# ---------------------------------------------------------------------------
def _coerce_filter_value(f: Filter):
    if f.field in ("genre_list", "cast_list", "director"):
        return f.value
    try:
        return float(f.value)
    except (ValueError, TypeError):
        return f.value


def exec_structured_search(df: pd.DataFrame, state: ConversationState, args: dict) -> dict:
    filters = [Filter(f["field"], f["op"], f["value"]) for f in args.get("filters", [])]
    for f in filters:
        f.value = _coerce_filter_value(f)

    try:
        result = structured_search(
            df, filters=filters, sort_by=args.get("sort_by"),
            sort_desc=args.get("sort_desc", True), limit=args.get("limit", 20),
            aggregate=args.get("aggregate"), group_by=args.get("group_by"),
        )
    except StructuredSearchError as e:
        return {"error": str(e)}

    state.last_result_uids = result["all_uids"]
    state.last_filters = filters

    if result["mode"] == "count":
        return {"mode": "count", "count": result["count"]}

    if result["mode"] == "group_by":
        return {"mode": "group_by", "results": result["display"].to_dict(orient="records")}

    cols = [c for c in ["title", "release_year", "vote_average", "vote_count",
                         "revenue_clean", "budget_clean", "runtime_clean"]
            if c in result["display"].columns]
    return {
        "mode": "records",
        "total_matching": result["total_matching"],
        "shown": result["shown"],
        "results": result["display"][cols].to_dict(orient="records"),
    }


def exec_fuzzy_search(fuzzy_index: FuzzyIndex, args: dict) -> dict:
    r = fuzzy_index.find(args["query"])
    if r["status"] == "matched":
        return {"status": "matched", "title": r["match"].title, "movie_uid": r["match"].movie_uid, "score": r["match"].score}
    return {
        "status": r["status"],
        "candidates": [{"title": c.title, "score": c.score} for c in r["candidates"]],
    }


def exec_semantic_search(df: pd.DataFrame, semantic_index: SemanticIndex,
                          fuzzy_index: FuzzyIndex, state: ConversationState, args: dict) -> dict:
    seed_uid = None
    if args.get("seed_title"):
        r = fuzzy_index.find(args["seed_title"])
        if r["status"] != "matched":
            return {"error": f"Could not confidently resolve seed title '{args['seed_title']}' ({r['status']})",
                    "candidates": [c.title for c in r["candidates"]]}
        seed_uid = r["match"].movie_uid

    if seed_uid is None and not args.get("query"):
        return {"error": "semantic_search requires either query or seed_title"}

    result = semantic_index.search(query=args.get("query"), top_k=args.get("top_k", 8), seed_title_uid=seed_uid)

    retrieved = []
    for m in result["results"]:
        doc = df.loc[df["movie_uid"] == m["movie_uid"], "rag_document"].iloc[0]
        retrieved.append({"title": m["title"], "score": m["score"], "movie_uid": m["movie_uid"], "document": doc})

    state.last_retrieved = retrieved
    state.last_result_uids = [r["movie_uid"] for r in retrieved]
    return {
        "status": result["status"],
        "results": [{"title": r["title"], "score": round(r["score"], 3)} for r in retrieved],
    }


def exec_movie_details(df: pd.DataFrame, fuzzy_index: FuzzyIndex, state: ConversationState, args: dict) -> dict:
    uid = None
    if args.get("reference") in ("first", "last") and state.last_result_uids:
        uid = state.last_result_uids[0] if args["reference"] == "first" else state.last_result_uids[-1]
    elif args.get("title"):
        r = fuzzy_index.find(args["title"])
        if r["status"] != "matched":
            return {"status": r["status"], "candidates": [c.title for c in r["candidates"]]}
        uid = r["match"].movie_uid

    if uid is None:
        return {"error": "movie_details requires a resolvable title or reference"}

    row = df[df["movie_uid"] == uid]
    if row.empty:
        return {"status": "not_found"}
    r = row.iloc[0]
    return {
        "title": r["title"],
        "release_year": None if pd.isna(r["release_year"]) else int(r["release_year"]),
        "genres": list(r["genre_list"]),
        "director": r["director"] if r["director"] and not pd.isna(r["director"]) else None,
        "cast": list(r["cast_list"]),
        "overview": r["overview_filled"],
        "runtime": None if pd.isna(r["runtime_clean"]) else float(r["runtime_clean"]),
        "vote_average": float(r["vote_average"]) if not pd.isna(r["vote_average"]) else None,
        "vote_count": int(r["vote_count"]),
        "budget": None if pd.isna(r["budget_clean"]) else float(r["budget_clean"]),
        "revenue": None if pd.isna(r["revenue_clean"]) else float(r["revenue_clean"]),
    }


def exec_rag_answer(client: genai.Client, model_name: str, semantic_index: SemanticIndex,
                     df: pd.DataFrame, state: ConversationState, user_query: str) -> dict:
    retrieved = state.last_retrieved
    if not retrieved:
        result = semantic_index.search(query=user_query, top_k=6)
        retrieved = []
        for m in result["results"]:
            doc = df.loc[df["movie_uid"] == m["movie_uid"], "rag_document"].iloc[0]
            retrieved.append({"title": m["title"], "score": m["score"], "document": doc})

    try:
        answer = generate_rag_answer(client, model_name, user_query, retrieved)
    except RuntimeError as e:
        return {"error": str(e)}
    return {"answer": answer, "grounded_on": [r["title"] for r in retrieved]}


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------
@dataclass
class AgentTurnResult:
    text_response: str
    trace: list[dict]


def run_agent_turn(
    user_message: str,
    df: pd.DataFrame,
    fuzzy_index: FuzzyIndex,
    semantic_index: SemanticIndex,
    client: genai.Client,
    model_name: str,
    state: ConversationState,
    history: list | None = None,
) -> tuple[AgentTurnResult, list]:
    history = history or []
    trace: list[dict] = []
    tools = build_tools()
    config = types.GenerateContentConfig(tools=[tools], system_instruction=AGENT_SYSTEM_INSTRUCTION)

    contents = history + [types.Content(role="user", parts=[types.Part(text=user_message)])]

    try:
        response = client.models.generate_content(model=model_name, contents=contents, config=config)
    except Exception as e:  # noqa: BLE001
        return AgentTurnResult(f"Sorry, the routing model failed to respond: {e}", trace), contents

    max_hops = 4
    for _ in range(max_hops):
        candidate = response.candidates[0]
        function_calls = [p.function_call for p in candidate.content.parts if p.function_call]

        if not function_calls:
            final_text = "".join(p.text for p in candidate.content.parts if p.text)
            return AgentTurnResult(final_text or "I don't have a response for that.", trace), contents + [candidate.content]

        contents.append(candidate.content)
        tool_response_parts = []
        for fc in function_calls:
            name, args = fc.name, dict(fc.args)
            try:
                if name == "structured_search":
                    result = exec_structured_search(df, state, args)
                elif name == "fuzzy_movie_search":
                    result = exec_fuzzy_search(fuzzy_index, args)
                elif name == "semantic_search":
                    result = exec_semantic_search(df, semantic_index, fuzzy_index, state, args)
                elif name == "movie_details":
                    result = exec_movie_details(df, fuzzy_index, state, args)
                elif name == "rag_answer":
                    result = exec_rag_answer(client, model_name, semantic_index, df, state, user_message)
                else:
                    result = {"error": f"Unknown tool {name}"}
            except Exception as e:  # noqa: BLE001 - never let a tool crash the whole turn
                result = {"error": f"Tool '{name}' failed: {e}"}

            trace.append({"tool": name, "args": args, "result": result})
            tool_response_parts.append(
                types.Part.from_function_response(name=name, response={"result": clean_for_json(result)})
            )

        contents.append(types.Content(role="user", parts=tool_response_parts))
        try:
            response = client.models.generate_content(model=model_name, contents=contents, config=config)
        except Exception as e:  # noqa: BLE001
            return AgentTurnResult(f"Sorry, the model failed mid-conversation: {e}", trace), contents

    return AgentTurnResult(
        "I wasn't able to settle on an answer within the allotted tool-call budget -- "
        "try rephrasing or narrowing your question.", trace,
    ), contents