"""
Answer evaluator: scores answer faithfulness given sources.
This module encapsulates the evaluation logic so you can swap models easily.
"""

from typing import List, Dict, Any
from configurations.config import LLM_MODEL_NAME, OPENAI_API_KEY
from loggers.logger import logger
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# ------------------------------
# Utility to safely extract text from ChatOpenAI response
# ------------------------------
def extract_llm_output(resp: Any) -> str:
    """
    Safely extract the text from a ChatOpenAI response.
    Handles AIMessage, list of AIMessage, dicts, and plain strings.
    """
    if isinstance(resp, list) and hasattr(resp[0], "content"):
        return resp[0].content
    elif hasattr(resp, "content"):
        return resp.content
    elif isinstance(resp, dict):
        if "text" in resp:
            return resp["text"]
        elif "generations" in resp:
            return resp["generations"][0]["message"]["content"]
    return str(resp)

# ------------------------------
# Answer evaluation helper
# ------------------------------
def evaluate_answer(answer: str, sources: List[Any], question: str) -> Dict[str, Any]:
    """
    Use the LLM to score the answer's faithfulness to the provided sources.

    Returns a dict:
        {
          "score": int (0-100),
          "verdict": "faithful"|"not_faithful"|"partial",
          "comment": "short explanation"
        }

    Note: This will make an extra API call. Keep responses short to reduce cost.
    """
    # Build a compact context summarizing sources for the evaluator
    # We'll provide numbered source blocks so the evaluator can compare.
    numbered_sources = []
    for i, src in enumerate(sources, start=1):
        # src is typically a Document with .page_content and .metadata
        metadata = src.metadata if hasattr(src, "metadata") else {}
        preview = src.page_content[:1000].replace("\n", " ")
        numbered_sources.append(f"[{i}] Meta: {metadata}\n{preview}")

    eval_context = "\n\n".join(numbered_sources)

    # Create a short evaluation prompt
    eval_system = (
        "You are an evaluator that judges whether a given ANSWER is faithful to the provided SOURCES. "
        "Give a score from 0 to 100 (100 = fully faithful and supported by sources, 0 = not at all supported). "
        "Then provide a one-line verdict: faithful / partial / not_faithful, and a two-sentence rationale."
    )

    eval_user = (
        f"SOURCES:\n{eval_context}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}\n\n"
        "Please provide a JSON object with keys: score (int 0-100), verdict (string), comment (string)."
    )

    # Use the same ChatOpenAI but create a short-lived instance (we can reuse OPENAI settings)
    evaluator = ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=0.0, openai_api_key=OPENAI_API_KEY)
    # We will pass messages directly
    from langchain.schema import SystemMessage, HumanMessage

    # Prepare messages properly
    messages = [
        SystemMessage(content=eval_system),
        HumanMessage(content=eval_user)
    ]

    # Call the evaluator
    resp = evaluator(messages)
    # resp may be a string; try to parse JSON out of it. Keep parsing robust.
    # raw = resp.content if hasattr(resp, "content") else str(resp)
    # raw = resp['generations'][0]['message']['content']
    raw = extract_llm_output(resp)
    # raw = resp
    # Try to extract JSON - but keep fallback to a simple heuristic
    import json
    try:
        # Sometimes model outputs JSON directly; attempt parsing
        parsed = json.loads(raw.strip())
        # Ensure keys exist
        score = int(parsed.get("score", 0))
        verdict = parsed.get("verdict", "unknown")
        comment = parsed.get("comment", "")
    except Exception:
        # Fallback: do a simple heuristic parse (look for numbers)
        # This is a best-effort fallback; you can refine parsing/prompting for strict JSON output.
        score = 0
        verdict = "unknown"
        comment = raw.strip().replace("\n", " ")[:300]

    return {"score": score, "verdict": verdict, "comment": comment}
