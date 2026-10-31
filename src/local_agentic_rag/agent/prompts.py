"""
Prompts and JSON schemas for each agent step.

Written for 7B-class local models: short, imperative, one task per call,
structured steps constrained by JSON schema, free text only for the final
answer (small models write worse prose when forced into JSON).
"""

from __future__ import annotations

from local_agentic_rag.schema import ANSWER_NOT_FOUND, ScoredChunk

# --------------------------------------------------------------------------- planner

PLAN_SYSTEM = "You turn a user's question about a document into precise search queries."

PLAN_USER = """Document title: {title}

Conversation so far:
{history}

User question: {question}

Do two things:
1. "standalone_question": rewrite the user question so it can be understood without the conversation (resolve words like "it", "they", "that table"). If it is already standalone, copy it unchanged.
2. "search_queries": write 1 to 3 short keyword-rich search queries for finding the passages needed to answer it. If the question asks about several things (a comparison, a list of items, multiple parts), write one query per thing. Use terms likely to appear in the document.

Respond in JSON."""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "standalone_question": {"type": "string"},
        "search_queries": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
    "required": ["standalone_question", "search_queries"],
}

# --------------------------------------------------------------------------- translation

TRANSLATE_SYSTEM = "You translate search questions into the language of a document."

TRANSLATE_USER = """A user asked a question about a document written in another language.

Sample of the document's text:
{sample}

User question: {question}

Translate the question into the language of the document sample so it can be used to search the document. Keep names, numbers and technical terms. Respond in JSON with "translation"."""

TRANSLATE_SCHEMA = {
    "type": "object",
    "properties": {"translation": {"type": "string"}},
    "required": ["translation"],
}

TRANSLATE_ANSWER_SYSTEM = "You translate short answers between languages accurately."

TRANSLATE_ANSWER_USER = """Translate this answer from {source} into {target}.

Rules:
- Translate the meaning faithfully. Do not add, drop or change any fact.
- Keep every citation marker such as [1] or [2][3] exactly, right after the sentence it belongs to.
- Keep numbers as digits. Write names in the script of the target language.
- Do not add words about the passages (no "according to passage [1]"); translate only what the answer says.
- Reply with the translation only.

Answer in {source}:
{answer}

Translation:"""

# --------------------------------------------------------------------------- grader

GRADE_SYSTEM = "You judge whether retrieved document passages contain the information needed to answer a question."

GRADE_USER = """Question: {question}

Passages:
{passages}

Respond in JSON with:
- "relevant": the numbers of the passages that contain information useful for answering the question (empty list if none).
- "sufficient": true if the relevant passages together contain everything needed to answer the question fully, otherwise false.
- "missing_query": if not sufficient, a short search query for the missing information; otherwise "".
"""

GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {"type": "array", "items": {"type": "integer"}},
        "sufficient": {"type": "boolean"},
        "missing_query": {"type": "string"},
    },
    "required": ["relevant", "sufficient", "missing_query"],
}

# --------------------------------------------------------------------------- generator

ANSWER_SYSTEM = (
    "You are a careful assistant that answers questions about a document using only the "
    "numbered passages provided. You never use outside knowledge."
)

ANSWER_USER = """Passages from the document "{title}":

{passages}

Question: {question}

Rules:
- Use ONLY the passages above. If they do not contain the answer, reply with exactly {sentinel} and nothing else.
- Be concise and specific. Quote exact numbers, names and terms from the passages.
- Cite the passage number in square brackets after each sentence that uses it, e.g. [2] or [1][3].
- If the question has several parts and the passages answer only some of them, answer those parts (with citations) and say which part the document does not cover.
- Write the answer in {answer_language}, even if the passages are in another language. Keep names, numbers and the [n] citation markers.
{feedback}
Answer:"""

REGENERATE_FEEDBACK = (
    "- A previous draft made claims that the passages do not support: {claims}. "
    "Do not repeat them; only state what the passages say."
)

# --------------------------------------------------------------------------- verifier

VERIFY_SYSTEM = "You check whether an answer is fully supported by source passages."

VERIFY_USER = """Passages:
{passages}

Question: {question}

Answer to check:
{answer}

Is every factual claim in the answer (numbers, names, dates, statements) directly supported by the passages? Ignore the bracketed citation numbers themselves.

Respond in JSON with:
- "supported": true if every claim is supported, otherwise false.
- "unsupported_claims": list of the claims that are not supported (empty if none).
"""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["supported", "unsupported_claims"],
}


def format_passages(passages: list[ScoredChunk], max_chars: int | None = None) -> str:
    """Numbered passage block; numbers are the citation markers the model must use."""
    parts = []
    for i, sc in enumerate(passages, start=1):
        c = sc.chunk
        label = f"page {c.page}"
        if c.section:
            label += f" | section: {c.section[:80]}"
        if c.kind == "table":
            label += " | table"
        text = c.text if max_chars is None else c.text[:max_chars]
        parts.append(f"[{i}] ({label})\n{text}")
    return "\n\n".join(parts)


def format_history(history: list[tuple[str, str]] | None, turns: int) -> str:
    if not history or turns <= 0:
        return "(none)"
    lines = []
    for user, assistant in history[-turns:]:
        lines.append(f"User: {user}")
        lines.append(f"Assistant: {assistant[:400]}")
    return "\n".join(lines)


__all__ = [
    "ANSWER_NOT_FOUND",
    "ANSWER_SYSTEM",
    "ANSWER_USER",
    "GRADE_SCHEMA",
    "GRADE_SYSTEM",
    "GRADE_USER",
    "PLAN_SCHEMA",
    "PLAN_SYSTEM",
    "PLAN_USER",
    "REGENERATE_FEEDBACK",
    "VERIFY_SCHEMA",
    "VERIFY_SYSTEM",
    "VERIFY_USER",
    "format_history",
    "format_passages",
]
