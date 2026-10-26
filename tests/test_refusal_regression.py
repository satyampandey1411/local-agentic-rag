"""Refusal detection regressions (cases observed in the final Colab benchmark run)."""

from __future__ import annotations

import pytest

from local_agentic_rag.agent import prompts as P
from local_agentic_rag.agent.citations import is_refusal, strip_trailing_sentinel


@pytest.mark.parametrize(
    "answer",
    [
        # nist-ai-600-1-q21 (unanswerable) was scored as an answer because it carried citations
        "The document does not specify the exact number of people who took part in NIST's "
        "Generative AI Public Working Group [3][2]. It mentions that the group was open.",
        "The passage does not provide a specific Hugging Face identifier for the model [2][4]",
        "दस्तावेज़ में इसका उल्लेख नहीं है [1]।",
        "या प्रश्नाचे उत्तर दस्तऐवजात दिलेले नाही.",
        f"{P.ANSWER_NOT_FOUND}",
    ],
)
def test_declining_answers_are_refusals(answer):
    assert is_refusal(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "The corporate profits in Q2 2024 were -82.3 billion dollars [1]. "
        "The document does not specify the exact profits for Q2 2025.",
        "Retail trade contributed -0.54 percentage points [1]. Its rate is not mentioned in the passages.",
        "The document does not state the total, but it lists 9 actions [1].",
        "Brevity bias, literal bias and position bias [1].",
        "The document does not mention a dividend cut; it says a dividend of 0.50 was approved [4].",
    ],
)
def test_substantive_answers_are_not_refusals(answer):
    assert not is_refusal(answer)


def test_prompt_has_a_single_refusal_rule():
    # A second, stricter rule ("if the specific fact is not in the passages, reply exactly
    # <sentinel>: do not describe related content") was tried and measured on the Colab T4:
    # false refusals rose from 3.8% to 21.2% (single pass) because the 7B model refused
    # answerable questions. Unanswerable questions are handled by is_refusal() instead.
    prompt = P.ANSWER_USER.format(
        title="t",
        passages="[1] x",
        question="q",
        sentinel=P.ANSWER_NOT_FOUND,
        feedback="",
        answer_language="English",
    )
    assert "do not describe related content" not in prompt
    assert prompt.count(P.ANSWER_NOT_FOUND) == 1


def test_cited_answer_with_trailing_sentinel_is_kept():
    # Kaggle run, ml-hi-tajmahal-q06: a correct, cited two-part answer followed by the sentinel.
    raw = (
        "ताजमहल शाहजहाँ ने अपनी पत्नी मुमताज़ महल की याद में बनवाया था [1]. "
        f"यह 1983 में युनेस्को विश्व धरोहर स्थल घोषित किया गया था [4]. {P.ANSWER_NOT_FOUND}"
    )
    cleaned = strip_trailing_sentinel(raw)
    assert P.ANSWER_NOT_FOUND not in cleaned and cleaned.endswith("[4].")
    assert not is_refusal(cleaned)


@pytest.mark.parametrize(
    "raw",
    [
        f"{P.ANSWER_NOT_FOUND} [The passages do not provide the Hugging Face identifier.]",
        f"I looked through the passages. {P.ANSWER_NOT_FOUND}",  # no citation before it
        f"The passages do not provide the identifier [2]. {P.ANSWER_NOT_FOUND}",
    ],
)
def test_sentinel_without_a_cited_answer_is_still_a_refusal(raw):
    assert is_refusal(strip_trailing_sentinel(raw))
