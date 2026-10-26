"""F11 presupposition gate: the entity extractor and the agent hook (scripted LLM, no models)."""

from __future__ import annotations

import pytest

from local_agentic_rag.agent.presupposition import (
    extract_entities,
    find_missing_presupposition,
    missing_entities,
)
from local_agentic_rag.config import PRESETS, RAGConfig, preset
from local_agentic_rag.schema import REFUSAL_MESSAGE
from tests.helpers import scripted
from tests.test_agent import VERIFY_OK, cite, index, make_agent, steps  # noqa: F401

UNSUPPORTED = {"supported": False, "unsupported_claims": ["revenue doubled"]}


def pairs(question: str) -> list[tuple[str, str]]:
    return [(e.kind, e.text) for e in extract_entities(question)]


# ------------------------------------------------------------------ extractor


def test_years_in_range_only():
    assert pairs("What happened in 1999 and 2024?") == [("year", "1999"), ("year", "2024")]
    assert pairs("Revenue was 1,2005 or 3.2010 or 1750 or 2150 or 12023.") == []


def test_year_in_devanagari_digits_is_normalised():
    assert pairs("ताजमहल २००४ में क्या हुआ?") == [("year", "2004")]


def test_multiword_proper_noun_phrases():
    got = pairs("What powers does the Intelligence Reform Act of 2004 give the DNI?")
    assert ("phrase", "Intelligence Reform Act") in got
    assert ("year", "2004") in got


def test_leading_interrogative_and_imperative_are_skipped():
    assert pairs("Which Table shows the result?") == []
    assert pairs("Explain Quantum Gravity in detail") == [("phrase", "Quantum Gravity")]
    assert pairs("What Natural Questions score?") == [("phrase", "Natural Questions")]
    assert pairs("What Natural score?") == []  # "What" ignored, one capitalised word left


def test_single_capitalised_word_is_not_an_entity():
    assert pairs("How did Contriever perform on retrieval?") == []


def test_possessive_and_punctuation_end_a_phrase():
    assert pairs("How many people took part in NIST's Generative AI Public Working Group?") == [
        ("phrase", "Generative AI Public Working Group")
    ]
    assert pairs("Did Ada Lovelace, Charles Babbage meet?") == [
        ("phrase", "Ada Lovelace"),
        ("phrase", "Charles Babbage"),
    ]


def test_codes_with_letters_and_digits():
    assert ("code", "RIP-4471-B") in pairs("What was the budget of project RIP-4471-B?")
    assert ("code", "BM25") in pairs("How good was BM25 here?")


def test_units_ordinals_and_short_tokens_are_not_codes():
    assert pairs("Is the 3rd 10k run on a 5G link with 4.2B parameters and Q2 data?") == []


def test_devanagari_text_is_unaffected():
    assert pairs("ताजमहल किसने और किसकी याद में बनवाया था?") == []
    assert pairs("शनिवारवाडा पाहण्यासाठी प्रवेश शुल्क किती आहे?") == []


def test_matching_ignores_case_space_dashes_and_nfkc():
    ents = extract_entities("Who signed the Intelligence Reform Act and project RIP-4471-B?")
    passage = "the INTELLIGENCE\n reform  act was passed. See RIP‑4471–B (ﬁnal)."
    assert missing_entities(ents, [passage]) == []
    missing = missing_entities(ents, ["Intelligence Reform was passed."])
    assert [e.text for e in missing] == ["Intelligence Reform Act", "RIP-4471-B"]


def test_phrase_is_checked_across_all_passages_and_years_in_any_script():
    ents = extract_entities("Was the Berlin Wall standing in 1989?")
    assert missing_entities(ents, ["The Berlin Wall fell.", "It was 1989."]) == []
    assert missing_entities(extract_entities("What happened in 1989?"), ["वर्ष १९८९ में"]) == []


def test_latin_entity_is_not_tested_against_a_devanagari_document():
    ents = extract_entities("Taj Mahal kisne banwaya aur yeh UNESCO World Heritage Site kab bana?")
    assert [e.text for e in ents] == ["UNESCO World Heritage Site"]
    assert missing_entities(ents, ["ताजमहल यूनेस्को विश्व धरोहर स्थल है।"]) == []


def test_find_missing_returns_first_or_none():
    assert find_missing_presupposition("What was revenue in 2023?", ["revenue 2023"]) is None
    hit = find_missing_presupposition("What was revenue in 2031?", ["revenue 2023"])
    assert hit is not None and hit.text == "2031" and hit.kind == "year"


# ------------------------------------------------------------------ config


def test_flag_defaults_off_and_preset_turns_it_on():
    assert RAGConfig().presupposition_check is False
    assert all(
        not v.get("presupposition_check") for k, v in PRESETS.items() if k != "agentic_presup"
    )
    assert preset("agentic_presup").presupposition_check is True
    assert preset("agentic_presup").agent_mode == "adaptive"


# ------------------------------------------------------------------ agent hook

ABSENT_YEAR = "What was revenue in fiscal 2031?"
PRESENT_YEAR = "What was revenue in fiscal 2023?"


def run(question, index, verify, **cfg):  # noqa: F811
    cfg.setdefault("agent_mode", "adaptive")
    cfg.setdefault("max_retrieval_rounds", 1)  # no escalation: the verdict is final
    agent, llm = make_agent(
        scripted(generate=cite("Revenue increased", "Revenue doubled [{n}]."), verify=verify), **cfg
    )
    return agent.run(question, [index]), llm


def test_flag_off_leaves_unsupported_answer_unchanged(index):  # noqa: F811
    ans, _ = run(ABSENT_YEAR, index, UNSUPPORTED)
    assert ans.refused is False and ans.grounded is False
    assert ans.answer == "Revenue doubled [1]." and ans.citations
    assert "presupposition_refusal" not in steps(ans)


def test_supported_verdict_is_never_overridden(index):  # noqa: F811
    ans, _ = run(ABSENT_YEAR, index, VERIFY_OK, presupposition_check=True)
    assert ans.refused is False and ans.grounded is True
    assert "presupposition_refusal" not in steps(ans)


def test_unsupported_and_entity_missing_refuses(index):  # noqa: F811
    ans, llm = run(ABSENT_YEAR, index, UNSUPPORTED, presupposition_check=True)
    assert ans.refused is True
    assert ans.answer == REFUSAL_MESSAGE
    assert ans.citations == []
    assert ans.grounded is None and ans.unsupported_claims == []  # a refusal carries no verdict
    step = ans.trace[-1]
    assert step.step == "presupposition_refusal"
    assert step.detail["entity"] == "2031" and step.detail["kind"] == "year"
    assert step.detail["unsupported"] == ["revenue doubled"]  # the trace keeps what was dropped
    assert len(llm.calls) == 2  # generate + verify only: the gate adds no LLM call


def test_unsupported_but_entity_present_is_unchanged(index):  # noqa: F811
    ans, _ = run(PRESENT_YEAR, index, UNSUPPORTED, presupposition_check=True)
    assert ans.refused is False and ans.answer == "Revenue doubled [1]."
    assert "presupposition_refusal" not in steps(ans)


def test_unverified_answer_is_never_gated(index):  # noqa: F811
    ans, _ = run(ABSENT_YEAR, index, VERIFY_OK, presupposition_check=True, use_verifier=False)
    assert ans.grounded is None and ans.refused is False


def test_already_refused_answer_is_not_touched(index):  # noqa: F811
    agent, _ = make_agent(
        scripted(generate="ANSWER_NOT_FOUND", verify=UNSUPPORTED),
        agent_mode="adaptive",
        max_retrieval_rounds=1,
        presupposition_check=True,
    )
    ans = agent.run(ABSENT_YEAR, [index])
    assert ans.refused is True
    assert "presupposition_refusal" not in steps(ans)


@pytest.mark.parametrize("mode", ["adaptive", "full"])
def test_gate_runs_in_both_agent_modes(index, mode):  # noqa: F811
    ans, _ = run(
        ABSENT_YEAR,
        index,
        UNSUPPORTED,
        presupposition_check=True,
        agent_mode=mode,
        max_retrieval_rounds=1,
        max_answer_attempts=1,
        use_planner=False,
        use_grader=False,
    )
    assert ans.refused is True and ans.trace[-1].step == "presupposition_refusal"
