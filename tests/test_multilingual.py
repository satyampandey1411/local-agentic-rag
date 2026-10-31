"""Multilingual support: tokenisation, script detection, translation step, localised output."""

from __future__ import annotations

import json

from local_agentic_rag import lang
from local_agentic_rag.agent import prompts as P
from local_agentic_rag.eval.metrics import all_facts_present, normalize_text
from local_agentic_rag.retrieval.bm25 import BM25Index, tokenize
from tests.helpers import FakeEmbedder, make_index, scripted
from tests.test_agent import VERIFY_OK, cite, make_agent, steps


def test_devanagari_words_keep_vowel_signs():
    assert tokenize("ताजमहल का निर्माण शाहजहाँ ने करवाया") == [
        "ताजमहल",
        "का",
        "निर्माण",
        "शाहजहाँ",
        "ने",
        "करवाया",
    ]


def test_bm25_matches_hindi_and_marathi_terms():
    index = BM25Index(["ताजमहल आगरा में है", "शनिवारवाडा पुण्यातील वास्तू आहे", "unrelated english text"])
    assert index.search("ताजमहल कहाँ है", 3)[0][0] == 0
    assert index.search("शनिवारवाडा", 3)[0][0] == 1


def test_script_and_hinglish_detection():
    assert lang.script_of("ताजमहल किसने बनवाया?") == "devanagari"
    assert lang.script_of("Who built it?") == "latin"
    assert lang.is_hinglish("Taj Mahal kisne banwaya tha?")
    assert not lang.is_hinglish("Who built the Taj Mahal?")
    # regression: English with repeated "the" / "me" must not look like Hinglish
    assert not lang.is_hinglish(
        "According to the bias-interplay analysis, which biases were the most problematic for the retrievers?"
    )
    assert not lang.is_hinglish("Tell me what the paper says about the dataset and the queries")
    assert lang.is_hinglish(
        "Authors ne kaunsa dataset repurpose kiya aur har setting mein kitni queries thi?"
    )
    assert lang.needs_translation("Who built it?", "devanagari")
    assert lang.needs_translation("ताजमहल किसने बनवाया?", "latin")
    assert lang.needs_translation("Re-DocRED kya hai aur kitne queries hai?", "latin")
    assert not lang.needs_translation("Who built it?", "latin")


def test_grader_accepts_devanagari_digits():
    assert normalize_text("१६३२ में") == "1632 में"
    assert all_facts_present("निर्माण १६३२ में शुरू हुआ", [["1632"]])


HINDI_CORPUS = [
    (1, "ताजमहल आगरा में यमुना नदी के किनारे स्थित एक मकबरा है।"),
    (2, "इसका निर्माण मुगल सम्राट शाहजहाँ ने अपनी पत्नी मुमताज़ महल की याद में करवाया था।"),
    (3, "निर्माण कार्य 1632 में शुरू हुआ और लगभग 1653 में पूरा हुआ।"),
]


def test_cross_lingual_question_gets_translated_query_and_answer_language_rule():
    index = make_index(HINDI_CORPUS, FakeEmbedder(), doc_id="hi", title="ताजमहल")
    agent, llm = make_agent(
        scripted(
            # role_of() treats unknown schemas as errors, so the translation reply is routed by schema below
            generate=cite("शाहजहाँ", "The Taj Mahal was built by Shah Jahan [{n}]."),
            verify=VERIFY_OK,
        ),
        agent_mode="adaptive",
    )
    original = llm.responder

    def responder(messages, schema):
        if schema is P.TRANSLATE_SCHEMA:
            return json.dumps({"translation": "ताजमहल का निर्माण किसने करवाया शाहजहाँ"})
        return original(messages, schema)

    llm.responder = responder
    ans = agent.run("Who built the Taj Mahal?", [index])
    assert steps(ans)[:2] == ["translate", "retrieve"]
    assert ans.search_queries[1].startswith("ताजमहल")
    assert ans.cited_pages == [2]
    generate_prompt = [c for c in llm.calls if c["json_schema"] is None][0]["messages"][-1][
        "content"
    ]
    assert "Write the answer in English" in generate_prompt


def test_answer_language_is_explicit():
    assert lang.answer_language("Who built the Taj Mahal?") == "English"
    assert lang.answer_language("Taj Mahal kisne banwaya tha?").startswith("Hinglish")
    assert "Devanagari" in lang.answer_language("ताजमहल किसने बनवाया?")


def test_same_language_question_skips_translation():
    index = make_index(HINDI_CORPUS, FakeEmbedder(), doc_id="hi", title="ताजमहल")
    agent, llm = make_agent(
        scripted(generate=cite("शाहजहाँ", "शाहजहाँ ने [{n}]।"), verify=VERIFY_OK),
        agent_mode="adaptive",
    )
    ans = agent.run("ताजमहल किसने बनवाया?", [index])
    assert "translate" not in steps(ans)


def test_hindi_refusal_is_localised():
    index = make_index(HINDI_CORPUS, FakeEmbedder(), doc_id="hi", title="ताजमहल")
    agent, _ = make_agent(
        scripted(
            generate=P.ANSWER_NOT_FOUND,
            plan={"standalone_question": "x", "search_queries": []},
            grade={"relevant": [], "sufficient": False, "missing_query": ""},
            verify=VERIFY_OK,
        ),
        agent_mode="adaptive",
    )
    ans = agent.run("ताजमहल की ऊँचाई कितनी है?", [index])
    assert ans.refused and "नहीं मिला" in ans.answer
