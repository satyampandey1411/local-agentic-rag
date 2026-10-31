"""Languages beyond English/Hindi/Marathi: answer language, explicit requests, localised
refusals, keyword tokens for every script, sentence ends, page references, digits, OCR packs
and cross-script translation over a mixed library."""

from __future__ import annotations

import json
import re
import subprocess

import pytest

from local_agentic_rag import lang
from local_agentic_rag.agent import prompts as P
from local_agentic_rag.eval.metrics import all_facts_present, normalize_text
from local_agentic_rag.ingest import pdf
from local_agentic_rag.ingest.chunker import _SENTENCE_SPLIT, _split_units
from local_agentic_rag.retrieval.bm25 import BM25Index, tokenize
from local_agentic_rag.retrieval.hybrid import page_refs
from tests.helpers import FakeEmbedder, make_index, scripted
from tests.test_agent import VERIFY_OK, cite, make_agent, steps
from tests.test_multilingual import HINDI_CORPUS

# ----------------------------------------------------------------------------- answer language


@pytest.mark.parametrize(
    ("question", "language"),
    [
        ("Who built the Taj Mahal?", "English"),
        ("Charminar height?", "English"),
        ("Charminar evaru kattaru? Ekkada undi?", "Telugu (Latin letters)"),
        ("Thanjavur kovil yaar kattinaar? enna varusham?", "Tamil (Latin letters)"),
        ("Des Moines population?", "English"),
        ("Who was Le Corbusier?", "English"),
        ("Taj Mahal kisne banwaya tha?", "Hinglish"),
        ("ताजमहल किसने बनवाया?", "Hindi"),
        ("शनिवारवाडा कोणी बांधला?", "Marathi"),
        ("सगरमाथाको उचाइ कति छ?", "Nepali"),
        ("ताजमहलः केन निर्मितः आसीत्?", "Sanskrit"),
        ("चारमीनार के बनवले रहल?", "Bhojpuri"),
        ("ई मंदिर कउन बनवाइस?", "Awadhi"),
        ("मिथिलाक राजधानी कोन छल?", "Maithili"),
        ("چارمینار کس نے بنوایا؟", "Urdu"),
        ("من بنى المسجد؟", "Arabic"),
        ("కోటను ఎవరు నిర్మించారు?", "Telugu"),
        ("கோயிலைக் கட்டியவர் யார்?", "Tamil"),
        ("ಅರಮನೆಯನ್ನು ಯಾರು ಕಟ್ಟಿಸಿದರು?", "Kannada"),
        ("കൊട്ടാരം ആരാണ് നിർമ്മിച്ചത്?", "Malayalam"),
        ("મંદિર કોણે બંધાવ્યું?", "Gujarati"),
        ("ਮੰਦਰ ਕਿਸ ਨੇ ਬਣਾਇਆ?", "Punjabi"),
        ("ମନ୍ଦିର କିଏ ତିଆରି କରିଥିଲେ?", "Odia"),
        ("মন্দিরটি কে নির্মাণ করেন?", "Bengali"),
        ("মন্দিৰটো কোনে নিৰ্মাণ কৰিছিল?", "Assamese"),
        ("Qui a construit le Taj Mahal ?", "French"),
        ("¿Quién construyó el Taj Mahal?", "Spanish"),
        ("Wer hat das Taj Mahal gebaut?", "German"),
        ("誰がタージ・マハルを建てましたか？", "Japanese"),
        ("泰姬陵是谁建造的？", "Chinese"),
        ("Кто построил Тадж-Махал?", "Russian"),
    ],
)
def test_question_language(question, language):
    assert lang.choose_answer_language(question).name == language


def test_unclear_devanagari_question_is_not_forced_into_hindi():
    choice = lang.choose_answer_language("ताजमहल?")
    assert choice.name is None
    assert "not automatically in Hindi" in choice.instruction


@pytest.mark.parametrize(
    ("question", "requested"),
    [
        ("Who built the Taj Mahal? Answer in Urdu.", "Urdu"),
        ("Who built it? In Urdu please.", "Urdu"),
        ("ताजमहल किसने बनवाया? उर्दू में बताइए", "Urdu"),
        ("Taj Mahal kisne banwaya? English mein batao", "English"),
        ("కోటను ఎవరు నిర్మించారు? హిందీలో చెప్పండి", "Hindi"),
        ("கோயிலைக் கட்டியவர் யார்? தமிழில் பதில் சொல்லுங்கள்", "Tamil"),
        ("মন্দিরটি কে নির্মাণ করেন? ইংরেজিতে উত্তর দিন", "English"),
        ("കൊട്ടാരം ആരാണ് നിർമ്മിച്ചത്? മലയാളത്തിൽ ഉത്തരം പറയൂ", "Malayalam"),
        ("शनिवारवाडा कोणी बांधला? इंग्रजीत नाही, हिंदीत उत्तर द्या", "Hindi"),
        # mentions of a language that are not requests
        ("What is mango called in Hindi?", None),
        ("उर्दू भाषा किस देश की राष्ट्रभाषा है? बताइए", None),
        ("Which temple in Tamil Nadu is tallest? Answer briefly.", None),
        ("Who built the Taj Mahal?", None),
        ("Which poems did he write in Urdu?", None),
        ("Tell me how many books were written in Hindi.", None),
        ("Give the number of films released in Tamil in 2010.", None),
        ("कौन सी किताबें उर्दू में लिखी गईं? बताइए।", None),
        ("Explain the result in Hindi.", "Hindi"),
        ("Can you answer in Bengali?", "Bengali"),
    ],
)
def test_explicit_answer_language_request(question, requested):
    assert lang.requested_language(question) == requested


def test_request_overrides_question_language_in_the_instruction():
    choice = lang.choose_answer_language("Who built the Taj Mahal? Answer in Urdu.")
    assert choice.requested and choice.name == "Urdu"
    assert choice.instruction.startswith("Urdu")


def test_refusal_follows_the_answer_language():
    assert "కనబడలేదు" in lang.refusal_message("కోటను ఎవరు నిర్మించారు?", "EN")
    assert "نہیں ملا" in lang.refusal_message("چارمینار کس نے بنوایا؟", "EN")
    assert "सापडले नाही" in lang.refusal_message("शनिवारवाडा कोणी बांधला?", "EN")
    assert "नहीं मिला" in lang.refusal_message("ताजमहल किसने बनवाया?", "EN")
    assert "نہیں ملا" in lang.refusal_message("Who built it? Answer in Urdu.", "EN")
    assert lang.refusal_message("Who built the Taj Mahal?", "EN") == "EN"


def test_every_scripts_digits():
    assert lang.to_ascii_digits("१६३२ ১৬৩২ ۱۶۳۲ ٤٠ ౧౬ ௧௬ ๑๒") == "1632 1632 1632 40 16 16 12"
    assert normalize_text("١٦٣٢ ౧౬౩౨") == "1632 1632"
    assert all_facts_present("నిర్మాణం ౧౫౯౧లో", [["1591"]])


# ----------------------------------------------------------------------------- keyword tokens


def test_danda_is_not_part_of_the_word():
    assert tokenize("यह आगरा में है।")[-1] == "है"
    assert tokenize("দিল্লি ভারতের রাজধানী।")[-1] == "রাজধানী"


def test_spaceless_scripts_get_bigrams():
    zh = tokenize("泰姬陵是谁建造的")
    assert "泰" in zh and "泰姬" in zh and "建造" in zh
    ja = tokenize("東京タワーは完成した")
    assert "東京" in ja and "タワ" in ja  # kana no longer swallowed into one long token
    th = tokenize("ปราสาทหินพิมาย")
    assert "ปร" in th and len(th) > 5


def test_urdu_presentation_forms_match_letters():
    # Some PDFs store Arabic-script text as presentation forms; NFKC folds them.
    assert tokenize("ﻻ") == tokenize("لا")


def test_bm25_finds_words_at_sentence_end_and_in_cjk():
    index = BM25Index(["शाहजहाँ ने ताजमहल बनवाया।", "यह आगरा में है।", "泰姬陵位于阿格拉。"])
    assert index.search("बनवाया", 3)[0][0] == 0
    assert index.search("阿格拉", 3)[0][0] == 2


# ----------------------------------------------------------------------------- chunking


def test_sentence_ends_in_other_scripts():
    assert _SENTENCE_SPLIT.split("आगरा में है। इसे शाहजहाँ ने बनवाया।") == [
        "आगरा में है।",
        "इसे शाहजहाँ ने बनवाया।",
    ]
    assert _SENTENCE_SPLIT.split("泰姬陵位于印度。它建于1632年！") == [
        "泰姬陵位于印度。",
        "它建于1632年！",
    ]
    assert _SENTENCE_SPLIT.split("e.g. this stays. Next One.") == ["e.g. this stays.", "Next One."]


def test_long_spaceless_paragraph_is_cut_to_size():
    units = _split_units("泰" * 2000, 320, len)
    assert len(units) > 1 and all(len(u) <= 320 for u in units)
    assert "".join(units) == "泰" * 2000


# ----------------------------------------------------------------------------- page references


@pytest.mark.parametrize(
    ("question", "pages"),
    [
        ("পৃষ্ঠা ৫ এ কী আছে?", [5]),
        ("صفحہ ۳ پر کیا ہے؟", [3]),
        ("పేజీ ౪ లో ఏముంది?", [4]),
        ("பக்கம் 2 இல் என்ன உள்ளது?", [2]),
        ("पृष्ठ 3 से 5", [3, 4, 5]),
        ("In 1,234 cases", []),
        ("Was steht auf Seite 12?", [12]),
        ("Wie viele Besucher hatte die Webseite 2023?", []),
        ("Startseite 12", []),
    ],
)
def test_page_references_in_other_languages(question, pages):
    assert page_refs(question) == pages


# ----------------------------------------------------------------------------- OCR packs


def test_ocr_packs_follow_the_detected_script():
    installed = frozenset({"eng", "hin", "mar", "osd", "urd", "ara", "tel", "script/Tamil"})
    assert pdf.ocr_language_for_script("Telugu", installed) == "tel+eng"
    assert pdf.ocr_language_for_script("Arabic", installed) == "urd+ara+eng"
    assert pdf.ocr_language_for_script("Tamil", installed) == "script/Tamil+eng"
    # Devanagari, Latin and unknown keep the measured default
    for script in ("Devanagari", "Latin", "Kannada", ""):
        assert pdf.ocr_language_for_script(script, installed) == pdf.DEFAULT_OCR_LANGUAGE


def test_auto_ocr_language_reads_tesseract_osd(monkeypatch):
    monkeypatch.setattr(pdf, "_installed_ocr_languages", lambda: frozenset({"eng", "osd", "tel"}))

    def fake_run(cmd, **kwargs):
        assert "osd" in cmd
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Script: Telugu\nScript confidence: 9.1\n"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert pdf._auto_ocr_language("page.png") == "tel+eng"


def test_auto_ocr_language_without_osd_pack(monkeypatch):
    monkeypatch.setattr(pdf, "_installed_ocr_languages", lambda: frozenset({"eng", "hin"}))
    assert pdf._auto_ocr_language("page.png") == pdf.DEFAULT_OCR_LANGUAGE


def test_ocr_fallback_uses_the_detected_script(monkeypatch):
    """When the CLI OCR fails (run 9: timeouts on a busy CPU), the fallback OCR must use the
    packs picked from the page's script, not the Latin/Devanagari default."""
    seen = {}
    monkeypatch.setattr(pdf, "_page_ocr_language", lambda page: "tel+eng")
    monkeypatch.setattr(
        pdf, "_ocr_cli_paragraphs", lambda page, no, lang: seen.setdefault("cli", lang) and []
    )
    monkeypatch.setattr(
        pdf, "_ocr_textpage", lambda page, no, lang: seen.setdefault("fallback", lang) and None
    )

    class _ScanPage:
        rect = type("R", (), {"width": 595.0, "height": 842.0})()

        def get_text(self, kind="text", textpage=None):
            return "" if kind == "text" else {"blocks": []}

        def get_images(self, full=False):
            return [(1,)]

    pdf._extract_page(_ScanPage(), 1, [], True, "auto")
    assert seen == {"cli": "tel+eng", "fallback": "tel+eng"}


def test_ocr_timeout_retries_with_one_pack_at_lower_dpi(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd[-1], kwargs.get("timeout")))
        if sum(1 for c in calls if c[0] != "dpi") == 1:  # the first Tesseract run overruns
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))
        return subprocess.CompletedProcess(cmd, 0, stdout="पहला अनुच्छेद\n\nदूसरा अनुच्छेद\n")

    class _Pix:
        def save(self, path):
            open(path, "wb").close()

    class _Page:
        def get_pixmap(self, dpi):
            calls.append(("dpi", dpi))
            return _Pix()

    monkeypatch.setattr(pdf.shutil, "which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr(subprocess, "run", fake_run)
    paragraphs = pdf._ocr_cli_paragraphs(_Page(), 1, "tel+eng")
    assert paragraphs == ["पहला अनुच्छेद", "दूसरा अनुच्छेद"]
    assert ("dpi", pdf.OCR_DPI) in calls and ("dpi", pdf.OCR_RETRY_DPI) in calls
    assert [c for c in calls if c[0] != "dpi"] == [
        ("tel+eng", pdf.OCR_TIMEOUT_S),
        ("tel", pdf.OCR_RETRY_TIMEOUT_S),
    ]
    calls.clear()
    pdf._ocr_cli_paragraphs(
        _Page(), 2, pdf.DEFAULT_OCR_LANGUAGE
    )  # a Hindi scan keeps the Hindi pack
    assert [c[0] for c in calls if c[0] != "dpi"] == [pdf.DEFAULT_OCR_LANGUAGE, "eng+hin"]


class _LegacyFontPage:
    def __init__(self, font: str) -> None:
        self.font = font

    def get_fonts(self):
        return [(5, "ttf", "TrueType", f"ABCDEF+{self.font}", "F1", "WinAnsiEncoding")]

    def get_text(self, kind):
        span = {"font": f"ABCDEF+{self.font}", "text": "Hkkjr ljdkj dh ;kstuk"}
        return {"blocks": [{"lines": [{"spans": [span]}]}]}


def test_legacy_hindi_font_page_is_sent_to_ocr():
    assert pdf.uses_legacy_indic_font(_LegacyFontPage("KrutiDev010"), "Hkkjr ljdkj")
    assert not pdf.uses_legacy_indic_font(_LegacyFontPage("NotoSans-Regular"), "Hkkjr ljdkj")


# ----------------------------------------------------------------------------- agent


TAMIL_CORPUS = [
    (1, "பிரகதீசுவரர் கோயில் தஞ்சாவூரில் உள்ளது."),
    (2, "இக்கோயிலை முதலாம் இராஜராஜ சோழன் கட்டினார்."),
]


def test_mixed_library_translates_into_each_script():
    hindi = make_index(HINDI_CORPUS, FakeEmbedder(), doc_id="hi", title="ताजमहल")
    tamil = make_index(TAMIL_CORPUS, FakeEmbedder(), doc_id="ta", title="கோயில்")
    agent, llm = make_agent(
        scripted(generate=cite("शाहजहाँ", "షాజహాన్ [{n}]."), verify=VERIFY_OK),
        agent_mode="adaptive",
    )
    original = llm.responder

    def responder(messages, schema):
        if schema is P.TRANSLATE_SCHEMA:
            sample = messages[-1]["content"]
            text = "ताजमहल किसने बनवाया" if "ताजमहल" in sample else "கோயிலைக் கட்டியவர் யார்"
            return json.dumps({"translation": text})
        return original(messages, schema)

    llm.responder = responder
    ans = agent.run("తాజ్ మహల్‌ను ఎవరు నిర్మించారు?", [hindi, tamil])
    assert steps(ans).count("translate") == 2
    assert "ताजमहल किसने बनवाया" in ans.search_queries
    assert "கோயிலைக் கட்டியவர் யார்" in ans.search_queries
    generate_prompt = [c for c in llm.calls if c["json_schema"] is None][0]["messages"][-1][
        "content"
    ]
    assert "Write the answer in Telugu" in generate_prompt


def _with_translations(llm, query="ताजमहल किसने बनवाया", answer=None):
    """Route the query-translation and answer-translation calls of a scripted LLM."""
    original = llm.responder

    def responder(messages, schema):
        if schema is P.TRANSLATE_SCHEMA:
            return json.dumps({"translation": query})
        if messages[0]["content"] == P.TRANSLATE_ANSWER_SYSTEM:
            # keep the source answer's citation marker, as the prompt asks
            marker = re.findall(r"\[\d+\]", messages[-1]["content"].split("Answer in")[-1])
            text = answer if answer is not None else "TRANSLATED [1]"
            return re.sub(r"\[\d+\]", marker[0], text) if marker and text else text
        return original(messages, schema)

    llm.responder = responder


def _prompts(llm):
    free = [c for c in llm.calls if c["json_schema"] is None]
    return [c["messages"][-1]["content"] for c in free], [c["messages"][0]["content"] for c in free]


def test_explicit_request_answers_in_the_passages_language_then_translates():
    index = make_index(HINDI_CORPUS, FakeEmbedder(), doc_id="hi", title="ताजमहल")
    agent, llm = make_agent(
        scripted(generate=cite("शाहजहाँ", "शाहजहाँ ने बनवाया [{n}]।"), verify=VERIFY_OK),
        agent_mode="adaptive",
    )
    _with_translations(llm, answer="شاہجہاں نے بنوایا [1]۔")
    ans = agent.run("Who built the Taj Mahal? Answer in Urdu.", [index])
    prompts, systems = _prompts(llm)
    assert "Write the answer in Hindi, the language of the passages" in prompts[0]
    assert systems[-1] == P.TRANSLATE_ANSWER_SYSTEM
    assert "into Urdu (the user asked for the answer in Urdu)" in prompts[-1]
    assert ans.answer.startswith("شاہجہاں") and ans.cited_pages == [2]
    assert "translate_answer" in steps(ans)


def test_failed_translation_keeps_the_checked_answer():
    index = make_index(HINDI_CORPUS, FakeEmbedder(), doc_id="hi", title="ताजमहल")
    agent, llm = make_agent(
        scripted(generate=cite("शाहजहाँ", "शाहजहाँ ने बनवाया [{n}]।"), verify=VERIFY_OK),
        agent_mode="adaptive",
    )
    _with_translations(llm)
    routed, attempts = llm.responder, []

    def responder(messages, schema):
        if messages[0]["content"] == P.TRANSLATE_ANSWER_SYSTEM:
            attempts.append(1)
            raise RuntimeError("connection reset")  # not an LLMError
        return routed(messages, schema)

    llm.responder = responder
    ans = agent.run("Who built the Taj Mahal? Answer in Urdu.", [index])
    assert attempts, "the answer translation was not attempted"
    assert ans.answer.startswith("शाहजहाँ") and ans.cited_pages == [2] and not ans.error


def test_english_answers_stay_direct():
    index = make_index(HINDI_CORPUS, FakeEmbedder(), doc_id="hi", title="ताजमहल")
    agent, llm = make_agent(
        scripted(generate=cite("शाहजहाँ", "Shah Jahan built it [{n}]."), verify=VERIFY_OK),
        agent_mode="adaptive",
    )
    _with_translations(llm)
    ans = agent.run("Who built the Taj Mahal?", [index])
    assert "translate_answer" not in steps(ans)
    assert "Write the answer in English" in _prompts(llm)[0][0]


def test_same_language_needs_no_pivot():
    index = make_index(HINDI_CORPUS, FakeEmbedder(), doc_id="hi", title="ताजमहल")
    agent, llm = make_agent(
        scripted(generate=cite("शाहजहाँ", "शाहजहाँ ने [{n}]।"), verify=VERIFY_OK),
        agent_mode="adaptive",
    )
    ans = agent.run("ताजमहल किसने बनवाया?", [index])
    assert "translate_answer" not in steps(ans)


def test_menu_choice_overrides_the_question_language():
    index = make_index(HINDI_CORPUS, FakeEmbedder(), doc_id="hi", title="ताजमहल")
    agent, llm = make_agent(
        scripted(generate=cite("शाहजहाँ", "शाहजहाँ ने बनवाया [{n}]।"), verify=VERIFY_OK),
        agent_mode="adaptive",
    )
    _with_translations(llm, answer="ஷாஜஹான் கட்டினார் [1].")
    ans = agent.run("ताजमहल किसने बनवाया?", [index], answer_language="Tamil")
    prompts, _ = _prompts(llm)
    assert "into Tamil (the user chose it)" in prompts[-1]
    assert ans.answer.startswith("ஷாஜஹான்")


def test_translation_still_runs_after_a_suffix_was_appended():
    index = make_index(HINDI_CORPUS, FakeEmbedder(), doc_id="hi", title="ताजमहल")
    agent, llm = make_agent(
        scripted(generate=cite("शाहजहाँ", "शाहजहाँ ने बनवाया [{n}]।"), verify=VERIFY_OK),
        agent_mode="adaptive",
    )
    _with_translations(llm, answer="షాజహాన్ నిర్మించారు [1].")
    original = agent._name_cited_table

    def add_suffix(ans):
        original(ans)
        ans.answer += " (Table 2)"

    agent._name_cited_table = add_suffix
    ans = agent.run("తాజ్ మహల్‌ను ఎవరు నిర్మించారు?", [index])
    assert ans.answer.startswith("షాజహాన్")


def test_menu_choice_localises_the_refusal():
    index = make_index(HINDI_CORPUS, FakeEmbedder(), doc_id="hi", title="ताजमहल")
    agent, llm = make_agent(
        scripted(
            generate=P.ANSWER_NOT_FOUND,
            plan={"standalone_question": "x", "search_queries": []},
            grade={"relevant": [], "sufficient": False, "missing_query": ""},
            verify=VERIFY_OK,
        ),
        agent_mode="adaptive",
    )
    _with_translations(llm)
    ans = agent.run("How tall is it?", [index], answer_language="Urdu")
    assert ans.refused and "نہیں ملا" in ans.answer


def test_telugu_refusal_is_localised():
    index = make_index(HINDI_CORPUS, FakeEmbedder(), doc_id="hi", title="ताजमहल")
    agent, llm = make_agent(
        scripted(
            generate=P.ANSWER_NOT_FOUND,
            plan={"standalone_question": "x", "search_queries": []},
            grade={"relevant": [], "sufficient": False, "missing_query": ""},
            verify=VERIFY_OK,
        ),
        agent_mode="adaptive",
    )
    original = llm.responder

    def responder(messages, schema):
        if schema is P.TRANSLATE_SCHEMA:
            return json.dumps({"translation": "ताजमहल की ऊँचाई"})
        return original(messages, schema)

    llm.responder = responder
    ans = agent.run("తాజ్ మహల్ ఎత్తు ఎంత?", [index])
    assert ans.refused and "కనబడలేదు" in ans.answer
