"""
Lightweight language/script detection (no model, no dependency).

Used to (a) decide when a question and the document are in different languages,
so the agent adds a translated search query, (b) tell the generator which
language to answer in, and (c) localise the refusal message.

Script detection is reliable.  Inside a shared script the language is guessed from
small function-word lists: Hindi, Marathi, Nepali, Sanskrit, Bhojpuri, Awadhi and
Maithili all use Devanagari; Urdu, Arabic and Persian share the Arabic script;
English, French, Spanish, Hinglish and others share the Latin one.  When the words
do not settle it, the generator is told to answer in the question's own language
(and, for Devanagari, explicitly not to fall back to Hindi).  An explicit request
such as "answer in Urdu" or "उर्दू में बताइए" overrides the question's language.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

_SCRIPTS = [
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("gurmukhi", 0x0A00, 0x0A7F),
    ("gujarati", 0x0A80, 0x0AFF),
    ("oriya", 0x0B00, 0x0B7F),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
    ("sinhala", 0x0D80, 0x0DFF),
    ("thai", 0x0E00, 0x0E7F),
    ("lao", 0x0E80, 0x0EFF),
    ("tibetan", 0x0F00, 0x0FFF),
    ("myanmar", 0x1000, 0x109F),
    ("georgian", 0x10A0, 0x10FF),
    ("hangul", 0x1100, 0x11FF),
    ("ethiopic", 0x1200, 0x139F),
    ("khmer", 0x1780, 0x17FF),
    ("arabic", 0x0600, 0x06FF),
    ("arabic", 0x0750, 0x077F),  # Arabic Supplement
    ("arabic", 0xFB50, 0xFDFF),  # presentation forms: some PDFs store Urdu/Arabic like this
    ("arabic", 0xFE70, 0xFEFF),
    ("hebrew", 0x0590, 0x05FF),
    ("armenian", 0x0530, 0x058F),
    ("cyrillic", 0x0400, 0x04FF),
    ("greek", 0x0370, 0x03FF),
    ("cjk", 0x3400, 0x4DBF),
    ("cjk", 0x4E00, 0x9FFF),
    ("kana", 0x3040, 0x30FF),
    ("hangul", 0xAC00, 0xD7AF),
]

# Scripts used by one main language: the script names the language.
_SCRIPT_LANGUAGE = {
    "gurmukhi": "Punjabi",
    "gujarati": "Gujarati",
    "oriya": "Odia",
    "tamil": "Tamil",
    "telugu": "Telugu",
    "kannada": "Kannada",
    "malayalam": "Malayalam",
    "sinhala": "Sinhala",
    "thai": "Thai",
    "lao": "Lao",
    "tibetan": "Tibetan",
    "myanmar": "Burmese",
    "georgian": "Georgian",
    "hangul": "Korean",
    "khmer": "Khmer",
    "hebrew": "Hebrew",
    "armenian": "Armenian",
    "greek": "Greek",
    "kana": "Japanese",
    "cjk": "Chinese",
}

# Hindi function words written in Latin script.  Words that are also common English words
# ("the", "me", "par", "to", "is") are deliberately excluded: with them, ordinary English
# questions were classified as Hinglish (observed on Colab: answers came back with "aur").
_HINGLISH_TEXT = """hai hain tha thi thay kya kaun kaunsa kaunse kis kisne kisko kitna kitne kitni kab kahan
    kaise kyun kyon ka ki ke ko se mein nahi nahin bhi yeh woh unka unki uska uski iska iski
    batao bataiye banwaya banaya banane hua hui gaya gayi saal ne kiya liye wala wali aur
    raha rahe rahi kar karte karta lagbhag kaam"""
_HINGLISH_WORDS = frozenset(_HINGLISH_TEXT.split())


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _words(text: str) -> frozenset[str]:
    return frozenset(_nfc(text).lower().split())


# Function words that tell the Devanagari languages apart.  Only words that are typical of
# one language are listed; words shared by several (का, के, से, कहाँ) are left out.
_DEVANAGARI_MARKERS = {
    "Hindi": _words(
        "है हैं था थे थी क्या किस किसने किसकी किसके किसका किसको कौन कौनसा कौनसी कितना कितने "
        "कितनी में और लिए गया गई नहीं यह वह बताइए बताइये बताओ बताएं बनवाया करवाया हुआ हुई "
        "द्वारा किया किसलिए"
    ),
    "Marathi": _words(
        "आहे आहेत होते होता होती काय कोणी कोणता कोणती कोणते कोणत्या किती मध्ये आणि केले केली "
        "केला कधी कुठे कसे नाही साठी झाले झाला झाली बांधले बांधला सांगा कोणाला कोणाच्या"
    ),
    "Nepali": _words(
        "छ छन् छैन हो थियो थिए थिइन् कुन कहिले कसले कसको कस्तो गरे गरेको गरिएको भएको हुन्छ "
        "पनि लागि बनाएको बनाइएको कहाँबाट"
    ),
    "Sanskrit": _words(
        "अस्ति सन्ति आसीत् आसन् अभवत् किम् कः कस्य कस्मिन् कस्याः कुत्र कदा केन च अपि इति एव "
        "तस्य अयम् इयम् इदम् भवति वर्तते स्थितम् अस्य अस्याः कानि काः"
    ),
    "Bhojpuri": _words(
        "बा बाटे बाड़े बाड़ी रहल रहली रहलें कवन कवना केकरा केकर ओकर एकर भइल गइल कइल बनवले "
        "बनवलें बनल होला हवे नइखे खातिर आपन इहाँ उहाँ कतना केतना"
    ),
    "Awadhi": _words(
        "हय अहै रहिन रहिस कउन कउनु केहि केहिका अउर भवा गवा बनवाइस बनवाइन किहिस किहिन कीन "
        "कीन्ह होत नाहीं जउन तउन"
    ),
    "Maithili": _words(
        "अछि छथि छल छलाह छलै छलथि कोन कतय कतए कहिया कियो मे सँ हुनक एहि ओहि भेल गेल कयल "
        "केलक केलनि नहि बनाओल बनौलनि"
    ),
}

_LATIN_MARKERS = {
    "English": _words(
        "the is are was were what which who whom whose when where why how many much of in on at "
        "did does do has have had an and to for by with from that this it its be been built "
        "year name called tell give please answer list there"
    ),
    "French": _words(
        "le la les des du est sont était quelle quel quels quelles qui combien où quand comment "
        "pourquoi une dans avec pour été ont au aux cette"
    ),
    "Spanish": _words(
        "el los las es son fue qué cuál cuáles cuántos cuántas cuánto dónde cuándo cómo quién "
        "para con una del al está están tiene"
    ),
    "German": _words(
        "der die das ist sind wurde wann wie welche welcher welches wo wer was und ein eine im "
        "von mit für auf hat viele"
    ),
    "Portuguese": _words(
        "os é são foi qual quais quantos quantas onde quando como quem do da dos das uma em no "
        "na com"
    ),
    "Italian": _words(
        "il lo gli è sono fu quale quali quanti quante dove quando come chi della del dei delle "
        "nel nella"
    ),
    "Indonesian": _words(
        "apa yang berapa kapan di dan adalah siapa mana dimana bagaimana ini itu dengan untuk "
        "dari pada tahun"
    ),
    "Turkish": _words("ne hangi kaç nerede zaman ve bir bu mi mı mu mü kim nasıl yılında için ile"),
}

_URDU_LETTERS = frozenset("ٹڈڑںےۓہھۂ")
_PERSIAN_LETTERS = frozenset("پچژگکی")
_ARABIC_LETTERS = frozenset("ةىيك")


def script_of(text: str) -> str:
    """Dominant script of ``text``: 'latin', 'devanagari', ... or 'unknown'."""
    counts: Counter[str] = Counter()
    for ch in text:
        cp = ord(ch)
        if ch.isascii():
            if ch.isalpha():
                counts["latin"] += 1
            continue
        if 0x00C0 <= cp <= 0x024F or 0x1E00 <= cp <= 0x1EFF:
            counts["latin"] += 1
            continue
        for name, lo, hi in _SCRIPTS:
            if lo <= cp <= hi:
                counts[name] += 1
                break
    if counts["kana"]:  # Japanese mixes kana with kanji; any kana means Japanese
        counts["kana"] += counts.pop("cjk", 0)
    return counts.most_common(1)[0][0] if counts else "unknown"


def is_hinglish(text: str) -> bool:
    """Latin-script text that reads like Hindi (>= 2 common Hindi function words)."""
    if script_of(text) != "latin":
        return False
    words = re.findall(r"[a-z]+", text.lower())
    hits = {w for w in words if w in _HINGLISH_WORDS}
    return len(hits) >= 2 and len([w for w in words if w in _HINGLISH_WORDS]) >= 0.2 * len(words)


def dominant_script(texts: list[str], sample: int = 200) -> str:
    counts: Counter[str] = Counter()
    for t in texts[:sample]:
        counts[script_of(t)] += 1
    counts.pop("unknown", None)
    return counts.most_common(1)[0][0] if counts else "unknown"


def needs_translation(question: str, doc_script: str) -> bool:
    """True when the question is unlikely to share vocabulary with the document."""
    if doc_script in ("unknown", ""):
        return False
    q_script = script_of(question)
    if q_script == "unknown":
        return False
    if q_script != doc_script:
        return True
    return doc_script == "latin" and is_hinglish(question)


#: First code point (the zero) of each non-ASCII decimal digit block: Arabic, Urdu/Persian,
#: N'Ko, Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam,
#: Sinhala, Thai, Lao, Tibetan, Myanmar, Khmer, Mongolian.
_DIGIT_ZEROS = (
    0x0660, 0x06F0, 0x07C0, 0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6,
    0x0D66, 0x0DE6, 0x0E50, 0x0ED0, 0x0F20, 0x1040, 0x17E0, 0x1810,
)  # fmt: skip
DIGITS_TO_ASCII = {zero + i: str(i) for zero in _DIGIT_ZEROS for i in range(10)}


def to_ascii_digits(text: str) -> str:
    """Map every script's decimal digits (०-९, ০-৯, ٠-٩, ۰-۹, ௦-௯, ...) to 0-9."""
    return text.translate(DIGITS_TO_ASCII)


# --------------------------------------------------------------------------- language of a text

# Letters plus combining marks (Indic vowel signs, Arabic harakat); the danda "।" ends a word.
_TOKEN_RE = re.compile(
    r"[^\W\d_](?:[^\W\d_]|[\u0300-\u036f\u0900-\u0963\u0966-\u0dff\u064b-\u065f\u0670])*"
)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(_nfc(text))]


def _best(scores: dict[str, float], default: str | None = None) -> str | None:
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not ranked or ranked[0][1] <= 0:
        return default
    if len(ranked) > 1 and ranked[1][1] == ranked[0][1]:
        return default if default in (ranked[0][0], ranked[1][0]) else None
    return ranked[0][0]


def _devanagari_language(tokens: list[str]) -> str | None:
    scores = {
        name: float(sum(t in words for t in tokens)) for name, words in _DEVANAGARI_MARKERS.items()
    }
    scores["Marathi"] += sum(t.endswith(("च्या", "ाचे", "ाची", "ाचा")) for t in tokens)
    # Sanskrit inflections end in visarga or a bare consonant far more often than Hindi words.
    final = sum(t.endswith(("ः", "्")) for t in tokens)
    if tokens and final >= 2 and final >= 0.3 * len(tokens):
        scores["Sanskrit"] += 2
    return _best(scores)


def _latin_language(text: str, tokens: list[str]) -> str | None:
    if is_hinglish(text):
        return "Hinglish"
    if any(ch in "ơưđạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ" for ch in text.lower()):
        return "Vietnamese"
    scores = {
        name: float(sum(t in words for t in tokens)) for name, words in _LATIN_MARKERS.items()
    }
    if not any(scores.values()):
        return "English"  # short or keyword-only questions ("Charminar height?") stay English
    best = _best(scores, default="English")
    # One foreign-looking word is not enough: "Des Moines population?" is English.
    if best and best != "English" and scores[best] < 2:
        return "English"
    return best


def _arabic_language(text: str) -> str | None:
    letters = set(text)
    if letters & _URDU_LETTERS:
        return "Urdu"
    if letters & _PERSIAN_LETTERS:
        return "Persian"
    if letters & _ARABIC_LETTERS:
        return "Arabic"
    return None


def _cyrillic_language(text: str) -> str | None:
    letters = set(text.lower())
    if letters & set("іїєґ"):
        return "Ukrainian"
    if letters & set("ђјљњћџѓќ"):
        return None  # Serbian or Macedonian: let the model read it
    if letters & set("әғқңөұүһ"):
        return None
    return "Russian"  # by far the most common; Bulgarian has no letter of its own to test


def language_of(text: str) -> str | None:
    """Best guess at the language of ``text`` ('Hindi', 'Telugu', 'Hinglish', ...), or None."""
    script = script_of(text)
    if script in _SCRIPT_LANGUAGE:
        return _SCRIPT_LANGUAGE[script]
    if script == "latin":
        return _latin_language(text, _tokens(text))
    if script == "devanagari":
        return _devanagari_language(_tokens(text))
    if script == "bengali":
        return "Assamese" if set(text) & set("ৰৱ") else "Bengali"
    if script == "arabic":
        return _arabic_language(text)
    if script == "cyrillic":
        return _cyrillic_language(text)
    return None


# --------------------------------------------------------------------------- explicit requests

_LANGUAGE_NAMES = {
    "English": "english angrezi अंग्रेज़ी अंग्रेजी इंग्लिश انگریزی ইংরেজি ఆంగ్లం ఇంగ్లీష్ ஆங்கிலம் "
    "ಇಂಗ್ಲಿಷ್ ഇംഗ്ലീഷ് અંગ્રેજી ਅੰਗਰੇਜ਼ੀ ଇଂରାଜୀ anglais inglés englisch",
    "Hindi": "hindi हिंदी हिन्दी ہندی হিন্দি హిందీ இந்தி ಹಿಂದಿ ഹിന്ദി હિન્દી ਹਿੰਦੀ ହିନ୍ଦୀ",
    "Urdu": "urdu उर्दू اردو উর্দু ఉర్దూ உருது ಉರ್ದು ഉറുദു ઉર્દૂ ਉਰਦੂ ଉର୍ଦ୍ଦୁ",
    "Marathi": "marathi मराठी",
    "Bengali": "bengali bangla बंगाली बांग्ला বাংলা بنگالی",
    "Telugu": "telugu तेलुगु తెలుగు تیلگو",
    "Tamil": "tamil तमिल தமிழ் تامل",
    "Kannada": "kannada कन्नड़ कन्नड ಕನ್ನಡ",
    "Malayalam": "malayalam मलयालम മലയാളം",
    "Gujarati": "gujarati गुजराती ગુજરાતી",
    "Punjabi": "punjabi panjabi पंजाबी ਪੰਜਾਬੀ پنجابی",
    "Odia": "odia oriya ओड़िया उड़िया ଓଡ଼ିଆ",
    "Assamese": "assamese असमिया অসমীয়া",
    "Nepali": "nepali नेपाली",
    "Sanskrit": "sanskrit संस्कृत संस्कृतम्",
    "Bhojpuri": "bhojpuri भोजपुरी",
    "Awadhi": "awadhi अवधी",
    "Maithili": "maithili मैथिली",
    "Arabic": "arabic अरबी عربي العربية عربی",
    "Persian": "persian farsi फ़ारसी फारसी فارسی",
    "French": "french français francais फ्रेंच फ़्रेंच",
    "Spanish": "spanish español espanol स्पेनिश",
    "German": "german deutsch जर्मन",
    "Portuguese": "portuguese português",
    "Italian": "italian italiano",
    "Chinese": "chinese 中文 汉语 चीनी",
    "Japanese": "japanese 日本語 जापानी",
    "Korean": "korean 한국어",
    "Russian": "russian русский रूसी",
}
_ALIASES = {alias: name for name, text in _LANGUAGE_NAMES.items() for alias in _words(text)}
# Native names written without their final virama/anusvara, so "தமிழ்" also matches
# "தமிழில்" and "മലയാളം" matches "മലയാളത്തിൽ".
_STEMS = {
    alias.rstrip("\u094d\u09cd\u0a4d\u0acd\u0b4d\u0bcd\u0c4d\u0ccd\u0d4d\u0d02"): name
    for alias, name in _ALIASES.items()
    if not alias.isascii()
}

# Words that ask for an answer: a language name next to one of these is a request.
_ANSWER_WORDS = _words(
    "answer reply respond write explain tell say give translate describe summarize summarise "
    "उत्तर जवाब बताओ बताइए बताइये बताएं बताएँ लिखो लिखिए लिखें समझाओ समझाइए दीजिए दें दो "
    "batao bataiye bataye jawab javab likho uttar samjhao "
    "सांगा लिहा جواب بتائیں بتاؤ بتایئے لکھیں لکھو উত্তর বলুন বলো লিখুন লেখো "
    "సమాధానం జవాబు చెప్పండి చెప్పు రాయండి பதில் சொல்லுங்கள் சொல் எழுதுங்கள் "
    "ಉತ್ತರ ಹೇಳಿ ಬರೆಯಿರಿ ഉത്തരം പറയൂ എഴുതുക જવાબ ઉત્તર કહો લખો ਜਵਾਬ ਉੱਤਰ ਦੱਸੋ ਲਿਖੋ "
    "ଉତ୍ତର କୁହ ଲେଖ réponds répondez réponse écris responde respuesta escribe antworte antwort"
)
# "in" written after the language name (Hindi में, Urdu میں, Hinglish mein ...).
_IN_AFTER = _words("में मे मा मध्ये میں ਵਿੱਚ ਵਿਚ mein me main mai")
# ... or glued to it as a locative suffix (తెలుగులో, தமிழில், ಕನ್ನಡದಲ್ಲಿ, বাংলায়, ગુજરાતીમાં).
_LOCATIVE_SUFFIXES = _words("లో ில் யில் ல் ದಲ್ಲಿ ನಲ್ಲಿ ಯಲ್ಲಿ ത്തിൽ യിൽ ിൽ য় তে ে માં ରେ ত त ीत")
_END_REQUEST_RE = re.compile(
    r"(?:^|[.?!,;:]\s*)(?:only\s+)?in(?:to)?\s+([a-zà-ÿ]+)(?:\s+(?:please|only|language))?\s*[.?!]*\s*$",
    re.IGNORECASE,
)


def requested_language(question: str) -> str | None:
    """Language the question explicitly asks the answer to be in, e.g. "... Answer in Urdu."."""
    text = _nfc(question or "")
    m = _END_REQUEST_RE.search(text.strip())
    if m and m.group(1).lower() in _ALIASES:
        return _ALIASES[m.group(1).lower()]
    tokens = _tokens(text)
    if not any(t in _ANSWER_WORDS for t in tokens):
        return None
    for i, tok in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if tok in _ALIASES:
            name = _ALIASES[tok]
            prev = tokens[i - 1] if i else ""
            if tok.isascii():
                # "in Urdu", "into Hindi", "Hindi mein" — and not "Tamil Nadu"
                if (prev in ("in", "into", "using") or nxt in _IN_AFTER) and nxt != "nadu":
                    return name
            elif nxt in _IN_AFTER:
                return name
            continue
        if tok.isascii():
            continue
        for stem, name in _STEMS.items():
            if tok.startswith(stem) and tok[len(stem) :] in _LOCATIVE_SUFFIXES:
                return name
    return None


# --------------------------------------------------------------------------- answer language


@dataclass(frozen=True)
class AnswerLanguage:
    name: str | None  # 'Telugu', 'English', ... or None when only the script is known
    instruction: str  # what the generator is told ("Write the answer in {instruction}")
    requested: bool = False


_SCRIPT_NAMES = {
    "devanagari": "Devanagari",
    "arabic": "Arabic",
    "cyrillic": "Cyrillic",
    "bengali": "Bengali",
    "ethiopic": "Ethiopic",
    "latin": "Latin",
}


def choose_answer_language(question: str) -> AnswerLanguage:
    requested = requested_language(question)
    if requested:
        return AnswerLanguage(
            requested, f"{requested} (the user asked for the answer in {requested})", requested=True
        )
    script = script_of(question)
    name = language_of(question)
    if name == "Hinglish":
        return AnswerLanguage(
            "Hinglish",
            "Hinglish (Hindi written in Latin script), the same way the question is written",
        )
    if name:
        if script == "devanagari":
            return AnswerLanguage(name, f"{name} (Devanagari script), the language of the question")
        return AnswerLanguage(name, name)
    if script == "devanagari":
        return AnswerLanguage(
            None,
            "the same language as the question. It is written in Devanagari, which Hindi, Marathi, "
            "Nepali, Sanskrit, Bhojpuri, Awadhi and Maithili all use: answer in the question's own "
            "language, not automatically in Hindi",
        )
    if script == "unknown":
        return AnswerLanguage("English", "English")
    label = _SCRIPT_NAMES.get(script, script)
    return AnswerLanguage(None, f"the same language as the question ({label} script)")


def answer_language(question: str) -> str:
    """Explicit language instruction for the generator (vague wording made a 7B model
    answer an English question in Hindi on Colab)."""
    return choose_answer_language(question).instruction


_NOT_FOUND_EN = "(The answer was not found in the document.)"
REFUSALS = {
    "Hindi": "दस्तावेज़ में इस प्रश्न का उत्तर नहीं मिला।",
    "Marathi": "दस्तऐवजात या प्रश्नाचे उत्तर सापडले नाही.",
    "Nepali": "कागजातमा यो प्रश्नको उत्तर भेटिएन।",
    "Sanskrit": "अस्य प्रश्नस्य उत्तरं प्रलेखे न लब्धम्।",
    "Bhojpuri": "दस्तावेज़ में एह सवाल के जवाब ना मिलल।",
    "Awadhi": "दस्तावेज़ मा ई सवाल कै जवाब नाहीं मिला।",
    "Maithili": "दस्तावेज़ मे एहि प्रश्नक उत्तर नहि भेटल।",
    "Urdu": "دستاویز میں اس سوال کا جواب نہیں ملا۔",
    "Bengali": "নথিতে এই প্রশ্নের উত্তর পাওয়া যায়নি।",
    "Assamese": "নথিত এই প্ৰশ্নৰ উত্তৰ পোৱা নগ'ল।",
    "Telugu": "ఈ ప్రశ్నకు సమాధానం పత్రంలో కనబడలేదు.",
    "Tamil": "இந்தக் கேள்விக்கான பதில் ஆவணத்தில் இல்லை.",
    "Kannada": "ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರ ದಾಖಲೆಯಲ್ಲಿ ಸಿಗಲಿಲ್ಲ.",
    "Malayalam": "ഈ ചോദ്യത്തിനുള്ള ഉത്തരം രേഖയിൽ കണ്ടെത്തിയില്ല.",
    "Gujarati": "દસ્તાવેજમાં આ પ્રશ્નનો જવાબ મળ્યો નથી.",
    "Punjabi": "ਦਸਤਾਵੇਜ਼ ਵਿੱਚ ਇਸ ਸਵਾਲ ਦਾ ਜਵਾਬ ਨਹੀਂ ਮਿਲਿਆ।",
    "Odia": "ଦଲିଲରେ ଏହି ପ୍ରଶ୍ନର ଉତ୍ତର ମିଳିଲା ନାହିଁ।",
    "Hinglish": "Document mein is sawaal ka jawab nahi mila.",
    "Arabic": "لم يتم العثور على إجابة هذا السؤال في المستند.",
    "Persian": "پاسخ این پرسش در سند یافت نشد.",
    "French": "La réponse à cette question n'a pas été trouvée dans le document.",
    "Spanish": "No se encontró la respuesta a esta pregunta en el documento.",
    "German": "Die Antwort auf diese Frage wurde im Dokument nicht gefunden.",
    "Portuguese": "A resposta a esta pergunta não foi encontrada no documento.",
    "Italian": "La risposta a questa domanda non è stata trovata nel documento.",
    "Russian": "Ответ на этот вопрос в документе не найден.",
    "Ukrainian": "Відповідь на це питання в документі не знайдено.",
    "Chinese": "文档中未找到该问题的答案。",
    "Japanese": "この質問の答えは文書の中に見つかりませんでした。",
    "Korean": "문서에서 이 질문에 대한 답을 찾을 수 없습니다.",
}


def refusal_message(question: str, default: str) -> str:
    """'Not found' in the language the answer would have been written in."""
    name = choose_answer_language(question).name
    text = REFUSALS.get(name or "")
    return f"{text} {_NOT_FOUND_EN}" if text else default
