"""
Language detection for en / hi / mr.

The actual problem
------------------
Script detection is trivial for English (Latin) but USELESS for separating
Hindi from Marathi: both are written in Devanagari, share most of their
Sanskrit-derived vocabulary, and MSMARCO-XI queries are short (a handful of
words). So the discriminator has to be morphological.

Marathi and Hindi diverge reliably in exactly the places short queries touch --
the copula, negation, and genitive postpositions:

    concept     Hindi              Marathi
    ---------   ----------------   -------------------
    "is/are"    है / हैं / था      आहे / आहेत / होता
    "what"      क्या               काय
    "not"       नहीं               नाही
    genitive    का / की / के       चा / ची / चे / च्या
    "means"     मतलब               म्हणजे

We score marker hits for each language and fall back to Hindi (the higher-prior
language) when a Devanagari string yields no markers at all, reporting LOW
confidence so the caller can prefer Sarvam's own language_code.

Precedence at runtime: Sarvam STT returns a detected language_code, and for the
VOICE path that wins -- it hears phonology, we only see text. This detector is
the fallback for the text path (benchmarks, typed input) and the cross-check.

Accuracy is measured on 2,000 parallel query triples in
evaluation/language_detection.py -- not asserted here.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import regex

_DEVANAGARI = regex.compile(r"\p{Devanagari}")
_LATIN = regex.compile(r"\p{Latin}")
_WORD = regex.compile(r"[\p{L}\p{N}\p{M}]+")

# Marker sets. Marathi first: its markers are rarer in Hindi than vice versa,
# so a Marathi hit is stronger evidence than a Hindi hit.
_MR_MARKERS = {
    "आहे", "आहेत", "आहात", "नाही", "नाहीत", "काय", "म्हणजे", "म्हणून",
    "मध्ये", "किंवा", "आणि", "पण", "होते", "होती", "करणे", "असे", "कसे",
    "कोण", "कुठे", "केव्हा", "त्याच्या", "तिच्या", "माझा", "तुमचा",
    "शकते", "पाहिजे", "अशा", "किती", "कशी", "कोणत्या", "यांनी", "असते",
    # Marathi finite-verb forms. Hindi conjugates with ता/ती/ते + copula
    # ("करता है"); Marathi fuses person into the verb ("करतो", "करतात").
    "करतो", "करते", "करतात", "होतो", "झाले", "झाली", "झाला", "असतो",
    "देते", "येते", "जाते", "मिळते", "लागते", "वापरले", "केले", "दिले",
}
_HI_MARKERS = {
    "है", "हैं", "हूँ", "हूं", "था", "थे", "थी", "नहीं", "क्या", "मतलब",
    "की", "के", "को", "में", "और", "या", "लेकिन", "कैसे", "कौन",
    "कहाँ", "कहां", "कब", "क्यों", "उसके", "उनके", "मेरा", "आपका",
    "सकता", "सकते", "चाहिए", "करना", "वाले", "गया", "कितना", "कितने",
}

# Tokens that LOOK discriminating but are not. Measured cost of getting these
# wrong (evaluation/language_detection.py, first run): Marathi recall 0.644.
#   ने   Hindi ergative ("X ने कहा") -- extremely common in Hindi, so listing
#        it as Marathi-only misfired at confidence 1.0 on Hindi queries.
#   का   Hindi genitive AND the Marathi interrogative particle ("मदत करतो का?").
#   होता shared past copula.
#   ला   Marathi dative but a frequent Hindi verb stem fragment.
_AMBIGUOUS = {"ने", "का", "होता", "ला", "त", "ही", "तो", "ते"}
_SHARED = (_MR_MARKERS & _HI_MARKERS) | _AMBIGUOUS
_MR_ONLY = _MR_MARKERS - _SHARED
_HI_ONLY = _HI_MARKERS - _SHARED

# THE structural difference between the two languages for short queries:
# Marathi agglutinates case/genitive postpositions onto the noun, Hindi keeps
# them as separate words.
#     mr  पोटॅशियमचे  खाद्यपदार्थांचा  फ्रँकने
#     hi  पोटैशियम के  खाद्य पदार्थों का  फ़्रैंक ने
# So Marathi evidence must be matched as a SUFFIX, not as a token. Without
# this, Marathi markers simply never fire on noun-phrase queries.
_MR_SUFFIXES = (
    "चा", "ची", "चे", "च्या", "चं", "ंचा", "ंची", "ंचे", "ंच्या",
    "ांचा", "ांची", "ांचे", "ांच्या", "ावर", "ाला", "ाने", "ांना",
    "तील", "मधील", "साठी", "कडे", "पासून", "पर्यंत", "शी", "णे",
)
_MIN_SUFFIX_STEM = 3   # avoid matching the bare postposition as a whole word

# Marathi-specific characters.
#   ळ  (U+0933) standard in Marathi, effectively absent from Hindi
#   ॲ  (U+0972) and ॅ (U+0945, candra E) are how Marathi writes English
#      loanword vowels -- Hindi uses ऑ/ॉ (U+0911/U+0949) for the same job.
#   ऱ  (U+0931) Marathi eyelash-ra
_MR_CHARS = ("ळ", "ऱ", "ॲ", "ॅ")
_HI_CHARS = ("ॉ", "ऑ", "ज़", "फ़", "क़", "ख़", "ग़")


@dataclass
class LanguageResult:
    lang: str
    confidence: float
    method: str
    scores: dict


def detect_language(text: str, default: str = "en") -> LanguageResult:
    text = unicodedata.normalize("NFC", (text or "").strip())
    if not text:
        return LanguageResult(default, 0.0, "empty", {})

    deva = len(_DEVANAGARI.findall(text))
    latin = len(_LATIN.findall(text))

    if deva == 0:
        conf = 0.99 if latin else 0.3
        return LanguageResult("en", conf, "script:latin",
                              {"latin": latin, "devanagari": 0})

    # Mixed script (code-mixed / transliterated) -- Devanagari still wins,
    # because an English speaker does not emit Devanagari, but a Hindi or
    # Marathi speaker frequently emits Latin loanwords.
    tokens = _WORD.findall(text)
    token_set = set(tokens)

    mr_tok = len(token_set & _MR_ONLY)
    hi_tok = len(token_set & _HI_ONLY)

    # Suffix evidence -- the decisive signal for Marathi noun-phrase queries.
    mr_suf = sum(
        1 for t in tokens
        if len(t) > _MIN_SUFFIX_STEM and t not in _HI_ONLY
        and t.endswith(_MR_SUFFIXES))

    mr_chr = sum(text.count(ch) for ch in _MR_CHARS)
    hi_chr = sum(text.count(ch) for ch in _HI_CHARS)

    mr = mr_tok + mr_suf + mr_chr
    hi = hi_tok + hi_chr

    scores = {"mr_markers": mr_tok, "mr_suffixes": mr_suf, "mr_chars": mr_chr,
              "hi_markers": hi_tok, "hi_chars": hi_chr,
              "devanagari": deva, "latin": latin}

    if mr == hi == 0:
        # Devanagari but no discriminating marker: short noun-phrase queries
        # like "कॉर्पोरेशन" are genuinely ambiguous between hi and mr.
        return LanguageResult("hi", 0.34, "devanagari:ambiguous", scores)

    total = mr + hi
    if mr > hi:
        return LanguageResult("mr", round(0.5 + 0.5 * mr / total, 3),
                              "markers", scores)
    if hi > mr:
        return LanguageResult("hi", round(0.5 + 0.5 * hi / total, 3),
                              "markers", scores)
    return LanguageResult("hi", 0.5, "markers:tie", scores)


def resolve_language(requested: str, transcript: str,
                     stt_lang: str | None = None) -> LanguageResult:
    """
    Combine the three signals in precedence order.

    1. An explicit user choice in the UI is authoritative -- if someone picks
       Marathi, we answer in Marathi even if the text looks Hindi.
    2. Sarvam's STT language_code, which is derived from audio.
    3. Our text heuristic.
    """
    if requested and requested != "auto":
        return LanguageResult(requested, 1.0, "user_selected", {})

    if stt_lang:
        code = stt_lang.split("-")[0].lower()
        if code in ("en", "hi", "mr"):
            text_guess = detect_language(transcript)
            # Sarvam cannot distinguish hi/mr from audio perfectly either; if
            # our markers disagree CONFIDENTLY, trust the text.
            if (text_guess.lang != code and text_guess.confidence >= 0.75
                    and {text_guess.lang, code} == {"hi", "mr"}):
                return LanguageResult(text_guess.lang, text_guess.confidence,
                                      "text_override_stt", text_guess.scores)
            return LanguageResult(code, 0.9, "sarvam_stt", {})

    return detect_language(transcript)
