"""
Input guardrail: empty / unintelligible / unsafe / prompt-injection.

Design stance
-------------
This layer runs BEFORE retrieval, so it must be cheap (sub-millisecond) and it
must not need a model. It is a filter, not a classifier: it catches the clear
cases and lets everything ambiguous through to the retrieval-confidence
guardrail, which has actual evidence to reason about. Over-blocking here would
refuse legitimate questions, which is a worse failure than passing a borderline
one to a stage that can still refuse.

Prompt injection specifically
-----------------------------
Injection reaches us through a *transcript* -- someone speaking instructions at
the microphone. It cannot arrive through the retrieved passages in the usual
way, because our corpus is a fixed MS MARCO snapshot we control. So we check
the transcript only, and we check it in all three languages: an English-only
pattern list would leave Hindi and Marathi injection unguarded, which is the
kind of gap that looks like working security until someone tries it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import regex

from app.schemas import RefusalReason

MIN_CHARS = 2
MIN_ALNUM_RATIO = 0.35


@dataclass
class GuardResult:
    allowed: bool
    reason: RefusalReason | None = None
    detail: str = ""
    matched: list[str] = field(default_factory=list)


# --- prompt injection ------------------------------------------------------
# Multilingual on purpose. The Hindi/Marathi entries are the natural spoken
# renderings of the same intents, not transliterations of the English strings.
_INJECTION = [
    # English
    r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+instructions?",
    r"disregard\s+(?:all\s+|the\s+)?(?:previous|prior|above)\s+",
    r"forget\s+(?:everything|all|your)\s+(?:instructions?|rules?|prompt)",
    r"you\s+are\s+(?:now|no\s+longer)\s+",
    r"(?:reveal|show|print|repeat|tell\s+me)\s+(?:your|the)\s+"
    r"(?:system\s+)?(?:prompt|instructions?|rules)",
    r"act\s+as\s+(?:if\s+you\s+are\s+)?(?:a\s+)?(?:dan|jailbreak|unrestricted)",
    r"developer\s+mode",
    r"pretend\s+(?:that\s+)?you\s+(?:are|have)\s+no\s+(?:rules|restrictions)",
    r"\bsudo\b|\broot\s+access\b",
    r"override\s+(?:your\s+)?(?:safety|guardrails?|restrictions?)",
    # Hindi
    r"पिछले\s+(?:सभी\s+)?निर्देश",
    r"निर्देश(?:ों)?\s+को\s+(?:भूल|अनदेखा|नज़रअंदाज)",
    r"अपना\s+(?:सिस्टम\s+)?प्रॉम्प्ट\s+(?:बताओ|दिखाओ|बता)",
    r"तुम\s+अब\s+.{0,20}\s*हो",
    # Marathi
    r"मागील\s+(?:सर्व\s+)?सूचना",
    r"सूचना\s+(?:विसर|दुर्लक्ष)",
    r"तुझा\s+(?:सिस्टम\s+)?प्रॉम्प्ट\s+(?:सांग|दाखव)",
]
_INJECTION_RX = [regex.compile(p, regex.IGNORECASE) for p in _INJECTION]

# --- unsafe content --------------------------------------------------------
# Deliberately narrow: clear harm-seeking intent only. MS MARCO is full of
# medical, legal and historical-violence queries that are perfectly legitimate
# ("what is the lethal dose of acetaminophen" is a real health question), so a
# broad keyword list would refuse the corpus's own content.
_UNSAFE = [
    r"how\s+(?:do\s+i|to)\s+(?:make|build|synthesi[sz]e)\s+"
    r"(?:a\s+)?(?:bomb|explosive|nerve\s+agent|meth|ricin)",
    r"how\s+(?:do\s+i|to)\s+(?:kill|murder|poison)\s+(?:a\s+|my\s+|someone)",
    r"(?:child|minor)\s+(?:porn|sexual)",
    r"how\s+to\s+(?:hack|ddos|breach)\s+(?:into\s+)?(?:someone|a\s+bank|their)",
    r"बम\s+कैसे\s+बना",
    r"बॉम्ब\s+कसा\s+बनव",
]
_UNSAFE_RX = [regex.compile(p, regex.IGNORECASE) for p in _UNSAFE]

_ALNUM = regex.compile(r"[\p{L}\p{N}]")
_WORD = regex.compile(r"[\p{L}\p{N}\p{M}]+")

# Sarvam returns these for silence / undecodable audio.
_STT_NULL = {"", "...", "null", "none", "<unk>", "[inaudible]", "n/a"}


def check_input(text: str | None) -> GuardResult:
    raw = (text or "").strip()

    if not raw or raw.lower() in _STT_NULL:
        return GuardResult(False, RefusalReason.empty_input,
                           "no speech detected")

    if len(raw) < MIN_CHARS:
        return GuardResult(False, RefusalReason.unintelligible,
                           f"transcript shorter than {MIN_CHARS} chars")

    # Mostly-punctuation transcripts are STT noise, not questions.
    alnum = len(_ALNUM.findall(raw))
    if alnum / max(len(raw), 1) < MIN_ALNUM_RATIO:
        return GuardResult(False, RefusalReason.unintelligible,
                           f"alphanumeric ratio {alnum / len(raw):.2f} "
                           f"below {MIN_ALNUM_RATIO}")

    if not _WORD.findall(raw):
        return GuardResult(False, RefusalReason.unintelligible,
                           "no word characters")

    hits = [rx.pattern for rx in _INJECTION_RX if rx.search(raw)]
    if hits:
        return GuardResult(False, RefusalReason.prompt_injection,
                           "instruction-override attempt in transcript", hits)

    hits = [rx.pattern for rx in _UNSAFE_RX if rx.search(raw)]
    if hits:
        return GuardResult(False, RefusalReason.unsafe,
                           "harm-seeking request", hits)

    return GuardResult(True)


def sanitize_for_prompt(text: str, max_chars: int = 600) -> str:
    """
    Neutralise text before it enters a prompt.

    Even after the injection check passes, the transcript is untrusted input.
    We strip characters used to fake prompt structure (role headers, fences)
    and hard-cap length so a long spoken monologue cannot push the grounding
    instructions out of the model's attention.
    """
    cleaned = regex.sub(r"[`\p{Cc}]", " ", text)
    cleaned = regex.sub(r"(?im)^\s*(system|assistant|user|human)\s*:", r"\1 -",
                        cleaned)
    cleaned = regex.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_chars]
