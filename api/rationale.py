"""Claude-backed rationale generator with validators and retries."""
from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

from .schemas import Attendee

MODEL_ID = "claude-sonnet-4-5"
MAX_TOKENS = 400
TEMPERATURE = 0.7
MAX_RETRIES = 5

SYSTEM_PROMPT = """You are writing a short paragraph that will be shown to one person (the member) about another person (the partner) they should meet at an event. You have access to screening details for both.

VOICE RULES — violating any of these makes the output invalid:
- Addressed to the member as "you", about the partner in third person ("she", "he", "they").
- Do NOT use any proper nouns that are names of people. Use pronouns or role nouns ("the paralegal", "the local"). Blind 8 renders the names in the surrounding UI.
- 3 to 5 sentences. Hard cap 800 characters.
- Short, declarative sentences. Periods do the work.
- Ground in specific details from screening (venues, cities, years, exact phrases). Never generic personality claims like "both thoughtful" or "both love to travel".
- NO em-dashes. No "—", no "--" used as an em-dash, no en-dashes. Use a period or a comma.
- NO hedge language: no "might", "could be", "potentially", "seems to", "likely".
- NO AI pattern phrases: no "not just X but Y", "in a world where", "dynamic blend", "perfect match", "shared passion for", "unique opportunity", "rich tapestry".
- NO meta commentary: no "here is why this works", no "this match stands out".
- NO superlatives without specifics. "Amazing" is banned unless followed immediately by the specific thing.
- NO corporate verbs: "leverage", "align", "embrace", "curate".
- Close with a specific action or prediction, not a feeling.

STRUCTURE: one observation about the member. one observation about the partner. one line about where their truths meet. one action or prediction to close. Short rationales are fine when the match is clean; do not pad.

OUTPUT: the rationale paragraph only. No preamble, no explanation, no quotes around it."""


FIVE_EXAMPLES = """FIVE EXAMPLES of the voice we want (read these before writing):

Example 1:
You have a 24-year map of Austin and you are still finding new pockets of it. She landed in January with nothing. No friends, no spots, no shortcuts, and she got quietly good at arriving alone. She wants the city you grew up in. You will remember parts of it by telling her.

Example 2:
She pulls people into acro yoga and slacklining on the first hangout. Your idea of a weekend is a fantasy draft and a show, not a couch. Two Austinites the same age who both refuse to sit still. She will have a Saturday plan before coffee cools.

Example 3:
You both left DC within a year of each other and are quietly rebuilding from scratch. You moved last summer and jumped into a fantasy league you are still learning. He baked bread through his first Austin winter and flew somewhere alone for his 31st. Neither of you is making a show of the reset. Ask him what winter here taught him.

Example 4:
You arrived four weeks ago and you already have a private rule about third places. She has been here six years and thirty-seven countries and still writes a list of people she wants to meet this week. You are both running on appetite. She will be the one who texts you at noon asking if you have eaten yet, and then takes you somewhere you did not know existed.

Example 5:
You both moved from a job you loved for a reason you do not fully want to explain yet. She left publishing in March. You left teaching in June. If you want to skip the small talk, start with whichever one of you is more tired today."""


BANNED_PHRASES = [
    "might", "could be", "potentially", "seems to", "likely",
    "not just", "in a world where", "dynamic blend", "perfect match",
    "shared passion for", "unique opportunity", "rich tapestry",
    "here is why", "this match stands out", "what makes this special",
    "leverage", "align", "embrace", "curate",
]

# Common multi-word place names that should NOT trigger the "name" heuristic
KNOWN_OK_TITLECASE = {
    "New York", "Los Angeles", "San Francisco", "San Diego", "San Jose",
    "New Jersey", "New Mexico", "New Orleans", "New Hampshire",
    "Hong Kong", "Rhode Island", "North Carolina", "South Carolina",
    "North Dakota", "South Dakota", "West Virginia", "Salt Lake",
    "Las Vegas", "Santa Fe", "Santa Monica", "Santa Barbara",
    "United States", "Silicon Valley", "Bay Area", "Wall Street",
    "Central Park", "Times Square", "Fifth Avenue",
    "Mexico City", "Cape Town", "Tel Aviv",
}


# ---------- Validators ----------

def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _has_name_like_sequence(text: str) -> bool:
    """
    Heuristic: two consecutive Title-Case words not at the start of a sentence.
    Allow known place names.
    """
    # Build list of sentence-start positions in the original text
    sentences = _split_sentences(text)
    # For each sentence, look for TitleCase TitleCase that is NOT the first two words
    name_re = re.compile(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b")
    for sent in sentences:
        # Find token index of each match
        tokens = re.findall(r"\S+", sent)
        # Find matches anywhere in the sentence
        for m in name_re.finditer(sent):
            phrase = f"{m.group(1)} {m.group(2)}"
            if phrase in KNOWN_OK_TITLECASE:
                continue
            # Determine if this match starts at the first word of the sentence
            start_char = m.start()
            leading = sent[:start_char]
            leading_tokens = re.findall(r"\S+", leading)
            if len(leading_tokens) == 0:
                # At sentence start — allowed (e.g. "She has been...")
                continue
            return True
    return False


def validate_rationale(text: str) -> Tuple[bool, str]:
    """Return (ok, reason)."""
    if not text or not text.strip():
        return False, "empty"
    if len(text) > 800:
        return False, "too_long"
    if "\u2014" in text:
        return False, "em_dash"
    if "–" in text:  # U+2013
        # fail if used between words (has non-space around it)
        if re.search(r"\S\s*\u2013\s*\S", text):
            return False, "en_dash"
    if " -- " in text:
        return False, "double_dash"
    sentences = _split_sentences(text)
    if not (2 <= len(sentences) <= 6):
        return False, f"sentence_count:{len(sentences)}"
    lower = text.lower()
    for phrase in BANNED_PHRASES:
        if phrase in lower:
            return False, f"banned:{phrase}"
    if _has_name_like_sequence(text):
        return False, "name_like"
    return True, "ok"


# ---------- User prompt builder ----------

def _fmt_host_notes(a: Attendee) -> str:
    if not a.hostNotes:
        return "none"
    return " | ".join(hn.notes for hn in a.hostNotes if hn.notes) or "none"


def _fmt_feedback(a: Attendee) -> str:
    if not a.feedback:
        return "none"
    parts = []
    for f in a.feedback:
        bits = []
        if f.highlight:
            bits.append(f"highlight: {f.highlight}")
        if f.myDinner:
            bits.append(f"dinner: {f.myDinner}")
        if f.nextBigEvent:
            bits.append(f"next: {f.nextBigEvent}")
        if bits:
            parts.append("; ".join(bits))
    return " || ".join(parts) if parts else "none"


def _fmt_person_block(a: Attendee) -> str:
    return (
        f"- Age: {a.age if a.age is not None else 'unknown'}\n"
        f"- City: {a.city or 'unknown'}\n"
        f"- Occupation: {a.occupation or 'unknown'}\n"
        f"- Archetype: {a.guestType or 'unknown'}, {', '.join(a.guestArchetype) or 'none'}\n"
        f"- Why they joined: {a.whyJoin or 'unknown'}\n"
        f"- Social bravery example: {a.socialBravery or 'unknown'}\n"
        f"- Passion: {a.passion or 'unknown'}\n"
        f"- Screening notes: {a.screeningNotes or 'none'}\n"
        f"- Host notes from past events: {_fmt_host_notes(a)}\n"
        f"- Post-event feedback highlights: {_fmt_feedback(a)}"
    )


def build_user_prompt(member: Attendee, partner: Attendee) -> str:
    return (
        "MEMBER (the reader, addressed as \"you\"):\n"
        f"{_fmt_person_block(member)}\n\n"
        "PARTNER (the person being suggested):\n"
        f"{_fmt_person_block(partner)}\n\n"
        f"{FIVE_EXAMPLES}\n\n"
        "Now write the rationale."
    )


# ---------- Claude call ----------

_client = None


def _get_client():
    global _client
    if _client is None:
        from anthropic import Anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _client = Anthropic(api_key=api_key)
    return _client


def _call_claude(user_prompt: str) -> str:
    client = _get_client()
    resp = client.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    # concatenate all text blocks
    out = "".join(
        getattr(block, "text", "") for block in resp.content
        if getattr(block, "type", None) == "text"
    ).strip()
    return out


def generate_rationale(
    member: Attendee,
    partner: Attendee,
    max_retries: int = MAX_RETRIES,
) -> Optional[str]:
    """Return a validated rationale or None after exhausting retries."""
    user_prompt = build_user_prompt(member, partner)
    last_err: Optional[str] = None
    for _ in range(max_retries):
        try:
            text = _call_claude(user_prompt)
        except Exception as e:  # API error, network, etc.
            last_err = f"api_error:{e}"
            continue
        ok, reason = validate_rationale(text)
        if ok:
            return text
        last_err = reason
    return None
