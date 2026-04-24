"""Claude-backed rationale generator with validators and retries."""
from __future__ import annotations

import logging
import os
import re
from typing import List, Optional, Tuple

from .schemas import Attendee

logger = logging.getLogger("peoplerank.rationale")

MODEL_ID = "claude-sonnet-4-5"
MAX_TOKENS = 400
TEMPERATURE = 0.7
MAX_RETRIES = 3

SYSTEM_PROMPT = """You are writing a short paragraph that will be shown to one person (the member) about another person (the partner) they should meet at an event. You have access to screening details for both.

VOICE RULES — violating any of these makes the output invalid:
- Addressed to the member as "you", about the partner in third person ("she", "he", "they").
- Do NOT use any first names or full names of people. Refer to the partner by pronoun or role noun only ("the paralegal", "the local"). Place names like "Austin", "East Side", "Rainey Street" ARE allowed.
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


# Banned as full substrings: these are AI-tell phrases that should never appear naturally
BANNED_SUBSTRINGS = [
    "in a world where", "dynamic blend", "perfect match",
    "rich tapestry", "here is why", "this match stands out",
    "what makes this special", "could be potentially", "seems to likely",
]

# Banned only when matching a specific regex pattern (avoids false positives on common words)
# Each entry: (pattern, human_reason)
BANNED_PATTERNS = [
    # "not just X but Y" structure — the real AI tell
    (r"\bnot just\b[^.!?]*\bbut\b", "not_just_but"),
    # Hedge words, as whole words only
    (r"\bmight be\b", "might_be"),
    (r"\bcould be\b", "could_be"),
    (r"\bpotentially\b", "potentially"),
    (r"\bseems to\b", "seems_to"),
    # Corporate verbs only as standalone words
    (r"\bleverage\b", "leverage"),
    (r"\bleveraging\b", "leveraging"),
    (r"\balign with\b", "align_with"),
    (r"\baligned with\b", "aligned_with"),
    (r"\balignment with\b", "alignment_with"),
    (r"\bembrace\b", "embrace"),
    (r"\bembraces\b", "embraces"),
    (r"\bembracing\b", "embracing"),
    # "shared passion for" — exact phrase
    (r"\bshared passion for\b", "shared_passion_for"),
    # "unique opportunity" — exact phrase
    (r"\bunique opportunity\b", "unique_opportunity"),
]

# Common first names we should flag if they appear followed by Title Case.
# We only flag TWO-Title-Case phrases where the FIRST word is a plausible first name.
COMMON_FIRST_NAMES = {
    "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph",
    "Thomas", "Charles", "Christopher", "Daniel", "Matthew", "Anthony", "Mark",
    "Donald", "Steven", "Paul", "Andrew", "Joshua", "Kenneth", "Kevin", "Brian",
    "George", "Edward", "Ronald", "Timothy", "Jason", "Jeffrey", "Ryan", "Jacob",
    "Gary", "Nicholas", "Eric", "Jonathan", "Stephen", "Larry", "Justin", "Scott",
    "Brandon", "Benjamin", "Samuel", "Gregory", "Frank", "Alexander", "Raymond",
    "Patrick", "Jack", "Dennis", "Jerry", "Tyler", "Aaron", "Jose", "Henry",
    "Adam", "Douglas", "Nathan", "Peter", "Zachary", "Kyle", "Noah", "Ethan",
    "Jeremy", "Walter", "Christian", "Sean", "Alan", "Keith", "Mohammed", "Ahmed",
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan",
    "Jessica", "Sarah", "Karen", "Nancy", "Lisa", "Betty", "Helen", "Sandra",
    "Donna", "Carol", "Ruth", "Sharon", "Michelle", "Laura", "Emily", "Kimberly",
    "Deborah", "Dorothy", "Amy", "Angela", "Ashley", "Brenda", "Emma", "Olivia",
    "Cynthia", "Marie", "Janet", "Catherine", "Frances", "Christine", "Samantha",
    "Debra", "Rachel", "Carolyn", "Virginia", "Maria", "Heather", "Diane",
    "Julie", "Joyce", "Victoria", "Kelly", "Christina", "Joan", "Evelyn",
    "Lauren", "Judith", "Megan", "Andrea", "Cheryl", "Hannah", "Jacqueline",
    "Martha", "Gloria", "Teresa", "Ann", "Sara", "Madison", "Frances", "Kathryn",
    "Janice", "Jean", "Abigail", "Alice", "Julia", "Judy", "Sophia", "Grace",
    "Denise", "Amber", "Doris", "Marilyn", "Danielle", "Beverly", "Isabella",
    "Theresa", "Diana", "Natalie", "Brittany", "Charlotte", "Marie", "Kayla",
    "Alexis", "Lori", "Carlo", "Karthik", "Kevin", "Mike", "Maggie", "Gwen",
    "Sagar", "Caleb", "Kaitlyn", "Angela", "Reem", "Rania"
}


# ---------- Validators ----------

def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _has_person_name(text: str) -> bool:
    """
    Only flag if we find a COMMON_FIRST_NAME as a standalone Title-Case token.
    This avoids false positives on place names like "East Side", "Rainey Street".
    """
    # Find all Title-Case words in the text (anywhere)
    words = re.findall(r"\b[A-Z][a-z]+\b", text)
    for w in words:
        if w in COMMON_FIRST_NAMES:
            return True
    return False


def validate_rationale(text: str) -> Tuple[bool, str]:
    """Return (ok, reason)."""
    if not text or not text.strip():
        return False, "empty"
    if len(text) > 800:
        return False, f"too_long:{len(text)}"
    if "\u2014" in text:
        return False, "em_dash"
    if re.search(r"\S\s*\u2013\s*\S", text):
        return False, "en_dash_between_words"
    if " -- " in text:
        return False, "double_dash"
    sentences = _split_sentences(text)
    if not (2 <= len(sentences) <= 7):
        return False, f"sentence_count:{len(sentences)}"
    lower = text.lower()
    for sub in BANNED_SUBSTRINGS:
        if sub in lower:
            return False, f"banned_sub:{sub}"
    for pattern, reason in BANNED_PATTERNS:
        if re.search(pattern, lower):
            return False, f"banned_pat:{reason}"
    if _has_person_name(text):
        return False, "person_name_detected"
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
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT + "\n\n" + FIVE_EXAMPLES,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_prompt}],
    )
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
    pair_tag = f"{member.id}->{partner.id}"
    for attempt in range(1, max_retries + 1):
        try:
            text = _call_claude(user_prompt)
        except Exception as e:
            logger.warning("rationale %s attempt %d api_error: %s", pair_tag, attempt, e)
            continue
        ok, reason = validate_rationale(text)
        if ok:
            if attempt > 1:
                logger.info("rationale %s succeeded on attempt %d", pair_tag, attempt)
            return text
        snippet = text[:120].replace("\n", " ")
        logger.warning(
            "rationale %s attempt %d failed validator: %s | text: %s...",
            pair_tag, attempt, reason, snippet,
        )
    logger.warning("rationale %s EXHAUSTED %d retries", pair_tag, max_retries)
    return None
