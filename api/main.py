"""FastAPI entrypoint for PeopleRank UC1."""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

# Make src/ importable
_ROOT = Path(__file__).resolve().parent.parent
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from .adapter import attendee_to_person  # noqa: E402
from .auth import check_bearer  # noqa: E402
from .cache import idempotency_cache  # noqa: E402
from .rationale import generate_rationale  # noqa: E402
from .schemas import (  # noqa: E402
    Attendee,
    MatchRequest,
    MatchResponse,
    PairOut,
    SkippedOut,
)
from .scoring import extract_signals, score_people_openai  # noqa: E402

logger = logging.getLogger("peoplerank.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="PeopleRank UC1", version="1.0")


# ---------- helpers ----------

SCREENING_MIN_WORDS = 50


def _word_count(*texts: str) -> int:
    total = 0
    for t in texts:
        if not t:
            continue
        total += len(t.split())
    return total


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clamp01(v: float) -> float:
    if v < 0:
        return 0.0
    if v > 1:
        return 1.0
    return v


# ---------- routes ----------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "peoplerank-uc1", "time": _iso_now()}


@app.post("/v1/match/community-event")
async def match_community_event(request: Request):
    # 1. Auth
    check_bearer(request)

    # Parse body manually so we can return 400 with runId on validation error
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_json", "runId": None},
        )

    run_id = payload.get("runId") if isinstance(payload, dict) else None

    try:
        req = MatchRequest(**payload)
    except ValidationError as e:
        return JSONResponse(
            status_code=400,
            content={"error": "validation_error", "runId": run_id, "detail": e.errors()},
        )

    # 2. Idempotency
    cached = idempotency_cache.get(req.runId)
    if cached is not None:
        return cached

    # 3 + 4. Screening sufficiency
    attendees_by_id: Dict[str, Attendee] = {a.id: a for a in req.attendees}
    insufficient: Set[str] = set()
    for a in req.attendees:
        wc = _word_count(a.whyJoin or "", a.socialBravery or "", a.screeningNotes or "")
        if wc < SCREENING_MIN_WORDS:
            insufficient.add(a.id)

    scoring_pool = [a for a in req.attendees if a.id not in insufficient]

    skipped: List[SkippedOut] = [
        SkippedOut(memberId=aid, reason="insufficient_screening", note=f"<{SCREENING_MIN_WORDS} screening words")
        for aid in insufficient
    ]

    # Need >=2 attendees to score. If not, short-circuit — remaining members get no_viable_partners.
    if len(scoring_pool) < 2:
        for a in scoring_pool:
            skipped.append(SkippedOut(memberId=a.id, reason="no_viable_partners"))
        resp = MatchResponse(
            runId=req.runId,
            model="peoplerank-uc1-v1",
            generatedAt=_iso_now(),
            pairs=[],
            skipped=skipped,
        ).model_dump()
        idempotency_cache.set(req.runId, resp)
        return resp

    # 5. Score with OpenAI backend
    people = [attendee_to_person(a) for a in scoring_pool]
    try:
        scored_pairs = score_people_openai(people)
    except Exception as e:
        logger.exception("scoring failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"scoring_failed: {e}",
        )

    # Index pairs by member id for fast lookup.
    # Each pair in scored_pairs covers both directions (symmetric).
    pairs_by_member: Dict[str, List[Dict]] = {}
    for sp in scored_pairs:
        a_id = sp["person_a"]["id"]
        b_id = sp["person_b"]["id"]
        pairs_by_member.setdefault(a_id, []).append({"other": b_id, "pair": sp})
        pairs_by_member.setdefault(b_id, []).append({"other": a_id, "pair": sp})

    # 6. Pick top-2 per member
    selected: List[Dict] = []  # items: {memberId, partnerId, pair}
    no_viable: Set[str] = set()
    for member in scoring_pool:
        mid = member.id
        excluded = set(member.excludedPartnerIds or [])
        candidates = [
            c for c in pairs_by_member.get(mid, [])
            if c["other"] not in excluded and c["other"] not in insufficient
        ]
        candidates.sort(key=lambda c: c["pair"]["final_score"], reverse=True)
        top = candidates[:2]
        if not top:
            no_viable.add(mid)
            continue
        for c in top:
            selected.append({"memberId": mid, "partnerId": c["other"], "pair": c["pair"]})

    for mid in no_viable:
        skipped.append(SkippedOut(memberId=mid, reason="no_viable_partners"))

    # 7 + 8. Generate + validate rationales
    successful_pair_entries: List[Dict] = []  # holds final dict before PairOut
    member_success_counts: Dict[str, int] = {}

    # Maintain stable per-member rank assignment (by score order)
    per_member_rank_counter: Dict[str, int] = {}
    # Sort selected so each member's picks process in score order
    selected.sort(key=lambda s: (s["memberId"], -s["pair"]["final_score"]))

    for sel in selected:
        mid = sel["memberId"]
        pid = sel["partnerId"]
        member = attendees_by_id.get(mid)
        partner = attendees_by_id.get(pid)
        if member is None or partner is None:
            continue

        rationale = generate_rationale(member, partner)
        if rationale is None:
            # This specific rank is dropped.
            continue

        rank = per_member_rank_counter.get(mid, 0) + 1
        if rank > 2:
            continue
        per_member_rank_counter[mid] = rank

        final_score = sel["pair"].get("final_score", 0.0)
        compat_score = _clamp01(final_score / 100.0)

        successful_pair_entries.append({
            "memberId": mid,
            "partnerId": pid,
            "rank": rank,
            "rationale": rationale,
            "compatibilityScore": compat_score,
            "signals": extract_signals(sel["pair"]),
        })
        member_success_counts[mid] = member_success_counts.get(mid, 0) + 1

    # Members with 0 successful rationales -> low_confidence skipped
    selected_member_ids = {s["memberId"] for s in selected}
    for mid in selected_member_ids:
        if member_success_counts.get(mid, 0) == 0:
            skipped.append(SkippedOut(memberId=mid, reason="low_confidence"))

    # 9. Final validation pass
    attendee_ids = set(attendees_by_id.keys())
    final_pairs: List[PairOut] = []
    # Track rank uniqueness per member
    seen_ranks: Dict[str, Set[int]] = {}
    for entry in successful_pair_entries:
        mid = entry["memberId"]
        pid = entry["partnerId"]
        if mid == pid:
            continue
        if mid not in attendee_ids or pid not in attendee_ids:
            continue
        member_attendee = attendees_by_id[mid]
        if pid in (member_attendee.excludedPartnerIds or []):
            continue
        rank = entry["rank"]
        if rank not in (1, 2):
            continue
        if rank in seen_ranks.get(mid, set()):
            continue
        score = entry["compatibilityScore"]
        if not (0.0 <= score <= 1.0):
            score = _clamp01(score)
        final_pairs.append(PairOut(
            memberId=mid,
            partnerId=pid,
            rank=rank,
            rationale=entry["rationale"],
            compatibilityScore=score,
            signals=entry["signals"],
        ))
        seen_ranks.setdefault(mid, set()).add(rank)

    # 10. Build + cache + return
    response = MatchResponse(
        runId=req.runId,
        model="peoplerank-uc1-v1",
        generatedAt=_iso_now(),
        pairs=final_pairs,
        skipped=skipped,
    ).model_dump()

    idempotency_cache.set(req.runId, response)
    return response
