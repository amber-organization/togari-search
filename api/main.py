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
    ScoreBreakdown,
    SkippedOut,
)
from .scoring import extract_signals, score_people_openai  # noqa: E402

logger = logging.getLogger("peoplerank.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="PeopleRank UC1", version="1.0")


# ---------- helpers ----------

SCREENING_MIN_WORDS = 50
CONF_HIGH = 0.70
CONF_LOW = 0.45


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


def _readiness_harmony(pair: Dict) -> float:
    """
    Derive a readiness harmony value in [0,1]. We use min(readiness_a, readiness_b)
    over 100 as a simple proxy: a pair is only as ready as its least-ready side.
    """
    ra = (pair.get("person_a") or {}).get("readiness") or 0.0
    rb = (pair.get("person_b") or {}).get("readiness") or 0.0
    low = min(ra, rb)
    if low <= 0:
        return 0.0
    return _clamp01(low / 100.0)


def _compute_calibrated_score(pair: Dict) -> float:
    """
    Calibrated 0-1 compatibility for the response.
    Formula: 0.7 * textSim + 0.3 * structSim, clamped.
    This lands naturally in 0.3 - 0.9 for real matches and matches the
    brief's example scores (0.87, 0.74).
    """
    text_sim = float(pair.get("vec_sim", 0.0) or 0.0)
    struct_sim = float(pair.get("struct_sim", 0.0) or 0.0)
    return _clamp01(0.70 * text_sim + 0.30 * struct_sim)


def _confidence_band(score: float) -> str:
    if score >= CONF_HIGH:
        return "high"
    if score >= CONF_LOW:
        return "medium"
    return "low"


def _explain(breakdown_text: float, breakdown_struct: float, trust: float, readiness: float, score: float) -> str:
    """Template-based one-line explanation of the score."""
    parts: List[str] = []

    if breakdown_text >= 0.55:
        parts.append("strong written overlap")
    elif breakdown_text >= 0.35:
        parts.append("moderate written overlap")
    else:
        parts.append("limited written overlap")

    if breakdown_struct >= 0.70:
        parts.append("aligned demographics and archetypes")
    elif breakdown_struct >= 0.45:
        parts.append("partial demographic alignment")
    else:
        parts.append("weak demographic alignment")

    if trust >= 0.70:
        trust_phrase = "prior connection in the graph"
    elif trust >= 0.35:
        trust_phrase = "some prior context"
    else:
        trust_phrase = "no prior relationship"
    parts.append(trust_phrase)

    if readiness >= 0.70:
        parts.append("both socially ready")
    elif readiness >= 0.40:
        parts.append("mixed readiness")
    else:
        parts.append("low readiness signal")

    return "; ".join(parts) + "."


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

    # 5. Score
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
    pairs_by_member: Dict[str, List[Dict]] = {}
    for sp in scored_pairs:
        a_id = sp["person_a"]["id"]
        b_id = sp["person_b"]["id"]
        pairs_by_member.setdefault(a_id, []).append({"other": b_id, "pair": sp})
        pairs_by_member.setdefault(b_id, []).append({"other": a_id, "pair": sp})

    # 6. Pick top-2 per member (ranking still uses the internal full score)
    selected: List[Dict] = []
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

    # 7 + 8. Generate + validate rationales (in parallel)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    successful_pair_entries: List[Dict] = []
    member_success_counts: Dict[str, int] = {}
    per_member_rank_counter: Dict[str, int] = {}
    selected.sort(key=lambda s: (s["memberId"], -s["pair"]["final_score"]))

    # Fire all rationale calls in parallel, up to 10 concurrent
    def _gen(sel):
        mid = sel["memberId"]
        pid = sel["partnerId"]
        member = attendees_by_id.get(mid)
        partner = attendees_by_id.get(pid)
        if member is None or partner is None:
            return None
        rationale = generate_rationale(member, partner)
        return (sel, rationale)

    rationale_results: List[tuple] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_gen, sel) for sel in selected]
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                rationale_results.append(result)

    # Preserve original ordering: sort results by (memberId, -score)
    rationale_results.sort(key=lambda r: (r[0]["memberId"], -r[0]["pair"]["final_score"]))

    for sel, rationale in rationale_results:
        mid = sel["memberId"]
        pid = sel["partnerId"]
        if rationale is None:
            continue

        rank = per_member_rank_counter.get(mid, 0) + 1
        if rank > 2:
            continue
        per_member_rank_counter[mid] = rank

        pair = sel["pair"]

        compat_score = _compute_calibrated_score(pair)
        confidence = _confidence_band(compat_score)

        text_sim = float(pair.get("vec_sim", 0.0) or 0.0)
        struct_sim = float(pair.get("struct_sim", 0.0) or 0.0)
        trust = float(pair.get("trust", 0.0) or 0.0)
        readiness = _readiness_harmony(pair)

        breakdown = ScoreBreakdown(
            textSimilarity=round(_clamp01(text_sim), 4),
            structuredSimilarity=round(_clamp01(struct_sim), 4),
            trust=round(_clamp01(trust), 4),
            readinessHarmony=round(readiness, 4),
            explanation=_explain(text_sim, struct_sim, trust, readiness, compat_score),
        )

        successful_pair_entries.append({
            "memberId": mid,
            "partnerId": pid,
            "rank": rank,
            "rationale": rationale,
            "compatibilityScore": round(compat_score, 4),
            "confidence": confidence,
            "scoreBreakdown": breakdown,
            "signals": extract_signals(pair),
        })
        member_success_counts[mid] = member_success_counts.get(mid, 0) + 1

    selected_member_ids = {s["memberId"] for s in selected}
    for mid in selected_member_ids:
        if member_success_counts.get(mid, 0) == 0:
            skipped.append(SkippedOut(memberId=mid, reason="low_confidence"))

    # 9. Final validation pass
    attendee_ids = set(attendees_by_id.keys())
    final_pairs: List[PairOut] = []
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
            confidence=entry["confidence"],
            scoreBreakdown=entry["scoreBreakdown"],
            signals=entry["signals"],
        ))
        seen_ranks.setdefault(mid, set()).add(rank)

    response = MatchResponse(
        runId=req.runId,
        model="peoplerank-uc1-v1",
        generatedAt=_iso_now(),
        pairs=final_pairs,
        skipped=skipped,
    ).model_dump()

    idempotency_cache.set(req.runId, response)
    return response


@app.get("/test")
async def test_harness():
    from .test_page import test_page
    return test_page()
