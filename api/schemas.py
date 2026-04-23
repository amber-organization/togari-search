"""Pydantic schemas for UC1 community-event matching."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ---------- Request ----------

class EventInfo(BaseModel):
    id: str
    type: Literal["community_event"] = "community_event"
    name: str
    city: str
    venue: str
    startsAt: str
    attendeeCount: int


class HostNote(BaseModel):
    eventId: str
    notes: str


class FeedbackEntry(BaseModel):
    eventId: str
    oneWord: Optional[str] = None
    myDinner: Optional[str] = None
    highlight: Optional[str] = None
    improvement: Optional[str] = None
    nextBigEvent: Optional[str] = None
    anythingElse: Optional[str] = None


class Attendee(BaseModel):
    id: str
    age: Optional[int] = None
    ageGroupPreference: Optional[
        Literal["22-25", "26-30", "31-35", "no_preference"]
    ] = None
    gender: Optional[Literal["man", "woman", "non_binary"]] = None
    city: Optional[str] = None
    occupation: Optional[str] = None
    guestType: Optional[Literal["storyteller", "investigator", "listener"]] = None
    guestArchetype: List[str] = Field(default_factory=list)
    whyJoin: Optional[str] = None
    socialBravery: Optional[str] = None
    passion: Optional[str] = None
    screeningNotes: Optional[str] = None
    hostNotes: List[HostNote] = Field(default_factory=list)
    feedback: List[FeedbackEntry] = Field(default_factory=list)
    welcomeDinnerEventId: Optional[str] = None
    eventsAttended: List[str] = Field(default_factory=list)
    excludedPartnerIds: List[str] = Field(default_factory=list)


class MatchRequest(BaseModel):
    runId: str
    event: EventInfo
    attendees: List[Attendee]


# ---------- Response ----------

class PairOut(BaseModel):
    memberId: str
    partnerId: str
    rank: Literal[1, 2]
    rationale: str
    compatibilityScore: float
    signals: List[str] = Field(default_factory=list)


class SkippedOut(BaseModel):
    memberId: str
    reason: Literal["no_viable_partners", "insufficient_screening", "low_confidence"]
    note: Optional[str] = None


class MatchResponse(BaseModel):
    runId: str
    model: str = "peoplerank-uc1-v1"
    generatedAt: str
    pairs: List[PairOut]
    skipped: List[SkippedOut]
