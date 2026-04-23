"""Browser-based test harness for the UC1 endpoint."""
from fastapi.responses import HTMLResponse

TEST_PAGE_HTML = """<!DOCTYPE html>
<html>
<head>
<title>PeopleRank UC1 · Test Harness</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 24px; background: #0f1a2e; color: #e2e8f0; }
  h1 { color: #60a5fa; font-size: 22px; margin-bottom: 4px; }
  .sub { color: #94a3b8; font-size: 13px; margin-bottom: 24px; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 16px; }
  .card { background: #1e293b; border-radius: 10px; padding: 20px; border: 1px solid #334155; }
  .card h3 { color: #93c5fd; font-size: 14px; margin: 0 0 14px; letter-spacing: 1px; text-transform: uppercase; }
  label { display: block; color: #cbd5e1; font-size: 12px; margin: 10px 0 4px; font-weight: 600; }
  input, textarea, select { width: 100%; padding: 8px 10px; background: #0f172a; border: 1px solid #334155; color: #e2e8f0; border-radius: 6px; font-size: 13px; font-family: inherit; }
  textarea { min-height: 60px; resize: vertical; }
  button { background: #2563eb; color: white; border: none; padding: 12px 28px; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; margin-top: 20px; }
  button:hover { background: #1d4ed8; }
  button:disabled { background: #475569; cursor: not-allowed; }
  #result { background: #020617; border: 1px solid #334155; border-radius: 10px; padding: 20px; margin-top: 24px; white-space: pre-wrap; font-family: 'SF Mono', Monaco, monospace; font-size: 12px; line-height: 1.6; color: #a5f3fc; max-height: 500px; overflow-y: auto; }
  .pair { background: #1e293b; border-left: 3px solid #60a5fa; padding: 20px 24px; margin: 14px 0; border-radius: 8px; }
  .pair-header { color: #e2e8f0; font-size: 15px; margin-bottom: 6px; font-weight: 600; }
  .pair-sub { color: #94a3b8; font-size: 12px; margin-bottom: 14px; }
  .what-member-sees { color: #fbbf24; font-size: 10px; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px; font-weight: 700; }
  .rationale { color: #f1f5f9; font-size: 15px; line-height: 1.75; font-family: Georgia, serif; padding: 14px 18px; background: #0f172a; border-radius: 6px; border: 1px solid #334155; }
  .audit { margin-top: 14px; padding-top: 14px; border-top: 1px solid #334155; }
  .audit-label { color: #64748b; font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 6px; }
  .audit-row { color: #cbd5e1; font-size: 12px; line-height: 1.8; }
  .audit-row code { background: #020617; padding: 1px 6px; border-radius: 3px; color: #93c5fd; font-size: 11px; }
  .signal { display: inline-block; background: #0f172a; border: 1px solid #334155; padding: 2px 8px; border-radius: 4px; margin: 2px 4px 2px 0; font-size: 11px; color: #a5f3fc; font-family: 'SF Mono', Monaco, monospace; }
  .explainer { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px 20px; margin: 20px 0; font-size: 13px; color: #cbd5e1; line-height: 1.7; }
  .explainer strong { color: #fbbf24; }
  .score-badge { display: inline-block; background: #0f172a; border: 1px solid #334155; padding: 3px 10px; border-radius: 4px; color: #93c5fd; font-size: 11px; font-family: 'SF Mono', Monaco, monospace; }
  .loading { color: #fbbf24; }
  .error { color: #f87171; }
</style>
</head>
<body>
<h1>PeopleRank UC1 · Test Harness</h1>
<div class="sub">Fill in two attendees, hit Run. Pre-filled with the Drew/Paralegal example from the brief.</div>

<div class="row">
  <div class="card">
    <h3>Attendee A</h3>
    <label>ID</label><input id="a_id" value="att_A">
    <label>Age</label><input id="a_age" type="number" value="29">
    <label>Gender</label>
    <select id="a_gender"><option>woman</option><option>man</option><option>non_binary</option></select>
    <label>City</label><input id="a_city" value="Austin">
    <label>Occupation</label><input id="a_occupation" value="paralegal">
    <label>Guest Type</label>
    <select id="a_guestType"><option>storyteller</option><option>investigator</option><option>listener</option></select>
    <label>Why Join</label><textarea id="a_whyJoin">I moved to Austin in January with no friends and I am trying to find people who actually want to talk about things that matter. I have been arriving alone at events and getting quietly good at it.</textarea>
    <label>Social Bravery</label><textarea id="a_socialBravery">Last month I walked into a stranger house party because I recognized a song from the sidewalk and I stayed for two hours.</textarea>
    <label>Passion</label><input id="a_passion" value="Live music and bookstores that smell like rain.">
    <label>Screening Notes</label><textarea id="a_screeningNotes">Recently relocated, actively building community, high openness, low guardedness.</textarea>
  </div>
  <div class="card">
    <h3>Attendee B</h3>
    <label>ID</label><input id="b_id" value="att_B">
    <label>Age</label><input id="b_age" type="number" value="31">
    <label>Gender</label>
    <select id="b_gender"><option>man</option><option>woman</option><option>non_binary</option></select>
    <label>City</label><input id="b_city" value="Austin">
    <label>Occupation</label><input id="b_occupation" value="software engineer">
    <label>Guest Type</label>
    <select id="b_guestType"><option>investigator</option><option>storyteller</option><option>listener</option></select>
    <label>Why Join</label><textarea id="b_whyJoin">I have lived here 24 years and I am still discovering pockets of the city. I want to meet people who are new enough to make me see it fresh again, and curious enough to ask questions I forgot to ask.</textarea>
    <label>Social Bravery</label><textarea id="b_socialBravery">I hosted a supper club for strangers for eighteen months straight and only knew two guests on the first night.</textarea>
    <label>Passion</label><input id="b_passion" value="East side taco trucks, fantasy football drafts, long walks along the greenbelt.">
    <label>Screening Notes</label><textarea id="b_screeningNotes">Austin native, deep local knowledge, understated, genuinely enjoys introducing newcomers to the city.</textarea>
  </div>
</div>

<label>API Key (Bearer)</label>
<input id="apiKey" value="" placeholder="paste your BLIND8_API_KEY here">

<button onclick="runTest()" id="runBtn">Run Match</button>

<div id="result"></div>

<script>
function buildAttendee(prefix) {
  const v = (id) => document.getElementById(prefix + "_" + id).value;
  return {
    id: v("id"),
    age: parseInt(v("age")),
    gender: v("gender"),
    city: v("city"),
    occupation: v("occupation"),
    guestType: v("guestType"),
    guestArchetype: [],
    whyJoin: v("whyJoin"),
    socialBravery: v("socialBravery"),
    passion: v("passion"),
    screeningNotes: v("screeningNotes"),
    hostNotes: [],
    feedback: [],
    eventsAttended: [],
    excludedPartnerIds: []
  };
}

async function runTest() {
  const btn = document.getElementById("runBtn");
  const out = document.getElementById("result");
  const key = document.getElementById("apiKey").value.trim();
  if (!key) { out.innerHTML = '<span class="error">Paste your API key above first.</span>'; return; }

  btn.disabled = true;
  out.innerHTML = '<span class="loading">Running... this takes 10–30 seconds (scoring + rationale generation)</span>';

  const runId = crypto.randomUUID();
  const payload = {
    runId,
    event: {
      id: "evt_test_" + Date.now(),
      type: "community_event",
      name: "Test Event",
      city: "Austin",
      venue: "Test Venue",
      startsAt: new Date(Date.now() + 7200000).toISOString(),
      attendeeCount: 2
    },
    attendees: [buildAttendee("a"), buildAttendee("b")]
  };

  try {
    const res = await fetch("/v1/match/community-event", {
      method: "POST",
      headers: {
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
        "Idempotency-Key": runId
      },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    if (!res.ok) {
      out.innerHTML = '<span class="error">HTTP ' + res.status + '</span>\\n\\n' + JSON.stringify(data, null, 2);
      return;
    }

    let html = '';
    for (const p of (data.pairs || [])) {
      html += '<div class="pair"><div class="pair-header">For ' + p.memberId + ' (about ' + p.partnerId + ', rank ' + p.rank + ')</div>';
      html += '<div class="rationale">' + p.rationale + '</div>';
      html += '<div class="score">score: ' + p.compatibilityScore.toFixed(3) + ' · signals: ' + (p.signals || []).join(", ") + '</div></div>';
    }
    for (const s of (data.skipped || [])) {
      html += '<div class="pair" style="border-left-color:#f87171"><div class="pair-header">Skipped: ' + s.memberId + '</div><div class="rationale">' + s.reason + (s.note ? ' — ' + s.note : '') + '</div></div>';
    }
    if (!html) html = '<span class="error">No pairs and no skips returned.</span>';
    html += '\\n\\n<details><summary style="cursor:pointer;color:#64748b;margin-top:20px">Raw JSON</summary><pre>' + JSON.stringify(data, null, 2) + '</pre></details>';
    out.innerHTML = html;
  } catch (e) {
    out.innerHTML = '<span class="error">Network error: ' + e.message + '</span>';
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""


def test_page() -> HTMLResponse:
    return HTMLResponse(TEST_PAGE_HTML)
