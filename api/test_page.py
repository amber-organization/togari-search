"""Browser-based test harness for the UC1 endpoint."""
from fastapi.responses import HTMLResponse

TEST_PAGE_HTML = """<!DOCTYPE html>
<html>
<head>
<title>PeopleRank UC1 Test</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 24px; background: #0f1a2e; color: #e2e8f0; }
  h1 { color: #60a5fa; font-size: 24px; margin-bottom: 4px; }
  .sub { color: #94a3b8; font-size: 13px; margin-bottom: 24px; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 16px; }
  .card { background: #1e293b; border-radius: 10px; padding: 20px; border: 1px solid #334155; }
  .card h3 { color: #93c5fd; font-size: 14px; margin: 0 0 14px; letter-spacing: 1px; text-transform: uppercase; }
  label { display: block; color: #cbd5e1; font-size: 12px; margin: 10px 0 4px; font-weight: 600; }
  input, textarea, select { width: 100%; padding: 8px 10px; background: #0f172a; border: 1px solid #334155; color: #e2e8f0; border-radius: 6px; font-size: 13px; font-family: inherit; box-sizing: border-box; }
  textarea { min-height: 60px; resize: vertical; }
  button { background: #2563eb; color: white; border: none; padding: 12px 28px; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; margin-top: 20px; }
  button:hover { background: #1d4ed8; }
  button:disabled { background: #475569; cursor: not-allowed; }

  .explainer { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 20px 24px; margin-top: 24px; font-size: 14px; color: #cbd5e1; line-height: 1.7; }
  .explainer strong { color: #fbbf24; }

  .match { background: #1e293b; border-left: 4px solid #60a5fa; border-radius: 8px; padding: 24px; margin: 20px 0; }
  .match-title { color: #fbbf24; font-size: 16px; font-weight: 700; margin-bottom: 4px; }
  .match-sub { color: #94a3b8; font-size: 13px; margin-bottom: 20px; }

  .section-label { color: #fbbf24; font-size: 10px; letter-spacing: 2px; text-transform: uppercase; font-weight: 700; margin-bottom: 10px; }
  .rationale-box { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 20px 24px; color: #f1f5f9; font-family: Georgia, serif; font-size: 16px; line-height: 1.8; }

  .audit { margin-top: 24px; padding-top: 20px; border-top: 1px solid #334155; }
  .audit .section-label { color: #64748b; }
  .metric-row { display: flex; align-items: baseline; gap: 12px; margin: 10px 0; font-size: 13px; }
  .metric-label { color: #94a3b8; min-width: 200px; }
  .metric-value { color: #93c5fd; font-family: 'SF Mono', Monaco, monospace; font-weight: 600; }
  .metric-note { color: #64748b; font-size: 12px; }

  .signal-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 12px; }
  .signal { background: #0f172a; border: 1px solid #334155; padding: 10px 14px; border-radius: 6px; }
  .signal-name { color: #a5f3fc; font-size: 12px; font-weight: 600; font-family: 'SF Mono', Monaco, monospace; margin-bottom: 4px; }
  .signal-value { color: #e2e8f0; font-size: 18px; font-weight: 700; font-family: 'SF Mono', Monaco, monospace; }
  .signal-desc { color: #64748b; font-size: 11px; margin-top: 2px; }

  .skipped { border-left-color: #f87171; }
  .error { color: #f87171; }
  .loading { color: #fbbf24; padding: 20px; }

  details { margin-top: 20px; }
  summary { cursor: pointer; color: #64748b; font-size: 12px; }
  details pre { margin-top: 12px; background: #020617; padding: 16px; border-radius: 6px; color: #a5f3fc; font-size: 11px; overflow-x: auto; }
</style>
</head>
<body>
<h1>PeopleRank UC1 · Test Harness</h1>
<div class="sub">Fill in two attendees, paste your API key, hit Run.</div>

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
<input id="apiKey" placeholder="paste your BLIND8_API_KEY here">

<button onclick="runTest()" id="runBtn">Run Match</button>

<div id="result"></div>

<script>
const SIGNAL_DESCRIPTIONS = {
  "identity": "Similarity in who they are (city, job, core identity text)",
  "personality": "Similarity in temperament and social bravery",
  "experience": "Similarity in life experiences and meeting context",
  "interest": "Similarity in passions and what they are drawn to",
  "age_proximity": "How close in age they are (1.00 = same age, falls off over 40 years)",
  "same_city": "1.00 if both in the same city, 0 otherwise",
  "complementary_type": "How well their conversation archetypes pair (storyteller + listener = 1.00, two of same type lower)",
  "shared_event_count": "How many past events they have both attended",
  "connection_exists": "Whether they already know each other",
  "referral_chain": "Whether one referred the other",
  "qualification_proximity": "Similar screening depth",
  "readiness_harmony": "Similar social readiness scores"
};

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

function renderSignals(signals) {
  let html = '<div class="signal-grid">';
  for (const s of signals) {
    const [name, val] = s.split(':');
    const desc = SIGNAL_DESCRIPTIONS[name] || '';
    html += '<div class="signal">';
    html += '<div class="signal-name">' + name + '</div>';
    html += '<div class="signal-value">' + val + '</div>';
    if (desc) html += '<div class="signal-desc">' + desc + '</div>';
    html += '</div>';
  }
  html += '</div>';
  return html;
}

async function runTest() {
  const btn = document.getElementById("runBtn");
  const out = document.getElementById("result");
  const key = document.getElementById("apiKey").value.trim();
  if (!key) { out.innerHTML = '<div class="error">Paste your API key above first.</div>'; return; }

  btn.disabled = true;
  out.innerHTML = '<div class="loading">Running... takes 10–30 seconds.</div>';

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
      out.innerHTML = '<div class="error">HTTP ' + res.status + '</div><pre>' + JSON.stringify(data, null, 2) + '</pre>';
      return;
    }

    const pairs = data.pairs || [];
    const skipped = data.skipped || [];

    let html = '';

    if (pairs.length) {
      html += '<div class="explainer">';
      html += '<strong>How to read this:</strong> Each card below is one match suggestion. A mutual match produces TWO cards — one written for each side, because the rationale addresses each person directly. ';
      html += 'The yellow box is the ONLY thing your members actually see. Everything in the Internal Audit section is metadata you can store but never render to members.';
      html += '</div>';
    }

    for (const p of pairs) {
      html += '<div class="match">';
      html += '<div class="match-title">' + p.memberId + ' → will be shown ' + p.partnerId + '</div>';
      html += '<div class="match-sub">Rank ' + p.rank + ' match · each member receives up to 2 rationales per event</div>';

      html += '<div class="section-label">What the member reads</div>';
      html += '<div class="rationale-box">' + p.rationale + '</div>';

      html += '<div class="audit">';
      html += '<div class="section-label">Internal audit (not rendered)</div>';

      html += '<div class="metric-row">';
      html += '<div class="metric-label">Compatibility score:</div>';
      html += '<div class="metric-value">' + p.compatibilityScore.toFixed(4) + '</div>';
      html += '<div class="metric-note">0–1 range · store for analysis, never show to members</div>';
      html += '</div>';

      html += '<div class="metric-row" style="display:block">';
      html += '<div class="metric-label" style="margin-bottom:8px">Signals that produced the match:</div>';
      html += renderSignals(p.signals || []);
      html += '</div>';

      html += '</div></div>';
    }

    for (const s of skipped) {
      html += '<div class="match skipped">';
      html += '<div class="match-title" style="color:#f87171">' + s.memberId + ' was skipped</div>';
      html += '<div class="match-sub">Reason: ' + s.reason + (s.note ? ' · ' + s.note : '') + '</div>';
      html += '<div style="color:#cbd5e1;margin-top:14px;font-size:13px;line-height:1.7">';
      html += 'This member will not receive a suggestion for this event. Possible reasons: not enough screening text to reason over, all potential partners were on the exclusion list, or the rationale writer could not produce prose that passed voice validation after 5 tries.';
      html += '</div></div>';
    }

    if (!html) html = '<div class="error">No pairs and no skips returned.</div>';

    html += '<details><summary>Show raw JSON response</summary><pre>' + JSON.stringify(data, null, 2) + '</pre></details>';
    out.innerHTML = html;
  } catch (e) {
    out.innerHTML = '<div class="error">Network error: ' + e.message + '</div>';
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
