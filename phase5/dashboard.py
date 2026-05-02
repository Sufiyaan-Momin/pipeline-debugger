"""
phase5/dashboard.py
─────────────────────────────────────────────────────────────────
PHASE 5 — Pipeline Monitor Dashboard

A Streamlit dashboard that visualizes:
  - Live Airflow DAG status
  - AI-generated incident diagnoses from Phase 4
  - Knowledge base stats from Phase 3
  - Pipeline health over time

INSTALL:
  pip install streamlit plotly watchdog

RUN:
  cd D:\pipeline-debugger
  streamlit run phase5/dashboard.py

Then open http://localhost:8501 in your browser.
─────────────────────────────────────────────────────────────────
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Page config (must be first Streamlit call) ────────────────────
st.set_page_config(
    page_title="Pipeline Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — industrial/terminal aesthetic ────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;800&display=swap');

  /* Base */
  :root {
    --bg:        #0a0c0f;
    --bg2:       #111418;
    --bg3:       #181d24;
    --border:    #1e2530;
    --text:      #c8d0dc;
    --text-dim:  #5a6578;
    --accent:    #00d4ff;
    --green:     #00ff88;
    --red:       #ff3b5c;
    --yellow:    #ffb800;
    --purple:    #9d6fff;
    --mono:      'JetBrains Mono', monospace;
    --display:   'Syne', sans-serif;
  }

  html, body, [class*="css"] {
    font-family: var(--mono);
    background-color: var(--bg);
    color: var(--text);
  }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 1.5rem 2rem 2rem; max-width: 100%; }

  /* Header */
  .dash-header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
    margin-bottom: 24px;
  }
  .dash-title {
    font-family: var(--display);
    font-weight: 800;
    font-size: 1.6rem;
    color: #fff;
    letter-spacing: -0.02em;
    margin: 0;
  }
  .dash-subtitle {
    font-size: 0.7rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.12em;
  }
  .live-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--green);
    display: inline-block;
    animation: pulse 1.5s infinite;
    margin-left: auto;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0,255,136,0.4); }
    50% { opacity: 0.8; box-shadow: 0 0 0 6px rgba(0,255,136,0); }
  }

  /* Metric cards */
  .metric-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 24px; }
  .metric-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px 20px;
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 2px;
  }
  .metric-card.green::before  { background: var(--green); }
  .metric-card.red::before    { background: var(--red); }
  .metric-card.yellow::before { background: var(--yellow); }
  .metric-card.blue::before   { background: var(--accent); }
  .metric-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-dim);
    margin-bottom: 8px;
  }
  .metric-value {
    font-family: var(--display);
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1;
    color: #fff;
  }
  .metric-sub { font-size: 0.65rem; color: var(--text-dim); margin-top: 6px; }

  /* DAG status table */
  .dag-row {
    display: grid;
    grid-template-columns: 24px 1fr 100px 120px 80px;
    gap: 12px;
    align-items: center;
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    font-size: 0.8rem;
  }
  .dag-row:hover { background: var(--bg3); }
  .dag-row.header {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-dim);
    background: var(--bg2);
    border-radius: 4px 4px 0 0;
    border-bottom: 1px solid var(--border);
  }
  .status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
  }
  .status-success { background: var(--green); }
  .status-failed  { background: var(--red); }
  .status-running { background: var(--yellow); animation: pulse 1s infinite; }
  .status-unknown { background: var(--text-dim); }
  .dag-id { color: var(--accent); font-weight: 600; }
  .tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
  }
  .tag-failure { background: rgba(255,59,92,0.15); color: var(--red); border: 1px solid rgba(255,59,92,0.3); }
  .tag-healthy { background: rgba(0,255,136,0.1); color: var(--green); border: 1px solid rgba(0,255,136,0.2); }
  .tag-timeout { background: rgba(255,184,0,0.1); color: var(--yellow); border: 1px solid rgba(255,184,0,0.2); }

  /* Incident cards */
  .incident-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--red);
    border-radius: 0 6px 6px 0;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 0.8rem;
  }
  .incident-card.high   { border-left-color: var(--red); }
  .incident-card.medium { border-left-color: var(--yellow); }
  .incident-card.low    { border-left-color: var(--text-dim); }
  .incident-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
  }
  .incident-dag { font-weight: 700; color: #fff; font-size: 0.85rem; }
  .incident-task { color: var(--text-dim); }
  .confidence-badge {
    margin-left: auto;
    padding: 2px 10px;
    border-radius: 3px;
    font-size: 0.6rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }
  .conf-high   { background: rgba(0,255,136,0.15); color: var(--green); }
  .conf-medium { background: rgba(255,184,0,0.15); color: var(--yellow); }
  .conf-low    { background: rgba(255,59,92,0.15); color: var(--red); }
  .incident-field { margin-bottom: 6px; }
  .field-label { color: var(--text-dim); font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; }
  .field-value { color: var(--text); margin-top: 2px; line-height: 1.5; }
  .fix-box {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 10px 14px;
    margin-top: 10px;
    font-size: 0.75rem;
    color: var(--green);
  }

  /* Section headers */
  .section-header {
    font-family: var(--display);
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 24px 0 16px;
  }

  /* KB stats */
  .kb-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 8px; }
  .kb-item {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 12px 16px;
    text-align: center;
  }
  .kb-count { font-size: 1.4rem; font-weight: 700; color: var(--accent); font-family: var(--display); }
  .kb-label { font-size: 0.6rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--bg2);
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] .stButton button {
    background: var(--bg3);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    font-family: var(--mono);
    font-size: 0.75rem;
    width: 100%;
  }
  [data-testid="stSidebar"] .stButton button:hover {
    border-color: var(--accent);
    color: var(--accent);
  }

  /* Plotly chart background */
  .js-plotly-plot { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────
AIRFLOW_BASE_URL = "http://localhost:8080/api/v1"
AIRFLOW_AUTH     = ("admin", "admin")
REPORTS_DIR      = Path("phase4/reports")
CHROMA_PATH      = "phase3/data/chroma_db"


# ── Data fetchers ─────────────────────────────────────────────────

@st.cache_data(ttl=15)
def fetch_dag_status():
    """Fetch all DAGs and their latest run status from Airflow."""
    try:
        dags_resp = requests.get(f"{AIRFLOW_BASE_URL}/dags", auth=AIRFLOW_AUTH, timeout=5)
        dags_resp.raise_for_status()
        dags = dags_resp.json().get("dags", [])

        results = []
        for dag in dags:
            dag_id = dag["dag_id"]
            try:
                runs_resp = requests.get(
                    f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns",
                    params={"limit": 1, "order_by": "-start_date"},
                    auth=AIRFLOW_AUTH, timeout=5,
                )
                runs = runs_resp.json().get("dag_runs", [])
                last_run = runs[0] if runs else {}
                results.append({
                    "dag_id":     dag_id,
                    "is_active":  dag.get("is_active", False),
                    "tags":       [t["name"] for t in dag.get("tags", [])],
                    "last_state": last_run.get("state", "no runs"),
                    "last_run":   last_run.get("start_date", "—"),
                    "schedule":   dag.get("schedule_interval", "—"),
                })
            except Exception:
                results.append({
                    "dag_id": dag_id, "is_active": False,
                    "tags": [], "last_state": "unknown",
                    "last_run": "—", "schedule": "—",
                })
        return results, None
    except requests.RequestException as e:
        return [], str(e)


def load_reports() -> list[dict]:
    """Load all diagnosis reports from phase4/reports/."""
    if not REPORTS_DIR.exists():
        return []
    reports = []
    for path in sorted(REPORTS_DIR.glob("*.json"), reverse=True):
        try:
            with open(path) as f:
                report = json.load(f)
                report["_filename"] = path.name
                reports.append(report)
        except Exception:
            pass
    return reports


def load_kb_stats() -> dict:
    """Load ChromaDB collection stats."""
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collections = ["incidents", "runbooks", "schema_docs", "sql_models"]
        stats = {}
        for name in collections:
            try:
                col = client.get_collection(name)
                stats[name] = col.count()
            except Exception:
                stats[name] = 0
        return stats
    except Exception:
        return {}


# ── Header ────────────────────────────────────────────────────────

st.markdown("""
<div class="dash-header">
  <div>
    <div class="dash-title">⚡ Pipeline Monitor</div>
    <div class="dash-subtitle">Self-Healing Data Pipeline Debugger</div>
  </div>
  <span class="live-dot" title="Live"></span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Controls")

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    auto_refresh = st.toggle("Auto-refresh (30s)", value=False)

    st.markdown("---")
    st.markdown("### Run Agent")
    st.markdown('<div style="font-size:0.7rem;color:#5a6578">Diagnose new failures with AI</div>', unsafe_allow_html=True)

    if st.button("▶ Run Agent (once)"):
        with st.spinner("Running agent..."):
            result = os.popen("python phase4/agent.py --once 2>&1").read()
        st.code(result, language="text")

    st.markdown("---")
    st.markdown("### Stack")
    st.markdown("""
<div style="font-size:0.7rem;color:#5a6578;line-height:2">
  🟢 Airflow 2.8<br>
  🟢 Postgres 15<br>
  🟢 ChromaDB<br>
  🟢 Groq LLaMA 3.3<br>
  🟢 LangGraph<br>
  🟢 dbt Core
</div>
""", unsafe_allow_html=True)

# ── Fetch data ────────────────────────────────────────────────────

dags, airflow_error = fetch_dag_status()
reports = load_reports()
kb_stats = load_kb_stats()

# ── Metric cards ──────────────────────────────────────────────────

total_dags    = len(dags)
failed_dags   = sum(1 for d in dags if d["last_state"] == "failed")
success_dags  = sum(1 for d in dags if d["last_state"] == "success")
total_reports = len(reports)
high_conf     = sum(1 for r in reports if r.get("diagnosis", {}).get("confidence") == "high")

st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card {'red' if failed_dags > 0 else 'green'}">
    <div class="metric-label">Active Failures</div>
    <div class="metric-value" style="color:{'#ff3b5c' if failed_dags > 0 else '#00ff88'}">{failed_dags}</div>
    <div class="metric-sub">{total_dags} DAGs total</div>
  </div>
  <div class="metric-card green">
    <div class="metric-label">Successful Runs</div>
    <div class="metric-value" style="color:#00ff88">{success_dags}</div>
    <div class="metric-sub">Last 24h</div>
  </div>
  <div class="metric-card blue">
    <div class="metric-label">AI Diagnoses</div>
    <div class="metric-value" style="color:#00d4ff">{total_reports}</div>
    <div class="metric-sub">{high_conf} high confidence</div>
  </div>
  <div class="metric-card yellow">
    <div class="metric-label">KB Chunks</div>
    <div class="metric-value" style="color:#ffb800">{sum(kb_stats.values()) if kb_stats else '—'}</div>
    <div class="metric-sub">Vector embeddings</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────

col_left, col_right = st.columns([3, 2])

with col_left:

    # ── DAG Status Table ─────────────────────────────────────────
    st.markdown('<div class="section-header">DAG Status</div>', unsafe_allow_html=True)

    if airflow_error:
        st.error(f"Cannot connect to Airflow: {airflow_error}")
        st.info("Make sure Airflow is running: `docker compose up -d`")
    elif not dags:
        st.warning("No DAGs found. Check Airflow connection.")
    else:
        st.markdown("""
        <div class="dag-row header">
          <div></div>
          <div>DAG ID</div>
          <div>Last State</div>
          <div>Last Run</div>
          <div>Schedule</div>
        </div>
        """, unsafe_allow_html=True)

        for dag in dags:
            state = dag["last_state"]
            dot_class = {
                "success": "status-success",
                "failed":  "status-failed",
                "running": "status-running",
            }.get(state, "status-unknown")

            # Format timestamp
            last_run = dag["last_run"]
            if last_run and last_run != "—":
                try:
                    dt = datetime.fromisoformat(last_run.replace("Z", "+00:00"))
                    last_run = dt.strftime("%m/%d %H:%M")
                except Exception:
                    last_run = last_run[:16]

            # Tag for failure type
            tags = dag.get("tags", [])
            tag_html = ""
            if "failure" in tags:
                if "timeout" in tags:
                    tag_html = '<span class="tag tag-timeout">timeout</span>'
                else:
                    tag_html = '<span class="tag tag-failure">failure</span>'
            elif "healthy" in tags:
                tag_html = '<span class="tag tag-healthy">healthy</span>'

            st.markdown(f"""
            <div class="dag-row">
              <span class="status-dot {dot_class}"></span>
              <span class="dag-id">{dag['dag_id']}</span>
              <span style="color:{'#ff3b5c' if state=='failed' else '#00ff88' if state=='success' else '#ffb800'}">{state}</span>
              <span style="color:#5a6578;font-size:0.75rem">{last_run}</span>
              <span>{tag_html}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Pipeline health chart ────────────────────────────────────
    if reports:
        st.markdown('<div class="section-header">Failure Distribution</div>', unsafe_allow_html=True)

        error_types = [r.get("diagnosis", {}).get("error_type", "unknown") for r in reports]
        type_counts = {}
        for t in error_types:
            type_counts[t] = type_counts.get(t, 0) + 1

        colors = {
            "schema_mismatch":    "#00d4ff",
            "data_quality_failure": "#ff3b5c",
            "task_timeout":       "#ffb800",
            "connection_failure": "#9d6fff",
            "unknown":            "#5a6578",
        }

        fig = go.Figure(go.Bar(
            x=list(type_counts.keys()),
            y=list(type_counts.values()),
            marker_color=[colors.get(k, "#5a6578") for k in type_counts.keys()],
            marker_line_width=0,
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono", color="#5a6578", size=11),
            margin=dict(l=0, r=0, t=10, b=0),
            height=200,
            xaxis=dict(gridcolor="#1e2530", tickfont=dict(color="#5a6578")),
            yaxis=dict(gridcolor="#1e2530", tickfont=dict(color="#5a6578")),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

with col_right:

    # ── Knowledge base stats ─────────────────────────────────────
    st.markdown('<div class="section-header">Knowledge Base</div>', unsafe_allow_html=True)

    if kb_stats:
        labels = {"incidents": "Incidents", "runbooks": "Runbooks",
                  "schema_docs": "Schema", "sql_models": "SQL Models"}
        cols = st.columns(4)
        for i, (key, count) in enumerate(kb_stats.items()):
            with cols[i]:
                st.markdown(f"""
                <div class="kb-item">
                  <div class="kb-count">{count}</div>
                  <div class="kb-label">{labels.get(key, key)}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Knowledge base not loaded. Run ingest.py first.")

    # ── Confidence donut chart ───────────────────────────────────
    if reports:
        conf_counts = {"high": 0, "medium": 0, "low": 0}
        for r in reports:
            c = r.get("diagnosis", {}).get("confidence", "low")
            conf_counts[c] = conf_counts.get(c, 0) + 1

        fig2 = go.Figure(go.Pie(
            labels=list(conf_counts.keys()),
            values=list(conf_counts.values()),
            hole=0.65,
            marker_colors=["#00ff88", "#ffb800", "#ff3b5c"],
            textfont=dict(family="JetBrains Mono", size=10),
            showlegend=True,
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono", color="#5a6578", size=10),
            margin=dict(l=0, r=0, t=10, b=0),
            height=180,
            legend=dict(
                font=dict(color="#5a6578", size=10),
                bgcolor="rgba(0,0,0,0)",
                orientation="h",
                yanchor="bottom", y=-0.2,
            ),
            annotations=[dict(
                text=f"{total_reports}<br><span style='font-size:10px'>total</span>",
                x=0.5, y=0.5, font_size=18,
                font_color="#ffffff",
                showarrow=False,
            )],
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── Incident Reports ──────────────────────────────────────────────

st.markdown('<div class="section-header">AI Diagnosis Reports</div>', unsafe_allow_html=True)

if not reports:
    st.markdown("""
    <div style="text-align:center;padding:40px;color:#5a6578;font-size:0.8rem">
      No reports yet.<br>
      Run the agent: <code>python phase4/agent.py --once</code>
    </div>
    """, unsafe_allow_html=True)
else:
    # Filter controls
    fcol1, fcol2, fcol3 = st.columns([2, 2, 1])
    with fcol1:
        filter_type = st.selectbox(
            "Filter by error type",
            ["All"] + list({r.get("diagnosis", {}).get("error_type", "unknown") for r in reports}),
            label_visibility="collapsed",
        )
    with fcol2:
        filter_conf = st.selectbox(
            "Filter by confidence",
            ["All confidence", "high", "medium", "low"],
            label_visibility="collapsed",
        )
    with fcol3:
        st.markdown(f'<div style="font-size:0.7rem;color:#5a6578;padding-top:8px">{len(reports)} reports</div>', unsafe_allow_html=True)

    # Apply filters
    filtered = reports
    if filter_type != "All":
        filtered = [r for r in filtered if r.get("diagnosis", {}).get("error_type") == filter_type]
    if filter_conf != "All confidence":
        filtered = [r for r in filtered if r.get("diagnosis", {}).get("confidence") == filter_conf]

    for report in filtered[:10]:  # show max 10
        d = report.get("diagnosis", {})
        confidence = d.get("confidence", "low")
        error_type = d.get("error_type", "unknown")
        dag_id     = report.get("dag_id", "unknown")
        task_id    = report.get("task_id", "unknown")
        generated  = report.get("report_generated_at", "")[:16].replace("T", " ")

        conf_class = f"conf-{confidence}"
        card_class = confidence

        st.markdown(f"""
        <div class="incident-card {card_class}">
          <div class="incident-header">
            <span class="incident-dag">{dag_id}</span>
            <span class="incident-task">› {task_id}</span>
            <span class="confidence-badge {conf_class}">{confidence} confidence</span>
          </div>

          <div class="incident-field">
            <div class="field-label">Error Type</div>
            <div class="field-value" style="color:#00d4ff">{error_type}</div>
          </div>

          <div class="incident-field">
            <div class="field-label">Root Cause</div>
            <div class="field-value">{d.get('root_cause', '—')}</div>
          </div>

          <div class="fix-box">
            <span style="color:#5a6578;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em">Recommended Fix › </span>
            {d.get('recommended_fix', '—')}
          </div>

          <div style="display:flex;gap:16px;margin-top:10px;font-size:0.65rem;color:#5a6578">
            <span>📄 {d.get('relevant_runbook', 'none')}</span>
            <span>🕐 {generated}</span>
            <span>{'🚨 Immediate action' if d.get('needs_immediate_action') else 'ℹ️ Monitor'}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Expandable log excerpt
        with st.expander(f"View log excerpt — {task_id}"):
            st.code(report.get("log_excerpt", "No log available")[-2000:], language="text")

# ── Auto-refresh ──────────────────────────────────────────────────

if auto_refresh:
    time.sleep(30)
    st.cache_data.clear()
    st.rerun()