"""
Phase 9: ACMGS Dashboard

Modern real-time control center for the Autonomous Carbon-Aware
Manufacturing Genome System.

Run:
    cd C:\\Users\\HP\\ACMGS
    streamlit run src/dashboard/app.py
"""

import os
import sys
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─── Path setup (works regardless of cwd) ────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import DB_PATH, CARBON_HIGH_THRESHOLD, CARBON_LOW_THRESHOLD
from src.carbon_scheduler import classify_carbon_zone, get_recommendation

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="ACMGS | Control Center",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "ACMGS v9.0 — Autonomous Carbon-Aware Manufacturing Genome System"},
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Global ─────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1426 100%) !important;
    font-family: 'Inter', sans-serif !important;
}
.main .block-container {
    padding-top: 0.8rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* ── Header banner ───────────────────────────────────────── */
.acmgs-header {
    background: linear-gradient(135deg,
        rgba(0,212,255,0.1) 0%,
        rgba(0,255,136,0.06) 50%,
        rgba(0,80,200,0.08) 100%
    );
    border: 1px solid rgba(0,212,255,0.22);
    border-radius: 16px;
    padding: 20px 32px;
    margin-bottom: 16px;
}
.acmgs-header h1 {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d4ff 0%, #00ff88 70%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.02em;
}
.acmgs-header p {
    color: rgba(255,255,255,0.5);
    font-size: 0.875rem;
    margin: 6px 0 10px 0;
}
.hbadge {
    display: inline-block;
    background: rgba(0,212,255,0.12);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.7rem;
    font-weight: 600;
    color: #00d4ff;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-right: 6px;
}
.hbadge-green  { background: rgba(0,255,136,0.12); border-color: rgba(0,255,136,0.3); color: #00ff88; }
.hbadge-yellow { background: rgba(255,214,0,0.1);  border-color: rgba(255,214,0,0.3);  color: #ffd600; }
.hbadge-red    { background: rgba(255,75,75,0.12); border-color: rgba(255,75,75,0.3);  color: #ff6b6b; }

/* ── Metrics ─────────────────────────────────────────────── */
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #00d4ff !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    color: rgba(255,255,255,0.45) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.038) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(0,212,255,0.35) !important;
}

/* ── Tabs ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 5px;
    gap: 3px;
    width: 100%;
    box-sizing: border-box;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: rgba(255,255,255,0.5);
    font-weight: 500;
    font-size: 0.83rem;
    padding: 7px 16px;
    transition: all 0.2s;
    flex: 1 1 0;
    justify-content: center;
    text-align: center;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,212,255,0.16) !important;
    color: #00d4ff !important;
    font-weight: 600 !important;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1426 0%, #090d1a 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

/* ── Section label ───────────────────────────────────────── */
.slabel {
    font-size: 0.72rem;
    font-weight: 600;
    color: rgba(255,255,255,0.3);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding-bottom: 5px;
    margin: 18px 0 10px 0;
}

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.22); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,212,255,0.4); }

/* ── Inputs / Sliders ────────────────────────────────────── */
.stSlider > div > div > div { background: rgba(0,212,255,0.18) !important; }
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}

/* ── Dataframe ───────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* ── Expander ────────────────────────────────────────────── */
details > summary {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 8px !important;
    color: rgba(255,255,255,0.7) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────────
GENOME_LABELS = (
    ["Temp", "Pressure", "Speed", "FeedRate", "Humidity"]
    + ["Density", "Hardness", "Grade"]
    + [f"EdNA{i:02d}" for i in range(16)]
    + ["CarbonInt"]
)

ZONE_COLORS = {"LOW": "#00ff88", "MEDIUM": "#ffd600", "HIGH": "#ff4b4b"}
ZONE_BG     = {"LOW": "rgba(0,255,136,0.08)", "MEDIUM": "rgba(255,214,0,0.08)",  "HIGH": "rgba(255,75,75,0.08)"}
ZONE_BORDER = {"LOW": "rgba(0,255,136,0.3)",  "MEDIUM": "rgba(255,214,0,0.3)",   "HIGH": "rgba(255,75,75,0.3)"}
ZONE_EMOJI  = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
ZONE_TITLE  = {"LOW": "CLEAN GRID", "MEDIUM": "MIXED GRID", "HIGH": "DIRTY GRID"}
ZONE_DESC   = {
    "LOW":    "Renewable energy dominant — Maximize production output at full capacity.",
    "MEDIUM": "Balanced energy mix — Optimize for efficiency and sustainability.",
    "HIGH":   "Heavy fossil fuel load — Activate conservation mode, minimize energy.",
}

TABLE_ICONS = {
    "batches":           "📦",   # product batches
    "energy_embeddings": "🔋",   # energy vectors / stored embeddings
    "genome_vectors":    "🧬",   # DNA genome
    "predictions":       "🔮",   # forecasting / ML predictions
    "pareto_solutions":  "⚖️",   # multi-objective trade-off balance
    "carbon_schedules":  "🌍",   # carbon / climate scheduling
    "pipeline_runs":     "🚀",   # pipeline execution
}

_CYAN    = "#00d4ff"
_GREEN   = "#00ff88"
_YELLOW  = "#ffd600"
_RED     = "#ff4b4b"
_PURPLE  = "#a855f7"
_ORANGE  = "#f97316"

CARBON_24H = [120, 100, 85, 75, 70, 65, 60, 55, 50, 55, 65, 80,
              100, 130, 160, 200, 260, 320, 420, 500, 460, 380, 280, 180]


# ─── Data loading (all cached) ────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _load_batches() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM batches ORDER BY batch_id", conn)
    conn.close()
    df["zone"] = df["carbon_intensity"].apply(classify_carbon_zone)
    return df


@st.cache_data(ttl=300)
def _load_pareto() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM pareto_solutions ORDER BY pred_yield DESC", conn)
    conn.close()
    return df


@st.cache_data(ttl=300)
def _load_predictions() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    return df


@st.cache_data(ttl=300)
def _load_schedules() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM carbon_schedules ORDER BY id", conn)
    conn.close()
    return df


@st.cache_data(ttl=300)
def _load_pipeline_runs() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT 50", conn)
    conn.close()
    return df


@st.cache_data(ttl=300)
def _load_db_summary() -> dict:
    conn = sqlite3.connect(DB_PATH)
    tables = ["batches", "energy_embeddings", "genome_vectors", "predictions",
              "pareto_solutions", "carbon_schedules", "pipeline_runs"]
    summary = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    summary["db_size_mb"] = round(os.path.getsize(DB_PATH) / 1_048_576, 2)
    conn.close()
    return summary


@st.cache_data(ttl=600)
def _load_genomes(n: int = 80) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT batch_id, genome FROM genome_vectors LIMIT {n}", conn)
    conn.close()
    return df


# ─── Chart helpers ────────────────────────────────────────────────────────────
def dark_layout(fig: go.Figure, height: int = None, margin: dict = None) -> go.Figure:
    """Apply the dashboard dark theme to any Plotly figure."""
    kw: dict = {}
    if height:
        kw["height"] = height
    if margin:
        kw["margin"] = margin
    else:
        kw["margin"] = dict(l=16, r=16, t=44, b=16)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="rgba(255,255,255,0.75)", family="Inter, sans-serif"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.07)",
            zerolinecolor="rgba(255,255,255,0.12)",
            linecolor="rgba(255,255,255,0.08)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.07)",
            zerolinecolor="rgba(255,255,255,0.12)",
            linecolor="rgba(255,255,255,0.08)",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.35)",
            bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1,
        ),
        title=dict(font=dict(size=13, color="rgba(255,255,255,0.7)")),
        **kw,
    )
    return fig


def make_gauge(val: float, zone: str) -> go.Figure:
    """Build a Plotly radial gauge for carbon intensity."""
    col = ZONE_COLORS[zone]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        title={
            "text": (
                "Grid Carbon Intensity<br>"
                "<span style='font-size:0.8em;color:rgba(255,255,255,0.45)'>gCO₂ / kWh</span>"
            ),
            "font": {"size": 14, "color": "rgba(255,255,255,0.6)"},
        },
        number={
            "font": {"size": 54, "color": col, "family": "JetBrains Mono, monospace"},
            "suffix": "",
        },
        gauge={
            "axis": {
                "range": [0, 600],
                "tickwidth": 1,
                "tickcolor": "rgba(255,255,255,0.2)",
                "tickfont": {"color": "rgba(255,255,255,0.35)", "size": 9},
                "nticks": 7,
            },
            "bar": {"color": col, "thickness": 0.2},
            "bgcolor": "rgba(255,255,255,0.02)",
            "borderwidth": 1,
            "bordercolor": "rgba(255,255,255,0.08)",
            "steps": [
                {"range": [0,   150], "color": "rgba(0,255,136,0.12)"},
                {"range": [150, 400], "color": "rgba(255,214,0,0.10)"},
                {"range": [400, 600], "color": "rgba(255,75,75,0.13)"},
            ],
            "threshold": {
                "line": {"color": col, "width": 3},
                "thickness": 0.82,
                "value": val,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "rgba(255,255,255,0.65)", "family": "Inter"},
        height=300,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig


# ─── Load all data once ───────────────────────────────────────────────────────
df_batches   = _load_batches()
df_pareto    = _load_pareto()
df_preds     = _load_predictions()
df_schedules = _load_schedules()
df_runs      = _load_pipeline_runs()
db_summary   = _load_db_summary()
df_genomes   = _load_genomes(80)

# Merge batches + predictions for combined analytics
df_merged = df_batches.merge(df_preds, on="batch_id", how="left")


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="text-align:center;padding:18px 0 14px 0;
            border-bottom:1px solid rgba(255,255,255,0.08);
            margin-bottom:18px;">
  <svg width="72" height="72" viewBox="0 0 68 68" fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="lg1" x1="0" y1="0" x2="68" y2="68" gradientUnits="userSpaceOnUse">
        <stop stop-color="#00d4ff"/><stop offset="1" stop-color="#00ff88"/>
      </linearGradient>
      <filter id="fw" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="2" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    </defs>
    <!-- Hexagon = Carbon (C) -->
    <polygon points="34,3 62,19 62,49 34,65 6,49 6,19"
             stroke="url(#lg1)" stroke-width="1.5" fill="rgba(0,212,255,0.04)"/>
    <!-- Vertex dots = Manufacturing (M) gear nodes -->
    <circle cx="34" cy="3"  r="2.5" fill="#00d4ff" filter="url(#fw)"/>
    <circle cx="62" cy="19" r="2.5" fill="#00d4ff" filter="url(#fw)"/>
    <circle cx="62" cy="49" r="2.5" fill="#00ff88" filter="url(#fw)"/>
    <circle cx="34" cy="65" r="2.5" fill="#00ff88" filter="url(#fw)"/>
    <circle cx="6"  cy="49" r="2.5" fill="#00ff88" filter="url(#fw)"/>
    <circle cx="6"  cy="19" r="2.5" fill="#00d4ff" filter="url(#fw)"/>
    <!-- DNA strands = Genome (G) -->
    <path d="M24,13 C22,21 26,25 24,34 C22,43 26,47 24,55"
          stroke="#00d4ff" stroke-width="2" fill="none"/>
    <path d="M44,13 C46,21 42,25 44,34 C46,43 42,47 44,55"
          stroke="#00ff88" stroke-width="2" fill="none"/>
    <!-- DNA rungs -->
    <line x1="24" y1="19" x2="44" y2="19" stroke="rgba(255,255,255,0.18)" stroke-width="1.2"/>
    <line x1="24" y1="27" x2="44" y2="27" stroke="rgba(255,255,255,0.18)" stroke-width="1.2"/>
    <line x1="24" y1="34" x2="44" y2="34" stroke="rgba(255,255,255,0.30)" stroke-width="1.5"/>
    <line x1="24" y1="41" x2="44" y2="41" stroke="rgba(255,255,255,0.18)" stroke-width="1.2"/>
    <line x1="24" y1="49" x2="44" y2="49" stroke="rgba(255,255,255,0.18)" stroke-width="1.2"/>
    <!-- Orbital ellipse = System (S) integration layer -->
    <ellipse cx="34" cy="34" rx="12" ry="6"
             stroke="rgba(0,212,255,0.32)" stroke-width="1" fill="none"
             transform="rotate(-35 34 34)"/>
    <!-- Central node = Autonomous (A) AI core -->
    <circle cx="34" cy="34" r="5.5" fill="url(#lg1)" filter="url(#fw)"/>
    <circle cx="34" cy="34" r="2.8" fill="#050e1f"/>
  </svg>
  <div style="font-size:1.3rem;font-weight:800;
              background:linear-gradient(90deg,#00d4ff,#00ff88);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              background-clip:text;letter-spacing:0.08em;margin-top:2px;">ACMGS</div>
  <div style="font-size:0.63rem;color:rgba(255,255,255,0.28);
              letter-spacing:0.15em;margin-top:3px;">CONTROL CENTER  v9.0</div>
</div>
""", unsafe_allow_html=True)

    # ── Carbon intensity slider ────────────────────────────────────────────────
    st.markdown('<div class="slabel">Live Carbon Monitor</div>', unsafe_allow_html=True)
    carbon_val = st.slider(
        "Grid Carbon Intensity (gCO₂/kWh)",
        min_value=0, max_value=600, value=220, step=5,
        label_visibility="collapsed",
    )
    zone = classify_carbon_zone(float(carbon_val))

    # Zone indicator card
    st.markdown(
        f'<div style="background:{ZONE_BG[zone]};border:1px solid {ZONE_BORDER[zone]};'
        'border-radius:12px;padding:14px;text-align:center;margin:8px 0 14px 0;">'
        f'<div style="font-size:1.9rem;line-height:1.2;">{ZONE_EMOJI[zone]}</div>'
        f'<div style="font-size:1.1rem;font-weight:700;color:{ZONE_COLORS[zone]};'
        'letter-spacing:0.05em;margin-top:2px;">'
        f'{zone} CARBON</div>'
        '<div style="font-size:0.82rem;color:rgba(255,255,255,0.5);margin-top:4px;">'
        f'{carbon_val} gCO₂/kWh</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # AI recommendation mini-panel
    try:
        rec = get_recommendation(float(carbon_val))
        sched = rec.get("recommended_schedule", {})
        rec_yield  = sched.get("pred_yield", 0.0)
        rec_energy = sched.get("pred_energy", 0.0)
        rec_carbon = sched.get("pred_carbon", 0.0)

        st.markdown(
            '<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);'
            'border-radius:8px;padding:13px;margin-bottom:14px;">'
            '<div style="font-size:0.69rem;color:rgba(255,255,255,0.28);text-transform:uppercase;'
            'letter-spacing:0.1em;margin-bottom:7px;">AI Recommendation</div>'
            f'<div style="font-size:0.82rem;color:rgba(255,255,255,0.8);line-height:1.5;">'
            f'{ZONE_DESC[zone]}</div>'
            '<div style="display:flex;gap:7px;margin-top:11px;">'
            '<div style="flex:1;background:rgba(0,212,255,0.08);border-radius:6px;padding:7px 4px;text-align:center;">'
            f'<div style="font-size:0.95rem;font-weight:600;color:#00d4ff;font-family:\'JetBrains Mono\',monospace;">'
            f'{rec_yield:.4f}</div>'
            '<div style="font-size:0.63rem;color:rgba(255,255,255,0.38);text-transform:uppercase;">Yield</div></div>'
            '<div style="flex:1;background:rgba(0,255,136,0.06);border-radius:6px;padding:7px 4px;text-align:center;">'
            f'<div style="font-size:0.95rem;font-weight:600;color:#00ff88;font-family:\'JetBrains Mono\',monospace;">'
            f'{rec_energy:.0f}</div>'
            '<div style="font-size:0.63rem;color:rgba(255,255,255,0.38);text-transform:uppercase;">kWh</div></div>'
            '<div style="flex:1;background:rgba(255,75,75,0.06);border-radius:6px;padding:7px 4px;text-align:center;">'
            f'<div style="font-size:0.95rem;font-weight:600;color:#ff6b6b;font-family:\'JetBrains Mono\',monospace;">'
            f'{rec_carbon:.0f}</div>'
            '<div style="font-size:0.63rem;color:rgba(255,255,255,0.38);text-transform:uppercase;">kg CO₂</div></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            f'<div style="font-size:0.8rem;color:rgba(255,255,255,0.5);'
            'padding:10px;background:rgba(255,255,255,0.02);border-radius:6px;">'
            f'{ZONE_DESC[zone]}</div>',
            unsafe_allow_html=True,
        )

    # ── DB row counts ──────────────────────────────────────────────────────────
    st.markdown('<div class="slabel">Database Status</div>', unsafe_allow_html=True)
    for key, count in [(k, v) for k, v in db_summary.items() if k != "db_size_mb"]:
        icon = TABLE_ICONS.get(key, "📄")
        st.markdown(
            '<div style="display:flex;justify-content:space-between;align-items:center;'
            'padding:5px 2px;border-bottom:1px solid rgba(255,255,255,0.05);">'
            f'<span style="font-size:0.77rem;color:rgba(255,255,255,0.44);">'
            f'{icon} {key.replace("_"," ").title()}</span>'
            f'<span style="font-size:0.77rem;font-weight:600;color:#00d4ff;'
            f'font-family:\'JetBrains Mono\',monospace;">{count:,}</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="margin-top:9px;padding:8px;background:rgba(255,255,255,0.02);'
        'border-radius:6px;display:flex;justify-content:space-between;">'
        '<span style="font-size:0.71rem;color:rgba(255,255,255,0.3);">DB Size</span>'
        f'<span style="font-size:0.71rem;font-weight:600;color:rgba(255,255,255,0.5);">'
        f'{db_summary["db_size_mb"]} MB</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="font-size:0.67rem;color:rgba(255,255,255,0.2);'
        'text-align:center;margin:12px 0 8px 0;">'
        f'{datetime.now().strftime("%b %d, %Y  ·  %H:%M:%S")}</div>',
        unsafe_allow_html=True,
    )

    if st.button("🔄  Refresh All Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ─── Main header ─────────────────────────────────────────────────────────────
badge_zone_cls = {"LOW": "hbadge-green", "MEDIUM": "hbadge-yellow", "HIGH": "hbadge-red"}[zone]
st.markdown(
    '<div class="acmgs-header"><div style="display:flex;justify-content:space-between;align-items:flex-start;">'
    '<div>'
    '<h1>🧬 ACMGS Control Center</h1>'
    '<p>Autonomous Carbon-Aware Manufacturing Genome System — Real-Time Intelligence Dashboard</p>'
    '<div>'
    '<span class="hbadge">Phase 9</span>'
    '<span class="hbadge hbadge-green">● Live</span>'
    f'<span class="hbadge">2,000 Batches</span>'
    f'<span class="hbadge">100 Pareto Solutions</span>'
    f'<span class="hbadge {badge_zone_cls}">{ZONE_EMOJI[zone]} {zone} Zone</span>'
    '</div></div>'
    f'<div style="text-align:right;font-size:0.73rem;color:rgba(255,255,255,0.3);padding-top:4px;">'
    f'<div style="font-size:1.4rem;">{ZONE_EMOJI[zone]}</div>'
    f'Updated {datetime.now().strftime("%H:%M:%S")}'
    '</div></div></div>',
    unsafe_allow_html=True,
)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎛️  Command Center",
    "📈  Production Analytics",
    "⚖️  Pareto Intelligence",
    "🧬  Genome Explorer",
    "🩺  System Health",
    "🤖  Digital Twin",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — COMMAND CENTER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Batches", f"{len(df_batches):,}", f"2,000 loaded")
    with k2:
        avg_yield = df_batches["yield"].mean()
        st.metric("Avg Batch Yield", f"{avg_yield:.4f}", f"σ = {df_batches['yield'].std():.4f}")
    with k3:
        avg_carbon = df_batches["carbon_intensity"].mean()
        st.metric("Avg Carbon Intensity", f"{avg_carbon:.1f}", "gCO₂/kWh", delta_color="inverse")
    with k4:
        best_yield = df_pareto["pred_yield"].max() if len(df_pareto) > 0 else 0.0
        st.metric("Best Pareto Yield", f"{best_yield:.4f}", f"{len(df_pareto)} solutions")

    st.markdown("<br>", unsafe_allow_html=True)

    # Gauge + schedule recommendation
    left, right = st.columns([4, 6])

    with left:
        fig_gauge = make_gauge(carbon_val, zone)
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

        # Zone description strip below gauge
        st.markdown(
            f'<div style="background:{ZONE_BG[zone]};border:1px solid {ZONE_BORDER[zone]};'
            'border-radius:10px;padding:12px 16px;margin-top:-10px;text-align:center;">'
            f'<span style="font-size:1.1rem;font-weight:700;color:{ZONE_COLORS[zone]};">'
            f'{ZONE_EMOJI[zone]}  {ZONE_TITLE[zone]}</span><br>'
            f'<span style="font-size:0.8rem;color:rgba(255,255,255,0.55);margin-top:4px;display:block;">'
            f'{ZONE_DESC[zone]}</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    with right:
        try:
            rec = get_recommendation(float(carbon_val))
            sched = rec["recommended_schedule"]

            st.markdown(
                '<div class="slabel" style="margin-top:4px;">Optimal Manufacturing Schedule</div>',
                unsafe_allow_html=True,
            )

            # Process parameters (4 per row)
            process_params = [
                ("Temperature", f"{sched.get('temperature', 0):.1f}", "°C",   _CYAN),
                ("Pressure",    f"{sched.get('pressure', 0):.2f}",    "bar",  _CYAN),
                ("Speed",       f"{sched.get('speed', 0):.0f}",       "rpm",  _CYAN),
                ("Feed Rate",   f"{sched.get('feed_rate', 0):.2f}",   "kg/h", _CYAN),
            ]
            material_params = [
                ("Density",     f"{sched.get('material_density', 0):.3f}",   "g/cm³",  _GREEN),
                ("Hardness",    f"{sched.get('material_hardness', 0):.1f}",  "HV",     _GREEN),
                ("Mat. Grade",  f"{int(sched.get('material_grade', 0))}",    "",       _GREEN),
                ("Humidity",    f"{sched.get('humidity', 0):.1f}",           "%",      _GREEN),
            ]
            outcome_params = [
                ("Pred Yield",   f"{sched.get('pred_yield', 0):.4f}",  "",      _YELLOW),
                ("Pred Quality", f"{sched.get('pred_quality', 0):.4f}", "",     _YELLOW),
                ("Pred Energy",  f"{sched.get('pred_energy', 0):.1f}",  "kWh",  _ORANGE),
                ("Pred Carbon",  f"{sched.get('pred_carbon', 0):.1f}",  "kg",   _RED),
            ]

            def _param_grid(params):
                cols = st.columns(4)
                for i, (label, val_str, unit, col_hex) in enumerate(params):
                    with cols[i]:
                        st.markdown(
                            '<div style="background:rgba(255,255,255,0.04);'
                            'border:1px solid rgba(255,255,255,0.08);border-radius:9px;'
                            'padding:11px 8px;text-align:center;margin-bottom:8px;">'
                            f'<div style="font-size:1.05rem;font-weight:600;color:{col_hex};'
                            "font-family:'JetBrains Mono',monospace;\">"
                            f'{val_str}'
                            f'<span style="font-size:0.62rem;color:rgba(255,255,255,0.35);'
                            f'margin-left:2px;">{unit}</span></div>'
                            f'<div style="font-size:0.65rem;color:rgba(255,255,255,0.38);'
                            'text-transform:uppercase;letter-spacing:0.05em;margin-top:3px;">'
                            f'{label}</div>'
                            '</div>',
                            unsafe_allow_html=True,
                        )

            _param_grid(process_params)
            _param_grid(material_params)
            _param_grid(outcome_params)

        except Exception as e:
            st.warning(f"Recommendation unavailable: {e}")

    # Schedule history
    if len(df_schedules) > 0:
        st.markdown('<div class="slabel" style="margin-top:24px;">Historical Schedule Decisions</div>',
                    unsafe_allow_html=True)
        disp = df_schedules[[
            "carbon_intensity", "zone",
            "schedule_pred_yield", "schedule_pred_quality",
            "schedule_pred_energy", "schedule_pred_carbon",
        ]].copy()
        disp.columns = ["Carbon Int.", "Zone", "Pred Yield", "Pred Quality", "Pred Energy (kWh)", "Pred Carbon (kg)"]
        disp = disp.round(4)
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PRODUCTION ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    # KPI row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Avg Yield",   f"{df_batches['yield'].mean():.4f}",
                  f"σ={df_batches['yield'].std():.4f}")
    with m2:
        st.metric("Avg Quality", f"{df_batches['quality'].mean():.4f}",
                  f"σ={df_batches['quality'].std():.4f}")
    with m3:
        st.metric("Avg Energy",  f"{df_batches['energy_consumption'].mean():.0f} kWh",
                  f"Max {df_batches['energy_consumption'].max():.0f}")
    with m4:
        st.metric("Avg Carbon",  f"{df_batches['carbon_intensity'].mean():.1f}",
                  "gCO₂/kWh", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: yield histogram + zone pie
    c1, c2 = st.columns([6, 4])

    with c1:
        fig_hist = px.histogram(
            df_batches, x="yield", nbins=60,
            title="Yield Distribution — 2,000 Batches",
            labels={"yield": "Batch Yield", "count": "Frequency"},
            color_discrete_sequence=[_CYAN],
        )
        fig_hist.update_traces(
            marker_line_color="rgba(0,212,255,0.5)",
            marker_line_width=0.5,
        )
        dark_layout(fig_hist, height=320)
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    with c2:
        zone_counts = df_batches["zone"].value_counts().reset_index()
        zone_counts.columns = ["zone", "count"]
        fig_pie = go.Figure(go.Pie(
            labels=zone_counts["zone"],
            values=zone_counts["count"],
            hole=0.5,
            marker=dict(colors=[ZONE_COLORS.get(z, _CYAN) for z in zone_counts["zone"]],
                        line=dict(color="rgba(0,0,0,0.4)", width=2)),
            textfont=dict(color="rgba(255,255,255,0.8)", size=11),
            hovertemplate="<b>%{label}</b><br>Batches: %{value}<br>Share: %{percent}<extra></extra>",
        ))
        fig_pie.update_layout(
            title=dict(text="Carbon Zone Distribution", font=dict(size=13, color="rgba(255,255,255,0.7)")),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
            legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.12)", borderwidth=1),
            height=320,
            margin=dict(l=16, r=16, t=44, b=16),
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    # Row 3: full-width scatter of all 2000 batches
    fig_scatter = px.scatter(
        df_batches, x="carbon_intensity", y="energy_consumption",
        color="zone",
        color_discrete_map=ZONE_COLORS,
        title="Carbon Intensity vs Energy Consumption — Full Production Fleet (2,000 Batches)",
        labels={
            "carbon_intensity": "Carbon Intensity (gCO₂/kWh)",
            "energy_consumption": "Energy Consumption (kWh)",
            "zone": "Carbon Zone",
        },
        hover_data=["batch_id", "yield", "quality"],
        opacity=0.6,
        category_orders={"zone": ["LOW", "MEDIUM", "HIGH"]},
    )
    fig_scatter.add_vline(
        x=CARBON_LOW_THRESHOLD, line_dash="dash",
        line_color="rgba(0,255,136,0.45)",
        annotation_text=f"LOW (<{CARBON_LOW_THRESHOLD})",
        annotation_position="top left",
        annotation_font=dict(color="rgba(0,255,136,0.7)", size=10),
    )
    fig_scatter.add_vline(
        x=CARBON_HIGH_THRESHOLD, line_dash="dash",
        line_color="rgba(255,75,75,0.45)",
        annotation_text=f"HIGH (>{CARBON_HIGH_THRESHOLD})",
        annotation_position="top right",
        annotation_font=dict(color="rgba(255,75,75,0.7)", size=10),
    )
    dark_layout(fig_scatter, height=380)
    st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

    # Row 4: two correlation scatters
    c3, c4 = st.columns(2)

    with c3:
        fig_tv = px.scatter(
            df_batches, x="temperature", y="yield",
            color="zone", color_discrete_map=ZONE_COLORS,
            title="Temperature vs Yield",
            labels={"temperature": "Temperature (°C)", "yield": "Yield"},
            opacity=0.55,
            hover_data=["batch_id"],
            category_orders={"zone": ["LOW", "MEDIUM", "HIGH"]},
        )
        dark_layout(fig_tv, height=320)
        st.plotly_chart(fig_tv, use_container_width=True, config={"displayModeBar": False})

    with c4:
        fig_se = px.scatter(
            df_batches, x="speed", y="energy_consumption",
            color="zone", color_discrete_map=ZONE_COLORS,
            title="Production Speed vs Energy Consumption",
            labels={"speed": "Speed (rpm)", "energy_consumption": "Energy (kWh)"},
            opacity=0.55,
            hover_data=["batch_id"],
            category_orders={"zone": ["LOW", "MEDIUM", "HIGH"]},
        )
        dark_layout(fig_se, height=320)
        st.plotly_chart(fig_se, use_container_width=True, config={"displayModeBar": False})

    # Row 5: Correlation heatmap + Batch explorer
    c5, c6 = st.columns([5, 5])

    with c5:
        st.markdown('<div class="slabel">Feature Correlation Matrix</div>', unsafe_allow_html=True)
        num_cols = ["temperature", "pressure", "speed", "feed_rate",
                    "humidity", "yield", "quality", "energy_consumption", "carbon_intensity"]
        corr = df_batches[num_cols].corr().round(2)
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=[c.replace("_", " ").title() for c in corr.columns],
            y=[c.replace("_", " ").title() for c in corr.index],
            colorscale=[[0,"#ff4b4b"], [0.5,"rgba(255,255,255,0.05)"], [1,"#00d4ff"]],
            zmid=0,
            text=corr.values,
            texttemplate="%{text:.2f}",
            textfont=dict(size=9, color="rgba(255,255,255,0.8)"),
            hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.3f}<extra></extra>",
            colorbar=dict(
                tickfont=dict(size=9, color="rgba(255,255,255,0.4)"),
                thickness=12, len=0.9,
            ),
        ))
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.65)", family="Inter", size=9),
            xaxis=dict(tickangle=40, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
            height=380,
            margin=dict(l=80, r=20, t=20, b=80),
        )
        st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})

    with c6:
        st.markdown('<div class="slabel">Batch Explorer</div>', unsafe_allow_html=True)
        search_term = st.text_input("Search by Batch ID prefix",
                                    placeholder="e.g.  BATCH_00",
                                    label_visibility="collapsed")
        df_search = (df_batches[df_batches["batch_id"].str.startswith(search_term)]
                     if search_term else df_batches.head(100))
        show_cols = ["batch_id", "temperature", "pressure", "speed",
                     "yield", "quality", "energy_consumption", "carbon_intensity", "zone"]
        st.dataframe(
            df_search[show_cols].round(4),
            use_container_width=True,
            hide_index=True,
            height=330,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PARETO INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if len(df_pareto) == 0:
        st.warning("No Pareto solutions in the database. Run Phase 5 first.")
    else:
        # Filter bar
        f1, f2, f3 = st.columns([3, 3, 4])
        with f1:
            min_yield = st.slider("Min Pred Yield", 0.0, 0.99, 0.0, 0.01)
        with f2:
            _c_min = float(df_pareto["pred_carbon"].min())
            _c_max = float(df_pareto["pred_carbon"].max())
            max_carbon = st.slider("Max Pred Carbon (kgCO₂)", _c_min, _c_max,
                                   _c_max, (_c_max - _c_min) / 20 or 1.0)
        with f3:
            color_by = st.selectbox(
                "Color dimension",
                ["pred_yield", "pred_quality", "pred_energy", "pred_carbon"],
                index=0,
            )

        df_fp = df_pareto[
            (df_pareto["pred_yield"]  >= min_yield) &
            (df_pareto["pred_carbon"] <= max_carbon)
        ].copy()

        st.markdown(
            f'<div style="font-size:0.8rem;color:rgba(255,255,255,0.38);margin-bottom:12px;">'
            f'Showing <b style="color:#00d4ff;">{len(df_fp)}</b> of '
            f'<b style="color:rgba(255,255,255,0.6);">{len(df_pareto)}</b> Pareto solutions</div>',
            unsafe_allow_html=True,
        )

        # 3D scatter ──────────────────────────────────────────────────────────
        color_scale_map = {
            "pred_yield":   [[0, _RED],    [0.5, _YELLOW], [1, _GREEN]],
            "pred_quality": [[0, _CYAN],   [0.5, _PURPLE], [1, _GREEN]],
            "pred_energy":  [[0, _GREEN],  [0.5, _YELLOW], [1, _RED]],
            "pred_carbon":  [[0, _GREEN],  [0.5, _YELLOW], [1, _RED]],
        }

        fig_3d = px.scatter_3d(
            df_fp,
            x="pred_yield", y="pred_energy", z="pred_carbon",
            color=color_by,
            color_continuous_scale=color_scale_map[color_by],
            title="Pareto Frontier — 3-Objective Trade-off Space",
            labels={
                "pred_yield":   "Yield",
                "pred_energy":  "Energy (kWh)",
                "pred_carbon":  "Carbon (kgCO₂)",
            },
            hover_data=["pred_quality", "temperature", "pressure", "speed"],
        )
        fig_3d.update_traces(marker=dict(size=7, opacity=0.9))
        fig_3d.update_layout(
            paper_bgcolor="rgba(13,17,23,1)",
            scene=dict(
                bgcolor="rgb(13,17,23)",
                xaxis=dict(
                    backgroundcolor="rgba(0,212,255,0.05)",
                    gridcolor="rgba(255,255,255,0.1)",
                    tickfont=dict(color="rgba(255,255,255,0.5)", size=9),
                    title=dict(font=dict(color="rgba(255,255,255,0.5)", size=10)),
                ),
                yaxis=dict(
                    backgroundcolor="rgba(0,255,136,0.02)",
                    gridcolor="rgba(255,255,255,0.1)",
                    tickfont=dict(color="rgba(255,255,255,0.5)", size=9),
                    title=dict(font=dict(color="rgba(255,255,255,0.5)", size=10)),
                ),
                zaxis=dict(
                    backgroundcolor="rgba(255,75,75,0.02)",
                    gridcolor="rgba(255,255,255,0.1)",
                    tickfont=dict(color="rgba(255,255,255,0.5)", size=9),
                    title=dict(font=dict(color="rgba(255,255,255,0.5)", size=10)),
                ),
            ),
            coloraxis_colorbar=dict(
                tickfont=dict(size=9, color="rgba(255,255,255,0.4)"),
                thickness=12,
            ),
            font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
            height=480,
            margin=dict(l=0, r=0, t=50, b=0),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        # 2D Pareto front + top-10 bar
        p1, p2 = st.columns([6, 4])

        with p1:
            fig_pareto2 = px.scatter(
                df_fp,
                x="pred_energy", y="pred_yield",
                color="pred_carbon",
                color_continuous_scale=[[0, _GREEN], [0.5, _YELLOW], [1, _RED]],
                title="Pareto Front — Yield vs Energy (size = quality)",
                labels={
                    "pred_energy": "Predicted Energy (kWh)",
                    "pred_yield":  "Predicted Yield",
                    "pred_carbon": "Carbon (kgCO₂)",
                },
                size="pred_quality",
                size_max=14,
                hover_data=["temperature", "pressure", "speed", "pred_quality"],
            )
            dark_layout(fig_pareto2, height=360)
            fig_pareto2.update_coloraxes(
                colorbar=dict(
                    tickfont=dict(size=9, color="rgba(255,255,255,0.4)"),
                    thickness=11, len=0.9,
                    title=dict(text="Carbon", font=dict(size=10)),
                )
            )
            st.plotly_chart(fig_pareto2, use_container_width=True,
                            config={"displayModeBar": False})

        with p2:
            top10 = df_pareto.head(10).copy()
            top10["rank"] = [f"#{i+1}" for i in range(len(top10))]
            fig_bar10 = go.Figure(go.Bar(
                x=top10["pred_yield"],
                y=top10["rank"],
                orientation="h",
                marker=dict(
                    color=top10["pred_yield"],
                    colorscale=[[0, "rgba(0,212,255,0.5)"], [1, "#00ff88"]],
                    line=dict(width=0),
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Yield: %{x:.4f}<br>"
                    "<extra></extra>"
                ),
            ))
            fig_bar10.update_layout(
                title=dict(text="Top 10 Solutions by Yield",
                           font=dict(size=13, color="rgba(255,255,255,0.7)")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
                xaxis=dict(title="Predicted Yield",
                           gridcolor="rgba(255,255,255,0.07)",
                           tickfont=dict(size=10)),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10)),
                height=360,
                margin=dict(l=50, r=16, t=44, b=30),
            )
            st.plotly_chart(fig_bar10, use_container_width=True,
                            config={"displayModeBar": False})

        # Filtered data table
        st.markdown('<div class="slabel">Filtered Pareto Solutions</div>', unsafe_allow_html=True)
        show_p_cols = [
            "temperature", "pressure", "speed", "feed_rate",
            "material_density", "material_hardness", "material_grade",
            "pred_yield", "pred_quality", "pred_energy", "pred_carbon",
        ]
        st.dataframe(
            df_fp[show_p_cols].reset_index(drop=True).round(4),
            use_container_width=True,
            hide_index=True,
            height=300,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — GENOME EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if len(df_genomes) == 0:
        st.warning("No genome data available in the database.")
    else:
        # Parse all loaded genomes into matrix
        genome_matrix = np.array([json.loads(g) for g in df_genomes["genome"]])  # (N, 25)
        batch_ids_gm  = df_genomes["batch_id"].tolist()

        # Batch selector
        st.markdown('<div class="slabel">Individual Batch Analysis</div>', unsafe_allow_html=True)
        g1, g2 = st.columns([3, 7])
        with g1:
            all_ids = df_batches["batch_id"].tolist()
            selected_batch = st.selectbox(
                "Select Batch ID",
                all_ids[:300],
                index=0,
                label_visibility="collapsed",
            )

        # Resolve genome for selected batch
        sel_genome = None
        if selected_batch in batch_ids_gm:
            sel_idx    = batch_ids_gm.index(selected_batch)
            sel_genome = genome_matrix[sel_idx]
        else:
            conn = sqlite3.connect(DB_PATH)
            row_ = conn.execute(
                "SELECT genome FROM genome_vectors WHERE batch_id=?", (selected_batch,)
            ).fetchone()
            conn.close()
            if row_:
                sel_genome = np.array(json.loads(row_[0]))

        with g2:
            if sel_genome is not None:
                batch_row = df_batches[df_batches["batch_id"] == selected_batch]
                if len(batch_row) > 0:
                    br = batch_row.iloc[0]
                    st.markdown(
                        '<div style="display:flex;gap:12px;flex-wrap:wrap;">'
                        + "".join([
                            f'<div style="background:rgba(255,255,255,0.04);border:1px solid '
                            f'rgba(255,255,255,0.09);border-radius:8px;padding:8px 14px;'
                            f'text-align:center;min-width:80px;">'
                            f'<div style="font-size:0.95rem;font-weight:600;color:{c};'
                            f'font-family:\'JetBrains Mono\',monospace;">{v}</div>'
                            f'<div style="font-size:0.62rem;color:rgba(255,255,255,0.38);'
                            f'text-transform:uppercase;letter-spacing:0.05em;margin-top:2px;">{lbl}</div>'
                            '</div>'
                            for lbl, v, c in [
                                ("Yield",   f"{br['yield']:.4f}",           _GREEN),
                                ("Quality", f"{br['quality']:.4f}",         _CYAN),
                                ("Energy",  f"{br['energy_consumption']:.0f} kWh", _YELLOW),
                                ("Carbon",  f"{br['carbon_intensity']:.1f}", _RED),
                                ("Zone",    br["zone"],                      ZONE_COLORS[br["zone"]]),
                            ]
                        ])
                        + '</div>',
                        unsafe_allow_html=True,
                    )

        # Individual batch charts
        if sel_genome is not None:
            ga, gb = st.columns([4, 6])

            with ga:
                # Radar chart of 5 process parameters (normalized to 0-100)
                batch_row = df_batches[df_batches["batch_id"] == selected_batch]
                if len(batch_row) > 0:
                    br = batch_row.iloc[0]
                    radar_cats = ["Temperature", "Pressure", "Speed", "Feed Rate", "Humidity"]
                    raw_vals   = [br["temperature"], br["pressure"], br["speed"],
                                  br["feed_rate"],  br["humidity"]]
                    feat_cols   = ["temperature", "pressure", "speed", "feed_rate", "humidity"]
                    norm_vals   = [
                        (v - df_batches[c].min()) / (df_batches[c].max() - df_batches[c].min() + 1e-9) * 100
                        for v, c in zip(raw_vals, feat_cols)
                    ]
                    # close the loop
                    r_vals  = norm_vals  + [norm_vals[0]]
                    r_theta = radar_cats + [radar_cats[0]]

                    fig_radar = go.Figure(go.Scatterpolar(
                        r=r_vals, theta=r_theta,
                        fill="toself",
                        fillcolor="rgba(0,212,255,0.15)",
                        line=dict(color=_CYAN, width=2),
                        marker=dict(color=_CYAN, size=6),
                        name=selected_batch,
                        hovertemplate="<b>%{theta}</b><br>Percentile: %{r:.1f}%<extra></extra>",
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            bgcolor="rgba(255,255,255,0.02)",
                            radialaxis=dict(
                                visible=True, range=[0, 100],
                                tickfont=dict(size=8, color="rgba(255,255,255,0.3)"),
                                gridcolor="rgba(255,255,255,0.08)",
                                linecolor="rgba(255,255,255,0.1)",
                            ),
                            angularaxis=dict(
                                tickfont=dict(size=10, color="rgba(255,255,255,0.6)"),
                                gridcolor="rgba(255,255,255,0.08)",
                                linecolor="rgba(255,255,255,0.1)",
                            ),
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
                        title=dict(text=f"Process Profile<br><span style='font-size:0.8em'>{selected_batch}</span>",
                                   font=dict(size=12, color="rgba(255,255,255,0.6)")),
                        height=360,
                        margin=dict(l=30, r=30, t=55, b=20),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_radar, use_container_width=True,
                                    config={"displayModeBar": False})

            with gb:
                # Bar chart of all 25 genome dimensions
                seg_colors = (
                    [_CYAN]   * 5   # Process
                    + [_GREEN]  * 3   # Material
                    + [_PURPLE] * 16  # EnergyDNA
                    + [_YELLOW] * 1   # Carbon
                )
                fig_gbar = go.Figure(go.Bar(
                    x=GENOME_LABELS,
                    y=sel_genome.tolist(),
                    marker=dict(
                        color=seg_colors,
                        opacity=0.88,
                        line=dict(width=0),
                    ),
                    hovertemplate="<b>%{x}</b><br>z-score: %{y:.4f}<extra></extra>",
                ))
                fig_gbar.add_hline(
                    y=0,
                    line=dict(color="rgba(255,255,255,0.2)", dash="dot", width=1),
                )
                # Segment dividers
                for xpos, label, col in [(4.5, "Process", _CYAN),
                                         (7.5, "Material", _GREEN),
                                         (23.5, "EdNA", _PURPLE),
                                         (24.5, "Ci", _YELLOW)]:
                    fig_gbar.add_vline(
                        x=xpos,
                        line=dict(color="rgba(255,255,255,0.12)", dash="dot", width=1),
                    )
                fig_gbar.update_layout(
                    title=dict(
                        text=f"25-Dimension Genome Vector — {selected_batch}",
                        font=dict(size=12, color="rgba(255,255,255,0.6)"),
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.02)",
                    font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
                    xaxis=dict(
                        tickangle=60, tickfont=dict(size=8),
                        gridcolor="rgba(255,255,255,0.04)",
                        linecolor="rgba(255,255,255,0.08)",
                    ),
                    yaxis=dict(
                        title="z-score",
                        gridcolor="rgba(255,255,255,0.07)",
                        zerolinecolor="rgba(255,255,255,0.15)",
                        linecolor="rgba(255,255,255,0.08)",
                    ),
                    height=360,
                    margin=dict(l=50, r=16, t=50, b=70),
                    showlegend=False,
                    bargap=0.25,
                )
                st.plotly_chart(fig_gbar, use_container_width=True,
                                config={"displayModeBar": False})

        # Population heatmap
        st.markdown(
            f'<div class="slabel">Genome Population Heatmap — '
            f'{len(df_genomes)} Batches × 25 Dimensions</div>',
            unsafe_allow_html=True,
        )
        fig_hm = go.Figure(go.Heatmap(
            z=genome_matrix.T,             # shape (25, N) — dims as rows
            x=[bid[-4:] for bid in batch_ids_gm],
            y=GENOME_LABELS,
            colorscale=[
                [0.0, "#ff4b4b"],
                [0.3, "rgba(200,80,80,0.5)"],
                [0.5, "rgba(255,255,255,0.06)"],
                [0.7, "rgba(0,160,255,0.5)"],
                [1.0, "#00d4ff"],
            ],
            zmid=0,
            hovertemplate="Batch: %{x}<br>Dimension: %{y}<br>z-score: %{z:.4f}<extra></extra>",
            colorbar=dict(
                tickfont=dict(size=9, color="rgba(255,255,255,0.4)"),
                outlinecolor="rgba(255,255,255,0.08)",
                outlinewidth=1,
                thickness=12,
                len=0.9,
                title=dict(text="z-score", font=dict(size=10, color="rgba(255,255,255,0.4)")),
            ),
        ))
        fig_hm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.65)", family="Inter"),
            xaxis=dict(
                title="Batch ID (last 4 chars)",
                tickfont=dict(size=7),
                tickangle=90,
                gridcolor="rgba(255,255,255,0.03)",
            ),
            yaxis=dict(tickfont=dict(size=9), gridcolor="rgba(255,255,255,0.03)"),
            height=500,
            margin=dict(l=90, r=20, t=20, b=70),
        )
        st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SYSTEM HEALTH
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="slabel">Database Overview</div>', unsafe_allow_html=True)

    # 7 metric cards
    h_cols = st.columns(7)
    for i, (key, count) in enumerate([(k, v) for k, v in db_summary.items() if k != "db_size_mb"]):
        icon = TABLE_ICONS.get(key, "📄")
        with h_cols[i]:
            st.metric(f"{icon} {key.replace('_', ' ').title()}", f"{count:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: DB bar chart + prediction accuracy
    ha, hb = st.columns([4, 6])

    with ha:
        df_counts = pd.DataFrame([
            {"Table": k.replace("_", " ").title(), "Rows": v}
            for k, v in db_summary.items() if k != "db_size_mb"
        ]).sort_values("Rows", ascending=True)

        fig_dbbar = go.Figure(go.Bar(
            x=df_counts["Rows"],
            y=df_counts["Table"],
            orientation="h",
            marker=dict(
                color=df_counts["Rows"],
                colorscale=[[0, "rgba(0,212,255,0.4)"], [1, "#00ff88"]],
                line=dict(width=0),
            ),
            hovertemplate="<b>%{y}</b><br>Rows: %{x:,}<extra></extra>",
            text=df_counts["Rows"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            textfont=dict(color="rgba(255,255,255,0.6)", size=10),
        ))
        fig_dbbar.update_layout(
            title=dict(text="Table Row Counts", font=dict(size=12, color="rgba(255,255,255,0.6)")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.015)",
            font=dict(color="rgba(255,255,255,0.65)", family="Inter"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=9)),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=9)),
            height=360,
            margin=dict(l=130, r=55, t=44, b=16),
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_dbbar, use_container_width=True, config={"displayModeBar": False})

    with hb:
        # Model prediction accuracy (yield pred vs actual, if data exists)
        df_pred_notnull = df_preds.dropna(subset=["pred_yield", "actual_yield"]) \
            if len(df_preds) > 0 and "actual_yield" in df_preds.columns else pd.DataFrame()

        if len(df_pred_notnull) > 0:
            mn = min(df_pred_notnull["actual_yield"].min(), df_pred_notnull["pred_yield"].min())
            mx = max(df_pred_notnull["actual_yield"].max(), df_pred_notnull["pred_yield"].max())
            fig_acc = px.scatter(
                df_pred_notnull,
                x="actual_yield", y="pred_yield",
                title="Yield Prediction Accuracy (Predicted vs Actual)",
                labels={"actual_yield": "Actual Yield", "pred_yield": "Predicted Yield"},
                color_discrete_sequence=[_CYAN],
                opacity=0.6,
            )
            fig_acc.add_shape(
                type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                line=dict(color="rgba(255,214,0,0.55)", dash="dash", width=2),
            )
            dark_layout(fig_acc, height=360)
            st.plotly_chart(fig_acc, use_container_width=True, config={"displayModeBar": False})
        else:
            # Empty state — show a note + DB info card
            st.markdown(
                '<div style="background:rgba(255,255,255,0.03);border:1px solid '
                'rgba(255,255,255,0.09);border-radius:12px;padding:24px;'
                'text-align:center;height:360px;display:flex;flex-direction:column;'
                'justify-content:center;align-items:center;">'
                '<div style="font-size:2rem;margin-bottom:10px;">🎯</div>'
                '<div style="font-size:0.9rem;color:rgba(255,255,255,0.6);">'
                'Prediction accuracy plot available after<br>'
                'running Phase 4 and loading <code>predictions</code> table.</div>'
                f'<div style="margin-top:16px;font-size:0.8rem;color:{_CYAN};">'
                f'Database: {db_summary["db_size_mb"]} MB  ·  '
                f'{db_summary["batches"]:,} batches loaded</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    # Pareto multi-objective metrics
    if len(df_pareto) > 0:
        st.markdown('<div class="slabel">Pareto Frontier Statistics</div>', unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric("Best Yield",    f"{df_pareto['pred_yield'].max():.4f}",
                      f"Avg {df_pareto['pred_yield'].mean():.4f}")
        with p2:
            st.metric("Best Quality",  f"{df_pareto['pred_quality'].max():.4f}",
                      f"Avg {df_pareto['pred_quality'].mean():.4f}")
        with p3:
            st.metric("Min Energy",    f"{df_pareto['pred_energy'].min():.1f} kWh",
                      f"Avg {df_pareto['pred_energy'].mean():.1f}", delta_color="inverse")
        with p4:
            st.metric("Min Carbon",    f"{df_pareto['pred_carbon'].min():.1f} kg",
                      f"Avg {df_pareto['pred_carbon'].mean():.1f}", delta_color="inverse")

    # Pipeline run audit log
    st.markdown('<div class="slabel">Pipeline Execution Log</div>', unsafe_allow_html=True)
    if len(df_runs) > 0:
        run_cols = [c for c in ["phase", "phase_name", "status", "details", "started_at", "finished_at"]
                    if c in df_runs.columns]
        st.dataframe(df_runs[run_cols], use_container_width=True, hide_index=True, height=260)
    else:
        st.info("No pipeline runs recorded yet.")

    # Footer
    st.markdown(
        '<div style="text-align:center;margin-top:30px;padding:18px;'
        'border-top:1px solid rgba(255,255,255,0.06);">'
        '<div style="font-size:0.77rem;color:rgba(255,255,255,0.22);">'
        'ACMGS v9.0 &nbsp;·&nbsp; Autonomous Carbon-Aware Manufacturing Genome System'
        '&nbsp;·&nbsp; Phase 9: Streamlit Dashboard<br>'
        f'Database: {db_summary.get("db_size_mb", 0)} MB &nbsp;·&nbsp; '
        f'{db_summary.get("batches", 0):,} Batches &nbsp;·&nbsp; '
        f'{db_summary.get("pareto_solutions", 0)} Pareto Solutions &nbsp;·&nbsp; '
        f'Rendered: {datetime.now().strftime("%Y-%m-%d  %H:%M:%S")}'
        '</div></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DIGITAL TWIN
# ══════════════════════════════════════════════════════════════════════════════
with tab6:

    # ── Load all DT data ───────────────────────────────────────────────────────
    try:
        _dt_conn = sqlite3.connect(DB_PATH)
        _dt_anomaly_count = _dt_conn.execute(
            "SELECT COUNT(*) FROM energy_embeddings WHERE is_anomaly=1"
        ).fetchone()[0]
        df_dt_health = pd.read_sql_query(
            "SELECT batch_id, recon_error, is_anomaly FROM energy_embeddings ORDER BY rowid",
            _dt_conn,
        )
        df_dt_genome_anom = pd.read_sql_query(
            """SELECT gv.batch_id, gv.genome, ee.is_anomaly
               FROM genome_vectors gv
               JOIN energy_embeddings ee ON gv.batch_id = ee.batch_id""",
            _dt_conn,
        )
        _dt_conn.close()
        df_dt_health["recon_error"] = pd.to_numeric(df_dt_health["recon_error"], errors="coerce")
        df_dt_health["is_anomaly"]  = pd.to_numeric(df_dt_health["is_anomaly"],  errors="coerce").fillna(0).astype(int)
        df_dt_genome_anom["is_anomaly"] = pd.to_numeric(df_dt_genome_anom["is_anomaly"], errors="coerce").fillna(0).astype(int)
    except Exception:
        _dt_anomaly_count = 0
        df_dt_health = pd.DataFrame(columns=["batch_id", "recon_error", "is_anomaly"])
        df_dt_genome_anom = pd.DataFrame(columns=["batch_id", "genome", "is_anomaly"])

    _dt_total      = db_summary.get("batches", 0)
    _dt_avg_yield  = df_preds["pred_yield"].mean() if len(df_preds) > 0 else 0.0
    _dt_avg_energy = df_batches["energy_consumption"].mean() if len(df_batches) > 0 else 0.0
    _dt_anom_rate  = _dt_anomaly_count / max(_dt_total, 1) * 100

    if _dt_anom_rate > 15:
        _dt_fstatus = "CRITICAL"; _dt_fc = _RED
        _dt_fbg = "rgba(255,75,75,0.08)"; _dt_fbd = "rgba(255,75,75,0.35)"; _dt_fi = "🔴"
    elif _dt_anom_rate > 7:
        _dt_fstatus = "DEGRADED"; _dt_fc = _YELLOW
        _dt_fbg = "rgba(255,214,0,0.08)"; _dt_fbd = "rgba(255,214,0,0.35)"; _dt_fi = "🟡"
    else:
        _dt_fstatus = "ONLINE"; _dt_fc = _GREEN
        _dt_fbg = "rgba(0,255,136,0.08)"; _dt_fbd = "rgba(0,255,136,0.35)"; _dt_fi = "🟢"

    _dt_latest = df_batches.iloc[-1] if len(df_batches) > 0 else None

    # ══ STATUS BANNER ══════════════════════════════════════════════════════════
    st.markdown(
        f'<div style="background:{_dt_fbg};border:1px solid {_dt_fbd};border-radius:12px;'
        f'padding:14px 24px;margin-bottom:18px;display:flex;justify-content:space-between;align-items:center;">'
        f'<div>'
        f'<div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.15em;color:rgba(255,255,255,0.35);">Digital Twin · Live Mirror</div>'
        f'<div style="font-size:1.5rem;font-weight:700;color:{_dt_fc};letter-spacing:0.06em;margin-top:2px;">'
        f'{_dt_fi} FACTORY {_dt_fstatus}</div>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'<div style="font-size:0.82rem;color:rgba(255,255,255,0.45);">'
        f'Mirroring {_dt_total:,} batches &nbsp;·&nbsp; {len(df_dt_health):,} embeddings analysed</div>'
        f'<div style="font-size:0.8rem;color:{_dt_fc};margin-top:4px;">'
        f'Anomaly Rate: {_dt_anom_rate:.1f}% &nbsp;·&nbsp; Active Zone: {ZONE_EMOJI[zone]} {zone} &nbsp;·&nbsp; CI: {carbon_val} gCO₂/kWh'
        f'</div></div></div>',
        unsafe_allow_html=True,
    )

    # ══ COMMAND PANEL KPIs ═════════════════════════════════════════════════════
    st.markdown('<div class="slabel">Command Panel</div>', unsafe_allow_html=True)
    _cp1, _cp2, _cp3, _cp4, _cp5 = st.columns(5)
    with _cp1:
        st.metric("🏭 Batches Processed", f"{_dt_total:,}", "production records")
    with _cp2:
        st.metric("⚠️ Anomalies Detected", f"{_dt_anomaly_count:,}",
                  f"{_dt_anom_rate:.1f}% rate", delta_color="inverse")
    with _cp3:
        st.metric("🎯 Fleet Avg Yield", f"{_dt_avg_yield:.4f}", "AI model")
    with _cp4:
        st.metric(f"{ZONE_EMOJI[zone]} Carbon Zone", zone, f"{carbon_val} gCO₂/kWh")
    with _cp5:
        st.metric("⚡ Avg Energy / Batch", f"{_dt_avg_energy:.0f} kWh", "fleet avg")

    st.markdown("<br>", unsafe_allow_html=True)

    # ══ LIVE FACTORY FLOOR ═════════════════════════════════════════════════════
    st.markdown('<div class="slabel">Live Factory Floor — 6-Station Production Line</div>',
                unsafe_allow_html=True)

    if _dt_latest is not None:
        _lat_id   = _dt_latest["batch_id"]
        _lat_anom = False
        if len(df_dt_health) > 0:
            _lmatch = df_dt_health[df_dt_health["batch_id"] == _lat_id]
            if len(_lmatch) > 0:
                _lat_anom = bool(_lmatch.iloc[0]["is_anomaly"])

        def _station_box(icon, name, rows, st_status="ok"):
            _sc = {
                "ok":   (_CYAN,   "rgba(0,212,255,0.10)",  "rgba(0,212,255,0.35)"),
                "warn": (_YELLOW, "rgba(255,214,0,0.09)",  "rgba(255,214,0,0.30)"),
                "crit": (_RED,    "rgba(255,75,75,0.09)",  "rgba(255,75,75,0.30)"),
            }[st_status]
            _dot = {"ok": _GREEN, "warn": _YELLOW, "crit": _RED}[st_status]
            _metrics = "".join([
                f'<div style="margin-top:6px;">'
                f'<div style="font-size:0.88rem;font-weight:700;color:{_sc[0]};">{v}'
                f'<span style="font-size:0.6rem;color:rgba(255,255,255,0.3);margin-left:2px;">{u}</span></div>'
                f'<div style="font-size:0.6rem;color:rgba(255,255,255,0.3);text-transform:uppercase;">{l}</div>'
                f'</div>'
                for l, v, u in rows
            ])
            return (
                f'<div style="flex:1;background:{_sc[1]};border:1.5px solid {_sc[2]};'
                f'border-radius:10px;padding:13px 8px;text-align:center;min-width:0;">'
                f'<div style="font-size:1.6rem;">{icon}</div>'
                f'<div style="font-size:0.63rem;font-weight:700;color:{_sc[0]};'
                f'text-transform:uppercase;letter-spacing:0.07em;margin:4px 0 3px;">{name}</div>'
                f'<div style="width:9px;height:9px;border-radius:50%;background:{_dot};'
                f'display:inline-block;box-shadow:0 0 7px {_dot};margin-bottom:6px;"></div>'
                f'{_metrics}</div>'
            )

        _arr = '<div style="display:flex;align-items:center;padding:0 5px;font-size:1.5rem;color:rgba(0,212,255,0.38);flex-shrink:0;">→</div>'

        _qval = float(_dt_latest.get("quality", 0.8))
        _qs = "crit" if _qval < 0.7 else ("warn" if _qval < 0.8 else "ok")
        _ds = "crit" if _lat_anom else "ok"

        st.markdown(
            '<div style="display:flex;align-items:stretch;gap:0;padding:4px 0;">'
            + _station_box("🏗️", "Raw Intake", [
                ("Temperature", f"{_dt_latest['temperature']:.1f}", "°C"),
                ("Humidity",    f"{_dt_latest['humidity']:.1f}",    "%"),
            ], "ok")
            + _arr
            + _station_box("⚙️", "Extrusion", [
                ("Pressure", f"{_dt_latest['pressure']:.2f}",  "bar"),
                ("Speed",    f"{_dt_latest['speed']:.0f}",     "rpm"),
            ], "ok")
            + _arr
            + _station_box("🧱", "Material Prep", [
                ("Feed Rate", f"{_dt_latest['feed_rate']:.2f}",         "kg/h"),
                ("Density",   f"{_dt_latest['material_density']:.3f}",  "g/cm³"),
            ], "ok")
            + _arr
            + _station_box("🔥", "Thermal Cure", [
                ("Hardness", f"{_dt_latest['material_hardness']:.1f}", "HV"),
                ("Grade",    f"{int(_dt_latest['material_grade'])}",   ""),
            ], "ok")
            + _arr
            + _station_box("🔬", "QC Scanner", [
                ("Quality", f"{_qval:.4f}",                        ""),
                ("Result",  "PASS" if _qval >= 0.7 else "FAIL",   ""),
            ], _qs)
            + _arr
            + _station_box("📦", "Dispatch", [
                ("Yield",   f"{_dt_latest['yield']:.4f}",                    ""),
                ("Energy",  f"{_dt_latest['energy_consumption']:.0f}",        "kWh"),
                ("Zone",    f"{ZONE_EMOJI[_dt_latest['zone']]} {_dt_latest['zone']}", ""),
            ], _ds)
            + '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:0.71rem;color:rgba(255,255,255,0.26);margin-top:7px;">'
            f'Mirroring latest batch: <span style="color:{_CYAN};">{_lat_id}</span>'
            f' &nbsp;·&nbsp; CI: {_dt_latest["carbon_intensity"]:.1f} gCO₂/kWh'
            f' &nbsp;·&nbsp; {ZONE_EMOJI[_dt_latest["zone"]]} {_dt_latest["zone"]} zone'
            f' &nbsp;·&nbsp; Anomaly: {"⚠️ YES" if _lat_anom else "✅ CLEAR"}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No batch data to display.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ══ CARBON-ZONE INTELLIGENCE ═══════════════════════════════════════════════
    st.markdown('<div class="slabel">Carbon-Aware Zone Intelligence — How the Factory Adapts</div>',
                unsafe_allow_html=True)

    _zrecs = {}
    for _zn, _zci in [("LOW", 100.0), ("MEDIUM", 250.0), ("HIGH", 500.0)]:
        try:
            _zrecs[_zn] = get_recommendation(float(_zci)).get("recommended_schedule", {})
        except Exception:
            _zrecs[_zn] = {}

    _zc1, _zc2, _zc3 = st.columns(3)
    for _zcol, _zn, _zci in zip([_zc1, _zc2, _zc3],
                                  ["LOW",  "MEDIUM", "HIGH"],
                                  [100,    250,      500]):
        _zr = _zrecs[_zn]
        _is_cur = (zone == _zn)
        _glow = f"box-shadow:0 0 20px {ZONE_COLORS[_zn]}33;" if _is_cur else ""
        with _zcol:
            st.markdown(
                f'<div style="background:{ZONE_BG[_zn]};border:1px solid {ZONE_BORDER[_zn]};'
                f'border-radius:12px;padding:16px 14px;{_glow}">'
                f'<div style="text-align:center;margin-bottom:12px;">'
                f'<div style="font-size:1.6rem;">{ZONE_EMOJI[_zn]}</div>'
                f'<div style="font-size:1.05rem;font-weight:700;color:{ZONE_COLORS[_zn]};">{_zn} CARBON</div>'
                f'<div style="font-size:0.75rem;color:rgba(255,255,255,0.38);">{_zci} gCO₂/kWh</div>'
                + (f'<div style="font-size:0.62rem;color:{ZONE_COLORS[_zn]};border:1px solid {ZONE_COLORS[_zn]};'
                   f'border-radius:4px;padding:2px 8px;display:inline-block;margin-top:5px;">▶ ACTIVE NOW</div>'
                   if _is_cur else '')
                + '</div>'
                + ''.join([
                    f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
                    f'border-bottom:1px solid rgba(255,255,255,0.05);">'
                    f'<span style="font-size:0.74rem;color:rgba(255,255,255,0.38);">{lbl}</span>'
                    f'<span style="font-size:0.77rem;font-weight:600;color:{ZONE_COLORS[_zn]};'
                    f'font-family:\'JetBrains Mono\',monospace;">{val}</span></div>'
                    for lbl, val in [
                        ("Pred Yield",   f"{_zr.get('pred_yield',  0):.4f}"),
                        ("Pred Quality", f"{_zr.get('pred_quality',0):.4f}"),
                        ("Pred Energy",  f"{_zr.get('pred_energy', 0):.0f} kWh"),
                        ("Pred Carbon",  f"{_zr.get('pred_carbon', 0):.1f} kg"),
                        ("Temperature",  f"{_zr.get('temperature', 0):.1f} °C"),
                        ("Speed",        f"{_zr.get('speed',       0):.0f} rpm"),
                        ("Pressure",     f"{_zr.get('pressure',    0):.2f} bar"),
                        ("Feed Rate",    f"{_zr.get('feed_rate',   0):.2f} kg/h"),
                    ]
                ])
                + '</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ══ BATCH FLOW SIMULATOR ═══════════════════════════════════════════════════
    st.markdown('<div class="slabel">Batch Flow Simulator — Step-Through All 6 Production Stations</div>',
                unsafe_allow_html=True)

    _sim_all_ids = df_batches["batch_id"].tolist()
    _bsim_c1, _bsim_c2 = st.columns([3, 7])
    with _bsim_c1:
        _bsim_id = st.selectbox("Batch to simulate", _sim_all_ids[:200], index=0,
                                key="dt_sim_bid", label_visibility="collapsed")
    with _bsim_c2:
        _run_bsim = st.button("▶  Simulate This Batch Through Factory", key="dt_run_bsim")

    _bsim_ph = st.empty()

    _SIM_STATIONS_DEF = [
        ("🏗️", "RAW INTAKE",    [("temperature","°C"),       ("humidity","%")]),
        ("⚙️", "EXTRUSION",     [("pressure","bar"),          ("speed","rpm")]),
        ("🧱", "MATERIAL PREP", [("feed_rate","kg/h"),        ("material_density","g/cm³")]),
        ("🔥", "THERMAL CURE",  [("material_hardness","HV"),  ("material_grade","")]),
        ("🔬", "QC SCANNER",    [("quality",""),              ("yield","")]),
        ("📦", "DISPATCH",      [("energy_consumption","kWh"),("carbon_intensity","gCO₂/kWh")]),
    ]

    def _bsim_row_html(brow, active):
        _h = '<div style="display:flex;gap:5px;align-items:stretch;">'
        for _si, (_ic, _sn, _flds) in enumerate(_SIM_STATIONS_DEF):
            if _si < active:
                _bg, _bc, _tc, _lbl = "rgba(0,255,136,0.07)", "rgba(0,255,136,0.35)", _GREEN, "✓ DONE"
            elif _si == active:
                _bg, _bc, _tc, _lbl = "rgba(0,212,255,0.13)", _CYAN, _CYAN, "▶ ACTIVE"
            else:
                _bg, _bc, _tc, _lbl = "rgba(255,255,255,0.02)", "rgba(255,255,255,0.1)", "rgba(255,255,255,0.25)", "PENDING"
            _mh = ""
            if _si <= active:
                for _fk, _fu in _flds:
                    _fv = brow.get(_fk, 0) if hasattr(brow, "get") else getattr(brow, _fk, 0)
                    try:
                        _fvs = f"{float(_fv):.4g}"
                    except Exception:
                        _fvs = str(_fv)
                    _mh += (f'<div style="font-size:0.82rem;font-weight:700;color:{_tc};">{_fvs}'
                            f'<span style="font-size:0.58rem;color:rgba(255,255,255,0.3);margin-left:2px;">{_fu}</span></div>')
            _h += (f'<div style="flex:1;background:{_bg};border:1.5px solid {_bc};border-radius:9px;'
                   f'padding:11px 6px;text-align:center;min-width:0;">'
                   f'<div style="font-size:1.3rem;">{_ic}</div>'
                   f'<div style="font-size:0.6rem;font-weight:700;color:{_tc};text-transform:uppercase;'
                   f'letter-spacing:0.06em;margin:3px 0;">{_sn}</div>'
                   f'<div style="font-size:0.58rem;color:{_tc};margin-bottom:5px;">{_lbl}</div>'
                   f'{_mh}</div>')
            if _si < len(_SIM_STATIONS_DEF) - 1:
                _h += '<div style="display:flex;align-items:center;font-size:1.1rem;color:rgba(0,212,255,0.35);flex-shrink:0;padding:0 2px;">→</div>'
        return _h + '</div>'

    if _run_bsim:
        _sbdf = df_batches[df_batches["batch_id"] == _bsim_id]
        if len(_sbdf) > 0:
            _sbr = _sbdf.iloc[0]
            for _step in range(len(_SIM_STATIONS_DEF) + 1):
                _active = min(_step, len(_SIM_STATIONS_DEF) - 1)
                _energy_pct = _step / len(_SIM_STATIONS_DEF)
                _estep = float(_sbr["energy_consumption"]) * _energy_pct
                _result_html = ""
                if _step == len(_SIM_STATIONS_DEF):
                    _rz = _sbr["zone"]
                    _result_html = (
                        f'<div style="background:rgba(0,255,136,0.07);border:1px solid rgba(0,255,136,0.3);'
                        f'border-radius:10px;padding:14px 20px;margin-top:12px;'
                        f'display:flex;gap:28px;flex-wrap:wrap;align-items:center;">'
                        f'<div><div style="font-size:0.62rem;color:rgba(255,255,255,0.35);text-transform:uppercase;">Final Yield</div>'
                        f'<div style="font-size:1.3rem;font-weight:700;color:{_GREEN};">{float(_sbr["yield"]):.4f}</div></div>'
                        f'<div><div style="font-size:0.62rem;color:rgba(255,255,255,0.35);text-transform:uppercase;">Quality</div>'
                        f'<div style="font-size:1.3rem;font-weight:700;color:{_CYAN};">{float(_sbr["quality"]):.4f}</div></div>'
                        f'<div><div style="font-size:0.62rem;color:rgba(255,255,255,0.35);text-transform:uppercase;">Energy Used</div>'
                        f'<div style="font-size:1.3rem;font-weight:700;color:{_YELLOW};">{float(_sbr["energy_consumption"]):.0f} kWh</div></div>'
                        f'<div><div style="font-size:0.62rem;color:rgba(255,255,255,0.35);text-transform:uppercase;">Carbon Int.</div>'
                        f'<div style="font-size:1.3rem;font-weight:700;color:{_RED};">{float(_sbr["carbon_intensity"]):.1f}</div></div>'
                        f'<div><div style="font-size:0.62rem;color:rgba(255,255,255,0.35);text-transform:uppercase;">Zone</div>'
                        f'<div style="font-size:1.3rem;font-weight:700;color:{ZONE_COLORS[_rz]};">'
                        f'{ZONE_EMOJI[_rz]} {_rz}</div></div>'
                        f'</div>'
                    )
                _bsim_ph.markdown(
                    f'<div style="background:rgba(0,0,0,0.3);border:1px solid rgba(0,212,255,0.15);'
                    f'border-radius:12px;padding:16px;">'
                    f'<div style="font-size:0.68rem;color:rgba(255,255,255,0.3);margin-bottom:8px;">'
                    f'Batch: <span style="color:{_CYAN};">{_bsim_id}</span>'
                    f' &nbsp;·&nbsp; Step {min(_step+1,len(_SIM_STATIONS_DEF))}/{len(_SIM_STATIONS_DEF)}'
                    f' &nbsp;·&nbsp; Energy consumed so far: <span style="color:{_YELLOW};">{_estep:.0f} kWh</span>'
                    f'</div>'
                    + _bsim_row_html(_sbr, _active)
                    + _result_html
                    + '</div>',
                    unsafe_allow_html=True,
                )
                time.sleep(1.1)
        else:
            _bsim_ph.warning(f"Batch {_bsim_id} not found.")
    else:
        _bsim_ph.markdown(
            '<div style="font-size:0.8rem;color:rgba(255,255,255,0.3);padding:12px;">'
            'Select a batch and click ▶ Simulate to watch it step through all 6 stations '
            'with real process parameters, energy usage, and final outcome.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ══ MACHINE HEALTH MONITOR ═════════════════════════════════════════════════
    st.markdown('<div class="slabel">Machine Health Monitor — Autoencoder Reconstruction Error</div>',
                unsafe_allow_html=True)

    _mh_l, _mh_r = st.columns([6, 4])

    with _mh_l:
        if len(df_dt_health) > 0:
            _dh100    = df_dt_health.tail(100).reset_index(drop=True)
            _thresh95 = float(df_dt_health["recon_error"].quantile(0.95))
            _fh = go.Figure()
            _fh.add_trace(go.Scatter(
                x=_dh100.index.tolist(), y=_dh100["recon_error"].tolist(),
                mode="lines", line=dict(color=_CYAN, width=1.8),
                fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
                name="Recon Error",
                hovertemplate="Batch %{x}<br>Recon Error: %{y:.6f}<extra></extra>",
            ))
            _am = _dh100["is_anomaly"] == 1
            if _am.any():
                _fh.add_trace(go.Scatter(
                    x=_dh100[_am].index.tolist(), y=_dh100[_am]["recon_error"].tolist(),
                    mode="markers",
                    marker=dict(color=_RED, size=10, symbol="x", line=dict(color=_RED, width=2)),
                    name="⚠️ Anomaly",
                    hovertemplate="⚠️ ANOMALY<br>Batch %{x}<br>Error: %{y:.6f}<extra></extra>",
                ))
            _fh.add_hline(y=_thresh95, line=dict(color=_RED, dash="dash", width=1.5),
                          annotation_text=f"95th pct ({_thresh95:.5f})",
                          annotation_font=dict(color=_RED, size=9))
            dark_layout(_fh, height=340)
            _fh.update_layout(
                title=dict(text="Reconstruction Error — Last 100 Batches  (× = anomaly)",
                           font=dict(size=12, color="rgba(255,255,255,0.55)")),
                legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)",
                            borderwidth=1, font=dict(size=10)),
            )
            st.plotly_chart(_fh, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No health data available.")

    with _mh_r:
        _ta = int((df_dt_health["is_anomaly"] == 1).sum()) if len(df_dt_health) > 0 else 0
        _te = len(df_dt_health)
        _ar = _ta / max(_te, 1) * 100
        _hsc = _RED if _ar > 15 else (_YELLOW if _ar > 7 else _GREEN)
        _t95 = float(df_dt_health["recon_error"].quantile(0.95)) if _te > 0 else 0.0
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);'
            f'border-radius:10px;padding:14px;margin-bottom:12px;">'
            f'<div style="font-size:0.65rem;color:rgba(255,255,255,0.32);text-transform:uppercase;'
            f'letter-spacing:0.1em;margin-bottom:8px;">Health Summary</div>'
            + ''.join([
                f'<div style="display:flex;justify-content:space-between;padding:5px 0;'
                f'border-bottom:1px solid rgba(255,255,255,0.05);">'
                f'<span style="font-size:0.77rem;color:rgba(255,255,255,0.4);">{lb}</span>'
                f'<span style="font-size:0.8rem;font-weight:600;color:{co};">{vl}</span></div>'
                for lb, vl, co in [
                    ("Total Embeddings",  f"{_te:,}",        _CYAN),
                    ("Anomalies Flagged", f"{_ta:,}",        _RED),
                    ("Anomaly Rate",      f"{_ar:.2f}%",     _hsc),
                    ("95th pct Threshold",f"{_t95:.5f}",     _YELLOW),
                    ("Factory Status",    _dt_fstatus,       _dt_fc),
                ]
            ])
            + '</div>',
            unsafe_allow_html=True,
        )
        if _ta > 0:
            _df_at = (df_dt_health[df_dt_health["is_anomaly"] == 1][["batch_id", "recon_error"]]
                      .sort_values("recon_error", ascending=False).head(15).copy())
            _df_at.columns = ["Batch ID", "Recon Error"]
            _df_at["Recon Error"] = _df_at["Recon Error"].round(6)
            st.markdown('<div style="font-size:0.65rem;color:rgba(255,255,255,0.3);'
                        'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">'
                        '⚠️ Top Anomalous Batches by Error</div>', unsafe_allow_html=True)
            st.dataframe(_df_at, use_container_width=True, hide_index=True, height=220)
        else:
            st.success("✅ No anomalies detected.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ══ GENOME ANOMALY FINGERPRINT ═════════════════════════════════════════════
    st.markdown('<div class="slabel">Genome Anomaly Fingerprint — 25-Dimension Manufacturing DNA</div>',
                unsafe_allow_html=True)

    if len(df_dt_genome_anom) > 0:
        _dfng = df_dt_genome_anom[df_dt_genome_anom["is_anomaly"] == 0]
        _dfag = df_dt_genome_anom[df_dt_genome_anom["is_anomaly"] == 1]
        _gnorm = np.mean([json.loads(g) for g in _dfng["genome"]], axis=0) if len(_dfng) > 0 else np.zeros(25)
        _ganom = np.mean([json.loads(g) for g in _dfag["genome"]], axis=0) if len(_dfag) > 0 else np.zeros(25)

        def _gbar(vals, title, col):
            _f = go.Figure(go.Bar(
                x=GENOME_LABELS, y=vals.tolist(),
                marker=dict(color=col, opacity=0.85, line=dict(width=0)),
                hovertemplate="<b>%{x}</b><br>z-score: %{y:.4f}<extra></extra>",
            ))
            _f.add_hline(y=0, line=dict(color="rgba(255,255,255,0.15)", dash="dot", width=1))
            _f.update_layout(
                title=dict(text=title, font=dict(size=11, color="rgba(255,255,255,0.55)")),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
                font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
                xaxis=dict(tickangle=60, tickfont=dict(size=8), gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(title="z-score", gridcolor="rgba(255,255,255,0.07)",
                           zerolinecolor="rgba(255,255,255,0.15)"),
                height=310, margin=dict(l=50, r=10, t=44, b=72),
                showlegend=False, bargap=0.25,
            )
            return _f

        _gfa, _gfb = st.columns(2)
        with _gfa:
            st.plotly_chart(_gbar(_gnorm, f"✅ Normal Batches ({len(_dfng)}) — Mean 25D Genome", _CYAN),
                            use_container_width=True, config={"displayModeBar": False})
        with _gfb:
            st.plotly_chart(_gbar(_ganom, f"⚠️ Anomalous Batches ({len(_dfag)}) — Mean 25D Genome", _RED),
                            use_container_width=True, config={"displayModeBar": False})

        _gdiff  = _ganom - _gnorm
        _gdc    = [_RED if v > 0 else _GREEN for v in _gdiff]
        _gfd    = go.Figure(go.Bar(
            x=GENOME_LABELS, y=_gdiff.tolist(),
            marker=dict(color=_gdc, opacity=0.85, line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>Δ z-score: %{y:.4f}<extra></extra>",
        ))
        _gfd.add_hline(y=0, line=dict(color="rgba(255,255,255,0.15)", dash="dot", width=1))
        _gfd.update_layout(
            title=dict(
                text="Δ Genome Delta: Anomalous − Normal  (Red = elevated in anomalies  ·  Green = suppressed)",
                font=dict(size=11, color="rgba(255,255,255,0.55)"),
            ),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
            font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
            xaxis=dict(tickangle=60, tickfont=dict(size=8), gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(title="Δ z-score", gridcolor="rgba(255,255,255,0.07)",
                       zerolinecolor="rgba(255,255,255,0.15)"),
            height=295, margin=dict(l=50, r=10, t=44, b=72), showlegend=False, bargap=0.25,
        )
        st.plotly_chart(_gfd, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Genome anomaly data unavailable.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ══ 24h CARBON GRID SIMULATION ═════════════════════════════════════════════
    st.markdown('<div class="slabel">24h Carbon Grid Simulation — AI-Driven Adaptive Scheduling</div>',
                unsafe_allow_html=True)

    _24h_btn   = st.button("▶  Animate 24h Carbon Forecast", key="dt_24h_anim")
    _24h_chart = st.empty()
    _24h_badge = st.empty()

    def _mk24(hx, cy, suffix=""):
        _zc = [ZONE_COLORS[classify_carbon_zone(float(c))] for c in cy]
        _fg = go.Figure()
        _fg.add_trace(go.Scatter(
            x=hx, y=cy, mode="lines+markers",
            line=dict(color=_CYAN, width=2),
            marker=dict(color=_zc, size=9, line=dict(color="rgba(0,0,0,0.4)", width=1)),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.07)",
            hovertemplate="Hour %{x}:00 — %{y} gCO₂/kWh<extra></extra>",
        ))
        _fg.add_hline(y=CARBON_LOW_THRESHOLD, line=dict(color=_GREEN, dash="dash", width=1),
                      annotation_text=f"LOW ≤{CARBON_LOW_THRESHOLD}",
                      annotation_font=dict(color=_GREEN, size=9))
        _fg.add_hline(y=CARBON_HIGH_THRESHOLD, line=dict(color=_RED, dash="dash", width=1),
                      annotation_text=f"HIGH >{CARBON_HIGH_THRESHOLD}",
                      annotation_font=dict(color=_RED, size=9))
        for _hh, _ci in enumerate(cy):
            _zz = classify_carbon_zone(float(_ci))
            _zfill = {"LOW":"rgba(0,255,136,0.06)","MEDIUM":"rgba(255,214,0,0.04)","HIGH":"rgba(255,75,75,0.06)"}[_zz]
            _fg.add_vrect(x0=_hh-0.5, x1=_hh+0.5, fillcolor=_zfill, line_width=0, layer="below")
        _fg.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
            font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
            xaxis=dict(title="Hour of Day", gridcolor="rgba(255,255,255,0.07)",
                       range=[-0.5, 23.5], tickvals=list(range(0, 24, 2)),
                       ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)]),
            yaxis=dict(title="Carbon Intensity (gCO₂/kWh)", gridcolor="rgba(255,255,255,0.07)"),
            title=dict(text=f"Simulated 24h Carbon Grid Load{suffix}",
                       font=dict(size=12, color="rgba(255,255,255,0.55)")),
            height=320, margin=dict(l=55, r=20, t=44, b=44), showlegend=False,
        )
        return _fg

    if _24h_btn:
        _hacc, _cacc = [], []
        for _hr in range(24):
            _hacc.append(_hr); _cacc.append(CARBON_24H[_hr])
            _sz = classify_carbon_zone(float(CARBON_24H[_hr]))
            _24h_chart.plotly_chart(_mk24(_hacc, _cacc, f" — Hour {_hr:02d}:00"),
                                    use_container_width=True, config={"displayModeBar": False})
            try:
                _hrec = get_recommendation(float(CARBON_24H[_hr]))
                _hrs  = _hrec.get("recommended_schedule", {})
                _hry  = _hrs.get("pred_yield", 0.0)
                _hre  = _hrs.get("pred_energy", 0.0)
            except Exception:
                _hry = _hre = 0.0
            _24h_badge.markdown(
                f'<div style="background:{ZONE_BG[_sz]};border:1px solid {ZONE_BORDER[_sz]};'
                f'border-radius:8px;padding:8px 20px;display:inline-flex;gap:20px;align-items:center;">'
                f'<span style="font-weight:700;font-size:1.0rem;color:{ZONE_COLORS[_sz]};">'
                f'{ZONE_EMOJI[_sz]}  Hour {_hr:02d}:00 — {_sz}</span>'
                f'<span style="font-size:0.8rem;color:rgba(255,255,255,0.45);">'
                f'CI = {CARBON_24H[_hr]} gCO₂/kWh &nbsp;·&nbsp; '
                f'Rec Yield = {_hry:.4f} &nbsp;·&nbsp; Rec Energy = {_hre:.0f} kWh</span></div>',
                unsafe_allow_html=True,
            )
            time.sleep(0.8)
        _24h_badge.markdown(
            f'<div style="font-size:0.8rem;color:{_GREEN};margin-top:6px;">'
            f'✅  24h simulation complete — {len(CARBON_24H)} hourly AI recommendations rendered.</div>',
            unsafe_allow_html=True,
        )
    else:
        _24h_chart.plotly_chart(_mk24(list(range(24)), CARBON_24H),
                                use_container_width=True, config={"displayModeBar": False})
        _24h_badge.markdown(
            '<div style="font-size:0.79rem;color:rgba(255,255,255,0.32);margin-top:2px;">'
            '↑ Click ▶ to animate hour by hour — AI recommends an optimal schedule at each step.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ══ WHAT-IF PIPELINE SIMULATOR ════════════════════════════════════════════
    st.markdown(
        '<div class="slabel">🔬 What-If Pipeline Simulator — In-Memory · No DB Write</div>',
        unsafe_allow_html=True,
    )

    _wi_col_l, _wi_col_r = st.columns([2, 3], gap="large")

    with _wi_col_l:
        st.markdown(
            '<div style="font-size:0.76rem;font-weight:600;color:rgba(255,255,255,0.45);'
            'letter-spacing:0.1em;margin:0 0 6px 0;">PROCESS PARAMETERS</div>',
            unsafe_allow_html=True,
        )
        _wi_temp     = st.slider("Temperature (°C)",       100,  300, 180,        key="wi_temp")
        _wi_pressure = st.slider("Pressure (bar)",         1.0, 10.0,  5.0, 0.1,  key="wi_pressure")
        _wi_speed    = st.slider("Speed (rpm)",             50,  300, 150,        key="wi_speed")
        _wi_feed     = st.slider("Feed Rate (kg/h)",        5.0, 50.0, 20.0, 0.5, key="wi_feed")
        _wi_humidity = st.slider("Humidity (%)",            20,   80,  45,        key="wi_humidity")
        st.markdown(
            '<div style="font-size:0.76rem;font-weight:600;color:rgba(255,255,255,0.45);'
            'letter-spacing:0.1em;margin:10px 0 6px 0;">MATERIAL PROPERTIES</div>',
            unsafe_allow_html=True,
        )
        _wi_density  = st.slider("Density (g/cm³)",        0.8,  3.5,  2.0, 0.1, key="wi_density")
        _wi_hardness = st.slider("Hardness (HRC)",          10,  100,  55,        key="wi_hardness")
        _wi_grade    = st.slider("Grade (1-10)",             1,   10,   5,        key="wi_grade")
        st.markdown(
            '<div style="font-size:0.76rem;font-weight:600;color:rgba(255,255,255,0.45);'
            'letter-spacing:0.1em;margin:10px 0 6px 0;">ENVIRONMENT</div>',
            unsafe_allow_html=True,
        )
        _wi_carbon = st.slider("Carbon Intensity (gCO₂/kWh)", 0, 600, 220, key="wi_carbon")
        st.markdown("<br>", unsafe_allow_html=True)
        _wi_run = st.button(
            "▶  Run Full Pipeline Simulation",
            key="wi_run", use_container_width=True,
        )

    with _wi_col_r:
        if _wi_run:
            _wi_status = st.empty()
            _wi_prog   = st.progress(0)
            _wi_result = st.empty()
            try:
                import pickle
                import torch
                from src.energy_dna.model import LSTMAutoencoder
                from config.settings import (
                    MODELS_DIR, ENERGY_INPUT_DIM, ENERGY_HIDDEN_DIM,
                    ENERGY_LATENT_DIM, ENERGY_NUM_LAYERS,
                )

                # ── Stage 1 : Synthetic energy signal ─────────────────────────
                _wi_status.markdown(
                    '<div style="font-size:0.82rem;color:#00d4ff;">'
                    '⚙️  Stage 1 / 5 — Generating energy signal...</div>',
                    unsafe_allow_html=True,
                )
                _wi_prog.progress(12)
                time.sleep(0.35)
                _rng   = np.random.default_rng(seed=int(_wi_temp * 100 + _wi_speed))
                _base  = 50 + (_wi_temp / 300) * 80 + (_wi_speed / 300) * 50 + (_wi_feed / 50) * 20
                _t_ax  = np.linspace(0, 4 * np.pi, 128)
                _sig   = _base + 10 * np.sin(_t_ax) + _rng.normal(0, 5, 128)

                # ── Stage 2 : LSTM Energy DNA encoding ────────────────────────
                _wi_status.markdown(
                    '<div style="font-size:0.82rem;color:#00d4ff;">'
                    '🔋  Stage 2 / 5 — Encoding Energy DNA via LSTM Autoencoder...</div>',
                    unsafe_allow_html=True,
                )
                _wi_prog.progress(30)
                time.sleep(0.45)
                _sig_mean = _sig.mean()
                _sig_std  = float(_sig.std()) if float(_sig.std()) > 0 else 1.0
                _sig_norm = (_sig - _sig_mean) / _sig_std

                _ae_path = os.path.join(MODELS_DIR, "lstm_autoencoder.pth")
                if os.path.exists(_ae_path):
                    _ae_wi = LSTMAutoencoder(
                        ENERGY_INPUT_DIM, ENERGY_HIDDEN_DIM,
                        ENERGY_LATENT_DIM, ENERGY_NUM_LAYERS,
                    )
                    _ae_wi.load_state_dict(
                        torch.load(_ae_path, map_location="cpu", weights_only=True)
                    )
                    _ae_wi.eval()
                    with torch.no_grad():
                        _x_t   = torch.tensor(
                            _sig_norm, dtype=torch.float32
                        ).unsqueeze(0).unsqueeze(2)          # (1, 128, 1)
                        _recon_t, _latent_t = _ae_wi(_x_t)
                        _emb_wi    = _latent_t.squeeze(0).numpy()          # (16,)
                        _recon_err = float(
                            torch.mean((_recon_t - _x_t) ** 2).item()
                        )
                else:
                    _emb_wi    = np.zeros(16, dtype=np.float32)
                    _recon_err = 0.0

                # ── Stage 3 : Genome assembly ──────────────────────────────────
                _wi_status.markdown(
                    '<div style="font-size:0.82rem;color:#00d4ff;">'
                    '🧬  Stage 3 / 5 — Assembling Batch Genome (25-D)...</div>',
                    unsafe_allow_html=True,
                )
                _wi_prog.progress(52)
                time.sleep(0.4)
                _genome_wi = np.array(
                    [
                        float(_wi_temp), float(_wi_pressure), float(_wi_speed),
                        float(_wi_feed), float(_wi_humidity),
                        float(_wi_density), float(_wi_hardness), float(_wi_grade),
                        *_emb_wi.tolist(),
                        float(_wi_carbon),
                    ],
                    dtype=np.float32,
                )  # (25,)

                # ── Stage 4 : Prediction model ─────────────────────────────────
                _wi_status.markdown(
                    '<div style="font-size:0.82rem;color:#00d4ff;">'
                    '🔮  Stage 4 / 5 — Running Multi-Target Prediction Model...</div>',
                    unsafe_allow_html=True,
                )
                _wi_prog.progress(72)
                time.sleep(0.4)
                _pred_path = os.path.join(MODELS_DIR, "predictor.pkl")
                if os.path.exists(_pred_path):
                    with open(_pred_path, "rb") as _pf:
                        _pred_wi = pickle.load(_pf)
                    _preds_wi = _pred_wi.predict(_genome_wi.reshape(1, -1))[0]
                    _py = float(_preds_wi[0])
                    _pq = float(_preds_wi[1])
                    _pe = float(_preds_wi[2])
                else:
                    _py = 0.5 + (_wi_temp  / 300) * 0.4
                    _pq = 0.4 + (_wi_speed / 300) * 0.5
                    _pe = 50  + (_wi_temp  / 300) * 450

                # ── Stage 5 : Risk assessment ──────────────────────────────────
                _wi_status.markdown(
                    '<div style="font-size:0.82rem;color:#00d4ff;">'
                    '🌍  Stage 5 / 5 — Carbon Zone & Anomaly Risk Assessment...</div>',
                    unsafe_allow_html=True,
                )
                _wi_prog.progress(92)
                time.sleep(0.35)

                _wi_zone   = classify_carbon_zone(float(_wi_carbon))
                _wi_anomaly = _recon_err > 0.199084
                _anom_lbl   = "⚠️  ANOMALY DETECTED"   if _wi_anomaly else "✅  NORMAL OPERATION"
                _anom_col   = "#ff4b4b"                 if _wi_anomaly else "#00ff88"
                _zc         = ZONE_COLORS[_wi_zone]

                _wi_prog.progress(100)
                time.sleep(0.15)
                _wi_status.empty()
                _wi_prog.empty()

                # ── Result card ────────────────────────────────────────────────
                _wi_result.markdown(
                    f'<div style="background:linear-gradient(135deg,'
                    f'rgba(0,212,255,0.07),rgba(0,255,136,0.04));'
                    f'border:1px solid rgba(0,212,255,0.25);border-radius:16px;'
                    f'padding:22px 24px;">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;margin-bottom:16px;">'
                    f'<div style="font-size:1.0rem;font-weight:700;color:#00d4ff;'
                    f'letter-spacing:0.05em;">SIMULATION RESULT</div>'
                    f'<div style="font-size:0.72rem;padding:3px 14px;border-radius:20px;'
                    f'border:1px solid {_anom_col};color:{_anom_col};font-weight:600;">'
                    f'{_anom_lbl}</div></div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;'
                    f'gap:10px;margin-bottom:12px;">'
                    f'<div style="background:rgba(255,255,255,0.04);border-radius:10px;'
                    f'padding:12px;text-align:center;">'
                    f'<div style="font-size:0.62rem;color:rgba(255,255,255,0.36);'
                    f'text-transform:uppercase;letter-spacing:0.1em;">Pred. Yield</div>'
                    f'<div style="font-size:1.45rem;font-weight:700;color:#00d4ff;'
                    f'font-family:\'JetBrains Mono\',monospace;">{_py:.4f}</div></div>'
                    f'<div style="background:rgba(255,255,255,0.04);border-radius:10px;'
                    f'padding:12px;text-align:center;">'
                    f'<div style="font-size:0.62rem;color:rgba(255,255,255,0.36);'
                    f'text-transform:uppercase;letter-spacing:0.1em;">Pred. Quality</div>'
                    f'<div style="font-size:1.45rem;font-weight:700;color:#00ff88;'
                    f'font-family:\'JetBrains Mono\',monospace;">{_pq:.4f}</div></div>'
                    f'<div style="background:rgba(255,255,255,0.04);border-radius:10px;'
                    f'padding:12px;text-align:center;">'
                    f'<div style="font-size:0.62rem;color:rgba(255,255,255,0.36);'
                    f'text-transform:uppercase;letter-spacing:0.1em;">Pred. Energy</div>'
                    f'<div style="font-size:1.45rem;font-weight:700;color:#ffd600;'
                    f'font-family:\'JetBrains Mono\',monospace;">{_pe:.0f} kWh</div></div>'
                    f'</div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;'
                    f'gap:10px;margin-bottom:14px;">'
                    f'<div style="background:rgba(255,255,255,0.04);border-radius:10px;'
                    f'padding:12px;text-align:center;">'
                    f'<div style="font-size:0.62rem;color:rgba(255,255,255,0.36);'
                    f'text-transform:uppercase;letter-spacing:0.1em;">Carbon Zone</div>'
                    f'<div style="font-size:1.05rem;font-weight:700;color:{_zc};'
                    f'margin-top:3px;">{ZONE_EMOJI[_wi_zone]}  {_wi_zone}</div></div>'
                    f'<div style="background:rgba(255,255,255,0.04);border-radius:10px;'
                    f'padding:12px;text-align:center;">'
                    f'<div style="font-size:0.62rem;color:rgba(255,255,255,0.36);'
                    f'text-transform:uppercase;letter-spacing:0.1em;">Anomaly Score</div>'
                    f'<div style="font-size:1.05rem;font-weight:700;color:{_anom_col};'
                    f'font-family:\'JetBrains Mono\',monospace;margin-top:3px;">'
                    f'{_recon_err:.5f}</div></div>'
                    f'</div>'
                    f'<div style="font-size:0.69rem;color:rgba(255,255,255,0.25);'
                    f'border-top:1px solid rgba(255,255,255,0.07);padding-top:9px;">'
                    f'Genome [{chr(34)} {chr(34).join(f"{v:.2f}" for v in _genome_wi[:5])}'  
                    f' … {_genome_wi[24]:.0f}] '
                    f'&nbsp;·&nbsp; Threshold = 0.1991 '
                    f'&nbsp;·&nbsp; In-memory only — DB unchanged</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # ── Genome fingerprint bar chart ───────────────────────────────
                _wi_norm_g = _genome_wi / (np.abs(_genome_wi).max() or 1)
                _wi_fg = go.Figure(go.Bar(
                    x=GENOME_LABELS, y=_genome_wi.tolist(),
                    marker=dict(
                        color=_wi_norm_g.tolist(),
                        colorscale=[[0, "#0050c8"], [0.5, "#00d4ff"], [1, "#00ff88"]],
                        showscale=False,
                    ),
                ))
                _wi_fg.update_layout(
                    title="Simulated Batch Genome Fingerprint (25-D)",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "rgba(255,255,255,0.65)", "family": "Inter"},
                    height=210, margin=dict(l=10, r=10, t=34, b=5),
                    xaxis=dict(tickfont=dict(size=8.5), gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                )
                st.plotly_chart(_wi_fg, use_container_width=True,
                                config={"displayModeBar": False})

            except Exception as _wi_ex:
                _wi_prog.empty()
                _wi_status.error(f"Simulation error: {_wi_ex}")
        else:
            st.markdown(
                '<div style="display:flex;align-items:center;justify-content:center;'
                'padding:60px 24px;text-align:center;'
                'background:rgba(255,255,255,0.02);'
                'border:1px dashed rgba(255,255,255,0.1);border-radius:14px;">'
                '<div>'
                '<div style="font-size:2.4rem;margin-bottom:10px;">🔬</div>'
                '<div style="font-size:0.92rem;font-weight:600;'
                'color:rgba(255,255,255,0.42);">What-If Simulator Ready</div>'
                '<div style="font-size:0.75rem;color:rgba(255,255,255,0.22);margin-top:8px;">'
                'Set process parameters on the left, then click<br>'
                '<strong style="color:rgba(0,212,255,0.55);">'
                '▶ Run Full Pipeline Simulation</strong></div>'
                '</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ══ REAL-TIME BATCH STREAM ═════════════════════════════════════════════════
    st.markdown('<div class="slabel">Real-Time Batch Stream — Last 20 Dispatched</div>',
                unsafe_allow_html=True)

    _str_c1, _str_c2 = st.columns([1, 9])
    with _str_c1:
        if st.button("🔄 Refresh", key="dt_refresh"):
            st.cache_data.clear()
            st.rerun()

    try:
        _stconn = sqlite3.connect(DB_PATH)
        df_stream = pd.read_sql_query(
            """SELECT b.batch_id, b.yield, b.quality, b.energy_consumption,
                      b.carbon_intensity, b.temperature, b.speed,
                      COALESCE(ee.is_anomaly, 0)  AS is_anomaly,
                      COALESCE(ee.recon_error, 0)  AS recon_error
               FROM batches b
               LEFT JOIN energy_embeddings ee ON b.batch_id = ee.batch_id
               ORDER BY b.batch_id DESC LIMIT 20""",
            _stconn,
        )
        _stconn.close()
        df_stream["carbon_intensity"] = pd.to_numeric(df_stream["carbon_intensity"], errors="coerce")
        df_stream["recon_error"]      = pd.to_numeric(df_stream["recon_error"],      errors="coerce")
        df_stream["is_anomaly"]       = pd.to_numeric(df_stream["is_anomaly"],       errors="coerce").fillna(0).astype(int)
    except Exception:
        df_stream = pd.DataFrame()

    if len(df_stream) > 0:
        df_stream["Zone"]   = df_stream["carbon_intensity"].apply(classify_carbon_zone)
        df_stream["Status"] = df_stream["is_anomaly"].apply(lambda x: "⚠️ Anomaly" if x == 1 else "✅ Normal")
        _ds = df_stream[["batch_id","yield","quality","energy_consumption",
                          "carbon_intensity","temperature","speed","Zone","Status","recon_error"]].copy()
        _ds.columns = ["Batch ID","Yield","Quality","Energy (kWh)","Carbon CI",
                       "Temp (°C)","Speed (rpm)","Zone","Status","Recon Err"]
        for _col in ["Yield","Quality","Carbon CI","Recon Err"]:
            _ds[_col] = _ds[_col].round(4)
        _ds["Energy (kWh)"] = _ds["Energy (kWh)"].round(1)
        _ds["Temp (°C)"]    = _ds["Temp (°C)"].round(1)
        _ds["Speed (rpm)"]  = _ds["Speed (rpm)"].round(0).astype(int)
        st.dataframe(_ds, use_container_width=True, hide_index=True, height=390)
        st.markdown(
            '<div style="font-size:0.73rem;color:rgba(255,255,255,0.28);margin-top:5px;">'
            '↑ Last 20 batches ordered by Batch ID descending. Click Refresh to reload from DB.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No batch stream data available.")
