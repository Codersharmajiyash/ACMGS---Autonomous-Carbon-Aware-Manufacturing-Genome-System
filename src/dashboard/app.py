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
    page_icon="⚡",
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
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: rgba(255,255,255,0.5);
    font-weight: 500;
    font-size: 0.83rem;
    padding: 7px 16px;
    transition: all 0.2s;
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
    "batches": "🏭", "energy_embeddings": "⚡", "genome_vectors": "🧬",
    "predictions": "🎯", "pareto_solutions": "📊", "carbon_schedules": "🌿",
    "pipeline_runs": "🔄",
}

_CYAN    = "#00d4ff"
_GREEN   = "#00ff88"
_YELLOW  = "#ffd600"
_RED     = "#ff4b4b"
_PURPLE  = "#a855f7"
_ORANGE  = "#f97316"


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
  <div style="font-size:2.6rem;line-height:1;">⚡</div>
  <div style="font-size:1.25rem;font-weight:800;
              background:linear-gradient(90deg,#00d4ff,#00ff88);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              background-clip:text;letter-spacing:-0.02em;">ACMGS</div>
  <div style="font-size:0.67rem;color:rgba(255,255,255,0.28);
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
    '<h1>⚡ ACMGS Control Center</h1>'
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚡  Command Center",
    "📊  Production Analytics",
    "🎯  Pareto Intelligence",
    "🧬  Genome Explorer",
    "🔧  System Health",
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
            max_carbon = st.slider("Max Pred Carbon (kgCO₂)", 200.0, 500.0,
                                   float(df_pareto["pred_carbon"].max()), 5.0)
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
        fig_3d.update_traces(marker=dict(size=5, opacity=0.85))
        fig_3d.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                bgcolor="rgba(255,255,255,0.01)",
                xaxis=dict(
                    backgroundcolor="rgba(0,212,255,0.03)",
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
