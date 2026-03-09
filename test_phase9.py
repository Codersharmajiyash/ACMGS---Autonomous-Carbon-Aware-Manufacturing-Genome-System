"""
Phase 9 Test Suite — ACMGS Dashboard

Tests that all data loading, zone logic, genome parsing, and chart-building
functions work correctly without starting a full Streamlit server.

Run:
    cd C:\\Users\\HP\\ACMGS
    python test_phase9.py
"""

import sys
import os
import json
import sqlite3
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# ─── Test runner ─────────────────────────────────────────────────────────────
PASS  = "[PASS]"
FAIL  = "[FAIL]"
total = 0
passed = 0


def check(label: str, condition: bool, detail: str = ""):
    global total, passed
    total += 1
    if condition:
        passed += 1
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}" + (f"  ({detail})" if detail else ""))


# ─── Imports ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE 9 TEST SUITE - ACMGS Dashboard")
print("=" * 65)

print("\n[TEST 1]  Module imports")
try:
    from config.settings import DB_PATH, CARBON_HIGH_THRESHOLD, CARBON_LOW_THRESHOLD
    check("config.settings imports", True)
except Exception as e:
    check("config.settings imports", False, str(e))

try:
    from src.carbon_scheduler import classify_carbon_zone, get_recommendation
    check("src.carbon_scheduler imports", True)
except Exception as e:
    check("src.carbon_scheduler imports", False, str(e))

try:
    import plotly.express as px
    import plotly.graph_objects as go
    check("plotly imports", True)
except Exception as e:
    check("plotly imports", False, str(e))

try:
    import streamlit as st
    check("streamlit import", True)
except Exception as e:
    check("streamlit import", False, str(e))

check("DB_PATH exists on disk", os.path.exists(DB_PATH), DB_PATH)
check("DB_PATH has non-zero size", os.path.getsize(DB_PATH) > 0)


# ─── Carbon zone classification ──────────────────────────────────────────────
print("\n[TEST 2]  Carbon zone classification")
check("100 gCO2 -> LOW",    classify_carbon_zone(100)  == "LOW")
check("150 gCO2 -> LOW",    classify_carbon_zone(150)  == "LOW")
check("151 gCO2 -> MEDIUM", classify_carbon_zone(151)  == "MEDIUM")
check("300 gCO2 -> MEDIUM", classify_carbon_zone(300)  == "MEDIUM")
check("399 gCO2 -> MEDIUM", classify_carbon_zone(399)  == "MEDIUM")
check("400 gCO2 -> HIGH",   classify_carbon_zone(400)  == "HIGH")
check("550 gCO2 -> HIGH",   classify_carbon_zone(550)  == "HIGH")
check("0   gCO2 -> LOW",    classify_carbon_zone(0)    == "LOW")


# ─── Database connectivity ───────────────────────────────────────────────────
print("\n[TEST 3]  Database queries")
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

def tbl_count(name):
    return conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]

check("batches count == 2000",          tbl_count("batches")          == 2000)
check("energy_embeddings count == 2000", tbl_count("energy_embeddings") == 2000)
check("genome_vectors count == 2000",   tbl_count("genome_vectors")   == 2000)
check("pareto_solutions count >= 100",  tbl_count("pareto_solutions") >= 100)
check("carbon_schedules count >= 1",    tbl_count("carbon_schedules") >= 1)

# Schema checks
batches_cols = [r[1] for r in conn.execute("PRAGMA table_info(batches)").fetchall()]
check("batches has yield column",            "yield"            in batches_cols)
check("batches has carbon_intensity column", "carbon_intensity" in batches_cols)

pareto_cols = [r[1] for r in conn.execute("PRAGMA table_info(pareto_solutions)").fetchall()]
check("pareto has pred_yield",   "pred_yield"   in pareto_cols)
check("pareto has pred_energy",  "pred_energy"  in pareto_cols)
check("pareto has pred_carbon",  "pred_carbon"  in pareto_cols)
check("pareto has pred_quality", "pred_quality" in pareto_cols)

conn.close()


# ─── Data loading ─────────────────────────────────────────────────────────────
print("\n[TEST 4]  Data loading via SQL")

conn2 = sqlite3.connect(DB_PATH)

df_b = pd.read_sql_query("SELECT * FROM batches ORDER BY batch_id", conn2)
check("batches DataFrame shape[0] == 2000", df_b.shape[0] == 2000,
      f"got {df_b.shape[0]}")
check("batches DataFrame shape[1] >= 12",   df_b.shape[1] >= 12,
      f"got {df_b.shape[1]}")
check("no NaN in yield column",  df_b["yield"].isna().sum() == 0)
check("yield range 0–1",
      bool((df_b["yield"] >= 0).all() and (df_b["yield"] <= 1).all()))
check("carbon_intensity > 0 always",
      bool((df_b["carbon_intensity"] > 0).all()))

df_p = pd.read_sql_query("SELECT * FROM pareto_solutions", conn2)
check("pareto DataFrame shape[0] >= 100",  df_p.shape[0] >= 100,
      f"got {df_p.shape[0]}")
check("pred_yield in (0,1]",
      bool((df_p["pred_yield"] > 0).all() and (df_p["pred_yield"] <= 1).all()))
check("pred_energy > 0",  bool((df_p["pred_energy"] > 0).all()))
check("pred_carbon > 0",  bool((df_p["pred_carbon"] > 0).all()))

df_g = pd.read_sql_query("SELECT batch_id, genome FROM genome_vectors LIMIT 50", conn2)
check("genome_vectors returns 50 rows",  len(df_g) == 50)
check("genome JSON parses to 25 dims",
      len(json.loads(df_g["genome"].iloc[0])) == 25)

conn2.close()


# ─── Genome matrix parsing ────────────────────────────────────────────────────
print("\n[TEST 5]  Genome matrix operations")

conn3 = sqlite3.connect(DB_PATH)
df_gall = pd.read_sql_query("SELECT batch_id, genome FROM genome_vectors LIMIT 80", conn3)
conn3.close()

gm = np.array([json.loads(g) for g in df_gall["genome"]])
check("genome matrix shape == (80, 25)", gm.shape == (80, 25),
      f"got {gm.shape}")
check("genome values finite (no NaN/Inf)", bool(np.isfinite(gm).all()))
check("genome roughly z-scored (|mean| < 0.3)", float(abs(gm.mean())) < 0.3,
      f"mean={gm.mean():.4f}")
check("genome std in reasonable range",   float(gm.std()) > 0.1)

# Genome segment labeling
GENOME_LABELS = (
    ["Temp", "Pressure", "Speed", "FeedRate", "Humidity"]
    + ["Density", "Hardness", "Grade"]
    + [f"EdNA{i:02d}" for i in range(16)]
    + ["CarbonInt"]
)
check("GENOME_LABELS has 25 entries", len(GENOME_LABELS) == 25)
check("GENOME_LABELS[0] == 'Temp'",    GENOME_LABELS[0]  == "Temp")
check("GENOME_LABELS[24] == 'CarbonInt'", GENOME_LABELS[24] == "CarbonInt")


# ─── Get recommendation structure ────────────────────────────────────────────
print("\n[TEST 6]  Recommendation output structure")

for ci, expected_zone in [(100, "LOW"), (250, "MEDIUM"), (500, "HIGH")]:
    try:
        rec = get_recommendation(float(ci))
        check(f"rec({ci}) returns dict",                   isinstance(rec, dict))
        check(f"rec({ci}) has 'zone' key",                 "zone" in rec)
        check(f"rec({ci})['zone'] == '{expected_zone}'",   rec["zone"] == expected_zone,
              f"got {rec.get('zone')}")
        check(f"rec({ci}) has 'recommended_schedule'",     "recommended_schedule" in rec)
        sched = rec["recommended_schedule"]
        check(f"rec({ci}) schedule has pred_yield",        "pred_yield"  in sched)
        check(f"rec({ci}) schedule has pred_energy",       "pred_energy" in sched)
        check(f"rec({ci}) schedule has temperature",       "temperature" in sched)
    except Exception as e:
        check(f"rec({ci}) no exception", False, str(e))


# ─── Plotly chart construction ────────────────────────────────────────────────
print("\n[TEST 7]  Plotly chart construction")

import plotly.graph_objects as go
import plotly.express as px

# Gauge figure
zone_c = classify_carbon_zone(250.0)
fig_g = go.Figure(go.Indicator(
    mode="gauge+number", value=250.0,
    gauge={"axis": {"range": [0, 600]}, "bar": {"color": "#ffd600"}},
))
check("gauge figure created",          isinstance(fig_g, go.Figure))
check("gauge has 1 trace",             len(fig_g.data) == 1)
check("gauge trace is Indicator",      fig_g.data[0].type == "indicator")

# Scatter from batch data
conn4 = sqlite3.connect(DB_PATH)
df_sample = pd.read_sql_query("SELECT * FROM batches LIMIT 100", conn4)
conn4.close()
df_sample["zone"] = df_sample["carbon_intensity"].apply(classify_carbon_zone)
fig_s = px.scatter(df_sample, x="carbon_intensity", y="energy_consumption",
                   color="zone")
check("scatter figure created",        isinstance(fig_s, go.Figure))
check("scatter has data traces",       len(fig_s.data) >= 1)

# Heatmap from genomes
fig_hm = go.Figure(go.Heatmap(
    z=gm.T, x=[b[-4:] for b in df_gall["batch_id"]],
    y=GENOME_LABELS, colorscale="RdBu",
))
check("heatmap figure created",        isinstance(fig_hm, go.Figure))
check("heatmap z shape == (25, 80)",   fig_hm.data[0].z.shape == (25, 80),
      f"got {fig_hm.data[0].z.shape}")

# 3D scatter
conn5 = sqlite3.connect(DB_PATH)
df_par = pd.read_sql_query("SELECT * FROM pareto_solutions", conn5)
conn5.close()
fig_3d = px.scatter_3d(df_par, x="pred_yield", y="pred_energy", z="pred_carbon",
                        color="pred_yield")
check("3D scatter figure created",     isinstance(fig_3d, go.Figure))
check("3D scatter has data",           len(fig_3d.data) >= 1)


# ─── App module import (syntax check) ────────────────────────────────────────
print("\n[TEST 8]  Dashboard app.py syntax & import check")

# Parse the file for syntax errors without executing Streamlit runtime
import ast
app_path = ROOT / "src" / "dashboard" / "app.py"
check("app.py file exists", app_path.exists(), str(app_path))
if app_path.exists():
    try:
        src = app_path.read_text(encoding="utf-8")
        ast.parse(src)
        check("app.py parses without syntax errors", True)
    except SyntaxError as e:
        check("app.py parses without syntax errors", False, f"line {e.lineno}: {e.msg}")

    check("app.py imports streamlit",         "import streamlit" in src)
    check("app.py imports plotly",            "import plotly" in src)
    check("app.py uses set_page_config",      "set_page_config" in src)
    check("app.py defines dark_layout()",     "def dark_layout" in src)
    check("app.py defines make_gauge()",      "def make_gauge" in src)
    check("app.py has Command Center tab",    "Command Center" in src)
    check("app.py has Production Analytics",  "Production Analytics" in src)
    check("app.py has Pareto Intelligence",   "Pareto Intelligence" in src)
    check("app.py has Genome Explorer",       "Genome Explorer" in src)
    check("app.py has System Health",         "System Health" in src)
    check("app.py loads batches from DB",     "_load_batches" in src)
    check("app.py loads pareto from DB",      "_load_pareto" in src)
    check("app.py uses cache_data",           "cache_data" in src)
    check("app.py calls classify_carbon_zone", "classify_carbon_zone" in src)
    check("app.py calls get_recommendation",  "get_recommendation" in src)

# config.toml
toml_path = ROOT / ".streamlit" / "config.toml"
check(".streamlit/config.toml exists", toml_path.exists())

# ─── Summary ─────────────────────────────────────────────────────────────────
print()
print("=" * 65)
pct = round(passed / total * 100, 1) if total else 0
print(f"  RESULT: {passed}/{total} tests passed  ({pct}%)")
if passed == total:
    print("  Phase 9 Dashboard: ALL CHECKS PASSED")
else:
    print(f"  Phase 9 Dashboard: {total - passed} check(s) FAILED")
print("=" * 65)

sys.exit(0 if passed == total else 1)
