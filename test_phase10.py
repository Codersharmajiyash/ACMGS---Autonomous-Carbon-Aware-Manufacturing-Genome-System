"""
Phase 10 Test Suite — ACMGS System Integration

Verifies that main.py is fully functional: CLI parsing, status command,
artefact detection, phase runner imports, service launcher structure,
and end-to-end integration of all 9 preceding phases.

Run:
    cd C:\\Users\\HP\\ACMGS
    python test_phase10.py
"""

import sys
import os
import ast
sys.stdout.reconfigure(encoding="utf-8")
import subprocess
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


print("\n" + "=" * 65)
print("  PHASE 10 TEST SUITE — ACMGS System Integration")
print("=" * 65)


# ─── TEST 1: main.py file integrity ──────────────────────────────────────────
print("\n[TEST 1]  main.py file integrity")

main_path = ROOT / "main.py"
check("main.py exists",           main_path.exists())

if main_path.exists():
    src = main_path.read_text(encoding="utf-8")
    try:
        ast.parse(src)
        check("main.py parses without syntax errors", True)
    except SyntaxError as e:
        check("main.py parses without syntax errors", False, f"line {e.lineno}: {e.msg}")

    check("main.py has build_parser()",   "def build_parser" in src)
    check("main.py has cmd_status()",     "def cmd_status"   in src)
    check("main.py has cmd_full()",       "def cmd_full"     in src)
    check("main.py has cmd_phase()",      "def cmd_phase"    in src)
    check("main.py has cmd_api()",        "def cmd_api"      in src)
    check("main.py has cmd_dashboard()",  "def cmd_dashboard" in src)
    check("main.py has cmd_serve()",      "def cmd_serve"    in src)
    check("main.py has cmd_verify()",     "def cmd_verify"   in src)
    check("main.py has --status arg",     "--status"         in src)
    check("main.py has --full arg",       "--full"           in src)
    check("main.py has --phase arg",      "--phase"          in src)
    check("main.py has --api arg",        "--api"            in src)
    check("main.py has --dashboard arg",  "--dashboard"      in src)
    check("main.py has --serve arg",      "--serve"          in src)
    check("main.py has --verify arg",     "--verify"         in src)
    check("main.py has PHASE_ARTEFACTS",  "PHASE_ARTEFACTS"  in src)
    check("main.py has PHASE_NAMES",      "PHASE_NAMES"      in src)
    check("main.py has PHASE_RUNNERS",    "PHASE_RUNNERS"    in src)
    check("main.py has phase 1-7 runners","run_phase_1"       in src)
    check("main.py imports argparse",     "import argparse"  in src)
    check("main.py imports subprocess",   "import subprocess" in src)


# ─── TEST 2: CLI parser construction ─────────────────────────────────────────
print("\n[TEST 2]  CLI argument parser")

try:
    # Import main module without executing it
    import importlib.util
    spec = importlib.util.spec_from_file_location("main", main_path)
    main_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_mod)
    check("main module imports successfully", True)

    parser = main_mod.build_parser()
    check("build_parser() returns ArgumentParser", parser is not None)

    # Test --status
    args = parser.parse_args(["--status"])
    check("--status parsed correctly",    args.status is True)

    # Test --full
    args = parser.parse_args(["--full"])
    check("--full parsed correctly",      args.full is True)

    # Test --phase 4
    args = parser.parse_args(["--phase", "4"])
    check("--phase 4 parsed correctly",   args.phase == 4)

    # Test --api
    args = parser.parse_args(["--api"])
    check("--api parsed correctly",       args.api is True)

    # Test --dashboard
    args = parser.parse_args(["--dashboard"])
    check("--dashboard parsed correctly", args.dashboard is True)

    # Test --serve
    args = parser.parse_args(["--serve"])
    check("--serve parsed correctly",     args.serve is True)

    # Test --verify
    args = parser.parse_args(["--verify"])
    check("--verify parsed correctly",    args.verify is True)

    # Test --api --port 9000
    args = parser.parse_args(["--api", "--port", "9000"])
    check("--api --port 9000 parsed",     args.port == 9000)

    # Test mutual exclusion
    try:
        parser.parse_args(["--status", "--full"])
        check("mutually exclusive flags rejected", False, "should have raised SystemExit")
    except SystemExit:
        check("mutually exclusive flags rejected", True)

except Exception as e:
    check("main module imports successfully", False, str(e))


# ─── TEST 3: Artefact detection ───────────────────────────────────────────────
print("\n[TEST 3]  Phase artefact detection")

from config.settings import DB_PATH, SIMULATED_DIR, PROCESSED_DIR, MODELS_DIR

artefacts = {
    1: [
        os.path.join(SIMULATED_DIR, "batch_data.csv"),
        os.path.join(SIMULATED_DIR, "energy_signals.npy"),
    ],
    2: [
        os.path.join(SIMULATED_DIR, "energy_embeddings.npy"),
        os.path.join(MODELS_DIR, "lstm_autoencoder.pth"),
    ],
    3: [
        os.path.join(PROCESSED_DIR, "genome_vectors.npy"),
        os.path.join(PROCESSED_DIR, "batch_ids.npy"),
        os.path.join(PROCESSED_DIR, "genome_normalization.npz"),
    ],
    4: [
        os.path.join(MODELS_DIR, "predictor.pkl"),
        os.path.join(MODELS_DIR, "predictor_metrics.pkl"),
    ],
    5: [os.path.join(SIMULATED_DIR, "pareto_solutions.csv")],
    6: [os.path.join(SIMULATED_DIR, "carbon_schedule_demo.csv")],
    7: [DB_PATH],
    8: [str(ROOT / "src" / "api" / "main.py")],
    9: [str(ROOT / "src" / "dashboard" / "app.py")],
}

for phase, files in artefacts.items():
    for f in files:
        rel = os.path.relpath(f, ROOT)
        check(f"Phase {phase} artefact exists: {os.path.basename(f)}",
              os.path.exists(f) and os.path.getsize(f) > 0,
              f"missing: {rel}")


# ─── TEST 4: _phase_done() logic ─────────────────────────────────────────────
print("\n[TEST 4]  Phase completion detection")

for phase in range(1, 10):
    done = main_mod._phase_done(phase)
    check(f"Phase {phase} _phase_done() returns bool", isinstance(done, bool))

# All phases 1-9 should be done (artefacts exist)
all_phases_done = all(main_mod._phase_done(p) for p in range(1, 10))
check("All phases 1-9 detected as complete", all_phases_done,
      [p for p in range(1, 10) if not main_mod._phase_done(p)])


# ─── TEST 5: Phase module imports ─────────────────────────────────────────────
print("\n[TEST 5]  Phase module entry points importable")

try:
    from src.data_simulation.simulator import generate_full_dataset
    check("Phase 1: generate_full_dataset importable",    callable(generate_full_dataset))
except Exception as e:
    check("Phase 1: generate_full_dataset importable", False, str(e))

try:
    from src.energy_dna.trainer import run_energy_dna_pipeline
    check("Phase 2: run_energy_dna_pipeline importable",  callable(run_energy_dna_pipeline))
except Exception as e:
    check("Phase 2: run_energy_dna_pipeline importable", False, str(e))

try:
    from src.batch_genome.encoder import run_batch_genome_pipeline
    check("Phase 3: run_batch_genome_pipeline importable", callable(run_batch_genome_pipeline))
except Exception as e:
    check("Phase 3: run_batch_genome_pipeline importable", False, str(e))

try:
    from src.prediction.predictor import run_prediction_pipeline
    check("Phase 4: run_prediction_pipeline importable",  callable(run_prediction_pipeline))
except Exception as e:
    check("Phase 4: run_prediction_pipeline importable", False, str(e))

try:
    from src.optimization.optimizer import run_optimization_phase
    check("Phase 5: run_optimization_phase importable",   callable(run_optimization_phase))
except Exception as e:
    check("Phase 5: run_optimization_phase importable", False, str(e))

try:
    from src.carbon_scheduler.scheduler import run_scheduler_pipeline
    check("Phase 6: run_scheduler_pipeline importable",   callable(run_scheduler_pipeline))
except Exception as e:
    check("Phase 6: run_scheduler_pipeline importable", False, str(e))

try:
    from src.database.manager import run_database_pipeline
    check("Phase 7: run_database_pipeline importable",    callable(run_database_pipeline))
except Exception as e:
    check("Phase 7: run_database_pipeline importable", False, str(e))

try:
    from src.api.main import app
    check("Phase 8: FastAPI app importable",              app is not None)
except Exception as e:
    check("Phase 8: FastAPI app importable", False, str(e))

try:
    app_path = ROOT / "src" / "dashboard" / "app.py"
    check("Phase 9: dashboard app.py exists",             app_path.exists())
except Exception as e:
    check("Phase 9: dashboard app.py exists", False, str(e))


# ─── TEST 6: --status command runs successfully ──────────────────────────────
print("\n[TEST 6]  --status command (subprocess)")

result = subprocess.run(
    [sys.executable, str(main_path), "--status"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    encoding="utf-8",
    timeout=30,
)
check("--status exits with code 0",       result.returncode == 0,
      result.stderr[:200] if result.returncode != 0 else "")
check("--status prints phase 1",          "Phase 1" in result.stdout)
check("--status prints phase 7",          "Phase 7" in result.stdout)
check("--status prints phase 9",          "Phase 9" in result.stdout)
check("--status shows [DONE] for phase 1","[DONE]"   in result.stdout)
check("--status shows DB row counts",     "batches"  in result.stdout)
check("--status shows ALL PHASES COMPLETE", "ALL PHASES COMPLETE" in result.stdout)


# ─── TEST 7: --phase 6 (scheduler, fast, no retraining) ──────────────────────
print("\n[TEST 7]  --phase runner (phase 6 — fastest phase)")

result_p6 = subprocess.run(
    [sys.executable, str(main_path), "--phase", "6"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    encoding="utf-8",
    timeout=60,
)
check("--phase 6 exits successfully",         result_p6.returncode == 0,
      result_p6.stderr[:200] if result_p6.returncode != 0 else "")
check("--phase 6 produces output",            len(result_p6.stdout) > 0)
carbon_csv = os.path.join(SIMULATED_DIR, "carbon_schedule_demo.csv")
check("carbon_schedule_demo.csv still exists", os.path.exists(carbon_csv))


# ─── TEST 8: --phase 8/9 handled gracefully ──────────────────────────────────
print("\n[TEST 8]  Service phase handling (no crash)")

result_p8 = subprocess.run(
    [sys.executable, str(main_path), "--phase", "8"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    encoding="utf-8",
    timeout=10,
)
check("--phase 8 prints guidance (no crash)",
      result_p8.returncode == 0 and "service" in result_p8.stdout.lower(),
      result_p8.stdout[:100])

result_p9 = subprocess.run(
    [sys.executable, str(main_path), "--phase", "9"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    encoding="utf-8",
    timeout=10,
)
check("--phase 9 prints guidance (no crash)",
      result_p9.returncode == 0 and "service" in result_p9.stdout.lower(),
      result_p9.stdout[:100])

result_bad = subprocess.run(
    [sys.executable, str(main_path), "--phase", "99"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    encoding="utf-8",
    timeout=10,
)
check("--phase 99 handled without crash",     result_bad.returncode == 0)


# ─── TEST 9: Cross-phase data consistency ────────────────────────────────────
print("\n[TEST 9]  Cross-phase data consistency")

import numpy as np
import pandas as pd
import sqlite3
import json

# Batch IDs consistent across all phases
batch_ids   = np.load(os.path.join(PROCESSED_DIR, "batch_ids.npy"), allow_pickle=True)
df_batches  = pd.read_csv(os.path.join(SIMULATED_DIR, "batch_data.csv"))
genome_vecs = np.load(os.path.join(PROCESSED_DIR, "genome_vectors.npy"))
emb         = np.load(os.path.join(SIMULATED_DIR, "energy_embeddings.npy"))

check("batch_ids count == 2000",         len(batch_ids)  == 2000)
check("batch_data rows == 2000",         len(df_batches) == 2000)
check("genome_vectors rows == 2000",     genome_vecs.shape[0] == 2000)
check("genome_vectors dims == 25",       genome_vecs.shape[1] == 25)
check("energy_embeddings rows == 2000",  emb.shape[0] == 2000)
check("energy_embeddings dims == 16",    emb.shape[1] == 16)

# DB consistency
conn = sqlite3.connect(DB_PATH)
db_batches  = conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
db_genomes  = conn.execute("SELECT COUNT(*) FROM genome_vectors").fetchone()[0]
db_pareto   = conn.execute("SELECT COUNT(*) FROM pareto_solutions").fetchone()[0]
db_sched    = conn.execute("SELECT COUNT(*) FROM carbon_schedules").fetchone()[0]
conn.close()

check("DB batches == CSV batches",       db_batches == len(df_batches),
      f"DB={db_batches}, CSV={len(df_batches)}")
check("DB genome_vectors == 2000",       db_genomes == 2000)
check("DB pareto_solutions >= 100",      db_pareto  >= 100)
check("DB carbon_schedules >= 1",        db_sched   >= 1)

# Pareto solutions within valid ranges
df_pareto = pd.read_csv(os.path.join(SIMULATED_DIR, "pareto_solutions.csv"))
check("Pareto pred_yield in (0,1]",
      bool((df_pareto["pred_yield"] > 0).all() and
           (df_pareto["pred_yield"] <= 1).all()))
check("Pareto pred_energy > 0",          bool((df_pareto["pred_energy"] > 0).all()))
check("Pareto pred_carbon > 0",          bool((df_pareto["pred_carbon"] > 0).all()))


# ─── TEST 10: Full integration smoke test ─────────────────────────────────────
print("\n[TEST 10]  Full integration smoke test")

# Simulates what a real user would do: get a recommendation and verify
# the full chain from DB → scheduler → recommendation
from src.carbon_scheduler import classify_carbon_zone, get_recommendation
from src.database.manager import get_pareto_solutions, get_db_summary

# DB summary
summary = get_db_summary()
check("get_db_summary() returns dict",        isinstance(summary, dict))
check("DB summary batches == 2000",           summary.get("batches") == 2000)
check("DB summary pareto_solutions >= 100",   summary.get("pareto_solutions",0) >= 100)

# Zone classification
check("classify_carbon_zone(50) = LOW",    classify_carbon_zone(50)   == "LOW")
check("classify_carbon_zone(300) = MEDIUM",classify_carbon_zone(300)  == "MEDIUM")
check("classify_carbon_zone(450) = HIGH",  classify_carbon_zone(450)  == "HIGH")

# Full recommendation pipeline
for ci, zone in [(80, "LOW"), (300, "MEDIUM"), (500, "HIGH")]:
    rec = get_recommendation(float(ci))
    check(f"Full pipeline rec({ci}) zone={zone}",
          rec["zone"] == zone, f"got {rec.get('zone')}")
    sched = rec["recommended_schedule"]
    check(f"Full pipeline rec({ci}) yield>0",
          sched["pred_yield"] > 0)
    check(f"Full pipeline rec({ci}) energy>0",
          sched["pred_energy"] > 0)

# Pareto query
df_p = get_pareto_solutions()
check("get_pareto_solutions() returns DataFrame",  isinstance(df_p, pd.DataFrame))
check("get_pareto_solutions() has >= 100 rows",       len(df_p) >= 100)


# ─── TEST 11: cmd_status() output capture ────────────────────────────────────
print("\n[TEST 11]  cmd_status() internal function")

import io
from contextlib import redirect_stdout

buf = io.StringIO()
try:
    with redirect_stdout(buf):
        main_mod.cmd_status()
    out = buf.getvalue()
    check("cmd_status() runs without exception",  True)
    check("cmd_status() includes phase 1",        "Phase 1" in out)
    check("cmd_status() includes phase 9",        "Phase 9" in out)
    check("cmd_status() shows completion",        "COMPLETE" in out or "ALL PHASES" in out)
    check("cmd_status() shows batch_data.csv",    "batch_data.csv" in out)
    check("cmd_status() shows predictor.pkl",     "predictor.pkl" in out)
except Exception as e:
    check("cmd_status() runs without exception", False, str(e))


# ─── Summary ─────────────────────────────────────────────────────────────────
print()
print("=" * 65)
pct = round(passed / total * 100, 1) if total else 0
print(f"  RESULT: {passed}/{total} tests passed  ({pct}%)")
if passed == total:
    print("  Phase 10 System Integration: ALL CHECKS PASSED")
    print("  ACMGS is fully complete and operational.")
else:
    print(f"  Phase 10 System Integration: {total - passed} check(s) FAILED")
print("=" * 65)

sys.exit(0 if passed == total else 1)
