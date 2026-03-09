"""
ACMGS Main Entry Point — Phase 10: System Integration

Ties together all 10 phases of the Autonomous Carbon-Aware Manufacturing
Genome System into a single, clean command-line launcher.

WHAT THIS DOES:
  Provides one file to rule them all — you can run any single phase,
  the full end-to-end pipeline, start the API, start the dashboard,
  or inspect current system status, all from one place.

USAGE:
    python main.py --status            # Show what's built so far
    python main.py --full              # Run entire pipeline (phases 1-7)
    python main.py --phase 1           # Run just Phase 1
    python main.py --api               # Start FastAPI on port 8000
    python main.py --dashboard         # Start Streamlit on port 8501
    python main.py --serve             # Start both API + dashboard
    python main.py --verify            # Run all test suites
"""

import argparse
import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ─── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    DB_PATH, SIMULATED_DIR, PROCESSED_DIR, MODELS_DIR,
    CARBON_HIGH_THRESHOLD, CARBON_LOW_THRESHOLD,
)

# ─── Artefact map: phase → list of files it produces ─────────────────────────
PHASE_ARTEFACTS = {
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
    5: [
        os.path.join(SIMULATED_DIR, "pareto_solutions.csv"),
    ],
    6: [
        os.path.join(SIMULATED_DIR, "carbon_schedule_demo.csv"),
    ],
    7: [DB_PATH],
    8: [os.path.join(ROOT, "src", "api", "main.py")],
    9: [os.path.join(ROOT, "src", "dashboard", "app.py")],
}

PHASE_NAMES = {
    1: "Data Simulation",
    2: "Energy DNA (LSTM Autoencoder)",
    3: "Batch Genome Encoder",
    4: "XGBoost Prediction Model",
    5: "NSGA-II Evolutionary Optimizer",
    6: "Carbon-Aware Scheduler",
    7: "SQLite Database",
    8: "FastAPI REST Layer",
    9: "Streamlit Dashboard",
}

PYTHON = sys.executable


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _sep(char="-", width=65): print(char * width)
def _hdr(text):  _sep(); print(f"  {text}"); _sep()
def _ok(msg):    print(f"  [OK]   {msg}")
def _warn(msg):  print(f"  [WARN] {msg}")
def _err(msg):   print(f"  [FAIL] {msg}")
def _info(msg):  print(f"         {msg}")


def _phase_done(phase: int) -> bool:
    """Return True only if all artefacts for a phase exist and are non-empty."""
    files = PHASE_ARTEFACTS.get(phase, [])
    return bool(files) and all(
        os.path.exists(f) and os.path.getsize(f) > 0
        for f in files
    )


# ─── STATUS ───────────────────────────────────────────────────────────────────
def cmd_status():
    """Print a full system status report."""
    _hdr("ACMGS - System Status Report")
    print(f"  Project Root : {ROOT}")
    print(f"  Python       : {PYTHON}")
    print(f"  Timestamp    : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print()

    all_done = True
    for phase in range(1, 10):
        done = _phase_done(phase)
        if not done:
            all_done = False
        icon = "[DONE]" if done else "[MISS]"
        print(f"  Phase {phase}  {icon}  {PHASE_NAMES[phase]}")
        for f in PHASE_ARTEFACTS[phase]:
            rel = os.path.relpath(f, ROOT)
            if os.path.exists(f):
                size_kb = os.path.getsize(f) / 1024
                _info(f"  {rel}  ({size_kb:.1f} KB)")
            else:
                _info(f"  {rel}  <MISSING>")

    print()

    # DB summary if available
    if _phase_done(7):
        try:
            import sqlite3
            conn = sqlite3.connect(DB_PATH)
            tables = ["batches", "energy_embeddings", "genome_vectors",
                      "predictions", "pareto_solutions", "carbon_schedules",
                      "pipeline_runs"]
            print("  Database table counts:")
            for t in tables:
                n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                print(f"    {t:<28} {n:>6} rows")
            conn.close()
        except Exception as e:
            _warn(f"Could not read DB: {e}")

    print()
    _sep()
    if all_done:
        print("  ALL PHASES COMPLETE - system is fully operational")
    else:
        print("  Some phases are missing - run  python main.py --full  to build")
    _sep()


# ─── INDIVIDUAL PHASE RUNNERS ─────────────────────────────────────────────────
def run_phase_1():
    _hdr("Phase 1: Data Simulation")
    from src.data_simulation.simulator import generate_full_dataset
    result = generate_full_dataset()
    _ok(f"Generated batch_data + energy_signals")
    return result


def run_phase_2():
    _hdr("Phase 2: Energy DNA — LSTM Autoencoder")
    from src.energy_dna.trainer import run_energy_dna_pipeline
    result = run_energy_dna_pipeline()
    _ok("LSTM trained, energy_embeddings.npy saved")
    return result


def run_phase_3():
    _hdr("Phase 3: Batch Genome Encoder")
    from src.batch_genome.encoder import run_batch_genome_pipeline
    result = run_batch_genome_pipeline()
    _ok("genome_vectors.npy saved (2000 × 25)")
    return result


def run_phase_4():
    _hdr("Phase 4: XGBoost Prediction Model")
    from src.prediction.predictor import run_prediction_pipeline
    result = run_prediction_pipeline()
    _ok("predictor.pkl saved")
    return result


def run_phase_5():
    _hdr("Phase 5: NSGA-II Evolutionary Optimizer")
    from src.optimization.optimizer import run_optimization_phase
    result = run_optimization_phase()
    _ok("pareto_solutions.csv saved (100 solutions)")
    return result


def run_phase_6():
    _hdr("Phase 6: Carbon-Aware Scheduler")
    from src.carbon_scheduler.scheduler import run_scheduler_pipeline
    result = run_scheduler_pipeline()
    _ok("carbon_schedule_demo.csv saved")
    return result


def run_phase_7():
    _hdr("Phase 7: SQLite Database")
    from src.database.manager import run_database_pipeline
    result = run_database_pipeline()
    _ok(f"Database loaded: {result}")
    return result


PHASE_RUNNERS = {
    1: run_phase_1,
    2: run_phase_2,
    3: run_phase_3,
    4: run_phase_4,
    5: run_phase_5,
    6: run_phase_6,
    7: run_phase_7,
}


def cmd_phase(n: int):
    """Run a single phase by number (1-7)."""
    if n not in PHASE_RUNNERS:
        if n == 8:
            print("  Phase 8 (API) is a service - use:  python main.py --api")
        elif n == 9:
            print("  Phase 9 (Dashboard) is a service - use:  python main.py --dashboard")
        else:
            print(f"  Unknown phase: {n}. Valid range: 1-9")
        return
    started = time.time()
    try:
        PHASE_RUNNERS[n]()
        elapsed = time.time() - started
        _ok(f"Phase {n} finished in {elapsed:.1f}s")
    except Exception as e:
        _err(f"Phase {n} failed: {e}")
        raise


# ─── FULL PIPELINE ────────────────────────────────────────────────────────────
def cmd_full():
    """Run the complete data + model + DB pipeline (Phases 1-7)."""
    _sep("═")
    print("  ACMGS - FULL PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    _sep("═")

    pipeline_start = time.time()
    results = {}
    failed  = []

    for phase in range(1, 8):
        phase_start = time.time()
        try:
            results[phase] = PHASE_RUNNERS[phase]()
            elapsed = time.time() - phase_start
            _ok(f"Phase {phase} — {PHASE_NAMES[phase]} ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - phase_start
            _err(f"Phase {phase} — {PHASE_NAMES[phase]} FAILED: {e}")
            failed.append(phase)

    total = time.time() - pipeline_start
    _sep("═")
    print(f"  PIPELINE COMPLETE  -  {7 - len(failed)}/7 phases succeeded")
    print(f"  Total time: {total:.1f}s")
    if failed:
        print(f"  Failed phases: {failed}")
    else:
        print("  All phases succeeded. Run  python main.py --status  to verify.")
    _sep("═")
    return results


# ─── API SERVER ───────────────────────────────────────────────────────────────
def cmd_api(port: int = 8000, reload: bool = False):
    """Start the FastAPI server via uvicorn."""
    _hdr(f"Phase 8: FastAPI Server  ->  http://0.0.0.0:{port}")
    print("  Docs available at:  http://localhost:{}/docs".format(port))
    print("  Press Ctrl+C to stop\n")
    cmd = [
        PYTHON, "-m", "uvicorn",
        "src.api.main:app",
        "--host", "0.0.0.0",
        "--port", str(port),
    ]
    if reload:
        cmd.append("--reload")
    subprocess.run(cmd, cwd=ROOT)


# ─── DASHBOARD ────────────────────────────────────────────────────────────────
def cmd_dashboard(port: int = 8501):
    """Start the Streamlit dashboard."""
    _hdr(f"Phase 9: Streamlit Dashboard  ->  http://localhost:{port}")
    print("  Press Ctrl+C to stop\n")
    cmd = [
        PYTHON, "-m", "streamlit", "run",
        str(ROOT / "src" / "dashboard" / "app.py"),
        "--server.port", str(port),
        "--server.headless", "true",
    ]
    subprocess.run(cmd, cwd=ROOT)


# ─── SERVE BOTH ───────────────────────────────────────────────────────────────
def cmd_serve(api_port: int = 8000, dash_port: int = 8501):
    """Launch API and Dashboard in parallel background processes."""
    _sep("═")
    print("  ACMGS - Launching full stack")
    print(f"  API       ->  http://localhost:{api_port}")
    print(f"  API Docs  ->  http://localhost:{api_port}/docs")
    print(f"  Dashboard ->  http://localhost:{dash_port}")
    print("  Press Ctrl+C to stop both services\n")
    _sep("═")

    api_proc = subprocess.Popen(
        [PYTHON, "-m", "uvicorn",
         "src.api.main:app", "--host", "0.0.0.0", "--port", str(api_port)],
        cwd=ROOT,
    )

    time.sleep(2)  # Let API start before dashboard

    dash_proc = subprocess.Popen(
        [PYTHON, "-m", "streamlit", "run",
         str(ROOT / "src" / "dashboard" / "app.py"),
         "--server.port", str(dash_port),
         "--server.headless", "true"],
        cwd=ROOT,
    )

    print(f"\n  [PID {api_proc.pid}]  FastAPI server running")
    print(f"  [PID {dash_proc.pid}]  Streamlit dashboard running")
    print("\n  Ctrl+C to shut down both.\n")

    try:
        api_proc.wait()
        dash_proc.wait()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        api_proc.terminate()
        dash_proc.terminate()
        api_proc.wait()
        dash_proc.wait()
        print("  Both services stopped.")


# ─── VERIFY ───────────────────────────────────────────────────────────────────
def cmd_verify():
    """Run the full test suite across all phases."""
    _hdr("ACMGS — Full System Verification")
    test_files = [
        ROOT / "final_check.py",
        ROOT / "test_phase7.py",
        ROOT / "test_phase8.py",
        ROOT / "test_phase9.py",
        ROOT / "test_phase10.py",
    ]
    all_passed = True
    for tf in test_files:
        if not tf.exists():
            _warn(f"Test file not found: {tf.name}")
            continue
        print(f"\n  Running {tf.name} ...")
        result = subprocess.run([PYTHON, str(tf)], cwd=ROOT)
        if result.returncode == 0:
            _ok(f"{tf.name} — PASSED")
        else:
            _err(f"{tf.name} — FAILED (exit code {result.returncode})")
            all_passed = False

    print()
    _sep()
    if all_passed:
        print("  ALL TEST SUITES PASSED - ACMGS is fully operational")
    else:
        print("  Some tests failed - check output above")
    _sep()
    return all_passed


# ─── CLI ──────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python main.py",
        description="ACMGS — Autonomous Carbon-Aware Manufacturing Genome System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --status          Show pipeline status
  python main.py --full            Run full data + model pipeline (phases 1-7)
  python main.py --phase 4         Run only Phase 4 (XGBoost predictor)
  python main.py --api             Start FastAPI server on port 8000
  python main.py --dashboard       Start Streamlit dashboard on port 8501
  python main.py --serve           Start both API + dashboard
  python main.py --verify          Run all test suites
  python main.py --api --port 9000 Start API on custom port

        """,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--status",    action="store_true", help="Show system status")
    group.add_argument("--full",      action="store_true", help="Run full pipeline (phases 1-7)")
    group.add_argument("--phase",     type=int,            help="Run a single phase (1-9)", metavar="N")
    group.add_argument("--api",       action="store_true", help="Start FastAPI server")
    group.add_argument("--dashboard", action="store_true", help="Start Streamlit dashboard")
    group.add_argument("--serve",     action="store_true", help="Start API + dashboard")
    group.add_argument("--verify",    action="store_true", help="Run all test suites")

    p.add_argument("--port", type=int, default=None,
                   help="Port override for --api (default 8000) or --dashboard (default 8501)")
    p.add_argument("--reload", action="store_true",
                   help="Enable hot-reload for --api (dev mode)")
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.status:
        cmd_status()

    elif args.full:
        cmd_full()

    elif args.phase is not None:
        cmd_phase(args.phase)

    elif args.api:
        cmd_api(port=args.port or 8000, reload=args.reload)

    elif args.dashboard:
        cmd_dashboard(port=args.port or 8501)

    elif args.serve:
        api_port  = args.port or 8000
        dash_port = (args.port + 1) if args.port else 8501
        cmd_serve(api_port=api_port, dash_port=dash_port)

    elif args.verify:
        ok = cmd_verify()
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
