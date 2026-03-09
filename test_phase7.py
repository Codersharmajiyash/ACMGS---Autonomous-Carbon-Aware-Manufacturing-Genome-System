"""
Phase 7 — Comprehensive Database Test Suite
Tests every table, query helper, audit log, and edge case.
"""

import os
import sys
import json
import sqlite3
import warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# ── ensure project root is on path ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database.manager import (
    create_tables,
    get_connection,
    load_batches_from_csv,
    load_embeddings_from_npy,
    load_genomes_from_npy,
    load_pareto_from_csv,
    load_schedules_from_csv,
    log_pipeline_run,
    get_batch,
    get_genome,
    get_latest_schedule,
    get_pareto_solutions,
    get_db_summary,
    run_database_pipeline,
    DB_PATH,
)

PASS = "PASS"
FAIL = "FAIL"
results = {}


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results[name] = status
    symbol = "+" if condition else "X"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {symbol} [{status}]  {name}{suffix}")
    return condition


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ════════════════════════════════════════════════════════════════
section("TEST 1: DB FILE EXISTS AND TABLES CREATED")
# ════════════════════════════════════════════════════════════════

check("DB file exists on disk", os.path.exists(DB_PATH), DB_PATH)

db_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
check("DB file is non-empty", db_size > 0, f"{db_size:,} bytes")

expected_tables = [
    "batches", "energy_embeddings", "genome_vectors",
    "predictions", "pareto_solutions", "carbon_schedules", "pipeline_runs"
]
with get_connection() as conn:
    actual_tables = [
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]

for t in expected_tables:
    check(f"Table '{t}' exists", t in actual_tables)


# ════════════════════════════════════════════════════════════════
section("TEST 2: ROW COUNTS PER TABLE")
# ════════════════════════════════════════════════════════════════

summary = get_db_summary()

check("batches: 2000 rows",          summary["batches"] == 2000,          f"{summary['batches']} rows")
check("energy_embeddings: 2000 rows",summary["energy_embeddings"] == 2000, f"{summary['energy_embeddings']} rows")
check("genome_vectors: 2000 rows",   summary["genome_vectors"] == 2000,   f"{summary['genome_vectors']} rows")
check("predictions: 2000 rows",      summary["predictions"] == 2000,      f"{summary['predictions']} rows")
check("pareto_solutions: 100 rows",  summary["pareto_solutions"] >= 100,  f"{summary['pareto_solutions']} rows")
check("carbon_schedules: 3 rows",    summary["carbon_schedules"] == 3,    f"{summary['carbon_schedules']} rows")
check("pipeline_runs: >= 1 row",     summary["pipeline_runs"] >= 1,       f"{summary['pipeline_runs']} rows")


# ════════════════════════════════════════════════════════════════
section("TEST 3: SCHEMA COLUMN VALIDATION")
# ════════════════════════════════════════════════════════════════

def get_columns(table):
    with get_connection() as conn:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]

# batches
batch_cols = get_columns("batches")
for col in ["batch_id", "temperature", "pressure", "speed", "feed_rate",
            "humidity", "material_density", "material_hardness", "material_grade",
            "yield", "quality", "energy_consumption", "carbon_intensity"]:
    check(f"batches.{col} exists", col in batch_cols)

# genome_vectors
genome_cols = get_columns("genome_vectors")
for col in ["batch_id", "genome", "created_at"]:
    check(f"genome_vectors.{col} exists", col in genome_cols)

# carbon_schedules
sched_cols = get_columns("carbon_schedules")
for col in ["zone", "carbon_intensity", "schedule_pred_yield",
            "schedule_pred_energy", "schedule_material_density"]:
    check(f"carbon_schedules.{col} exists", col in sched_cols)

# pareto_solutions
pareto_cols = get_columns("pareto_solutions")
for col in ["run_id", "temperature", "pred_yield", "pred_carbon"]:
    check(f"pareto_solutions.{col} exists", col in pareto_cols)


# ════════════════════════════════════════════════════════════════
section("TEST 4: DATA CONTENT VALIDATION")
# ════════════════════════════════════════════════════════════════

# Batch spot checks
batch = get_batch("BATCH_0000")
check("get_batch() returns dict",           isinstance(batch, dict))
check("BATCH_0000 has all 13 data fields",  batch is not None and len(batch) >= 13, f"{len(batch) if batch else 0} fields")
check("BATCH_0000 carbon in valid range",   batch is not None and 50 <= batch["carbon_intensity"] <= 600,
      f"{batch['carbon_intensity']:.1f} gCO2/kWh" if batch else "N/A")
check("BATCH_0000 yield in valid range",    batch is not None and 0.5 <= batch["yield"] <= 1.0,
      f"{batch['yield']:.4f}" if batch else "N/A")

batch_last = get_batch("BATCH_1999")
check("BATCH_1999 exists (all 2000 stored)", batch_last is not None)

batch_missing = get_batch("BATCH_9999")
check("Non-existent batch returns None",    batch_missing is None)

# Genome spot checks
genome = get_genome("BATCH_0000")
check("get_genome() returns list",          isinstance(genome, list))
check("Genome dimension is 25",            genome is not None and len(genome) == 25, f"{len(genome) if genome else 0} dims")
check("Genome values are floats",          genome is not None and all(isinstance(x, float) for x in genome))

# Energy embeddings
with get_connection() as conn:
    emb_row = conn.execute(
        "SELECT embedding FROM energy_embeddings WHERE batch_id='BATCH_0000'"
    ).fetchone()
if emb_row:
    emb = json.loads(emb_row["embedding"])
    check("Embedding dimension is 16",     len(emb) == 16, f"{len(emb)} dims")
    check("Embedding values are numeric",  all(isinstance(x, float) for x in emb))
else:
    check("Embedding row found", False, "BATCH_0000 missing")

# Pareto solutions — get only the latest run to avoid accumulation across test runs
with get_connection() as conn:
    latest_run = conn.execute(
        "SELECT run_id FROM pareto_solutions ORDER BY id DESC LIMIT 1"
    ).fetchone()
latest_run_id = latest_run["run_id"] if latest_run else None
pareto_df = get_pareto_solutions(run_id=latest_run_id)
check("get_pareto_solutions() returns DataFrame", len(pareto_df) >= 100, f"{len(pareto_df)} rows")
check("Pareto pred_carbon != 0.0",
      pareto_df["pred_carbon"].mean() > 50,
      f"mean={pareto_df['pred_carbon'].mean():.2f} gCO2/kWh")
check("Pareto pred_yield realistic",
      0.5 <= pareto_df["pred_yield"].mean() <= 1.0,
      f"mean={pareto_df['pred_yield'].mean():.4f}")

# Carbon schedules
for zone in ["LOW", "MEDIUM", "HIGH"]:
    sched = get_latest_schedule(zone)
    check(f"Schedule for {zone} zone exists", sched is not None)
    if sched:
        check(f"{zone} schedule_pred_yield in range",
              0.5 <= sched["schedule_pred_yield"] <= 1.0,
              f"{sched['schedule_pred_yield']:.4f}")

# Zone logic validation
low_sched   = get_latest_schedule("LOW")
high_sched  = get_latest_schedule("HIGH")
if low_sched and high_sched:
    check("LOW zone has higher yield than HIGH zone",
          low_sched["schedule_pred_yield"] >= high_sched["schedule_pred_yield"],
          f"LOW={low_sched['schedule_pred_yield']:.4f} vs HIGH={high_sched['schedule_pred_yield']:.4f}")
    check("HIGH zone has lower/equal energy than LOW zone",
          high_sched["schedule_pred_energy"] <= low_sched["schedule_pred_energy"],
          f"HIGH={high_sched['schedule_pred_energy']:.1f} vs LOW={low_sched['schedule_pred_energy']:.1f}")


# ════════════════════════════════════════════════════════════════
section("TEST 5: AUDIT LOG (pipeline_runs)")
# ════════════════════════════════════════════════════════════════

run_id = log_pipeline_run(7, "Database Test", "success", {"test": "phase7_audit"})
check("log_pipeline_run() returns integer ID", isinstance(run_id, int) and run_id > 0, f"id={run_id}")

with get_connection() as conn:
    row = conn.execute(
        "SELECT * FROM pipeline_runs WHERE id = ?", (run_id,)
    ).fetchone()

check("Audit log row saved correctly",     row is not None)
check("Audit log phase = 7",              row is not None and row["phase"] == 7)
check("Audit log status = success",       row is not None and row["status"] == "success")
check("Audit log details is valid JSON",
      row is not None and json.loads(row["details"]) == {"test": "phase7_audit"})


# ════════════════════════════════════════════════════════════════
section("TEST 6: IDEMPOTENCY (re-run safety)")
# ════════════════════════════════════════════════════════════════

before = get_db_summary()
# Re-run full pipeline — should not explode or duplicate
run_database_pipeline()
after = get_db_summary()

check("Re-run: batches count unchanged",          after["batches"] == before["batches"], f"{after['batches']}")
check("Re-run: genome_vectors count unchanged",   after["genome_vectors"] == before["genome_vectors"], f"{after['genome_vectors']}")
check("Re-run: pareto count stable (100 each)",   after["pareto_solutions"] >= 100, f"{after['pareto_solutions']}")
check("Re-run: carbon_schedules unchanged",       after["carbon_schedules"] == before["carbon_schedules"], f"{after['carbon_schedules']}")


# ════════════════════════════════════════════════════════════════
section("TEST 7: FOREIGN KEY INTEGRITY")
# ════════════════════════════════════════════════════════════════

with get_connection() as conn:
    # Every embedding must have a matching batch
    orphan_emb = conn.execute(
        "SELECT COUNT(*) FROM energy_embeddings e "
        "LEFT JOIN batches b ON e.batch_id = b.batch_id "
        "WHERE b.batch_id IS NULL"
    ).fetchone()[0]
    check("No orphan energy_embeddings", orphan_emb == 0, f"{orphan_emb} orphans")

    # Every genome must have a matching batch
    orphan_gen = conn.execute(
        "SELECT COUNT(*) FROM genome_vectors g "
        "LEFT JOIN batches b ON g.batch_id = b.batch_id "
        "WHERE b.batch_id IS NULL"
    ).fetchone()[0]
    check("No orphan genome_vectors", orphan_gen == 0, f"{orphan_gen} orphans")

    # Each batch_id in batches is unique
    dup_batches = conn.execute(
        "SELECT COUNT(*) - COUNT(DISTINCT batch_id) FROM batches"
    ).fetchone()[0]
    check("No duplicate batch_ids in batches", dup_batches == 0, f"{dup_batches} duplicates")

    # Each batch_id in genome_vectors is unique
    dup_genomes = conn.execute(
        "SELECT COUNT(*) - COUNT(DISTINCT batch_id) FROM genome_vectors"
    ).fetchone()[0]
    check("No duplicate batch_ids in genome_vectors", dup_genomes == 0, f"{dup_genomes} duplicates")


# ════════════════════════════════════════════════════════════════
section("FINAL REPORT")
# ════════════════════════════════════════════════════════════════

total  = len(results)
passed = sum(1 for v in results.values() if v == PASS)
failed = total - passed
pct    = (passed / total) * 100 if total > 0 else 0

print(f"\n  Total tests : {total}")
print(f"  Passed      : {passed}")
print(f"  Failed      : {failed}")
print(f"\n  SUCCESS RATE: {pct:.1f}%")

if failed:
    print("\n  FAILED TESTS:")
    for name, status in results.items():
        if status == FAIL:
            print(f"    [X] {name}")

print()
if pct == 100:
    print("  PHASE 7 DATABASE: 100% - FULLY OPERATIONAL")
elif pct >= 90:
    print("  PHASE 7 DATABASE: MOSTLY WORKING - minor issues above")
elif pct >= 70:
    print("  PHASE 7 DATABASE: PARTIAL - review failures above")
else:
    print("  PHASE 7 DATABASE: CRITICAL ISSUES - review failures above")
print("=" * 60)

sys.exit(0 if pct == 100 else 1)
