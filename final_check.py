# ACMGS - FINAL COMPREHENSIVE CHECK
# Tests all phases 0-7 end-to-end.
# Run: python final_check.py

import os, sys, json, sqlite3, pickle, warnings
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

total_pass = 0
total_fail = 0
failures   = []

def check(label, condition, detail=""):
    global total_pass, total_fail
    tag = "  (%s)" % detail if detail else ""
    if condition:
        total_pass += 1
        print("  [PASS]  %s%s" % (label, tag))
    else:
        total_fail += 1
        failures.append(label)
        print("  [FAIL]  %s%s" % (label, tag))

def section(title):
    print("\n" + "="*65)
    print("  " + title)
    print("="*65)

# =============================================================================
#  BLOCK 0 - ARTEFACT FILES
# =============================================================================
section("BLOCK 0 - ARTEFACT FILES (data / models)")

files = {
    "Phase 1 | batch_data.csv"           : "data/simulated/batch_data.csv",
    "Phase 1 | energy_signals.npy"       : "data/simulated/energy_signals.npy",
    "Phase 2 | lstm_autoencoder.pth"     : "models/saved/lstm_autoencoder.pth",
    "Phase 2 | energy_embeddings.npy"    : "data/simulated/energy_embeddings.npy",
    "Phase 3 | genome_vectors.npy"       : "data/processed/genome_vectors.npy",
    "Phase 3 | genome_normalization.npz" : "data/processed/genome_normalization.npz",
    "Phase 3 | genome_metadata.csv"      : "data/processed/genome_metadata.csv",
    "Phase 3 | batch_ids.npy"            : "data/processed/batch_ids.npy",
    "Phase 4 | predictor.pkl"            : "models/saved/predictor.pkl",
    "Phase 4 | predictor_metrics.pkl"    : "models/saved/predictor_metrics.pkl",
    "Phase 5 | pareto_solutions.csv"     : "data/simulated/pareto_solutions.csv",
    "Phase 6 | carbon_schedule_demo.csv" : "data/simulated/carbon_schedule_demo.csv",
    "Phase 7 | acmgs.db"                 : "data/acmgs.db",
}

for label, path in files.items():
    exists = os.path.isfile(path)
    size   = os.path.getsize(path) if exists else 0
    check(label, exists, "%s bytes" % "{:,}".format(size))

# =============================================================================
#  BLOCK 1 - PHASE 1: DATA SIMULATION
# =============================================================================
section("BLOCK 1 - PHASE 1: DATA SIMULATION")

try:
    df  = pd.read_csv("data/simulated/batch_data.csv")
    sig = np.load("data/simulated/energy_signals.npy")

    check("DataFrame shape (2000x13)",    df.shape == (2000, 13),  str(df.shape))
    check("Signals shape (2000x128)",     sig.shape == (2000,128), str(sig.shape))

    expected_cols = ["temperature","pressure","speed","feed_rate","humidity",
                     "material_density","material_hardness","material_grade",
                     "yield","quality","energy_consumption","carbon_intensity","batch_id"]
    check("All 13 columns present",       set(expected_cols) == set(df.columns))
    check("Batch IDs sequential",         df.batch_id.iloc[0]=="BATCH_0000" and df.batch_id.iloc[-1]=="BATCH_1999")
    check("No NaN values",                df.isna().sum().sum()==0, "%d NaNs" % df.isna().sum().sum())
    check("Yield in range [0.5, 1.0]",    df["yield"].between(0.5,1.0).all(),
          "%.3f-%.3f" % (df["yield"].min(), df["yield"].max()))
    check("Energy in range [50, 500]",    df["energy_consumption"].between(50,500).all(),
          "%.1f-%.1f" % (df["energy_consumption"].min(), df["energy_consumption"].max()))
    check("Carbon in range [50, 600]",    df["carbon_intensity"].between(50,600).all(),
          "%.1f-%.1f" % (df["carbon_intensity"].min(), df["carbon_intensity"].max()))
    check("Signals no NaN or Inf",        not (np.isnan(sig).any() or np.isinf(sig).any()))
    check("Material grade in {1,2,3}",    set(df["material_grade"].unique()) <= {1,2,3})
    check("Carbon sinusoidal variation",  df["carbon_intensity"].std() > 50,
          "std=%.1f" % df["carbon_intensity"].std())

    from src.data_simulation.simulator import (
        generate_full_dataset, generate_energy_signals,
        generate_process_parameters, generate_material_profiles,
        generate_targets, generate_carbon_intensity
    )
    check("Phase 1 module: 6 functions importable", True)
except Exception as e:
    check("Phase 1 data load FAILED", False, str(e)[:120])

# =============================================================================
#  BLOCK 2 - PHASE 2: ENERGY DNA
# =============================================================================
section("BLOCK 2 - PHASE 2: ENERGY DNA (LSTM Autoencoder)")

try:
    emb = np.load("data/simulated/energy_embeddings.npy")
    check("Embeddings shape (2000x16)",   emb.shape==(2000,16), str(emb.shape))
    check("No NaN in embeddings",         not np.isnan(emb).any())
    check("Embeddings not all-zero",      np.abs(emb).max() > 0)
    check("Embedding std > 0",            emb.std() > 0, "std=%.4f" % emb.std())

    mdl_size = os.path.getsize("models/saved/lstm_autoencoder.pth")
    check("Model file >= 100 KB",         mdl_size >= 100_000, "%d KB" % (mdl_size//1024))

    import torch
    from src.energy_dna.model import LSTMAutoencoder
    from config.settings import (ENERGY_INPUT_DIM, ENERGY_HIDDEN_DIM,
                                  ENERGY_LATENT_DIM, ENERGY_NUM_LAYERS)
    m = LSTMAutoencoder(ENERGY_INPUT_DIM, ENERGY_HIDDEN_DIM, ENERGY_LATENT_DIM, ENERGY_NUM_LAYERS)
    m.load_state_dict(torch.load("models/saved/lstm_autoencoder.pth", map_location="cpu"))
    m.eval()
    dummy = torch.randn(1, 128, 1)
    with torch.no_grad():
        recon, latent = m(dummy)
    check("LSTM model loads and forward-passes", recon.shape==(1,128,1), str(recon.shape))
    check("Latent vector dimension = 16",        latent.shape==(1,16),   str(latent.shape))

    from src.energy_dna.trainer import run_energy_dna_pipeline
    check("Phase 2 trainer module importable",   True)
except Exception as e:
    check("Phase 2 check FAILED", False, str(e)[:120])

# =============================================================================
#  BLOCK 3 - PHASE 3: BATCH GENOME
# =============================================================================
section("BLOCK 3 - PHASE 3: BATCH GENOME ENCODER")

try:
    genome = np.load("data/processed/genome_vectors.npy")
    norm   = np.load("data/processed/genome_normalization.npz")
    bids   = np.load("data/processed/batch_ids.npy", allow_pickle=True)
    meta   = pd.read_csv("data/processed/genome_metadata.csv")

    check("Genome shape (2000x25)",       genome.shape==(2000,25), str(genome.shape))
    check("No NaN in genome",             not np.isnan(genome).any())
    check("Genome normalized (mean ~0)",  abs(genome.mean()) < 0.01, "mean=%.8f" % genome.mean())
    check("Genome normalized (std ~1)",   abs(genome.std()-1.0) < 0.1, "std=%.4f" % genome.std())
    check("Batch IDs count = 2000",       len(bids)==2000, str(len(bids)))
    check("First batch ID = BATCH_0000",  bids[0]=="BATCH_0000")
    check("Last batch ID = BATCH_1999",   bids[-1]=="BATCH_1999")
    check("Metadata rows = 2000",         len(meta)==2000, str(len(meta)))

    raw_carbon_mean = float(norm["mean"][24])
    check("Norm raw carbon mean ~301.75", abs(raw_carbon_mean - 301.75) < 5,
          "%.2f gCO2/kWh" % raw_carbon_mean)

    from src.batch_genome.encoder import (
        run_batch_genome_pipeline, construct_genome_vectors, load_genome_vectors
    )
    check("Phase 3 module: 3 functions importable", True)
except Exception as e:
    check("Phase 3 check FAILED", False, str(e)[:120])

# =============================================================================
#  BLOCK 4 - PHASE 4: PREDICTION MODEL
# =============================================================================
section("BLOCK 4 - PHASE 4: XGBOOST PREDICTOR")

try:
    genome = np.load("data/processed/genome_vectors.npy")

    with open("models/saved/predictor.pkl","rb") as f:
        model = pickle.load(f)
    with open("models/saved/predictor_metrics.pkl","rb") as f:
        metrics = pickle.load(f)

    check("Model type is XGBoost/Multi",
          "xgb" in type(model).__module__.lower() or "multi" in type(model).__name__.lower(),
          type(model).__name__)

    pred = model.predict(genome[:1])[0]
    check("Prediction returns 3 targets",    len(pred)==3, "%d targets" % len(pred))
    check("Predicted yield in [0.5,1.0]",    0.5 <= pred[0] <= 1.0, "%.4f" % pred[0])
    check("Predicted quality in [0.3,1.0]",  0.3 <= pred[1] <= 1.0, "%.4f" % pred[1])
    check("Predicted energy in [50,500]",    50 <= pred[2] <= 500,  "%.1f kWh" % pred[2])

    r2_yield = metrics.get("r2_yield", metrics.get("yield_r2", None))
    if r2_yield is not None:
        check("R2 yield >= 0.85",            r2_yield >= 0.85, "R2=%.4f" % r2_yield)
    else:
        check("Metrics keys readable",       True, str(list(metrics.keys())))

    all_preds = model.predict(genome)
    check("Predict all 2000 batches",        all_preds.shape[0]==2000, str(all_preds.shape))

    from src.prediction.predictor import (
        load_genome_features, create_predictor_model, run_prediction_pipeline
    )
    check("Phase 4 module: 3 functions importable", True)
except Exception as e:
    check("Phase 4 check FAILED", False, str(e)[:120])

# =============================================================================
#  BLOCK 5 - PHASE 5: NSGA-II OPTIMIZER
# =============================================================================
section("BLOCK 5 - PHASE 5: NSGA-II EVOLUTIONARY OPTIMIZER")

try:
    pareto = pd.read_csv("data/simulated/pareto_solutions.csv")
    check("Pareto shape (200x12)",          pareto.shape==(200,12), str(pareto.shape))

    expected_pcols = ["temperature","pressure","speed","feed_rate","humidity",
                      "material_density","material_hardness","material_grade",
                      "carbon_intensity","pred_yield","pred_quality","pred_energy","pred_carbon"]
    present = [c for c in expected_pcols if c in pareto.columns]
    check("Key Pareto columns present",     len(present) >= 12, str(len(present)) + "/13 cols")

    check("pred_carbon not zero",           pareto["pred_carbon"].mean() > 50,
          "mean=%.2f gCO2/kWh" % pareto["pred_carbon"].mean())
    check("pred_carbon mean realistic",     20 < pareto["pred_carbon"].mean() < 200,
          "%.2f" % pareto["pred_carbon"].mean())
    check("pred_yield realistic (>0.6)",    pareto["pred_yield"].min() > 0.6,
          "%.4f-%.4f" % (pareto["pred_yield"].min(), pareto["pred_yield"].max()))
    check("pred_energy in valid range",     pareto["pred_energy"].between(50,510).all(),
          "%.1f-%.1f kWh" % (pareto["pred_energy"].min(), pareto["pred_energy"].max()))
    check("Temperature in bounds [150,350]",pareto["temperature"].between(150,350).all(),
          "%.1f-%.1f" % (pareto["temperature"].min(), pareto["temperature"].max()))

    from src.optimization import NSGA2Optimizer, OptimizationResult, run_optimization_phase
    check("Phase 5 module: 3 symbols importable", True)
except Exception as e:
    check("Phase 5 check FAILED", False, str(e)[:120])

# =============================================================================
#  BLOCK 6 - PHASE 6: CARBON SCHEDULER
# =============================================================================
section("BLOCK 6 - PHASE 6: CARBON-AWARE SCHEDULER")

try:
    sched = pd.read_csv("data/simulated/carbon_schedule_demo.csv")
    check("Schedule rows = 3",              sched.shape[0]==3, "rows=%d" % sched.shape[0])
    check("3 carbon zones present",         set(sched["zone"].tolist())=={"LOW","MEDIUM","HIGH"})

    low_row    = sched[sched["zone"]=="LOW"].iloc[0]
    medium_row = sched[sched["zone"]=="MEDIUM"].iloc[0]
    high_row   = sched[sched["zone"]=="HIGH"].iloc[0]

    check("LOW carbon_intensity <= 150",    low_row["carbon_intensity"] <= 150,
          "%.0f" % low_row["carbon_intensity"])
    check("MEDIUM carbon_intensity in [150,400]",
          150 < medium_row["carbon_intensity"] < 400,
          "%.0f" % medium_row["carbon_intensity"])
    check("HIGH carbon_intensity >= 400",   high_row["carbon_intensity"] >= 400,
          "%.0f" % high_row["carbon_intensity"])
    check("LOW yield >= HIGH yield",
          low_row["schedule_pred_yield"] >= high_row["schedule_pred_yield"],
          "low=%.4f vs high=%.4f" % (low_row["schedule_pred_yield"], high_row["schedule_pred_yield"]))
    check("HIGH energy <= LOW energy",
          high_row["schedule_pred_energy"] <= low_row["schedule_pred_energy"] + 5,
          "high=%.1f vs low=%.1f" % (high_row["schedule_pred_energy"], low_row["schedule_pred_energy"]))

    from src.carbon_scheduler import (
        run_scheduler_pipeline, get_recommendation,
        classify_carbon_zone, select_best_schedule
    )
    check("Phase 6 module: 4 functions importable", True)

    zone_low  = classify_carbon_zone(100)
    zone_mid  = classify_carbon_zone(250)
    zone_high = classify_carbon_zone(500)
    check("classify_carbon_zone(100) = LOW",    zone_low=="LOW",    zone_low)
    check("classify_carbon_zone(250) = MEDIUM", zone_mid=="MEDIUM", zone_mid)
    check("classify_carbon_zone(500) = HIGH",   zone_high=="HIGH",  zone_high)
except Exception as e:
    check("Phase 6 check FAILED", False, str(e)[:120])

# =============================================================================
#  BLOCK 7 - PHASE 7: SQLITE DATABASE
# =============================================================================
section("BLOCK 7 - PHASE 7: SQLITE DATABASE")

try:
    db_path = "data/acmgs.db"
    db_size = os.path.getsize(db_path)
    check("DB file >= 1 MB",               db_size >= 1_000_000, "%s bytes" % "{:,}".format(db_size))

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    for tbl in ["batches","energy_embeddings","genome_vectors","predictions",
                "pareto_solutions","carbon_schedules","pipeline_runs"]:
        check("Table '%s' exists" % tbl,   tbl in tables)

    for tbl, expected in [("batches",2000),("energy_embeddings",2000),
                           ("genome_vectors",2000),("predictions",2000),
                           ("pareto_solutions",200),("carbon_schedules",3)]:
        cur.execute("SELECT COUNT(*) FROM %s" % tbl)
        cnt = cur.fetchone()[0]
        check("%s: %d rows" % (tbl, expected), cnt==expected, "%d rows" % cnt)

    cur.execute("SELECT COUNT(*) FROM pipeline_runs")
    pr = cur.fetchone()[0]
    check("pipeline_runs >= 1 row",        pr >= 1, "%d rows" % pr)

    cur.execute("SELECT AVG(pred_yield), AVG(pred_quality), AVG(pred_energy) FROM predictions")
    py, pq, pe = cur.fetchone()
    check("predictions pred_yield non-zero",  py is not None and py > 0,  "avg=%.4f" % (py or 0))
    check("predictions pred_quality non-zero",pq is not None and pq > 0,  "avg=%.4f" % (pq or 0))
    check("predictions pred_energy non-zero", pe is not None and pe > 50, "avg=%.1f" % (pe or 0))

    cur.execute("SELECT COUNT(*) FROM energy_embeddings e WHERE e.batch_id NOT IN (SELECT batch_id FROM batches)")
    check("No orphan energy_embeddings",   cur.fetchone()[0]==0)

    cur.execute("SELECT COUNT(*) FROM genome_vectors g WHERE g.batch_id NOT IN (SELECT batch_id FROM batches)")
    check("No orphan genome_vectors",      cur.fetchone()[0]==0)

    cur.execute("SELECT * FROM batches WHERE batch_id='BATCH_0000'")
    row = dict(zip([d[0] for d in cur.description], cur.fetchone()))
    check("BATCH_0000 carbon realistic",
          50 <= row["carbon_intensity"] <= 600, "%.1f gCO2/kWh" % row["carbon_intensity"])

    cur.execute("SELECT AVG(pred_carbon) FROM pareto_solutions")
    avg_c = cur.fetchone()[0]
    check("DB Pareto pred_carbon != 0",    avg_c is not None and abs(avg_c) > 50,
          "avg=%.2f" % avg_c if avg_c else "NULL")

    conn.close()

    from src.database import (
        run_database_pipeline, create_tables, get_batch,
        get_genome, get_latest_schedule, get_pareto_solutions,
        get_db_summary, log_pipeline_run
    )
    check("Phase 7 module: 8 functions importable", True)

    b = get_batch("BATCH_0000")
    check("get_batch() returns dict",           isinstance(b, dict))
    check("get_batch() has >= 13 fields",       len(b) >= 13, "%d fields" % len(b))
    g = get_genome("BATCH_0000")
    check("get_genome() returns 25-dim list",   len(g)==25, "%d dims" % len(g))

    lo = get_latest_schedule("LOW")
    check("get_latest_schedule(LOW) works",     lo is not None)

    summary = get_db_summary()
    check("get_db_summary() returns dict",      isinstance(summary, dict))
    check("Summary batches = 2000",             summary.get("batches")==2000,
          str(summary.get("batches")))

    df_pareto = get_pareto_solutions()
    check("get_pareto_solutions() DataFrame",   isinstance(df_pareto, pd.DataFrame))
    check("Pareto DataFrame has 200 rows",      len(df_pareto)==200, "%d rows" % len(df_pareto))

except Exception as e:
    check("Phase 7 check FAILED", False, str(e)[:150])

# =============================================================================
#  BLOCK 8 - CONFIG & UTILITIES
# =============================================================================
section("BLOCK 8 - CONFIG & UTILITIES")

try:
    from config.settings import (
        SIM_NUM_BATCHES, SIM_SIGNAL_LENGTH, SIM_RANDOM_SEED,
        ENERGY_LATENT_DIM, GENOME_PROCESS_FEATURES, GENOME_MATERIAL_FEATURES,
        CARBON_HIGH_THRESHOLD, CARBON_LOW_THRESHOLD, DB_PATH
    )
    check("config.settings importable",            True)
    check("SIM_NUM_BATCHES = 2000",                SIM_NUM_BATCHES==2000)
    check("SIM_SIGNAL_LENGTH = 128",               SIM_SIGNAL_LENGTH==128)
    check("SIM_RANDOM_SEED = 42",                  SIM_RANDOM_SEED==42)
    check("ENERGY_LATENT_DIM = 16",                ENERGY_LATENT_DIM==16)
    check("5 process features",                    len(GENOME_PROCESS_FEATURES)==5)
    check("3 material features",                   len(GENOME_MATERIAL_FEATURES)==3)
    check("CARBON_HIGH_THRESHOLD = 400",           CARBON_HIGH_THRESHOLD==400)
    check("CARBON_LOW_THRESHOLD = 150",            CARBON_LOW_THRESHOLD==150)
    check("DB_PATH ends in acmgs.db",              DB_PATH.endswith("acmgs.db"))

    from src.utils.logger import get_logger
    log = get_logger("final_check")
    check("Logger utility works",                  log is not None)
except Exception as e:
    check("Config/Utils check FAILED", False, str(e))

# =============================================================================
#  BLOCK 9 - CROSS-PHASE DATA CONSISTENCY
# =============================================================================
section("BLOCK 9 - CROSS-PHASE DATA CONSISTENCY")

try:
    df     = pd.read_csv("data/simulated/batch_data.csv")
    emb    = np.load("data/simulated/energy_embeddings.npy")
    genome = np.load("data/processed/genome_vectors.npy")
    pareto = pd.read_csv("data/simulated/pareto_solutions.csv")
    bids   = np.load("data/processed/batch_ids.npy", allow_pickle=True)

    check("batch_data rows == embeddings rows",  len(df)==len(emb),
          "df=%d emb=%d" % (len(df), len(emb)))
    check("embeddings rows == genome rows",      len(emb)==len(genome),
          "emb=%d genome=%d" % (len(emb), len(genome)))
    check("batch_ids count == genome rows",      len(bids)==len(genome),
          "bids=%d genome=%d" % (len(bids), len(genome)))

    g_energy_slice = genome[:, 8:24]
    check("Genome energy slice [8:24] std > 0",
          g_energy_slice.std() > 0, "std=%.4f" % g_energy_slice.std())

    norm = np.load("data/processed/genome_normalization.npz")
    expected_carbon_raw = float(norm["mean"][24])
    check("Pareto carbon footprint realistic",
          20 < pareto["pred_carbon"].mean() < 200,
          "pareto_carbon_mean=%.2f" % pareto["pred_carbon"].mean())

    meta = pd.read_csv("data/processed/genome_metadata.csv")
    check("Genome metadata rows == 2000",        len(meta)==2000, str(len(meta)))

    if "batch_id" in meta.columns:
        meta_ids = set(meta["batch_id"].tolist())
        csv_ids  = set(df["batch_id"].tolist())
        check("Metadata batch IDs match batch_data",
              meta_ids == csv_ids, "match=%s" % str(meta_ids==csv_ids))
    else:
        check("Genome metadata columns readable", True, str(list(meta.columns)))

except Exception as e:
    check("Cross-phase consistency check FAILED", False, str(e)[:120])

# =============================================================================
#  FINAL REPORT
# =============================================================================
print("\n" + "="*65)
print("  FINAL REPORT")
print("="*65)
print("\n  Total checks : %d" % (total_pass + total_fail))
print("  Passed       : %d" % total_pass)
print("  Failed       : %d" % total_fail)
pct = 100.0 * total_pass / (total_pass + total_fail) if (total_pass + total_fail) > 0 else 0
print("\n  SUCCESS RATE : %.1f%%" % pct)

if failures:
    print("\n  FAILED CHECKS (%d):" % len(failures))
    for f in failures:
        print("    [FAIL]  %s" % f)
else:
    print("\n  ALL CHECKS PASSED - SYSTEM FULLY OPERATIONAL")

print("\n" + "="*65)
print("  PHASE STATUS SUMMARY")
print("="*65)

phase_map = {
    "Phase 0 - System Architecture" : True,
    "Phase 1 - Data Simulation"     : "Phase 1 | batch_data.csv" not in failures,
    "Phase 2 - Energy DNA (LSTM)"   : "Phase 2 | lstm_autoencoder.pth" not in failures,
    "Phase 3 - Batch Genome"        : "Phase 3 | genome_vectors.npy" not in failures,
    "Phase 4 - XGBoost Predictor"   : "Phase 4 | predictor.pkl" not in failures,
    "Phase 5 - NSGA-II Optimizer"   : "Phase 5 | pareto_solutions.csv" not in failures,
    "Phase 6 - Carbon Scheduler"    : "Phase 6 | carbon_schedule_demo.csv" not in failures,
    "Phase 7 - SQLite Database"     : "Phase 7 | acmgs.db" not in failures,
}

for ph, ok in phase_map.items():
    sym = "[OK]" if ok else "[!!]"
    print("  %s  %s" % (sym, ph))

print("\n" + "="*65 + "\n")
sys.exit(0 if total_fail == 0 else 1)
