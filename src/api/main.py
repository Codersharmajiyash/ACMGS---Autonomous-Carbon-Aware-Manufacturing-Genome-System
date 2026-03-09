"""
Phase 8: ACMGS FastAPI REST API

Endpoints
---------
GET  /health                      — system status + DB table counts
GET  /batches/{batch_id}          — fetch one batch from DB
GET  /genome/{batch_id}           — fetch 25-dim genome vector from DB
GET  /schedule/{carbon_intensity} — carbon-aware manufacturing recommendation
GET  /pareto                      — all Pareto-optimal manufacturing solutions
POST /predict                     — run XGBoost prediction on a genome vector
GET  /db/summary                  — DB row counts + file size

Run with:
    python -m src.api.main
    or
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import pickle
import sqlite3
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from config.settings import DB_PATH, MODELS_DIR, PROCESSED_DIR
from src.api.schemas import (
    BatchResponse,
    DBSummaryResponse,
    GenomeResponse,
    HealthResponse,
    ParetoResponse,
    PredictRequest,
    PredictResponse,
    ScheduleResponse,
)
from src.carbon_scheduler import classify_carbon_zone
from src.database import (
    get_batch,
    get_db_summary,
    get_genome,
    get_latest_schedule,
    get_pareto_solutions,
)
from src.utils.logger import get_logger

logger = get_logger("api")

# ─── App creation ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="ACMGS API",
    description=(
        "Autonomous Carbon-aware Manufacturing Genome System — "
        "REST interface for all pipeline phases (1-7)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Lazy-loaded predictor ────────────────────────────────────────────────────

_predictor = None

def get_predictor():
    """Load the XGBoost predictor once and cache it."""
    global _predictor
    if _predictor is None:
        pkl_path = os.path.join(MODELS_DIR, "predictor.pkl")
        if not os.path.isfile(pkl_path):
            raise HTTPException(status_code=503, detail="Predictor model not found. Run Phase 4 first.")
        with open(pkl_path, "rb") as f:
            _predictor = pickle.load(f)
        logger.info("Predictor loaded from %s" % pkl_path)
    return _predictor


# =============================================================================
#  GET /health
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """
    System health check.
    Returns API status, DB connectivity, and table row counts.
    """
    try:
        summary = get_db_summary()
        db_ok = True
    except Exception as e:
        logger.error("DB health check failed: %s" % e)
        summary = {}
        db_ok = False

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        phase=8,
        db_connected=db_ok,
        table_counts=summary,
        message="ACMGS API is running — Phase 8 active",
    )


# =============================================================================
#  GET /batches/{batch_id}
# =============================================================================

@app.get("/batches/{batch_id}", response_model=BatchResponse, tags=["Data"])
def get_batch_endpoint(batch_id: str):
    """
    Retrieve a single manufacturing batch record by its ID.

    - **batch_id**: e.g. `BATCH_0042`
    """
    row = get_batch(batch_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Batch '%s' not found" % batch_id)
    # map 'yield' key to 'yield_' to avoid Python keyword clash
    row_out = dict(row)
    return BatchResponse(**row_out)


# =============================================================================
#  GET /genome/{batch_id}
# =============================================================================

@app.get("/genome/{batch_id}", response_model=GenomeResponse, tags=["Data"])
def get_genome_endpoint(batch_id: str):
    """
    Retrieve the 25-dimensional genome vector for a batch.

    Genome layout:
    - `[0:5]`  — process params (temperature, pressure, speed, feed_rate, humidity)
    - `[5:8]`  — material props (density, hardness, grade)
    - `[8:24]` — energy DNA (16-dim LSTM latent vector from Phase 2)
    - `[24]`   — carbon intensity
    """
    genome = get_genome(batch_id)
    if genome is None:
        raise HTTPException(status_code=404, detail="Genome for '%s' not found" % batch_id)
    return GenomeResponse(batch_id=batch_id, genome=genome, dims=len(genome))


# =============================================================================
#  GET /schedule/{carbon_intensity}
# =============================================================================

@app.get("/schedule/{carbon_intensity}", response_model=ScheduleResponse, tags=["Scheduling"])
def get_schedule_endpoint(carbon_intensity: float):
    """
    Get the optimal manufacturing schedule for the given grid carbon intensity.

    - Carbon ≤ 150 gCO2/kWh → **LOW** zone: maximize production
    - 150 < Carbon < 400   → **MEDIUM** zone: balance efficiency
    - Carbon ≥ 400 gCO2/kWh → **HIGH** zone: minimize energy/carbon

    - **carbon_intensity**: current grid carbon in gCO2/kWh (e.g. `250.0`)
    """
    if carbon_intensity < 0 or carbon_intensity > 1000:
        raise HTTPException(
            status_code=422,
            detail="carbon_intensity must be between 0 and 1000 gCO2/kWh"
        )

    zone = classify_carbon_zone(carbon_intensity)
    sched = get_latest_schedule(zone)

    if sched is None:
        raise HTTPException(
            status_code=503,
            detail="No schedule found for zone '%s'. Run Phase 6 first." % zone
        )

    # Build human-readable recommendation
    recs = {
        "LOW"    : "Grid is clean — run at FULL PRODUCTION. Maximize yield and quality.",
        "MEDIUM" : "Grid is moderate — BALANCED MODE. Optimize for efficiency.",
        "HIGH"   : "Grid is dirty — CONSERVATION MODE. Minimize energy and carbon emissions.",
    }

    return ScheduleResponse(
        zone=zone,
        carbon_intensity=carbon_intensity,
        schedule_temperature=sched.get("schedule_temperature"),
        schedule_pressure=sched.get("schedule_pressure"),
        schedule_speed=sched.get("schedule_speed"),
        schedule_feed_rate=sched.get("schedule_feed_rate"),
        schedule_humidity=sched.get("schedule_humidity"),
        schedule_pred_yield=sched.get("schedule_pred_yield"),
        schedule_pred_quality=sched.get("schedule_pred_quality"),
        schedule_pred_energy=sched.get("schedule_pred_energy"),
        schedule_pred_carbon=sched.get("schedule_pred_carbon"),
        recommendation=recs[zone],
    )


# =============================================================================
#  GET /pareto
# =============================================================================

@app.get("/pareto", response_model=ParetoResponse, tags=["Optimization"])
def list_pareto(
    limit: int = Query(100, ge=1, le=500, description="Max solutions to return"),
    min_yield: float = Query(0.0, ge=0.0, le=1.0, description="Filter: min predicted yield"),
    max_carbon: float = Query(1000.0, ge=0.0, description="Filter: max predicted carbon (gCO2/kWh)"),
):
    """
    List Pareto-optimal manufacturing configurations from Phase 5.

    Optional filters:
    - **min_yield**: only return solutions where `pred_yield >= min_yield`
    - **max_carbon**: only return solutions where `pred_carbon <= max_carbon`
    - **limit**: cap the number of results
    """
    df = get_pareto_solutions()
    if df is None or df.empty:
        raise HTTPException(status_code=503, detail="No Pareto solutions found. Run Phase 5 first.")

    # Apply filters
    df = df[df["pred_yield"] >= min_yield]
    df = df[df["pred_carbon"] <= max_carbon]
    df = df.head(limit)

    # Round floats for clean JSON
    df = df.round(4)

    run_id = str(df["run_id"].iloc[0]) if "run_id" in df.columns else None

    return ParetoResponse(
        count=len(df),
        run_id=run_id,
        solutions=df.to_dict(orient="records"),
    )


# =============================================================================
#  POST /predict
# =============================================================================

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_endpoint(req: PredictRequest):
    """
    Run the XGBoost predictor on a genome vector.

    Supply EITHER:
    - **batch_id**: looks up the stored genome from the DB
    - **genome**: a raw 25-dimensional float list

    Returns predicted yield, quality, and energy consumption.
    """
    model = get_predictor()

    if req.genome is not None:
        # Use provided genome directly
        if len(req.genome) != 25:
            raise HTTPException(
                status_code=422,
                detail="genome must have exactly 25 dimensions, got %d" % len(req.genome)
            )
        genome_arr = np.array(req.genome, dtype=np.float32).reshape(1, -1)
        batch_id = req.batch_id
    elif req.batch_id is not None:
        genome_vals = get_genome(req.batch_id)
        if genome_vals is None:
            raise HTTPException(
                status_code=404,
                detail="Genome for batch '%s' not found" % req.batch_id
            )
        genome_arr = np.array(genome_vals, dtype=np.float32).reshape(1, -1)
        batch_id = req.batch_id
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'batch_id' or 'genome' in the request body."
        )

    preds = model.predict(genome_arr)[0]

    return PredictResponse(
        batch_id=batch_id,
        pred_yield=round(float(preds[0]), 4),
        pred_quality=round(float(preds[1]), 4),
        pred_energy=round(float(preds[2]), 2),
    )


# =============================================================================
#  GET /db/summary
# =============================================================================

@app.get("/db/summary", response_model=DBSummaryResponse, tags=["System"])
def db_summary():
    """
    Database summary — row counts for all tables + DB file size in bytes.
    """
    try:
        summary = get_db_summary()
    except Exception as e:
        raise HTTPException(status_code=503, detail="DB unavailable: %s" % str(e))

    db_size = os.path.getsize(DB_PATH) if os.path.isfile(DB_PATH) else 0

    return DBSummaryResponse(
        batches=summary.get("batches", 0),
        energy_embeddings=summary.get("energy_embeddings", 0),
        genome_vectors=summary.get("genome_vectors", 0),
        predictions=summary.get("predictions", 0),
        pareto_solutions=summary.get("pareto_solutions", 0),
        carbon_schedules=summary.get("carbon_schedules", 0),
        pipeline_runs=summary.get("pipeline_runs", 0),
        db_size_bytes=db_size,
    )


# =============================================================================
#  Entry point — run directly
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting ACMGS API server on http://0.0.0.0:8000")
    logger.info("Docs available at: http://localhost:8000/docs")
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
