"""
Phase 8: Pydantic response/request schemas for ACMGS API.

All response models validate and document the shape of every endpoint's output.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    phase: int = Field(8, description="Current highest completed phase")
    db_connected: bool
    table_counts: Dict[str, int]
    message: str = Field(..., example="ACMGS API is running")


# ─── Batch ────────────────────────────────────────────────────────────────────

class BatchResponse(BaseModel):
    batch_id: str
    temperature: float
    pressure: float
    speed: float
    feed_rate: float
    humidity: float
    material_density: float
    material_hardness: float
    material_grade: int
    yield_: Optional[float] = Field(None, alias="yield")
    quality: float
    energy_consumption: float
    carbon_intensity: float

    model_config = {"populate_by_name": True}


# ─── Genome ───────────────────────────────────────────────────────────────────

class GenomeResponse(BaseModel):
    batch_id: str
    genome: List[float] = Field(..., description="25-dimensional genome vector")
    dims: int = Field(25, description="Always 25: 5 process + 3 material + 16 energy DNA + 1 carbon")
    layout: Dict[str, str] = Field(
        default={
            "0:5"  : "process params (temp, pressure, speed, feed_rate, humidity)",
            "5:8"  : "material props (density, hardness, grade)",
            "8:24" : "energy DNA (16-dim LSTM latent)",
            "24"   : "carbon intensity",
        }
    )


# ─── Schedule ─────────────────────────────────────────────────────────────────

class ScheduleResponse(BaseModel):
    zone: str = Field(..., description="LOW | MEDIUM | HIGH")
    carbon_intensity: float = Field(..., description="Input carbon intensity (gCO2/kWh)")
    schedule_temperature: Optional[float] = None
    schedule_pressure: Optional[float] = None
    schedule_speed: Optional[float] = None
    schedule_feed_rate: Optional[float] = None
    schedule_humidity: Optional[float] = None
    schedule_pred_yield: Optional[float] = None
    schedule_pred_quality: Optional[float] = None
    schedule_pred_energy: Optional[float] = None
    schedule_pred_carbon: Optional[float] = None
    recommendation: str = Field(..., description="Human-readable recommendation text")


# ─── Pareto ───────────────────────────────────────────────────────────────────

class ParetoSolution(BaseModel):
    temperature: float
    pressure: float
    speed: float
    feed_rate: float
    humidity: float
    material_density: float
    material_hardness: float
    material_grade: int
    carbon_intensity: float
    pred_yield: float
    pred_quality: float
    pred_energy: float
    pred_carbon: float


class ParetoResponse(BaseModel):
    count: int
    run_id: Optional[str] = None
    solutions: List[Dict[str, Any]]


# ─── Prediction ───────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    batch_id: Optional[str] = Field(None, description="Predict from stored genome")
    genome: Optional[List[float]] = Field(
        None,
        description="Raw 25-dim genome vector (overrides batch_id lookup)"
    )


class PredictResponse(BaseModel):
    batch_id: Optional[str]
    pred_yield: float
    pred_quality: float
    pred_energy: float
    model: str = "XGBoost MultiOutput"


# ─── DB Summary ───────────────────────────────────────────────────────────────

class DBSummaryResponse(BaseModel):
    batches: int
    energy_embeddings: int
    genome_vectors: int
    predictions: int
    pareto_solutions: int
    carbon_schedules: int
    pipeline_runs: int
    db_size_bytes: int
