# ACMGS Phase 8 - API Test Suite
# Tests all FastAPI endpoints using httpx TestClient (no server needed)
# Run: python test_phase8.py

import os, sys, json, warnings
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

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
#  TEST 1: GET /health
# =============================================================================
section("TEST 1: GET /health")

r = client.get("/health")
check("Status code 200",              r.status_code == 200, str(r.status_code))

body = r.json()
check("Response has 'status' field",  "status" in body)
check("status = 'ok'",                body.get("status") == "ok", body.get("status"))
check("db_connected = True",          body.get("db_connected") == True)
check("phase = 8",                    body.get("phase") == 8, str(body.get("phase")))
check("table_counts is dict",         isinstance(body.get("table_counts"), dict))
check("table_counts has batches=2000",body.get("table_counts", {}).get("batches") == 2000,
      str(body.get("table_counts", {}).get("batches")))
check("message field present",        "message" in body)

# =============================================================================
#  TEST 2: GET /batches/{batch_id}
# =============================================================================
section("TEST 2: GET /batches/{batch_id}")

r = client.get("/batches/BATCH_0000")
check("BATCH_0000 status 200",        r.status_code == 200, str(r.status_code))

body = r.json()
check("Returns batch_id field",       body.get("batch_id") == "BATCH_0000")
check("Has temperature field",        "temperature" in body)
check("Has carbon_intensity field",   "carbon_intensity" in body)
check("Carbon in valid range",        50 <= body.get("carbon_intensity", 0) <= 600,
      "%.1f" % body.get("carbon_intensity", 0))
check("Yield in valid range",         0.5 <= body.get("yield_", body.get("yield", 0)) <= 1.0)

# Last batch
r2 = client.get("/batches/BATCH_1999")
check("BATCH_1999 status 200",        r2.status_code == 200, str(r2.status_code))
check("BATCH_1999 has correct ID",    r2.json().get("batch_id") == "BATCH_1999")

# Non-existent batch -> 404
r3 = client.get("/batches/BATCH_9999")
check("Non-existent batch -> 404",    r3.status_code == 404, str(r3.status_code))
check("404 has detail message",       "detail" in r3.json())

# =============================================================================
#  TEST 3: GET /genome/{batch_id}
# =============================================================================
section("TEST 3: GET /genome/{batch_id}")

r = client.get("/genome/BATCH_0000")
check("BATCH_0000 genome status 200", r.status_code == 200, str(r.status_code))

body = r.json()
check("Returns batch_id",             body.get("batch_id") == "BATCH_0000")
check("Genome is list",               isinstance(body.get("genome"), list))
check("Genome has 25 dims",           len(body.get("genome", [])) == 25,
      "%d dims" % len(body.get("genome", [])))
check("dims field = 25",              body.get("dims") == 25)
check("layout field present",         isinstance(body.get("layout"), dict))
check("Genome values are floats",     all(isinstance(v, (int, float)) for v in body["genome"][:5]))

# Non-existent genome -> 404
r2 = client.get("/genome/BATCH_9999")
check("Non-existent genome -> 404",   r2.status_code == 404, str(r2.status_code))

# =============================================================================
#  TEST 4: GET /schedule/{carbon_intensity}
# =============================================================================
section("TEST 4: GET /schedule/{carbon_intensity}")

for ci, expected_zone in [(50, "LOW"), (100, "LOW"), (250, "MEDIUM"), (350, "MEDIUM"), (450, "HIGH"), (550, "HIGH")]:
    r = client.get("/schedule/%.1f" % ci)
    check("schedule(%.0f) status 200" % ci, r.status_code == 200, str(r.status_code))
    body = r.json()
    check("schedule(%.0f) zone=%s" % (ci, expected_zone),  body.get("zone") == expected_zone,
          body.get("zone"))
    check("schedule(%.0f) carbon_intensity echoed" % ci,
          abs(body.get("carbon_intensity", -1) - ci) < 0.01)

# LOW zone checks
r_low = client.get("/schedule/100.0")
body_low = r_low.json()
check("LOW zone has recommendation",  bool(body_low.get("recommendation")))
check("LOW recommendation mentions FULL PRODUCTION",
      "FULL PRODUCTION" in body_low.get("recommendation","") or
      "full" in body_low.get("recommendation","").lower())
check("LOW schedule_pred_yield in range",
      0.5 <= (body_low.get("schedule_pred_yield") or 0) <= 1.0,
      "%.4f" % (body_low.get("schedule_pred_yield") or 0))

# HIGH zone checks
r_high = client.get("/schedule/500.0")
body_high = r_high.json()
check("HIGH zone has recommendation", bool(body_high.get("recommendation")))
check("HIGH recommendation mentions CONSERVATION",
      "CONSERVATION" in body_high.get("recommendation","") or
      "conservation" in body_high.get("recommendation","").lower())

# Out of range
r_bad = client.get("/schedule/9999.0")
check("carbon_intensity=9999 -> 422", r_bad.status_code == 422, str(r_bad.status_code))

# =============================================================================
#  TEST 5: GET /pareto
# =============================================================================
section("TEST 5: GET /pareto")

r = client.get("/pareto")
check("GET /pareto status 200",       r.status_code == 200, str(r.status_code))
body = r.json()
check("Has 'count' field",            "count" in body)
check("Has 'solutions' field",        "solutions" in body)
check("count = 100",                  body.get("count") == 100, str(body.get("count")))
check("solutions is list",            isinstance(body.get("solutions"), list))
check("solutions length = count",     len(body["solutions"]) == body["count"])

if body["solutions"]:
    sol = body["solutions"][0]
    check("Solution has pred_yield",  "pred_yield" in sol)
    check("Solution has pred_carbon", "pred_carbon" in sol)
    check("Solution has temperature", "temperature" in sol)
    check("pred_carbon != 0",         abs(sol.get("pred_carbon", 0)) > 50,
          "%.2f" % sol.get("pred_carbon", 0))

# Filter by min_yield
r2 = client.get("/pareto?min_yield=0.84")
body2 = r2.json()
check("min_yield filter: status 200",  r2.status_code == 200)
check("min_yield filter returns list", isinstance(body2.get("solutions"), list))
if body2["solutions"]:
    check("min_yield filter: all yield>=0.84",
          all(s["pred_yield"] >= 0.84 for s in body2["solutions"]))

# Filter by max_carbon
r3 = client.get("/pareto?max_carbon=320")
body3 = r3.json()
check("max_carbon filter: status 200", r3.status_code == 200)
if body3["solutions"]:
    check("max_carbon filter applied",
          all(s["pred_carbon"] <= 320 for s in body3["solutions"]))

# Limit
r4 = client.get("/pareto?limit=10")
check("limit=10 returns <=10",        len(r4.json().get("solutions",[])) <= 10,
      str(len(r4.json().get("solutions",[]))))

# =============================================================================
#  TEST 6: POST /predict
# =============================================================================
section("TEST 6: POST /predict")

# Predict using batch_id
r = client.post("/predict", json={"batch_id": "BATCH_0000"})
check("predict(batch_id) status 200", r.status_code == 200, str(r.status_code))
body = r.json()
check("predict returns pred_yield",   "pred_yield" in body)
check("predict returns pred_quality", "pred_quality" in body)
check("predict returns pred_energy",  "pred_energy" in body)
check("pred_yield in [0.5,1.0]",      0.5 <= body.get("pred_yield",0) <= 1.0,
      "%.4f" % body.get("pred_yield",0))
check("pred_quality in [0.3,1.0]",    0.3 <= body.get("pred_quality",0) <= 1.0)
check("pred_energy in [50,600]",      50 <= body.get("pred_energy",0) <= 600)
check("model field = XGBoost MultiOutput", "XGBoost" in body.get("model",""))

# Predict using raw genome
import numpy as np
import pickle
genome = np.load("data/processed/genome_vectors.npy")[0].tolist()
r2 = client.post("/predict", json={"genome": genome})
check("predict(genome) status 200",   r2.status_code == 200, str(r2.status_code))
body2 = r2.json()
check("predict(genome) pred_yield in range", 0.5 <= body2.get("pred_yield",0) <= 1.0,
      "%.4f" % body2.get("pred_yield",0))

# Wrong dims -> 422
r3 = client.post("/predict", json={"genome": [0.1]*10})
check("Wrong genome dims -> 422",     r3.status_code == 422, str(r3.status_code))

# Non-existent batch_id -> 404
r4 = client.post("/predict", json={"batch_id": "BATCH_9999"})
check("Non-existent batch_id -> 404", r4.status_code == 404, str(r4.status_code))

# No body -> 422
r5 = client.post("/predict", json={})
check("Empty body -> 422",            r5.status_code == 422, str(r5.status_code))

# =============================================================================
#  TEST 7: GET /db/summary
# =============================================================================
section("TEST 7: GET /db/summary")

r = client.get("/db/summary")
check("GET /db/summary status 200",   r.status_code == 200, str(r.status_code))
body = r.json()
check("batches = 2000",               body.get("batches") == 2000,    str(body.get("batches")))
check("energy_embeddings = 2000",     body.get("energy_embeddings") == 2000)
check("genome_vectors = 2000",        body.get("genome_vectors") == 2000)
check("pareto_solutions >= 100",       body.get("pareto_solutions",0) >= 100)
check("carbon_schedules = 3",         body.get("carbon_schedules") == 3)
check("pipeline_runs >= 1",           body.get("pipeline_runs",0) >= 1)
check("db_size_bytes >= 1MB",         body.get("db_size_bytes",0) >= 1_000_000,
      "%s bytes" % "{:,}".format(body.get("db_size_bytes",0)))

# =============================================================================
#  TEST 8: SCHEMA / OPENAPI VALIDATION
# =============================================================================
section("TEST 8: OpenAPI Schema")

r = client.get("/openapi.json")
check("GET /openapi.json status 200",   r.status_code == 200)
schema = r.json()
check("OpenAPI title = ACMGS API",      schema.get("info",{}).get("title") == "ACMGS API")
check("OpenAPI version present",        bool(schema.get("info",{}).get("version")))

paths = schema.get("paths", {})
check("/health in paths",               "/health" in paths)
check("/batches/{batch_id} in paths",   "/batches/{batch_id}" in paths)
check("/genome/{batch_id} in paths",    "/genome/{batch_id}" in paths)
check("/schedule/{carbon_intensity} in paths",
      "/schedule/{carbon_intensity}" in paths)
check("/pareto in paths",               "/pareto" in paths)
check("/predict in paths",              "/predict" in paths)
check("/db/summary in paths",           "/db/summary" in paths)

r2 = client.get("/docs")
check("GET /docs (Swagger UI) is up",   r2.status_code == 200)

# =============================================================================
#  FINAL REPORT
# =============================================================================
print("\n" + "="*65)
print("  FINAL REPORT")
print("="*65)
print("\n  Total tests  : %d" % (total_pass + total_fail))
print("  Passed       : %d" % total_pass)
print("  Failed       : %d" % total_fail)
pct = 100.0 * total_pass / (total_pass + total_fail) if (total_pass + total_fail) > 0 else 0
print("\n  SUCCESS RATE : %.1f%%" % pct)

if failures:
    print("\n  FAILED TESTS (%d):" % len(failures))
    for f in failures:
        print("    [FAIL]  %s" % f)
else:
    print("\n  ALL TESTS PASSED - PHASE 8 API: FULLY OPERATIONAL")

print("\n" + "="*65 + "\n")
sys.exit(0 if total_fail == 0 else 1)
