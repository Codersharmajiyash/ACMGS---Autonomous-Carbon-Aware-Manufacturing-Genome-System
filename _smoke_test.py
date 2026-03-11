"""Quick smoke test for all 7 FastAPI endpoints."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from src.api.main import app
from fastapi.testclient import TestClient
import numpy as np
import warnings
warnings.filterwarnings("ignore")

client = TestClient(app)
results = []

def chk(label, status_code, detail=""):
    ok = status_code in (200, 201)
    sym = "[OK]  " if ok else "[FAIL]"
    print(f"  {sym} {label:<42}  HTTP {status_code}  {detail}")
    results.append(ok)

print("\n" + "=" * 70)
print("  FastAPI ENDPOINT SMOKE TEST  -  7 Core Endpoints")
print("=" * 70)

# 1. /health
r = client.get("/health")
d = r.json()
chk("GET /health", r.status_code, f'status={d.get("status")}  db={d.get("db_connected")}')

# 2. /batches/{batch_id}
r = client.get("/batches/BATCH_0001")
d = r.json()
chk("GET /batches/BATCH_0001", r.status_code, f'yield={d.get("yield_",d.get("yield","?")):.3f}' if r.status_code==200 else str(d))

# 3. /genome/{batch_id}
r = client.get("/genome/BATCH_0001")
d = r.json()
chk("GET /genome/BATCH_0001", r.status_code, f'dims={d.get("dims")}' if r.status_code==200 else str(d))

# 4. /schedule LOW (80 gCO2)
r = client.get("/schedule/80")
d = r.json()        
chk("GET /schedule/80  [LOW]", r.status_code,
    f'zone={d.get("zone")}  yield={d.get("schedule_pred_yield",0):.3f}' if r.status_code==200 else str(d))

# 5. /schedule MEDIUM (250 gCO2)
r = client.get("/schedule/250")
d = r.json()
chk("GET /schedule/250  [MEDIUM]", r.status_code,
    f'zone={d.get("zone")}  yield={d.get("schedule_pred_yield",0):.3f}' if r.status_code==200 else str(d))

# 6. /schedule HIGH (450 gCO2)
r = client.get("/schedule/450")
d = r.json()
chk("GET /schedule/450  [HIGH]", r.status_code,
    f'zone={d.get("zone")}  yield={d.get("schedule_pred_yield",0):.3f}' if r.status_code==200 else str(d))

# 7. /pareto
r = client.get("/pareto")
d = r.json()
chk("GET /pareto", r.status_code, f'count={d.get("count")} Pareto solutions' if r.status_code==200 else str(d))

# 8. /predict with genome
genome = np.load("data/processed/genome_vectors.npy")[0].tolist()
r = client.post("/predict", json={"genome": genome})
d = r.json()
chk("POST /predict  (genome vector)", r.status_code,
    f'yield={d.get("pred_yield",0):.3f}  quality={d.get("pred_quality",0):.3f}  energy={d.get("pred_energy",0):.1f}' if r.status_code==200 else str(d))

# 9. /predict with batch_id
r = client.post("/predict", json={"batch_id": "BATCH_0001"})
d = r.json()
chk("POST /predict  (batch_id lookup)", r.status_code,
    f'yield={d.get("pred_yield",0):.3f}  quality={d.get("pred_quality",0):.3f}' if r.status_code==200 else str(d))

# 10. /db/summary
r = client.get("/db/summary")
d = r.json()
chk("GET /db/summary", r.status_code,
    f'batches={d.get("batches")}  pareto={d.get("pareto_solutions")}  db={d.get("db_size_bytes")//1024}KB' if r.status_code==200 else str(d))

passed = sum(results)
total  = len(results)
print()
print(f"  {passed}/{total} endpoints OK" + ("  -  ALL PASS" if passed == total else "  -  FAILURES ABOVE"))
print("=" * 70)
sys.exit(0 if passed == total else 1)
