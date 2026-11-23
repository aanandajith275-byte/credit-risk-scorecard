"""What does each model cost to run as a serverless scoring function?

    python experiments/cloud_benchmark.py        (~3 min; needs run_benchmark.py artifacts)

Measured on this machine, single-threaded (OMP_NUM_THREADS=1, because a Lambda function at <= 1,769 MB
gets at most one vCPU):
  * deployment artifact size
  * cold start: a fresh Python process that imports its libraries, loads the model and scores one
    applicant (what the first request after a scale-out waits for)
  * peak memory of that process (VmHWM: ru_maxrss would include the parent's memory inherited at fork)
  * warm single-request latency (p50 / p99) and batch throughput

Then converted to AWS Lambda cost (us-east-1, x86, published prices):
  $0.20 per 1M requests + $0.0000166667 per GB-second, billed per ms; INIT (cold start) time is billed
  since 1 Aug 2025; CPU share is proportional to memory, with 1,769 MB = 1 vCPU, so a smaller memory
  setting is cheaper per second but slower.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
from creditrisk.data import load_taiwan  # noqa: E402
from creditrisk.runtime import ScorecardRuntime  # noqa: E402

RES, ART, SVC = os.path.join(ROOT, "results"), os.path.join(ROOT, "models"), os.path.join(ROOT, "service", "model")
PRICE_REQ, PRICE_GBS, VCPU_MB = 0.20 / 1e6, 0.0000166667, 1769
MEMORY_OPTIONS = [128, 256, 512, 1024, 1769, 3008]
COLD_FRACTION = 0.01            # assumed share of requests that hit a cold start

X, y, meta = load_taiwan()
P = pd.read_csv(os.path.join(RES, "holdout_predictions_taiwan.csv"))
Xte = X.iloc[P["row"].to_numpy()]
example = {k: (v.item() if hasattr(v, "item") else v) for k, v in Xte.iloc[0].to_dict().items()}
json.dump(example, open("/tmp/_example_applicant.json", "w"))

COLD = {
    "scorecard": f"""
import sys, json, resource, time
t0 = time.perf_counter()
sys.path.insert(0, {SRC!r})
from creditrisk.runtime import ScorecardRuntime
rt = ScorecardRuntime({os.path.join(SVC, 'champion_scorecard.json')!r})
rt.score_one(json.load(open('/tmp/_example_applicant.json')))
hwm = [l for l in open('/proc/self/status') if l.startswith('VmHWM')][0].split()[1]
print(json.dumps({{"inner": time.perf_counter() - t0, "rss_mb": int(hwm) / 1024}}))
""",
}
CHALLENGER_TEMPLATE = """
import os, sys, json, resource, time
os.environ["OMP_NUM_THREADS"] = "1"
t0 = time.perf_counter()
sys.path.insert(0, {src!r})
import joblib, pandas as pd
m = joblib.load({path!r})
m.predict_proba(pd.DataFrame([json.load(open('/tmp/_example_applicant.json'))]))
hwm = [l for l in open('/proc/self/status') if l.startswith('VmHWM')][0].split()[1]
print(json.dumps({{"inner": time.perf_counter() - t0, "rss_mb": int(hwm) / 1024}}))
"""
models = [c for c in P.columns if c not in ("row", "y", "exposure", "sex")]
for m in models:
    if m != "scorecard":
        COLD[m] = CHALLENGER_TEMPLATE.format(src=SRC, path=os.path.join(ART, f"taiwan_{m}.joblib"))


def run_cold(code, reps=7):
    walls, inner, rss = [], [], []
    for _ in range(reps):
        t = time.perf_counter()
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
        walls.append(time.perf_counter() - t)
        d = json.loads(out.strip().splitlines()[-1])
        inner.append(d["inner"])
        rss.append(d["rss_mb"])
    return float(np.median(walls)), float(np.median(inner)), float(np.max(rss))


bare = run_cold("import json; hwm = [l for l in open('/proc/self/status') if l.startswith('VmHWM')][0].split()[1]; "
                "print(json.dumps({'inner': 0, 'rss_mb': int(hwm) / 1024}))")
print(f"bare interpreter start: {bare[0] * 1000:.0f} ms, {bare[2]:.0f} MB")

rows = []
records = Xte.head(300).to_dict("records")
for m in models:
    if m == "scorecard":
        size = os.path.getsize(os.path.join(SVC, "champion_scorecard.json")) + os.path.getsize(os.path.join(SRC, "creditrisk", "runtime.py"))
        rt = ScorecardRuntime(os.path.join(SVC, "champion_scorecard.json"))
        one = lambda r: rt.score_one(r)                                       # noqa: E731
        t = time.perf_counter()
        for r in Xte.to_dict("records"):
            rt.score_one(r)
        batch_rate = len(Xte) / (time.perf_counter() - t)
        deps = "Python standard library only"
    else:
        path = os.path.join(ART, f"taiwan_{m}.joblib")
        size = os.path.getsize(path)
        est = joblib.load(path)
        for k in est.get_params():
            if k.endswith("n_jobs"):
                est.set_params(**{k: 1})
        one = lambda r: est.predict_proba(pd.DataFrame([r]))                  # noqa: E731
        t = time.perf_counter()
        est.predict_proba(Xte)
        batch_rate = len(Xte) / (time.perf_counter() - t)
        deps = "numpy, scipy, pandas, scikit-learn"
    for r in records[:20]:
        one(r)                                                                # warm up
    lat = []
    for r in records:
        t = time.perf_counter()
        one(r)
        lat.append(time.perf_counter() - t)
    wall, inner, rss = run_cold(COLD[m])
    p50 = float(np.median(lat))
    # cheapest Lambda memory setting that fits (1.25x peak RSS) -- duration scales with the CPU share
    best = None
    for mem in MEMORY_OPTIONS:
        if mem < 1.25 * rss:
            continue
        slow = max(1.0, VCPU_MB / mem)
        dur = math.ceil(p50 * slow * 1000) / 1000
        init = math.ceil(inner * slow * 1000) / 1000
        gb = mem / 1024
        cost = PRICE_REQ + (1 - COLD_FRACTION) * dur * gb * PRICE_GBS + COLD_FRACTION * (dur + init) * gb * PRICE_GBS
        cand = {"lambda_memory_mb": mem, "billed_ms": dur * 1000, "cold_start_s_at_memory": init,
                "cost_per_million_usd": cost * 1e6}
        if best is None or cand["cost_per_million_usd"] < best["cost_per_million_usd"]:
            best = cand
    batch_cost_per_million = 1e6 / batch_rate * (VCPU_MB / 1024) * PRICE_GBS
    rows.append({"model": m, "artifact_kb": size / 1024, "dependencies": deps, "cold_start_ms": wall * 1000,
                 "cold_start_in_process_ms": inner * 1000, "peak_rss_mb": rss, "p50_ms": p50 * 1000,
                 "p99_ms": float(np.quantile(lat, 0.99)) * 1000, "batch_rows_per_s": batch_rate,
                 **best, "batch_cost_per_million_usd": batch_cost_per_million})
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "cloud_benchmark.csv"), index=False)
json.dump({"bare_python_start_ms": bare[0] * 1000, "bare_python_rss_mb": bare[2], "cold_fraction": COLD_FRACTION,
           "price_per_request": PRICE_REQ, "price_per_gb_s": PRICE_GBS, "vcpu_mb": VCPU_MB},
          open(os.path.join(RES, "cloud_benchmark_meta.json"), "w"), indent=1)
