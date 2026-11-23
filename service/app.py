"""Credit decision API: champion scorecard decides, challenger runs in shadow.

    python service/app.py                      # local, http://127.0.0.1:8080
    gunicorn -b :8080 service.app:app          # in the container (see Dockerfile)

POST /v1/score   {"applicants": [{...characteristics...}, ...]}
  -> per applicant: PD, score, decision, adverse-action reason codes, model versions,
     and (if enabled) the challenger's PD, which is logged but never used for the decision.
GET  /health     liveness / readiness
GET  /v1/model   the champion's model card (points table)

Every decision is appended to a JSON-lines prediction log. In the cloud this is a log sink or object
storage (e.g. Cloud Logging -> BigQuery, or S3), which the scheduled monitoring job reads.
"""
import hashlib
import json
import os
import sys
import threading
import time
import uuid

from flask import Flask, jsonify, request
from pydantic import BaseModel, Field, ValidationError, field_validator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from creditrisk.runtime import ScorecardRuntime  # noqa: E402

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(HERE, "model"))
CUTOFF_PD = float(os.environ.get("CUTOFF_PD", "0.35"))        # approve if PD < cut-off (set by credit policy)
SHADOW = os.environ.get("SHADOW_CHALLENGER", "1") == "1"
LOG_PATH = os.environ.get("PREDICTION_LOG", os.path.join("/tmp", "prediction_log.jsonl"))

CARD_PATH = os.path.join(MODEL_DIR, "champion_scorecard.json")
CHAMPION = ScorecardRuntime(CARD_PATH)
CHAMPION_VERSION = "scorecard-" + hashlib.sha256(open(CARD_PATH, "rb").read()).hexdigest()[:12]
SCHEMA = json.load(open(os.path.join(MODEL_DIR, "schema.json")))
_challenger, _lock = None, threading.Lock()


def challenger():
    """Lazy-load the challenger so it never slows the champion's cold start."""
    global _challenger
    if _challenger is None:
        with _lock:
            if _challenger is None:
                import joblib
                _challenger = joblib.load(os.path.join(MODEL_DIR, "challenger_gbm_monotone.joblib"))
    return _challenger


class Applicant(BaseModel):
    LIMIT_BAL: float = Field(gt=0)
    EDUCATION: str
    MARRIAGE: str
    AGE: int = Field(ge=18, le=100)
    PAY_1: int = Field(ge=-2, le=9)
    PAY_2: int = Field(ge=-2, le=9)
    PAY_3: int = Field(ge=-2, le=9)
    PAY_4: int = Field(ge=-2, le=9)
    PAY_5: int = Field(ge=-2, le=9)
    PAY_6: int = Field(ge=-2, le=9)
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)

    @field_validator("EDUCATION", "MARRIAGE")
    @classmethod
    def lower(cls, v):
        return v.strip().lower()

    def features(self):
        """Derived characteristics, computed exactly as in training (creditrisk/data.py)."""
        d = self.model_dump()
        lim = max(d["LIMIT_BAL"], 1.0)
        bills = [d[f"BILL_AMT{i}"] for i in range(1, 7)]
        pays = [d[f"PAY_{i}"] for i in range(1, 7)]
        d["utilisation_1"] = d["BILL_AMT1"] / lim
        d["utilisation_avg6"] = sum(bills) / 6 / lim
        d["pay_ratio_1"] = d["PAY_AMT1"] / max(d["BILL_AMT2"], 1.0)
        d["months_delinquent_6"] = sum(p > 0 for p in pays)
        d["max_delinquency_6"] = max(0, max(pays))
        return d


app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(status="ok", champion=CHAMPION_VERSION, shadow=SHADOW)


@app.get("/v1/model")
def model_card():
    return jsonify(version=CHAMPION_VERSION, cutoff_pd=CUTOFF_PD, scorecard=CHAMPION.card)


@app.post("/v1/score")
def score():
    t0 = time.perf_counter()
    body = request.get_json(silent=True) or {}
    raw = body.get("applicants")
    if not isinstance(raw, list) or not raw or len(raw) > 1000:
        return jsonify(error="send {'applicants': [...]} with 1-1000 applicants"), 400
    try:
        apps = [Applicant(**a).features() for a in raw]
    except ValidationError as e:
        return jsonify(error="invalid applicant", details=json.loads(e.json())), 422
    results = []
    for a in apps:
        s, p, detail = CHAMPION.score_one(a)
        results.append({"decision_id": str(uuid.uuid4()), "pd": round(p, 6), "score": round(s, 1),
                        "decision": "approve" if p < CUTOFF_PD else "decline",
                        "reason_codes": CHAMPION.reason_codes(detail) if p >= CUTOFF_PD else [],
                        "model_version": CHAMPION_VERSION})
    if SHADOW:
        try:                                    # a challenger failure must never affect the decision
            import pandas as pd
            ch = challenger().predict_proba(pd.DataFrame(apps)[SCHEMA["columns"]])[:, 1]
            for r, c in zip(results, ch):
                r["shadow"] = {"model": "gbm_monotone", "pd": round(float(c), 6)}
        except Exception as e:                  # noqa: BLE001
            app.logger.warning("shadow challenger failed: %s", e)
    latency = (time.perf_counter() - t0) * 1000
    with open(LOG_PATH, "a") as f:
        for a, r in zip(apps, results):
            f.write(json.dumps({"ts": time.time(), **r, "features": a}) + "\n")
    return jsonify(results=results, latency_ms=round(latency, 2))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8080)))
