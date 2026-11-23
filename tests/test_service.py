"""API tests with Flask's test client (no network)."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
MODEL = os.path.join(HERE, "..", "service", "model")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(MODEL, "champion_scorecard.json")),
                                reason="run experiments/run_benchmark.py first")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path / "log.jsonl"))
    sys.path.insert(0, os.path.join(HERE, ".."))
    import importlib
    import service.app as appmod
    importlib.reload(appmod)
    return appmod.app.test_client(), tmp_path / "log.jsonl"


def example():
    ex = json.load(open(os.path.join(MODEL, "schema.json")))["example"]
    return {k: v for k, v in ex.items() if not k.startswith(("utilisation", "pay_ratio", "months_", "max_"))}


def test_health(client):
    c, _ = client
    assert c.get("/health").json["status"] == "ok"


def test_score_and_shadow_and_log(client):
    c, log = client
    r = c.post("/v1/score", json={"applicants": [example(), example()]})
    assert r.status_code == 200
    res = r.json["results"]
    assert len(res) == 2 and 0 < res[0]["pd"] < 1 and res[0]["decision"] in ("approve", "decline")
    assert "shadow" in res[0]
    assert len(open(log).read().strip().splitlines()) == 2


def test_risky_applicant_gets_reason_codes(client):
    c, _ = client
    bad = example() | {"PAY_1": 3, "PAY_2": 2, "PAY_3": 2, "PAY_AMT1": 0, "PAY_AMT2": 0}
    res = c.post("/v1/score", json={"applicants": [bad]}).json["results"][0]
    assert res["decision"] == "decline"
    assert res["reason_codes"][0]["characteristic"].startswith(("PAY", "max_", "months_"))


def test_validation(client):
    c, _ = client
    assert c.post("/v1/score", json={"applicants": [example() | {"AGE": 7}]}).status_code == 422
    assert c.post("/v1/score", json={}).status_code == 400
