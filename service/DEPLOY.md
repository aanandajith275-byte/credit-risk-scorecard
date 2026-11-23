# Deploying the scoring service

The repository is deploy-ready but is **not deployed** from here. Two routes, both serverless (pay per request, scale to zero):

## Google Cloud Run (container)

```bash
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT/credit/scoring:$(git rev-parse --short HEAD) -f service/Dockerfile .
gcloud run deploy credit-scoring --image REGION-docker.pkg.dev/PROJECT/credit/scoring:TAG \
    --region asia-northeast1 --memory 512Mi --cpu 1 --concurrency 40 --min-instances 0 \
    --set-env-vars CUTOFF_PD=0.35,SHADOW_CHALLENGER=1 --no-allow-unauthenticated
```

* Prediction log: write to stdout as JSON and route it with a Cloud Logging sink to BigQuery
  (set `PREDICTION_LOG=/dev/stdout`).
* Monitoring: `service/monitor_job.py` as a Cloud Run **job**, triggered daily by Cloud Scheduler; a non-zero exit code
  (ALERT) fails the job, which alerting picks up.
* `--min-instances 1` removes cold starts for the challenger image at the cost of an always-on instance.

## AWS Lambda (champion only, zip package)

The champion needs only the standard library: zip `src/creditrisk/runtime.py` + `service/model/champion_scorecard.json`
and a 15-line handler that calls `ScorecardRuntime.score_one`. At 128 MB this measured ~20 ms cold start and
4 microseconds per decision (see `results/cloud_benchmark.csv`). The challenger (scikit-learn, ~150 MB in memory) is
better run as a container image or as an asynchronous shadow consumer of the request stream, so it never delays a decision.

## Model versioning

`model_version` in every response is the first 12 hex digits of the SHA-256 of the scorecard file, so every logged
decision can be traced to the exact points table that made it. The challenger pickle must be loaded with the same
scikit-learn version that saved it (pinned in `service/requirements.txt`).
