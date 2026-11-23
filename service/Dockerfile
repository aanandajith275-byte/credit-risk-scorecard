# Champion-only image: the scorecard needs nothing beyond the standard library + Flask.
# Build from the repository root:   docker build -f service/Dockerfile -t credit-scoring .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080
WORKDIR /app
COPY service/requirements.txt /app/service/requirements.txt
RUN pip install --no-cache-dir -r /app/service/requirements.txt
COPY src/creditrisk /app/src/creditrisk
COPY service /app/service
# a non-root user, as Cloud Run / ECS best practice recommends
RUN useradd --create-home scorer && chown -R scorer /app
USER scorer
EXPOSE 8080
# 1 worker per vCPU; threads let the shadow challenger overlap with I/O
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 30 service.app:app
