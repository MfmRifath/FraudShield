# MLflow tracking + registry server with a Postgres backend (the upstream image has no driver).
FROM ghcr.io/mlflow/mlflow:v3.17.2
RUN pip install --no-cache-dir psycopg2-binary==2.9.13
