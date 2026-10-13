FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow

EXPOSE 5000

# --allowed-hosts: mlflow >= 3.x rejects unknown Host headers (DNS-rebinding protection),
#   and inside the compose network the other services reach it as "mlflow:5000".
# --serve-artifacts + --artifacts-destination: artifacts are uploaded and downloaded
#   *through* the server, so clients never touch /mlflow-data themselves. Without this a
#   client on the host would write to its own C:\mlflow-data and the containers would not
#   see the files.
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--allowed-hosts", "mlflow:5000,localhost:5000,127.0.0.1:5000", \
     "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", \
     "--serve-artifacts", \
     "--artifacts-destination", "/mlflow-data/mlruns"]
