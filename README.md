# Deployment of the FastAPE model 🚀

This project provides all the code necessary to deploy the APE classification model using FastAPI and MLflow. This repository empowers users to expose the model as an API, making it easily accessible for various applications and services.

## Prerequisites

- Python 3.12

## Setup

```bash
export MLFLOW_S3_ENDPOINT_URL="https://$AWS_S3_ENDPOINT"
export MLFLOW_TRACKING_URI=***
export MLFLOW_MODEL_NAME=***
export MLFLOW_MODEL_VERSION=***

uv sync
uv run pre-commit install  # Not necessary, only for dev
cd src
uv run uvicorn api.main:app --host 0.0.0.0 --port 5000
```

⚠️ The `--reload` flag in the last command significantly slows down the forward pass of the model, as it introduces multiprocessing, monitoring overhead, and potential thread contention — all of which degrade performance, especially for CPU-bound inference.

## License

This project is under the [Apache license](https://github.com/InseeFrLab/codif-ape-train/blob/main/LICENSE) to encourage collaboration and free use.
