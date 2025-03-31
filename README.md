# Deployment of the FastAPE model 🚀

This project provides all the code necessary to deploy the APE classification model using FastAPI and MLflow. This repository empowers users to expose the model as an API, making it easily accessible for various applications and services.

## Prerequisites

- Python 3.12
- Python libraries: see `pyproject.toml`

## Setup

```bash
export MLFLOW_S3_ENDPOINT_URL="https://$AWS_S3_ENDPOINT"
export MLFLOW_TRACKING_URI=***
export MLFLOW_MODEL_NAME=***
export MLFLOW_MODEL_VERSION=***

uv sync
uv run pre-commit install
uv run -m nltk.downloader stopwords
cd src
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 5000
```

## License

This project is under the [Apache license](https://github.com/InseeFrLab/codif-ape-train/blob/main/LICENSE) to encourage collaboration and free use.
