#!/bin/bash
git config --global credential.helper store

uv sync
uv run pre-commit install

export MLFLOW_S3_ENDPOINT_URL="https://$AWS_S3_ENDPOINT"
export MLFLOW_TRACKING_URI=https://projet-ape-mlflow.user.lab.sspcloud.fr
export MLFLOW_MODEL_NAME=FastText-pytorch-2025
export MLFLOW_MODEL_VERSION="6"
export API_USERNAME=username
export API_PASSWORD=password
export AUTH_API=False
