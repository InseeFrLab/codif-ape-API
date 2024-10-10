#!/bin/bash
git config --global credential.helper store

pip install -r requirements.txt
pip install pre-commit
pre-commit install

export MLFLOW_S3_ENDPOINT_URL="https://$AWS_S3_ENDPOINT"
export MLFLOW_TRACKING_URI=https://user-meilametayebjee-mlflow.user.lab.sspcloud.fr
export MLFLOW_MODEL_NAME=fasttext-pytorch
export MLFLOW_MODEL_VERSION=7
export API_USERNAME=username
export API_PASSWORD=password
export AUTH_API=False

python -m nltk.downloader stopwords
