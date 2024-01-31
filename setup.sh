#!/bin/bash
git config --global credential.helper store

pip install -r requirements.txt
pip install pre-commit
pre-commit install

AWS_ACCESS_KEY_ID=`vault kv get -field=ACCESS_KEY onyxia-kv/projet-ape/s3` && export AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=`vault kv get -field=SECRET_KEY onyxia-kv/projet-ape/s3` && export AWS_SECRET_ACCESS_KEY
unset AWS_SESSION_TOKEN

export MLFLOW_S3_ENDPOINT_URL="https://$AWS_S3_ENDPOINT"
export MLFLOW_TRACKING_URI=https://projet-ape-mlflow.user.lab.sspcloud.fr
export MLFLOW_MODEL_NAME=FastText-APE
export MLFLOW_MODEL_VERSION="Production"
export API_USERNAME=username
export API_PASSWORD=password
export AUTH_API=False

python -m nltk.downloader stopwords
