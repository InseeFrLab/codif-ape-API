import os

import mlflow
import nltk


def load_model():
    model_uri = f"models:/{os.environ['MLFLOW_MODEL_NAME']}/{os.environ['MLFLOW_MODEL_VERSION']}"

    # Step 1: Set the destination path for the model artifacts
    dst_path = "/tmp/my_model"

    # Step 2: Download/extract the model here *without loading it yet*
    mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=dst_path)

    # Step 3: Append the nltk_data/ folder to nltk path BEFORE loading the model
    nltk_data_path = os.path.join(dst_path, "artifacts", "nltk_data")
    nltk.data.path.append(nltk_data_path)

    # Step 4: Now safely load the model from the pre-downloaded path
    model = mlflow.pyfunc.load_model(dst_path)

    return model
