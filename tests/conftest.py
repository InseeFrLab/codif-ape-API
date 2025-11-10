import os
import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.services import load_model  # adapte l’import selon ton projet


@pytest.fixture(scope="session")
def client():
    """
    Charge le modèle défini via les variables d'environnement :
    MLFLOW_MODEL_NAME et MLFLOW_MODEL_VERSION.
    Initialise l'app de FastAPI et retourne le TestClient.
    """
    model_name = os.getenv("MLFLOW_MODEL_NAME")
    model_version = os.getenv("MLFLOW_MODEL_VERSION")
    assert model_name, "MLFLOW_MODEL_NAME doit être défini"
    assert model_version, "MLFLOW_MODEL_VERSION doit être défini"

    # Charger le modèle via la fonction existante
    app.state.model = load_model()

    client = TestClient(app)
    return client
