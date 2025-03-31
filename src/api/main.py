"""
Main file for the API.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, List

import mlflow
import yaml
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasicCredentials
from pydantic import BaseModel

from api.models.forms import BatchForms, SingleForm
from utils.logging import configure_logging
from utils.preprocessing import preprocess_inputs
from utils.utils import (
    get_credentials,
    get_model,
    process_response,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager for managing the lifespan of the API.

    This context manager is used to load the ML model and other resources
    when the API starts and clean them up when the API stops.

    Args:
        app (FastAPI): The FastAPI application.
    """
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting API lifespan")

    app.state.model = get_model(os.environ["MLFLOW_MODEL_NAME"], os.environ["MLFLOW_MODEL_VERSION"])
    run_data = mlflow.get_run(app.state.model.metadata.run_id).data.params
    app.state.training_names = [
        run_data["text_feature"],
        *(v for k, v in run_data.items() if k.startswith("textual_features")),
        *(v for k, v in run_data.items() if k.startswith("categorical_features")),
    ]

    app.state.libs = yaml.safe_load(Path("api/data/libs.yaml").read_text())

    yield
    logger.info("🛑 Shutting down API lifespan")


class BatchPredictionRequest(BaseModel):
    """
    Pydantic BaseModel for representing the input data for the API.

    This BaseModel defines the structure of the input data required
    for the API's "/predict-batch" endpoint.

    Attributes:
        description_activity (List[str]): The text description.
        other_nature_activity (List[str]): Other nature of the activity.
        precision_act_sec_agricole (List[str]): Precision of the activity in the agricultural sector.
        type_form (List[str]): The type of the form CERFA.
        nature (List[str]): The nature of the activity.
        surface (List[str]): The surface of activity.
        cj (List[str]): The legal category code.
        activity_permanence_status (List[str]): The activity permanence status (permanent or seasonal).

    """

    description_activity: List[str]
    other_nature_activity: List[str]
    precision_act_sec_agricole: List[str]
    type_form: List[str]
    nature: List[str]
    surface: List[str]
    cj: List[str]
    activity_permanence_status: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "description_activity": [
                    ("LOUEUR MEUBLE NON PROFESSIONNEL EN RESIDENCE DE SERVICES (CODE APE 6820A Location de logements)")
                ],
                "other_nature_activity": [""],
                "precision_act_sec_agricole": [""],
                "type_form": ["I"],
                "nature": [""],
                "surface": [""],
                "cj": [""],
                "activity_permanence_status": [""],
            }
        }


app = FastAPI(
    lifespan=lifespan,
    title="Prédiction code APE",
    description="Application de prédiction pour l'activité principale de l'entreprise (APE)",
    version="0.0.1",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Welcome"])
def show_welcome_page(
    credentials: Annotated[HTTPBasicCredentials, Depends(get_credentials)],
):
    """
    Show welcome page with model name and version.
    """
    return {
        "Message": "Codification de l'APE",
        "Model_name": f"{os.environ['MLFLOW_MODEL_NAME']}",
        "Model_version": f"{os.environ['MLFLOW_MODEL_VERSION']}",
    }


@app.post("/predict", tags=["Predict"])
async def predict(
    credentials: Annotated[HTTPBasicCredentials, Depends(get_credentials)],
    form: SingleForm,
    nb_echos_max: int = 5,
    prob_min: float = 0.01,
):
    """
    Predict code APE.

    This endpoint accepts input data as query parameters and uses the loaded
    ML model to predict the code APE based on the input data.

    Args:
        nb_echos_max (int): Maximum number of echoes to consider. Default is 5.
        prob_min (float): Minimum probability threshold. Default is 0.01.

    Returns:
        dict: Response containing APE codes.
    """

    query = preprocess_inputs(app.state.training_names, [form])

    predictions = app.state.model.predict(query, params={"k": nb_echos_max})

    response = process_response(predictions, 0, nb_echos_max, prob_min, app.state.libs)

    # Logging
    query_to_log = {key: value[0] for key, value in query.items()}
    logging.info(f"{{'Query': {query_to_log}, 'Response': {response}}}")

    return response


@app.post("/predict-batch", tags=["Predict"])
async def predict_batch(
    credentials: Annotated[HTTPBasicCredentials, Depends(get_credentials)],
    forms: BatchForms,
    nb_echos_max: int = 5,
    prob_min: float = 0.01,
):
    """
    Endpoint for predicting batches of data.

    Args:
        credentials (HTTPBasicCredentials): The credentials for authentication.
        forms (Forms): The input data in the form of Forms object.
        nb_echos_max (int, optional): The maximum number of predictions to return. Defaults to 5.
        prob_min (float, optional): The minimum probability threshold for predictions. Defaults to 0.01.

    Returns:
        list: The list of predicted responses.
    """
    query = preprocess_inputs(app.state.training_names, forms.forms)

    predictions = app.state.model.predict(query, params={"k": nb_echos_max})

    response = [process_response(predictions, i, nb_echos_max, prob_min, app.state.libs) for i in range(len(predictions[0]))]

    # Logging
    for line in range(len(query[app.state.training_names[0]])):
        query_line = {key: value[line] for key, value in query.items()}
        response_line = response[line]
        logging.info(f"{{'Query': {query_line}, 'Response': {response_line}}}")

    return response
