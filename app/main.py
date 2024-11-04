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

from app.utils import (
    get_model,
    optional_security,
    preprocess_batch,
    preprocess_query,
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
    global model, libs, training_names

    model_name: str = os.getenv("MLFLOW_MODEL_NAME")
    model_version: str = os.getenv("MLFLOW_MODEL_VERSION")
    # Load the ML model
    model = get_model(model_name, model_version)
    libs = yaml.safe_load(Path("app/libs.yaml").read_text())
    text_feature = [mlflow.get_run(model.metadata.run_id).data.params["text_feature"]]
    textual_features = [
        v
        for k, v in mlflow.get_run(model.metadata.run_id).data.params.items()
        if k.startswith("textual_features")
    ]
    categorical_features = [
        v
        for k, v in mlflow.get_run(model.metadata.run_id).data.params.items()
        if k.startswith("categorical_features")
    ]
    training_names = text_feature + textual_features + categorical_features
    yield


class Forms(BaseModel):
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
        event (List[str]): The event of the form.
        cj (List[str]): The legal category code.
        activity_permanence_status (List[str]): The activity permanence status (permanent or seasonal).

    """

    description_activity: List[str]
    other_nature_activity: List[str]
    precision_act_sec_agricole: List[str]
    type_form: List[str]
    nature: List[str]
    surface: List[str]
    event: List[str]
    cj: List[str]
    activity_permanence_status: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "description_activity": [
                    (
                        "LOUEUR MEUBLE NON PROFESSIONNEL EN RESIDENCE DE "
                        "SERVICES (CODE APE 6820A Location de logements)"
                    )
                ],
                "other_nature_activity": [""],
                "precision_act_sec_agricole": [""],
                "type_form": ["I"],
                "nature": [""],
                "surface": [""],
                "event": ["01P"],
                "cj": [""],
                "activity_permanence_status": [""],
            }
        }


codification_ape_app = FastAPI(
    lifespan=lifespan,
    title="Prédiction code APE",
    description="Application de prédiction pour \
                                            l'activité principale \
                                            de l'entreprise (APE)",
    version="0.0.1",
)


codification_ape_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("codification_ape_log_file.log"),
        logging.StreamHandler(),
    ],
)


@codification_ape_app.get("/", tags=["Welcome"])
def show_welcome_page(
    credentials: Annotated[HTTPBasicCredentials, Depends(optional_security)],
):
    """
    Show welcome page with model name and version.
    """
    model_name: str = os.getenv("MLFLOW_MODEL_NAME")
    model_version: str = os.getenv("MLFLOW_MODEL_VERSION")

    return {
        "Message": "Codification de l'APE",
        "Model_name": f"{model_name}",
        "Model_version": f"{model_version}",
    }


@codification_ape_app.get("/predict", tags=["Predict"])
async def predict(
    credentials: Annotated[HTTPBasicCredentials, Depends(optional_security)],
    description_activity: str,
    other_nature_activity: str | None = None,
    precision_act_sec_agricole: str | None = None,
    type_form: str | None = None,
    nature: str | None = None,
    surface: str | None = None,
    event: str | None = None,
    cj: str | None = None,
    activity_permanence_status: str | None = None,
    nb_echos_max: int = 5,
    prob_min: float = 0.01,
):
    """
    Predict code APE.

    This endpoint accepts input data as query parameters and uses the loaded
    ML model to predict the code APE based on the input data.

    Args:
        description_activity (str): The text description.
        other_nature_activity (str, optional): Other nature of the activity. Defaults to None.
        precision_act_sec_agricole (str, optional): Precision of the activity in the agricultural sector. Defaults to None.
        type_form (str, optional): The type of the form CERFA. Defaults to None.
        nature (str, optional): The nature of the activity. Defaults to None.
        surface (str, optional): The surface of activity. Defaults to None.
        event (str, optional): The event of the form. Defaults to None.
        cj (str, optional): The legal category code. Defaults to None.
        activity_permanence_status (str, optional): The activity permanence status (permanent or seasonal). Defaults to None.
        nb_echos_max (int): Maximum number of echoes to consider. Default is 5.
        prob_min (float): Minimum probability threshold. Default is 0.01.

    Returns:
        dict: Response containing APE codes.
    """

    query = preprocess_query(
        training_names,
        description_activity,
        other_nature_activity,
        precision_act_sec_agricole,
        type_form,
        nature,
        surface,
        event,
        cj,
        activity_permanence_status,
    )

    if nb_echos_max != 1:
        predictions = model.predict(query, params={"k": nb_echos_max})
    else:
        predictions = model.predict(query, params={"k": 2})

    response = process_response(predictions, 0, nb_echos_max, prob_min, libs)

    # Logging
    query_to_log = {key: value[0] for key, value in query.items()}
    logging.info(f"{{'Query': {query_to_log}, 'Response': {response}}}")

    return response


@codification_ape_app.post("/predict-batch", tags=["Predict"])
async def predict_batch(
    credentials: Annotated[HTTPBasicCredentials, Depends(optional_security)],
    forms: Forms,
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
    query = preprocess_batch(training_names, forms.dict())

    if nb_echos_max != 1:
        predictions = model.predict(query, params={"k": nb_echos_max})
    else:
        predictions = model.predict(query, params={"k": 2})

    response = [
        process_response(predictions, i, nb_echos_max, prob_min, libs)
        for i in range(len(predictions[0]))
    ]

    # Logging
    for line in range(len(query[training_names[0]])):
        query_line = {key: value[line] for key, value in query.items()}
        response_line = response[line]
        logging.info(f"{{'Query': {query_line}, 'Response': {response_line}}}")

    return response
