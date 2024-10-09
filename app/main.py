"""
Main file for the API.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, List

import yaml
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasicCredentials
from pydantic import BaseModel

from app.utils import get_model, optional_security, process_response, process_response_explain


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager for managing the lifespan of the API.

    This context manager is used to load the ML model and other resources
    when the API starts and clean them up when the API stops.

    Args:
        app (FastAPI): The FastAPI application.
    """
    # global model, libs, training_names
    global model, libs

    model_name: str = os.getenv("MLFLOW_MODEL_NAME")
    model_version: str = os.getenv("MLFLOW_MODEL_VERSION")
    # Load the ML model

    module = get_model(model_name, model_version)
    model = module.model
    libs = yaml.safe_load(Path("app/libs.yaml").read_text())
    yield


class Liasses(BaseModel):
    """
    Pydantic BaseModel for representing the input data for the API.

    This BaseModel defines the structure of the input data required
    for the API's "/predict-batch" endpoint.

    Attributes:
        text_description (List[str]): The text description.
        type_ (List[str]): The type of liasse.
        nature (List[str]): The nature of the liasse.
        surface (List[str]): The surface of the liasse.
        event (List[str]): The event of the liasse.

    """

    text_description: List[str]
    type_: List[str]
    nature: List[str]
    surface: List[str]
    event: List[str]

    class Config:
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "text_description": [
                    (
                        "LOUEUR MEUBLE NON PROFESSIONNEL EN RESIDENCE DE "
                        "SERVICES (CODE APE 6820A Location de logements)"
                    )
                ],
                "type_": ["I"],
                "nature": [""],
                "surface": [""],
                "event": ["01P"],
            }
        }


class LiassesEvaluation(BaseModel):
    """
    Pydantic BaseModel for representing the input data for the API.

    This BaseModel defines the structure of the input data required
    for the API's "/evaluation" endpoint.

    Attributes:
        text_description (List[str]): The text description.
        type_ (List[str]): The type of liasse.
        nature (List[str]): The nature of the liasse.
        surface (List[str]): The surface of the liasse.
        event (List[str]): The event of the liasse.
        code (List[str]): The true code of the liasse.

    """

    text_description: List[str]
    type_: List[str]
    nature: List[str]
    surface: List[str]
    event: List[str]
    code: List[str]

    class Config:
        schema_extra = {
            "example": {
                "text_description": [
                    (
                        "LOUEUR MEUBLE NON PROFESSIONNEL EN RESIDENCE DE "
                        "SERVICES (CODE APE 6820A Location de logements)"
                    )
                ],
                "type_": ["I"],
                "nature": [""],
                "surface": [""],
                "event": ["01P"],
            }
        }


codification_ape_app = FastAPI(
    lifespan=lifespan,
    title="Prédiction code APE avec fastText Pytorch",
    description="Application de prédiction pour \
                                            l'activité principale \
                                            de l'entreprise (APE)",
    version="0.0.1",
    # root_path="/app",
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
    text_description: str,
    type_liasse: str | None = None,
    nature: str | None = None,
    surface: str | None = None,
    event: str | None = None,
    top_k: int = 5,
    prob_min: float = 0.01,
):
    """
    Predict code APE.

    This endpoint accepts input data as query parameters and uses the loaded
    ML model to predict the code APE based on the input data.

    Args:
        text_description (str): The text description.
        type_liasse (str, optional): The type of liasse. Defaults to None.
        nature (str, optional): The nature of the liasse. Defaults to None.
        surface (str, optional): The surface of the liasse. Defaults to None.
        event: (str, optional): Event of the liasse. Optional.
        nb_echos_max (int): Maximum number of echoes to consider. Default is 5.
        prob_min (float): Minimum probability threshold. Default is 0.01.

    Returns:
        dict: Response containing APE codes.
    """

    text = [text_description]  # model needs a list of strings
    params = {"additional_var": [1] * len(text)}  # TBR

    preds, confidence = model.predict(text=text, params=params, top_k=top_k)

    response = process_response(
        predictions=preds,
        liasse_nb=0,
        confidence=confidence,
        top_k=top_k,
        prob_min=prob_min,
        libs=libs,
    )

    return response


@codification_ape_app.get("/predict-and-explain", tags=["Explain"])
async def predict_and_explain(
    credentials: Annotated[HTTPBasicCredentials, Depends(optional_security)],
    text_description: str,
    type_liasse: str | None = None,
    nature: str | None = None,
    surface: str | None = None,
    event: str | None = None,
    prob_min: float = 0.01,
):
    """
    Predict code APE.

    This endpoint accepts input data as query parameters and uses the loaded
    ML model to predict the code APE based on the input data.

    Args:
        text_description (str): The text description.
        type_liasse (str, optional): The type of liasse. Defaults to None.
        nature (str, optional): The nature of the liasse. Defaults to None.
        surface (str, optional): The surface of the liasse. Defaults to None.
        event: (str, optional): Event of the liasse. Optional.
        nb_echos_max (int): Maximum number of echoes to consider. Default is 5.
        prob_min (float): Minimum probability threshold. Default is 0.01.

    Returns:
        dict: Response containing APE codes.
    """

    text = [text_description]  # model needs a list of strings
    params = {"additional_var": [1] * len(text)}  # TBR

    pred, confidence, all_scores = model.predict_and_explain(text, params)

    response = process_response_explain(
        text=text,
        predictions=pred,
        liasse_nb=0,
        confidence=confidence,
        all_scores=all_scores,
        prob_min=prob_min,
        libs=libs,
    )

    return response


@codification_ape_app.post("/predict-batch", tags=["Predict"])
async def predict_batch(
    credentials: Annotated[HTTPBasicCredentials, Depends(optional_security)],
    liasses: Liasses,
    top_k: int = 5,
    prob_min: float = 0.01,
):
    """
    Endpoint for predicting batches of data.

    Args:
        credentials (HTTPBasicCredentials): The credentials for authentication.
        liasses (Liasses): The input data in the form of Liasses object.
        nb_echos_max (int, optional): The maximum number of predictions to return. Defaults to 5.
        prob_min (float, optional): The minimum probability threshold for predictions. Defaults to 0.01.

    Returns:
        list: The list of predicted responses.
    """
    # query = preprocess_batch(training_names, liasses.dict())
    query = liasses.dict()
    text = query["text_description"]
    params = {"additional_var": [1] * len(text)}

    preds, confidence = model.predict(text=text, params=params, top_k=top_k)

    response = [
        process_response(
            predictions=preds,
            liasse_nb=i,
            confidence=confidence,
            top_k=top_k,
            prob_min=prob_min,
            libs=libs,
        )
        for i in range(len(preds))
    ]

    # Logging
    # for line in range(len(query[training_names[0]])):
    #     query_line = {key: value[line] for key, value in query.items()}
    #     response_line = response[line]
    #     logging.info(f"{{'Query': {query_line}, 'Response': {response_line}}}")

    return response
