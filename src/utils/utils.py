import os

import mlflow
import numpy as np
from fastapi import HTTPException, Request
from fastapi.security import HTTPBasic


def get_model(model_name: str, model_version: str) -> object:
    """
    This function fetches a trained machine learning model from the MLflow
    model registry based on the specified model name and version.

    Args:
        model_name (str): The name of the model to fetch from the model
        registry.
        model_version (str): The version of the model to fetch from the model
        registry.

    Returns:
        model (object): The loaded machine learning model.

    Raises:
        Exception: If the model fetching fails, an exception is raised with an
        error message.
    """

    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
        return model
    except Exception as error:
        raise Exception(
            f"Failed to fetch model {model_name} version \
            {model_version}: {str(error)}"
        ) from error


async def get_credentials(request: Request):
    """
    Determines whether to apply optional security measures based on the value of the AUTH_API environment variable.

    Args:
        request (Request): The incoming request object.

    Returns:
        Union[HTTPBasic, None]: An instance of the HTTPBasic class if AUTH_API is set to "True", otherwise None.
    """
    if os.getenv("AUTH_API") == "True":
        return await HTTPBasic(request)
    else:
        return None


def process_response(
    predictions: tuple,
    liasse_nb: int,
    nb_echos_max: int,
    prob_min: float,
    libs: dict,
):
    """
    Processes model predictions and generates response.

    Args:
        predictions (tuple): The model predictions as a tuple of two numpy
        arrays.
        nb_echos_max (int): The maximum number of echo predictions.
        prob_min (float): The minimum probability threshold for predictions.
        libs (dict): A dictionary containing mapping of codes to labels.

    Returns:
        response (dict): The processed response as a dictionary containing
        the predicted results.

    Raises:
        HTTPException: If the minimal probability requested is higher than
        the highest prediction probability of the model, a HTTPException
        is raised with a 400 status code and a detailed error message.
    """
    k = max(nb_echos_max, 2)
    if predictions[1][liasse_nb][-1] < prob_min:
        k = np.min(
            [
                np.argmax(np.logical_not(predictions[1][liasse_nb] > prob_min)),
                nb_echos_max,
            ]
        )

    output_dict = {
        str(rank_pred + 1): {
            "code": predictions[0][liasse_nb][rank_pred].replace("__label__", ""),
            "probabilite": float(predictions[1][liasse_nb][rank_pred]),
            "libelle": libs[predictions[0][liasse_nb][rank_pred].replace("__label__", "")],
        }
        for rank_pred in range(k)
    }

    try:
        response = output_dict | {"IC": output_dict["1"]["probabilite"] - float(predictions[1][liasse_nb][1])}
        return response
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=("The minimal probability requested is higher than the highest prediction probability of the model."),
        )
