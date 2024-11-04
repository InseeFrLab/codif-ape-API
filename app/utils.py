import os
import re
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials


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


async def optional_security(request: Request):
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


def get_current_username(
    credentials: Optional[HTTPBasicCredentials] = Depends(optional_security),
):
    """
    Retrieves the current username based on the provided credentials.

    Args:
        credentials (Optional[HTTPBasicCredentials]): The credentials used for authentication.

    Returns:
        str: The username extracted from the credentials.

    Raises:
        HTTPException: If authentication fails.

    """
    if os.getenv("AUTH_API") == "True":
        if not (credentials.username == os.getenv("API_USERNAME")) or not (
            credentials.password == os.getenv("API_PASSWORD")
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed",
                headers={"WWW-Authenticate": "Basic"},
            )
    return credentials.username


def preprocess_query(
    training_names: list,
    description_activity: str,
    other_nature_activity: str | None,
    precision_act_sec_agricole: str | None,
    type_form: str | None,
    nature: str | None,
    surface: str | None,
    event: str | None,
    cj: str | None,
    activity_permanence_status: str | None,
    nb_echos_max: int = 5,
) -> dict:
    """
    This function preprocesses the input query parameters for making
    predictions using the fetched machine learning model.

    Args:
        training_names (list): A list of feature names used for training.
        description_activity (str): The text description.
        other_nature_activity (str, optional): Other nature of the activity. Defaults to None.
        precision_act_sec_agricole (str, optional): Precision of the activity in the agricultural sector. Defaults to None.
        type_form (str, optional): The type of the form CERFA. Defaults to None.
        nature (str, optional): The nature of the activity. Defaults to None.
        surface (str, optional): The surface of activity. Defaults to None.
        event (str, optional): The event of the form. Defaults to None.
        cj (str, optional): The legal category code. Defaults to None.
        activity_permanence_status (str, optional): The activity permanence status (permanent or seasonal). Defaults to None.
        nb_echos_max (int, optional): The maximum number of echo predictions.
        Default is 5.

    Returns:
        query (dict): The preprocessed query in the required format for
        making predictions.

    """
    type_form, nature, surface, event, cj, activity_permanence_status = (
        np.nan if v is None else v
        for v in (type_form, nature, surface, event, cj, activity_permanence_status)
    )

    list_ok = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "G",
        "I",
        "L",
        "M",
        "N",
        "P",
        "R",
        "S",
        "X",
        "Y",
        "Z",
    ]
    check_format_features(
        [type_form],
        "type_form",
        r"^(" + "|".join(list_ok) + r")$",
        list_ok=list_ok,
    )

    check_format_features([nature], "nature", r"^\d{2}$")

    list_ok = ["1", "2", "3", "4"]
    check_format_features(
        [surface],
        "surface",
        r"^(" + "|".join(list_ok) + r")$",
        list_ok=list_ok,
    )

    check_format_features([event], "event", r"^\d{2}[APMF]$")

    # TODO : Add check for cj and activity_permanence_status

    type_form, nature, surface, event, cj, activity_permanence_status = (
        "NaN" if not isinstance(v, str) else v
        for v in (type_form, nature, surface, event, cj, activity_permanence_status)
    )

    query = {
        training_names[0]: [description_activity],
        training_names[1]: [other_nature_activity],
        training_names[2]: [precision_act_sec_agricole],
        training_names[3]: [type_form],
        training_names[4]: [nature],
        training_names[5]: [surface],
        training_names[6]: [event],
        training_names[7]: [cj],
        training_names[8]: [activity_permanence_status],
    }
    return query


def preprocess_batch(training_names: list, query: dict) -> dict:
    """
    Preprocesses a batch of data in a dictionary format for prediction.

    Args:
        query (dict): A dictionary containing the batch of data.

    Returns:
        dict: A dictionary containing the preprocessed data ready for further
        processing.
    Raises:
        HTTPException: If the 'description_activity' field is missing for any
            liasses in the batch, a HTTPException is raised with a 400
            status code and a detailed error message.
    """

    df = pd.DataFrame(query)
    df = df.apply(lambda x: x.str.strip())
    df = df.replace(["null", "", "NA", "NAN", "nan", "None"], np.nan)

    if df["description_activity"].isna().any():
        matches = df.index[df["description_activity"].isna()].to_list()
        raise HTTPException(
            status_code=400,
            detail=(
                "The description of the activity is missing for some forms. "
                f"See line(s): {*matches,}"
            ),
        )

    list_ok = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "G",
        "I",
        "L",
        "M",
        "N",
        "P",
        "R",
        "S",
        "X",
        "Y",
        "Z",
    ]
    check_format_features(
        df["type_form"].to_list(),
        "type_form",
        r"^(" + "|".join(list_ok) + r")$",
        list_ok=list_ok,
    )

    check_format_features(df["nature"].to_list(), "nature", r"^\d{2}$")

    list_ok = ["1", "2", "3", "4"]
    check_format_features(
        df["surface"].to_list(),
        "surface",
        r"^(" + "|".join(list_ok) + r")$",
        list_ok=list_ok,
    )

    check_format_features(df["event"].to_list(), "event", r"^\d{2}[PMF]$")

    # TODO: Add check for cj and activity_permanence_status*
    # TODO: make it cleaner
    df.loc[:, ["other_nature_activity", "precision_act_sec_agricole"]] = df[
        ["other_nature_activity", "precision_act_sec_agricole"]
    ].replace(np.nan, "")
    df.loc[
        :,
        [
            "type_form",
            "nature",
            "surface",
            "event",
            "cj",
            "activity_permanence_status",
        ],
    ] = df[
        [
            "type_form",
            "nature",
            "surface",
            "event",
            "cj",
            "activity_permanence_status",
        ]
    ].replace(np.nan, "NaN")

    query = df.rename(
        columns={
            "description_activity": training_names[0],
            "other_nature_activity": training_names[1],
            "precision_act_sec_agricole": training_names[2],
            "type_form": training_names[3],
            "nature": training_names[4],
            "surface": training_names[5],
            "event": training_names[6],
            "cj": training_names[7],
            "activity_permanence_status": training_names[8],
        }
    ).to_dict("list")

    return query


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
    k = nb_echos_max
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
        response = output_dict | {
            "IC": output_dict["1"]["probabilite"] - float(predictions[1][liasse_nb][1])
        }
        return response
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=(
                "The minimal probability requested is "
                "higher than the highest prediction "
                "probability of the model."
            ),
        )


def check_format_features(values: list, feature: str, regex: str, list_ok: list = None) -> None:
    """
    Check the format of values for a specific feature using regex pattern.

    Args:
        values (list): A list of values to be checked.
        feature (str): The name of the feature being checked.
        regex (str): The regex pattern used to check the format of values.
        list_ok (list, optional): A list of accepted values for the feature.

    Raises:
        HTTPException: If the format of any value in the list does not match
         the regex pattern, a HTTPException is raised with a
         400 status code and a detailed error message.
    """

    matches = []

    for i, value in enumerate(values):
        if isinstance(value, str):
            if not re.match(regex, value):
                matches.append(i)

    errors = {
        "type_form": (
            "The format of type_liasse is incorrect. Accepted values are"
            f": {list_ok}. See line(s) : {*matches,}"
        ),
        "nature": (
            "The format of nature is incorrect. The nature is an "
            f"integer between 00 and 99. See line(s): {*matches,}"
        ),
        "surface": (
            "The format of surface is incorrect. Accepted values are: "
            f"{list_ok}. See line(s): {*matches,}"
        ),
        "event": (
            f"The format of event is incorrect. The event value is an "
            "integer between 00 and 99 plus the letter A, P, M or F. Example: "
            f"'01P'. See line(s): {*matches,}"
        ),
    }

    if matches:
        raise HTTPException(status_code=400, detail=errors[feature])
