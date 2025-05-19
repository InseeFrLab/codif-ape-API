from typing import Annotated, List

import pandas as pd
from fastapi import APIRouter, Depends, Request
from fastapi.security import HTTPBasicCredentials

from api.models.forms import SingleForm
from api.models.responses import PredictionResponse
from utils.preprocessing import mappings, preprocess_inputs
from utils.security import get_credentials

router = APIRouter(prefix="/single", tags=["Predict an activity"])

APE_NIV5_MAPPING = mappings["nace2025"]
INV_APE_NIV5_MAPPING = {v: k for k, v in APE_NIV5_MAPPING.items()}

router = APIRouter(prefix="/batch", tags=["Predict a batch of activity"])


@router.post("/predict", response_model=List[PredictionResponse])
async def predict(
    credentials: Annotated[HTTPBasicCredentials, Depends(get_credentials)],
    request: Request,
    forms: list[SingleForm],
    params_dict: dict = {
        "nb_echos_max": 5,
        "prob_min": 0.01,
        "dataloader_params": {"num_workers": 0, "pin_memory": False, "persistent_workers": False},
    },
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
    query = preprocess_inputs(forms.forms)

    input_data = pd.DataFrame(query)

    return request.app.state.model.predict(input_data, params=params_dict)
