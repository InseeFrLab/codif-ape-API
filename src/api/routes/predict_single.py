from typing import Annotated

import numpy as np
import torch
from fastapi import APIRouter, Depends, Request
from fastapi.security import HTTPBasicCredentials
from torchFastText.datasets import FastTextModelDataset

from api.models.forms import SingleForm
from api.models.responses import PredictionResponse
from utils.logging import log_prediction
from utils.prediction import process_response
from utils.preprocessing import categorical_features, mappings, preprocess_inputs, text_feature
from utils.security import get_credentials

import time

router = APIRouter(prefix="/single", tags=["Predict an activity"])

APE_NIV5_MAPPING = mappings["nace2025"]
INV_APE_NIV5_MAPPING = {v: k for k, v in APE_NIV5_MAPPING.items()}


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    credentials: Annotated[HTTPBasicCredentials, Depends(get_credentials)],
    request: Request,
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

    query = preprocess_inputs([form])

    text, categorical_variables = (
        query[text_feature].values,
        query[categorical_features].values,
    )

    dataset = FastTextModelDataset(
        texts=text,
        categorical_variables=categorical_variables,
        tokenizer=request.app.state.model.model.tokenizer,
    )
    dataloader = dataset.create_dataloader(batch_size=1, shuffle=False, num_workers=1)

    batch = next(iter(dataloader))
    with torch.no_grad():
        scores = request.app.state.model(batch).detach()

    probs = torch.nn.functional.softmax(scores, dim=1)
    sorted_probs, sorted_probs_indices = probs.sort(descending=True, axis=1)

    predicted_class = sorted_probs_indices[:, :nb_echos_max].numpy()
    predicted_probs = sorted_probs[:, :nb_echos_max].numpy()

    predicted_class = np.vectorize(INV_APE_NIV5_MAPPING.get)(predicted_class)
    predictions = (predicted_class, predicted_probs)

    response = process_response(predictions, 0, nb_echos_max, prob_min, request.app.state.libs)

    log_prediction(query, response, 0)

    return response
