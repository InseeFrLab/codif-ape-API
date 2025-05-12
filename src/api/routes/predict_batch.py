from typing import Annotated, List

import numpy as np
import torch
from fastapi import APIRouter, Depends, Request
from fastapi.security import HTTPBasicCredentials
from torchFastText.datasets import FastTextModelDataset

from api.models.forms import BatchForms
from api.models.responses import PredictionResponse
from utils.logging import log_prediction
from utils.prediction import process_response
from utils.preprocessing import categorical_features, mappings, preprocess_inputs, text_feature
from utils.security import get_credentials

router = APIRouter(prefix="/single", tags=["Predict an activity"])

APE_NIV5_MAPPING = mappings["nace2025"]
INV_APE_NIV5_MAPPING = {v: k for k, v in APE_NIV5_MAPPING.items()}

router = APIRouter(prefix="/batch", tags=["Predict a batch of activity"])


@router.post("/predict", response_model=List[PredictionResponse])
async def predict(
    credentials: Annotated[HTTPBasicCredentials, Depends(get_credentials)],
    request: Request,
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
    query = preprocess_inputs(forms.forms)

    text, categorical_variables = (
        query[text_feature].values,
        query[categorical_features].values,
    )

    dataset = FastTextModelDataset(
        texts=text,
        categorical_variables=categorical_variables,
        tokenizer=request.app.state.model.model.tokenizer,
    )

    batch_size = len(text) if len(text) < 256 else 256
    dataloader = dataset.create_dataloader(batch_size=batch_size, shuffle=False, num_workers=12)

    all_scores = []
    for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():
            scores = request.app.state.model(batch).detach()
            all_scores.append(scores)
    
    all_scores = torch.cat(all_scores)
    probs = torch.nn.functional.softmax(scores, dim=1)
    sorted_probs, sorted_probs_indices = probs.sort(descending=True, axis=1)

    predicted_class = sorted_probs_indices[:, :nb_echos_max].numpy()
    predicted_probs = sorted_probs[:, :nb_echos_max].numpy()

    predicted_class = np.vectorize(INV_APE_NIV5_MAPPING.get)(predicted_class)
    predictions = (predicted_class, predicted_probs)

    responses = []
    for i in range(len(predictions[0])):
        response = process_response(predictions, i, nb_echos_max, prob_min, request.app.state.libs)
        log_prediction(query, response, i)
        responses.append(response)

    return responses
