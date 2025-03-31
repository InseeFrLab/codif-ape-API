from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.security import HTTPBasicCredentials

from api.models.forms import BatchForms
from api.models.responses import PredictionResponse
from utils.logging import log_prediction
from utils.prediction import process_response
from utils.preprocessing import preprocess_inputs
from utils.security import get_credentials

router = APIRouter(prefix="/batch", tags=["Predict a batch of activity"])


@router.post("/predict", response_model=PredictionResponse)
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
    query = preprocess_inputs(request.app.state.training_names, forms.forms)

    predictions = request.app.state.model.predict(query, params={"k": nb_echos_max})

    response = [
        process_response(predictions, i, nb_echos_max, prob_min, request.app.state.libs) for i in range(len(predictions[0]))
    ]

    responses = []
    for i in range(len(predictions[0])):
        response = process_response(predictions, i, nb_echos_max, prob_min, request.app.state.libs)
        log_prediction(query, response, i)
        responses.append(response)

    return responses
