from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.security import HTTPBasicCredentials

from api.models.forms import SingleForm
from api.models.responses import PredictionResponse
from utils.logging import log_prediction
from utils.prediction import process_response
from utils.preprocessing import preprocess_inputs
from utils.security import get_credentials

router = APIRouter(prefix="/single", tags=["Predict an activity"])


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

    query = preprocess_inputs(request.app.state.training_names, [form])

    predictions = request.app.state.model.predict(query, params={"k": nb_echos_max})

    response = process_response(predictions, 0, nb_echos_max, prob_min, request.app.state.libs)

    log_prediction(query, response, 0)

    return response
