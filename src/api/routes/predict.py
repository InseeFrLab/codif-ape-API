from typing import Annotated, List

from fastapi import APIRouter, Depends, Request
from fastapi.security import HTTPBasicCredentials

from api.models.forms import BatchForms
from api.models.responses import PredictionResponse
from utils.preprocessing import mappings
from utils.security import get_credentials

APE_NIV5_MAPPING = mappings["nace2025"]
INV_APE_NIV5_MAPPING = {v: k for k, v in APE_NIV5_MAPPING.items()}

router = APIRouter(prefix="/predict", tags=["Predict NACE code for a list of activities"])


@router.post("/", response_model=List[PredictionResponse])
async def predict(
    credentials: Annotated[HTTPBasicCredentials, Depends(get_credentials)],
    request: Request,
    forms: BatchForms,
    nb_echos_max: int = 5,
    prob_min: float = 0.01,
    num_workers: int = 1,
    batch_size: int = 1,
):
    """
    Endpoint for predicting batches of data.

    Args:
        credentials (HTTPBasicCredentials): The credentials for authentication.
        forms (Forms): The input data in the form of Forms object.
        nb_echos_max (int, optional): The maximum number of predictions to return. Defaults to 5.
        prob_min (float, optional): The minimum probability threshold for predictions. Defaults to 0.01.
        num_workers (int, optional): Number of CPU for multiprocessing in Dataloader. Defaults to 1.
        batch_size (int, optional): Size of a batch for batch prediction.

    For single predictions, we recommend keeping num_workers and batch_size to 1 for better performance.
    For batched predictions, consider increasing these two parameters (num_workers can range from 4 to 12, batch size can be increased up to 256) to optimize performance.

    Returns:
        list: The list of predicted responses.
    """
    input_data = forms.forms

    params_dict = {
        "nb_echos_max": nb_echos_max,
        "prob_min": prob_min,
        "dataloader_params": {
            "pin_memory": False,
            "persistent_workers": False,
            "num_workers": num_workers,
            "batch_size": batch_size,
        },
    }

    output = request.app.state.model.predict(input_data, params=params_dict)
    return [out.model_dump() for out in output]
