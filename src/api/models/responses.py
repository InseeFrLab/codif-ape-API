from typing import Any, Dict

from pydantic import BaseModel, RootModel, model_validator


class Prediction(BaseModel):
    code: str
    probabilite: float
    libelle: str


class PredictionResponse(RootModel[Dict[str, Any]]):
    """
    Lightweight validation for prediction responses.

    Expected attributes:
    - 'predictions': dict of prediction indices to prediction details.
    - 'IC': float, confidence interval.
    - 'MLversion': optional str, model version.

    Note: Detailed validation is performed during training: 
    each prediction is produced by the model artifact 
    and **normalized** by the training-side code (process_response). 
    This model serves to maintain API schema consistency with minimal overhead 
    """
    pass
