from typing import Any, Dict, Mapping

from pydantic import BaseModel, RootModel, model_validator


class Prediction(BaseModel):
    code: str
    probabilite: float
    libelle: str


class PredictionResponse(RootModel[Dict[str, Any]]):
    """Shallow validation only (strong one in training), keeps API schema shape consistent"""
    pass
