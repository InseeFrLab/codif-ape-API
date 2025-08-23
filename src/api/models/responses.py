"""
Lightweight validation for prediction responses returned by the inference API.

- Ensures API schema consistency with minimal overhead.
- Strong validation and normalization are performed during training
  (via `process_response` in the model artifact).
- The output contract is defined per PredictionResponse.
"""
from typing import Dict, Union

from pydantic import BaseModel, RootModel


class Prediction(BaseModel):
    code: str
    probabilite: float
    libelle: str


