from typing import Optional

from pydantic import BaseModel, model_validator, validator

from api.constants.models import VALID_SURFACE, VALID_TYPE_FORM


class SingleForm(BaseModel):
    description_activity: str
    other_nature_activity: Optional[str] = None
    precision_act_sec_agricole: Optional[str] = None
    type_form: Optional[str] = None
    nature: Optional[str] = None
    surface: Optional[str] = None
    cj: Optional[str] = None
    activity_permanence_status: Optional[str] = None

    @validator("type_form")
    def validate_type_form(cls, v):
        if (v is not None) and (v not in VALID_TYPE_FORM):
            raise ValueError(f"Invalid type_form '{v}', must be one of {VALID_TYPE_FORM}")
        return v

    @validator("nature")
    def validate_nature(cls, v):
        if (v is not None) and (not v.isdigit() or len(v) != 2):
            raise ValueError("nature must be a two-digit number (e.g., '01')")
        return v

    @validator("surface")
    def validate_surface(cls, v):
        if (v is not None) and (v not in VALID_SURFACE):
            raise ValueError(f"Invalid surface '{v}', must be one of {VALID_SURFACE}")
        return v


class BatchForms(BaseModel):
    forms: list[SingleForm]

    @model_validator(mode="after")
    def check_description_not_empty(cls, values):
        forms = values.forms
        missing_indexes = [
            idx for idx, form in enumerate(forms) if not form.description_activity or form.description_activity.strip() == ""
        ]

        if missing_indexes:
            raise ValueError(f"The description_activity is missing at indices: {tuple(missing_indexes)}")

        return values
