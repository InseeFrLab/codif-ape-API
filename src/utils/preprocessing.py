import pandas as pd

from api.models.forms import SingleForm


def preprocess_inputs(training_names: list, inputs: list[SingleForm]) -> dict:
    """
    Preprocess both single and batch inputs using shared logic.
    """
    df = pd.DataFrame([form.model_dump() for form in inputs])
    df.fillna("NaN", inplace=True)
    df = df.astype(str)

    mapping = {
        "description_activity": training_names[0],
        "other_nature_activity": training_names[1],
        "precision_act_sec_agricole": training_names[2],
        "type_form": training_names[3],
        "nature": training_names[4],
        "surface": training_names[5],
        "cj": training_names[6],
        "activity_permanence_status": training_names[7],
    }

    df.rename(columns=mapping, inplace=True)
    return df.to_dict(orient="list")
