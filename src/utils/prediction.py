from fastapi import HTTPException

from api.models.responses import Prediction, PredictionResponse


def process_response(
    predictions: tuple,
    liasse_nb: int,
    nb_echos_max: int,
    prob_min: float,
    libs: dict,
) -> PredictionResponse:
    """
    Process model
    predictions into a structured response.
    """
    labels, probs = predictions
    pred_labels = labels[liasse_nb]
    pred_probs = probs[liasse_nb]

    valid_indices = [i for i, p in enumerate(pred_probs) if p >= prob_min]
    k = min(nb_echos_max, len(valid_indices)) if valid_indices else 0

    if k == 0:
        raise HTTPException(
            status_code=400,
            detail="No prediction exceeds the given minimum probability threshold.",
        )

    response_data = {
        str(i + 1): Prediction(
            code=label,
            probabilite=float(prob),
            libelle=libs[label],
        )
        for i in range(k)
        for label, prob in [(pred_labels[i].replace("__label__", ""), pred_probs[i])]
    }

    ic = response_data["1"].probabilite - float(pred_probs[1]) if k > 1 else 0.0
    response_data["IC"] = ic

    return PredictionResponse(__root__=response_data)
