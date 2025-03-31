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

    valid_predictions = [
        (label.replace("__label__", ""), prob) for label, prob in zip(pred_labels, pred_probs) if prob >= prob_min
    ][:nb_echos_max]

    if not valid_predictions:
        raise HTTPException(
            status_code=400,
            detail="No prediction exceeds the minimum probability threshold.",
        )

    response_data = {
        str(i + 1): Prediction(
            code=label,
            probabilite=float(prob),
            libelle=libs[label],
        )
        for i, (label, prob) in enumerate(valid_predictions)
    }

    ic = response_data["1"].probabilite - float(pred_probs[1])
    response_data["IC"] = ic

    return PredictionResponse(response_data)
