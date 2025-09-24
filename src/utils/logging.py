import logging

from api.models.responses import OutputResponse


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )


def log_prediction(query: dict, response: OutputResponse, index: int = 0):
    query_line = {key: value[index] for key, value in query.items()}
    logging.info(f"{{'Query': {query_line}, 'Response': {response.model_dump()}}}")
