"""
Main file for logging utils functions.
"""

import logging

from api.models.responses import OutputResponse


def configure_logging():
    """
    Configures the root Python logger to output INFO level messages
    and higher to the console, using a standard timestamped format.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )


def log_prediction(query: dict, response: OutputResponse, index: int = 0):
    """
    Logs the raw query data (for a specific index in a batch) and the
    final structured prediction response for monitoring and debugging.
    """
    query_line = {key: value[index] for key, value in query.items()}
    logging.info(
        "{'Query': %s, 'Response': %s}",
        str(query_line),
        response.model_dump()
    )
