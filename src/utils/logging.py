import logging


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("codification_ape_log_file.log"),
            logging.StreamHandler(),
        ],
    )
