"""
Main file for the API.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasicCredentials

from api.routes import predict
from utils.load_model import load_model
from utils.logging import configure_logging
from utils.security import get_credentials


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles FastAPI application lifespan (startup and shutdown).

    On startup, configures logging and loads the machine learning model
    into `app.state.model` and version into `app.state.run_id`
    for immediate use by prediction endpoints.
    """
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting API lifespan")

    app.state.model = load_model()
    app.state.run_id = app.state.model.metadata.run_id

    yield
    logger.info("🛑 Shutting down API lifespan")


app = FastAPI(
    lifespan=lifespan,
    title="Prédiction code APE",
    description="Application de prédiction pour l'activité principale de l'entreprise (APE)",
    version="0.0.1",
)

app.include_router(predict.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Welcome"])
def root(
    _credentials: Annotated[HTTPBasicCredentials, Depends(get_credentials)],
):
    """
    Show welcome page with model name and version.

    Returns the application title along with the current ML model's
    name and version, fetched from environment variables.
    """
    return {
        "Message": "Codification de l'APE",
        "Model_name": f"{os.environ['MLFLOW_MODEL_NAME']}",
        "Model_version": f"{os.environ['MLFLOW_MODEL_VERSION']}",
    }
