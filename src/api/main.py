"""
Main file for the API.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

import mlflow
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasicCredentials

from api.routes import predict
from utils.logging import configure_logging
from utils.security import get_credentials


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting API lifespan")

    model_uri = f"models:/{os.environ['MLFLOW_MODEL_NAME']}/{os.environ['MLFLOW_MODEL_VERSION']}"
    app.state.model = mlflow.pyfunc.load_model(model_uri)

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
    credentials: Annotated[HTTPBasicCredentials, Depends(get_credentials)],
):
    """
    Show welcome page with model name and version.
    """
    return {
        "Message": "Codification de l'APE",
        "Model_name": f"{os.environ['MLFLOW_MODEL_NAME']}",
        "Model_version": f"{os.environ['MLFLOW_MODEL_VERSION']}",
    }
