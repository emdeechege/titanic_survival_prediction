"""FastAPI entry point: python app.py or uvicorn app:app."""

import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, List, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from predictor import load_model, predict_passengers


class Passenger(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    Pclass: Literal[1, 2, 3]
    Sex: Literal["male", "female"]
    Age: Optional[float] = Field(default=None, ge=0)
    SibSp: int = Field(default=0, ge=0)
    Parch: int = Field(default=0, ge=0)
    Fare: Optional[float] = Field(default=None, ge=0)
    Embarked: Optional[Literal["S", "C", "Q"]] = None


class Prediction(BaseModel):
    survived: int
    survival_probability: float


@asynccontextmanager
async def lifespan(app):
    app.state.model = load_model()
    yield


app = FastAPI(title="Titanic Survival API", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # Python's JSON parser accepts NaN/Infinity, but JSON responses cannot contain them.
    errors = jsonable_encoder(
        exc.errors(), custom_encoder={float: lambda value: value if math.isfinite(value) else str(value)}
    )
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    # This small HTML page needs no file download or Range header handling.
    return HTMLResponse((Path(__file__).resolve().parent / "static" / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"status": "ok"}


def infer(passengers):
    try:
        data = pd.DataFrame([passenger.model_dump() for passenger in passengers])
        with np.errstate(over="raise"):
            results = predict_passengers(app.state.model, data)
    except (ValueError, OverflowError, FloatingPointError) as exc:
        raise HTTPException(
            status_code=422,
            detail="Passenger values cannot be processed by the model; check numeric ranges.",
        ) from exc
    return [Prediction(survived=int(row.Survived), survival_probability=float(row.SurvivalProbability))
            for row in results.itertuples()]


@app.post("/predict", response_model=Prediction)
def predict(passenger: Passenger):
    return infer([passenger])[0]


@app.post("/predict/batch", response_model=List[Prediction])
def predict_batch(passengers: Annotated[List[Passenger], Field(min_length=1, max_length=10000)]):
    return infer(passengers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
