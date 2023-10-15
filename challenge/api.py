"""
API implementation to consume the prediction model
"""

from enum import Enum

import pandas as pd
import fastapi
from fastapi import Body
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from challenge.model import DelayModel

app = fastapi.FastAPI()


class FlightType(Enum):
    " Enumeration that represents different types of flights. "
    N = "N"
    I = "I"


class Flight(BaseModel):
    """
    Subclass of BaseModel and represents a flight with attributes such as
    opera, tipo_vuelo, and mes.
    """
    opera: str
    tipo_vuelo: str
    mes: int


@app.get("/health", status_code=200)
def get_health() -> dict:
    """
    Returns a dictionary with the status "OK" when the "/health" endpoint is
    accessed.
    
    Returns:
        (Dict): A dictionary with the key "status" and the value "OK" is being
        returned.
    """
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200)
def post_predict(data = Body(...)) -> dict:
    """
    POST endpoint that takes flight data, preprocesses it,
    and uses a model to predict flight delays.

    Args:
        data (Body(...)): Request body that contains the flights data.

    Returns:
        (Dict): Dictionary with the key "predict" and the prediction made by
        the model.
    """
    flights: [Flight] = data['flights']
    delay_model = DelayModel()
    for flight in flights:
        if flight.get("MES") <= 0 or flight.get("MES") > 12:
            return JSONResponse(
                status_code=400,
                content={"message": "'MES' value is not valid"}
            )
        if flight.get("TIPOVUELO") not in ["N", "I"]:
            return JSONResponse(
                status_code=400,
                content={"message": "'TIPOVUELO' value is not valid"}
            )
    flights_df = pd.DataFrame(
        [list(flight.values()) for flight in flights],
        columns=flights[0].keys()
    )
    features = delay_model.preprocess(data=flights_df)
    pred = delay_model.predict(features)
    return {"predict": pred}
