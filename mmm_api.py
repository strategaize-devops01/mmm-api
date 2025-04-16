from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from sklearn.linear_model import Ridge
import numpy as np

app = FastAPI()

class MMMRequest(BaseModel):
    channels: List[str]
    spend: List[List[float]]
    sales: List[float]
    what_if_spend: Optional[List[float]] = None

class MMMResponse(BaseModel):
    roi: dict
    base_sales: float
    predicted_sales: List[float]
    what_if_prediction: Optional[float] = None

@app.post("/mmm", response_model=MMMResponse)
def run_mmm(data: MMMRequest):
    X = np.array(data.spend)
    y = np.array(data.sales)

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    coefs = model.coef_
    intercept = model.intercept_
    y_pred = model.predict(X)
    roi = {ch: round(c / s, 2) if s != 0 else 0 for ch, c, s in zip(data.channels, coefs, np.mean(X, axis=0))}

    what_if_prediction = None
    if data.what_if_spend:
        X_whatif = np.array(data.what_if_spend).reshape(1, -1)
        what_if_prediction = round(float(model.predict(X_whatif)[0]), 2)

    return MMMResponse(
        roi={ch: round(r, 2) for ch, r in roi.items()},
        base_sales=round(intercept, 2),
        predicted_sales=[round(p, 2) for p in y_pred],
        what_if_prediction=what_if_prediction
    )
