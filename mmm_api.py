from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from sklearn.linear_model import Ridge
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd

app = FastAPI()

# Original request/response classes
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

# Advanced MMM classes
class DataRow(BaseModel):
    dt: str
    channel_grouping_l_0: Optional[str]
    channel_grouping_l_2: Optional[str]
    paid_organic: Optional[str]
    total_spend: Optional[float]
    total_clicks: Optional[float]
    total_impressions: Optional[float]
    visitors: Optional[float]
    leads: Optional[float]
    opps: Optional[float]
    accounts: Optional[float]
    bwons: Optional[float]
    fams: Optional[float]

class AdvancedMMMRequest(BaseModel):
    data: List[DataRow]
    features: List[str]
    target_kpi: str

@app.post("/inspect")
def inspect_mmm(request: AdvancedMMMRequest):
    df = pd.DataFrame([row.dict() for row in request.data])
    X = df[request.features].fillna(0).values
    y = df[request.target_kpi].fillna(0).values

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    coefs = model.coef_
    intercept = model.intercept_

    roi = {
        feat: round(c / s, 4) if s != 0 else 0
        for feat, c, s in zip(request.features, coefs, X.mean(axis=0))
    }

    # Response curves
    response_curves = {}
    for i, feat in enumerate(request.features):
        avg_spend = np.mean(X[:, i])
        test_spend = np.linspace(0, avg_spend * 2, 10)
        pred_sales = []

        for val in test_spend:
            X_test = np.zeros((1, len(request.features)))
            X_test[0, i] = val
            pred_sales.append(round(model.predict(X_test)[0], 2))

        response_curves[feat] = {
            "spend": list(map(float, test_spend)),
            "predicted_sales": pred_sales
        }

    # Budget waste detection
    waste_detection = {
        feat: {
            "roi": roi[feat],
            "waste": roi[feat] < 0.1 * max(roi.values())
        } for feat in request.features
    }

    mean_spends = X.mean(axis=0)
    overall_avg = np.mean(mean_spends)

    recommended_channels = {
        feat: {
            "roi": roi[feat],
            "average_spend": round(mean_spends[i], 2)
        }
        for i, feat in enumerate(request.features)
        if roi[feat] > np.percentile(list(roi.values()), 75) and mean_spends[i] < overall_avg
    }

    return JSONResponse({
        "roi_summary": roi,
        "response_curves": response_curves,
        "waste_detection": waste_detection,
        "recommended_channels": recommended_channels
    })
