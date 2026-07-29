import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import json
import os

from WattPredictor.config.config import get_config
from WattPredictor.components.inference.predictor import Predictor

app = FastAPI(
    title="WattPredictor REST API",
    description="High-performance utility-scale electricity demand forecasting API",
    version="1.0.0"
)

config = get_config()

NYISO_ZONES = {
    0: "West", 1: "Genesee", 2: "Central", 3: "North", 4: "Mohawk Valley",
    5: "Capital", 6: "Hudson Valley", 7: "Millwood", 8: "Dunwoodie",
    9: "New York City", 10: "Long Island"
}


class PredictionItem(BaseModel):
    sub_region_code: int = Field(..., description="NYISO Sub-region numeric identifier (0-10)")
    zone_name: str = Field(..., description="Human-readable NYISO sub-region name")
    predicted_demand_mw: float = Field(..., description="Predicted demand in Megawatts (MW)")
    date: str = Field(..., description="Target prediction UTC timestamp")


class PredictResponse(BaseModel):
    status: str = "success"
    prediction_time: str
    record_count: int
    predictions: List[PredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    timestamp: str


@app.get("/", tags=["General"])
def root():
    return {
        "service": "WattPredictor REST API",
        "status": "online",
        "documentation": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    model_exists = os.path.exists(config.model_path)
    return HealthResponse(
        status="healthy" if model_exists else "degraded",
        model_loaded=model_exists,
        model_path=str(config.model_path),
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def get_predictions():
    if not os.path.exists(config.model_path):
        raise HTTPException(status_code=503, detail="Model artifact not found. Please train model first.")
    
    try:
        predictor = Predictor(config=config)
        raw_df = predictor.predict()
        
        items = []
        for _, row in raw_df.iterrows():
            code = int(row['sub_region_code'])
            items.append(PredictionItem(
                sub_region_code=code,
                zone_name=NYISO_ZONES.get(code, f"Zone {code}"),
                predicted_demand_mw=float(row['predicted_demand']),
                date=str(row['date'])
            ))
            
        return PredictResponse(
            status="success",
            prediction_time=datetime.now(timezone.utc).isoformat(),
            record_count=len(items),
            predictions=items
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics", tags=["Monitoring"])
def get_metrics() -> Dict[str, Any]:
    metrics_path = config.metrics_path
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Evaluation metrics file not found. Run training pipeline first.")
    
    try:
        with open(metrics_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metrics: {str(e)}")
