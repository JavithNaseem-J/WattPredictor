# WattPredictor

Real-time, 1-hour-ahead electricity demand forecasting for all 11 NYISO zones (New York), served as a Streamlit dashboard, a FastAPI REST API, and a Docker image.

🚀 **Live:** [Click Here](https://wattpredictor-dashboard.onrender.com)

## Problem

Grid operators need accurate short-term demand forecasts per zone to balance supply. WattPredictor predicts next-hour demand (MW) for each of the 11 NYISO sub-regions.

## Features

- **Live data ingestion** — hourly demand from the EIA API (`NYIS` region sub-BA data) + weather from Open-Meteo (archive + forecast endpoints), with file caching and retry sessions
- **Feature pipeline** — 672-hour (4-week) lag windows, calendar features (hour, day-of-week, month, weekend, US federal holidays), temperature, and an engineered `average_demand_last_4_weeks` feature
- **Model selection** — GridSearchCV over XGBoost and LightGBM with `TimeSeriesSplit` (3 folds), tracked in MLflow; winning model: XGBRegressor (`n_estimators=200, max_depth=5, learning_rate=0.1`)
- **Drift monitoring** — Evidently report comparing the last 30 days vs. the prior 335-day baseline (HTML + JSON reports)
- **Three ways to consume** — Streamlit dashboard with a live NYISO zone map, FastAPI endpoints (`POST /predict`, `GET /health`, `GET /metrics`), batch inference via DVC pipeline
- **MLOps** — DVC pipeline (`prepare_data → train_model → predict`), multi-stage Dockerfile (non-root user, healthcheck), GitHub Actions CI with weekly automated retraining (cron `0 0 * * 0`), 78 pytest test functions

## Architecture

### Full System Architecture

```mermaid
flowchart TB
    subgraph SOURCES["External Data Sources"]
        EIA["EIA API"]
        WX["Open-Meteo API"]
    end

    subgraph DATA["Data Pipeline (DVC)"]
        ING["Data Ingestion"]
        VAL["Data Validation"]
        FE["Feature Engineering"]
        DATASTORE[("Processed Datasets")]
        ING -->|Validate Raw Data| VAL
        VAL -->|Generate 672h Lags| FE
        FE -->|Save Features| DATASTORE
    end

    subgraph TRAIN["Model Training and Evaluation"]
        TRAINER["GridSearchCV Training<br/>(XGBoost vs LightGBM)"]
        MLFLOW["MLflow Tracking"]
        EVAL["Model Evaluation"]
        DRIFT["Evidently Drift Report"]
        MODELSTORE[("Trained Model Artifact<br/>model.joblib")]
        
        TRAINER -->|Log Metrics and Params| MLFLOW
        TRAINER -->|Save Best Model| MODELSTORE
        MODELSTORE -->|Calculate Holdout Metrics| EVAL
        MODELSTORE -->|Detect Data Drift| DRIFT
    end

    subgraph SERVE["Serving Layer"]
        ST["Streamlit Dashboard"]
        API["FastAPI REST API"]
        PRED[("Batch Predictions")]
    end

    subgraph DEVOPS["DevOps and CI/CD"]
        GHA["GitHub Actions CI/CD"]
        DOCK["Docker Container"]
    end

    EIA -->|Fetch Hourly Demand| ING
    WX -->|Fetch Weather Data| ING
    DATASTORE -->|Supply Features| TRAINER
    MODELSTORE -->|Load Model| ST
    MODELSTORE -->|Load Model| API
    MODELSTORE -->|Batch Inference| PRED
    ST -.->|Containerize| DOCK
    GHA -->|Automate Retraining| DOCK
```

### ML Pipeline Flow (training run)

```mermaid
flowchart TD
    A["Raw Processed Data"] -->|Extract Lags & Calendar Features| B["Feature Generation"]
    B -->|90-Day Holdout Split| C["Train/Test Split"]
    C -->|TimeSeriesSplit Cross-Validation| D{"GridSearchCV"}
    D -->|Evaluate Hyperparameters| E["XGBoost Model"]
    D -->|Evaluate Hyperparameters| F["LightGBM Model"]
    E -->|Select Winning Model| G["Best Model"]
    F -->|Select Winning Model| G
    G -->|Serialize Model| J[("model.joblib")]
    J -->|Compute MAPE / RMSE / R²| H["Holdout Evaluation"]
    J -->|Compare Baseline vs Current| I["Evidently Drift Detection"]
    J -->|Interactive Zone Map| K["Streamlit Dashboard"]
    J -->|REST Prediction Endpoints| L["FastAPI Service"]
    J -->|Generate CSV Output| M["Batch Predict Pipeline"]
```

## REST API Endpoints

Base: `uvicorn src.WattPredictor.api.main:app` — interactive docs at `/docs`.

| Method | Endpoint | Description | Response |
|---|---|---|---|
| `GET` | `/` | Service status + docs link | `{service, status, documentation}` |
| `GET` | `/health` | Health check — reports whether `model.joblib` is loaded | `{status, model_loaded, model_path, timestamp}` |
| `POST` | `/predict` | Next-hour demand prediction for all 11 NYISO zones | `{status, prediction_time, record_count, predictions: [{sub_region_code, zone_name, predicted_demand_mw, date}]}` |
| `GET` | `/metrics` | Latest evaluation metrics from `artifacts/evaluation/metrics.json` | `{mse, mae, mape, rmse, r2_score}` |

Errors: `/predict` returns `503` if the model artifact is missing, `500` on prediction failure; `/metrics` returns `404` if training hasn't produced metrics yet.

## Tech Stack

Python 3.12 · scikit-learn 1.5.2 · XGBoost 2.1.3 · LightGBM 4.5.0 · MLflow · Evidently 0.4.31 · DVC · FastAPI · Streamlit 1.40.2 · Plotly/PyDeck · Docker · GitHub Actions · uv · pytest

## Results

| Metric | Value |
|---|---|
| MAPE | 2.12% |
| RMSE | 59.87 MW |
| MAE | 34.95 MW |
| R² | 0.9984 |


## Setup & Run

Requires an EIA API key. Set `ELEC_API`, `WX_API`, and `ELEC_API_KEY` in a `.env` file (see `config/config.py` for defaults), then:

```bash
uv sync && uv pip install -e .                          
python src/WattPredictor/pipeline/feature_pipeline.py   
python src/WattPredictor/pipeline/training_pipeline.py  
streamlit run app.py                                    
uvicorn src.WattPredictor.api.main:app --reload         
pytest                                                  
docker build -t wattpredictor . && docker run -p 8501:8501 wattpredictor
```

## Deployment

WattPredictor is configured for automated cloud deployment via **Render Blueprints** (`render.yaml`) and Docker containerization.

### 1-Click Render Blueprint Deployment

1. Push your repository to GitHub.
2. Log into [Render Dashboard](https://dashboard.render.com/) and click **New +** → **Blueprint**.
3. Connect your repository `JavithNaseem-J/WattPredictor`.
4. Render will automatically detect `render.yaml` and provision both services:
   - **`wattpredictor-dashboard`**: Streamlit interactive UI web service (Docker runtime).
   - **`wattpredictor-api`**: FastAPI REST API service (`uvicorn`).
5. Under the Environment tab for each service, set your `ELEC_API_KEY`.

### Docker Hub CI/CD

On every merge to `main`, GitHub Actions (`.github/workflows/ci-cd.yml`) automatically builds and pushes the multi-stage Docker image to Docker Hub.

## Future Work

- Serve the FastAPI app in Docker (current image runs only the Streamlit dashboard)
- Populate the monitoring pipeline (`artifacts/monitoring/monitoring_df.csv` is currently header-only)
- Extend beyond 1-hour-ahead to multi-step horizons

## Keep in Mind

- The app fetches live data and needs a valid `ELEC_API_KEY`; without it, the dashboard stops with an error
- Inference reads `artifacts/engineering/preprocessed.csv`, so run the feature pipeline before predicting
- The 2.12% MAPE reflects the most recent training run committed in `artifacts/`; retraining on newer data will change it

## License

MIT © 2025 Javith Naseem
