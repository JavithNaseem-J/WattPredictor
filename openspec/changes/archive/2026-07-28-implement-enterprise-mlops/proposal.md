## Why

To elevate WattPredictor to a 10/10 enterprise-grade system, the platform requires three critical production features: a dedicated REST API service for microservice prediction consumption, automated experiment tracking and model registry for auditability, and CI/CD workflow automation for continuous testing and scheduled model retraining.

## What Changes

- **FastAPI REST API Service**: Implement a high-performance REST API wrapper in `src/WattPredictor/api/app.py` exposing `/health`, `/predict`, and `/metrics` endpoints with Pydantic request/response schemas.
- **MLflow Model Registry & Tracking**: Integrate MLflow tracking into `trainer.py` and `evaluator.py` to record hyperparameter grid search runs, evaluation metrics (RMSE, MAE, MAPE), and versioned model artifacts into `mlruns/`.
- **GitHub Actions CI/CD Pipeline**: Add `.github/workflows/ml_pipeline.yml` to automate unit testing (`pytest`), linting, and scheduled pipeline execution on pull requests and main branch updates.

## Capabilities

### New Capabilities
- `rest-api`: FastAPI REST service exposing production model inference and system metrics.
- `experiment-tracking`: MLflow experiment tracking and model registry integration.
- `cicd-automation`: GitHub Actions workflow for automated testing and scheduled pipeline runs.

### Modified Capabilities

## Impact

- **Affected Code**: `src/WattPredictor/api/app.py`, `src/WattPredictor/components/training/trainer.py`, `src/WattPredictor/components/training/evaluator.py`, `.github/workflows/ml_pipeline.yml`, `pyproject.toml`.
- **Dependencies**: Adds `fastapi`, `uvicorn`, `mlflow` to project dependencies.
- **APIs**: Exposes new REST endpoints on port 8000.
