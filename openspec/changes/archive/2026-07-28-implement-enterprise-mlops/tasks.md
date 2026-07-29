## 1. Dependency Management & FastAPI Microservice

- [x] 1.1 Add `fastapi`, `uvicorn`, `pydantic`, and `mlflow` to project dependencies in `pyproject.toml` and sync environment via `uv sync`.
- [x] 1.2 Implement `src/WattPredictor/api/main.py` with FastAPI app exposing `/health`, `/predict`, and `/metrics` REST endpoints.
- [x] 1.3 Add test coverage for FastAPI endpoints in `tests/test_api_endpoints.py`.

## 2. MLflow Experiment Tracking & Registry Integration

- [x] 2.1 Integrate MLflow experiment tracking in `src/WattPredictor/components/training/trainer.py` to record grid search runs, best hyperparams, and model artifacts.
- [x] 2.2 Integrate MLflow metric logging in `src/WattPredictor/components/training/evaluator.py` to record RMSE, MAE, MAPE, and business ROI results.

## 3. GitHub Actions CI/CD Pipeline

- [x] 3.1 Create `.github/workflows/ml_pipeline.yml` for automated dependency setup with `uv`, pytest execution, and lint checks.

## 4. Verification

- [x] 4.1 Run `.venv\Scripts\pytest` to verify all existing tests and new FastAPI endpoint tests pass clean.
- [x] 4.2 Start FastAPI server using `uvicorn` and verify HTTP response from `/health` and `/predict`.
