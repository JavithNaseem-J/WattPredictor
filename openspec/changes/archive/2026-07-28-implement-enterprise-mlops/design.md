## Context

To achieve enterprise 10/10 production readiness, the system requires:
- An asynchronous HTTP API service to serve predictions to external microservices without runningStreamlit or python CLI scripts.
- Experiment tracking & model registry capabilities to log hyperparameter tuning and model binaries for compliance and rollback.
- CI/CD workflow automation to continuously test code on commits and run scheduled model re-training.

## Goals / Non-Goals

**Goals:**
- Build a `FastAPI` service in `src/WattPredictor/api/main.py` with endpoints `/health`, `/predict`, and `/metrics`.
- Add `MLflow` experiment logging to `trainer.py` and `evaluator.py` to record parameters, validation metrics (RMSE, MAE, MAPE), and registered model artifacts into local/remote MLflow store.
- Configure `.github/workflows/ml_pipeline.yml` for automated CI/CD unit testing (`pytest`) and scheduled pipeline triggers.

**Non-Goals:**
- Setting up a paid Kubernetes cluster or cloud-specific IAM infrastructure.

## Decisions

- **Decision 1: FastAPI REST Microservice**
  - **Approach**: Create `src/WattPredictor/api/main.py` using `FastAPI` + `uvicorn` and Pydantic request/response models (`PredictRequest`, `PredictResponse`).
  - **Rationale**: FastAPI provides high-performance asynchronous HTTP serving, automatic Swagger OpenAPI documentation (`/docs`), and input validation.

- **Decision 2: MLflow Integration in Trainer & Evaluator**
  - **Approach**: Initialize `mlflow.set_experiment("WattPredictor")` in `trainer.py` and log parameters and metrics (`mlflow.log_params`, `mlflow.log_metrics`, `mlflow.sklearn.log_model`).
  - **Rationale**: Standardizes experiment history and enables model artifact version control.

- **Decision 3: GitHub Actions Workflow**
  - **Approach**: Create `.github/workflows/ml_pipeline.yml` triggering on push to main and pull requests, using `uv` to install dependencies and execute `pytest`.
  - **Rationale**: Guarantees pull requests never break test suites.

## Risks / Trade-offs

- **[Risk] MLflow local storage footprint**
  - **Mitigation**: Store experiments under local `mlruns/` and configure `.gitignore` to prevent tracking binary runs in git.
