# WattPredictor ⚡

## Project Overview

**WattPredictor** is an end-to-end machine learning pipeline for forecasting hourly electricity demand in New York ISO (NYISO) zones using weather and temporal data. It supports real-time inference, model retraining, and production-grade monitoring with drift detection.

### Key Workflow Stages:

* **Data Ingestion**: Aggregates electricity demand from the NYISO API and weather metrics from Open-Meteo.
* **Data Validation**: Schema checks, missing values, and type consistency.
* **Feature Engineering**: Temporal and weather-based transformations using Hopsworks Feature Store.
* **Model Training**: Hyperparameter-tuned ensemble models (XGBoost, LightGBM).
* **Model Evaluation**: RMSE, MAE, R² metrics with visualizations.
* **Inference**: Real-time and batch predictions exposed via Streamlit app.
* **Monitoring**: Drift detection via Evidently, Prometheus, Grafana.

---

## Technologies Used

| Category          | Tools/Technologies                           |
| ----------------- | -------------------------------------------- |
| **Language**      | Python 3.10                                  |
| **ML Models**     | XGBoost, LightGBM, Optuna for tuning         |
| **Orchestration** | DVC + `dvc.yaml` pipelines                   |
| **Feature Store** | Hopsworks                                    |
| **Monitoring**    | Evidently, Prometheus, Grafana               |
| **Deployment**    | Docker, Kubernetes, AWS (EC2, IAM, ECR, CFN) |
| **App Interface** | Streamlit                                    |
| **CI/CD**         | GitHub Actions                               |

---

## Prerequisites

* Python >= 3.10
* AWS credentials configured with access to EC2, IAM, ECR, and CloudFormation.
* Valid API keys for:

  * NYISO electricity demand
  * Open-Meteo weather forecast
  * Hopsworks project and API key

---

## Installation and Setup

### Docker Build & Run

```bash
git clone https://github.com/yourusername/WattPredictor.git
cd WattPredictor
docker build -t wattpredictor .
docker run -p 8501:8501 --env-file .env wattpredictor
```

### Kubernetes Deployment

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### Local Development

```bash
poetry install
poetry run python main.py --stage feature_pipeline
poetry run python main.py --stage training_pipeline
poetry run python main.py --stage inference_pipeline
poetry run python main.py --stage monitoring_pipeline
```

---

## Project Structure

```
WattPredictor/
├── config_file/         # YAML config, schema, hyperparams
├── notebooks/           # EDA, training notebooks
├── src/WattPredictor/   # Main ML pipeline logic
│   ├── components/      # ingestion, validation, transformation, etc.
│   ├── pipeline/        # Orchestrated pipelines
│   ├── config/          # Configuration managers
│   ├── entity/          # Data classes (pydantic/dataclass)
│   ├── utils/           # Helpers, logger, exceptions
├── app.py               # Streamlit visualization
├── main.py              # CLI interface for DVC stages
├── Dockerfile           # Container config
├── dvc.yaml             # ML pipeline orchestration
└── .github/workflows/   # CI/CD via GitHub Actions
```

---

## Machine Learning Pipeline Details

### Feature Engineering

* Uses holiday flags, hour, day of week, and weather features.
* Feature group stored in Hopsworks.
* View: `elec_wx_features_view`

### Model Training

* Optimized using **Optuna**
* Models: XGBoost, LightGBM
* Cross-validated using time-aware `KFold`
* Registered to Hopsworks Model Registry

### Evaluation Metrics

| Metric | Description                  |
| ------ | ---------------------------- |
| RMSE   | Root Mean Square Error       |
| MAE    | Mean Absolute Error          |
| R²     | Coefficient of Determination |

---

## Monitoring and Observability

### Drift Detection

* Uses **Evidently** to compare recent predictions with historical stats.
* Output stored in `artifacts/drift/` and monitored in Streamlit.

### Dashboards

* **Grafana** connects to Prometheus for monitoring system-level metrics.
* Custom panels for:

  * API latency
  * Prediction frequency
  * Model version tracking

---

## Contribution Guidelines

1. Fork the repository.
2. Create a new branch: `feature/your-feature-name`
3. Make your changes and test thoroughly.
4. Submit a Pull Request with detailed description.

### Reporting Bugs

* Use GitHub Issues with reproduction steps.

### Code Review

* Reviews are required before merging to `main`.
* Ensure unit tests and linting pass via CI.

---

## License

This project is licensed under the [MIT License](./LICENSE).


---

## ✨ Quick Start: One-Command Execution

```bash
poetry run python main.py --stage feature_pipeline && \
poetry run python main.py --stage training_pipeline && \
poetry run python main.py --stage inference_pipeline && \
poetry run python main.py --stage monitoring_pipeline
```

---

## 🛡️ Troubleshooting

* **Model Not Found**: Ensure it's registered in Hopsworks
* **Drift Output Empty**: Ensure inference pipeline ran before monitoring
* **Feature Store Error**: Check `.env` for valid API keys
* **No Predictions Shown**: Make sure data ingestion and feature pipeline are up-to-date

---
