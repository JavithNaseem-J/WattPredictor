## 1. Configuration System Consolidation

- [x] 1.1 Update `WattPredictorConfig` in `src/WattPredictor/config/config.py` to read paths and params directly from `config_file/config.yaml` and `config_file/params.yaml`.
- [x] 1.2 Remove `src/WattPredictor/entity/config_entity.py` and update component classes (`Ingestion`, `Validation`, `Engineering`, `Trainer`, `Evaluation`, `Predictor`, `Drift`, `Monitoring`) to take `WattPredictorConfig` directly.
- [x] 1.3 Update pipeline runners (`feature_pipeline.py`, `inference_pipeline.py`, `monitoring_pipeline.py`, `training_pipeline.py`) to pass `WattPredictorConfig` to component initializers.

## 2. Decouple Web Application & Timezone Standardization

- [x] 2.1 Refactor `app.py` to remove duplicate feature engineering and manual DST loop functions (`get_eastern_offset`, `get_current_times`, `utc_to_eastern`, `eastern_to_utc`).
- [x] 2.2 Standardize Eastern Time conversions in `app.py` using `pytz.timezone('US/Eastern')`.
- [x] 2.3 Import `Predictor` into `app.py` and delegate real-time demand prediction and feature vector construction directly to `Predictor.predict()`.

## 3. Standardize File Paths & Ingestion Logic

- [x] 3.1 Refactor `predictor.py`, `drift.py`, `monitoring.py`, and `evaluator.py` to retrieve file paths strictly via `self.config` properties instead of hardcoded string literals.
- [x] 3.2 Refactor `Ingestion` in `src/WattPredictor/components/features/ingestion.py` to use `EIAClient.fetch_day` and `EIAClient.fetch_range` directly.
- [x] 3.3 Replace `CustomException` in `src/WattPredictor/utils/exception.py` with standard Python exceptions and structured `logger.exception()` logging.

## 4. Codebase Cleanup & Test Verification

- [x] 4.1 Remove dead files: `main.py`, `notebooks/WattPredictor_Project.ipynb`, unused factory helpers in `api_client.py`, and unreferenced imports across components.
- [x] 4.2 Update test suite in `tests/` (`test_models.py`, `test_integration.py`, `test_features.py`, `test_api_client.py`) to match updated component constructors and configuration properties.
- [x] 4.3 Run `pytest` to verify all unit and integration tests pass cleanly.
