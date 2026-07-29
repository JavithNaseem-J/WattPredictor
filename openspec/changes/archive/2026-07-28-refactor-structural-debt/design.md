## Context

The WattPredictor codebase has accumulated structural technical debt across configuration, web presentation, path management, and data ingestion layers:
- Configuration is fragmented between hardcoded dataclass defaults (`WattPredictorConfig` in `config.py`), unused Pydantic models (`config_entity.py`), and un-integrated YAML files (`config.yaml` / `params.yaml`).
- `app.py` contains redundant feature assembly, NYISO zone mapping, historical data padding, and custom DST calculation loops, bypassing the pipeline `Predictor` component.
- Component modules (`predictor.py`, `drift.py`, `monitoring.py`, `evaluator.py`) hardcode path strings like `"artifacts/engineering/preprocessed.csv"` rather than referencing `self.config`.
- `Ingestion` re-implements raw HTTP request assembly instead of reusing `EIAClient`.
- `CustomException` in `exception.py` manually parses `sys.exc_info()`, adding unnecessary abstraction over standard Python traceback and logging.

## Goals / Non-Goals

**Goals:**
- Unify `WattPredictorConfig` to dynamically load and expose settings from `config_file/config.yaml` and `config_file/params.yaml`.
- Remove redundant Pydantic models in `config_entity.py` and update component constructors to accept `WattPredictorConfig`.
- Refactor `app.py` to delegate inference and feature construction to `Predictor` and use `pytz` for timezone handling.
- Standardize artifact path accessors across all component files via `self.config`.
- Refactor `Ingestion` to delegate HTTP data fetching directly to `EIAClient`.
- Remove `CustomException`, `main.py`, empty notebook `WattPredictor_Project.ipynb`, and dead import code.

**Non-Goals:**
- Changing underlying ML model architecture, grid search spaces, or training hyperparameter settings.
- Altering external DVC pipeline stage commands or output file locations.
- Modifying visual layout styling or color schemes of the Streamlit dashboard.

## Decisions

- **Decision 1: Direct YAML Configuration Binding**
  - **Approach**: Modify `WattPredictorConfig` to parse `config.yaml` and `params.yaml` via `read_yaml` helper upon initialization.
  - **Rationale**: Keeps a single, clean typed Python interface while honoring external YAML configuration as the single source of truth.
  - **Alternatives Considered**: Keeping `config_entity.py` Pydantic models. Rejected due to redundant boilerplate without runtime validation value.

- **Decision 2: App Refactoring & Predictor Integration**
  - **Approach**: Update `app.py` to import `Predictor` from `WattPredictor.components.inference.predictor` and invoke `predictor.predict()`. Replace manual DST loops with `pytz.timezone("America/New_York")`.
  - **Rationale**: Eliminates code drift between live Streamlit predictions and pipeline evaluation logic. Standardizes timezone conversion using battle-tested standard libraries.
  - **Alternatives Considered**: Keeping custom DST loop in `app.py`. Rejected due to fragility and unnecessary complexity.

- **Decision 3: Replacing CustomException with Standard Logging & Exceptions**
  - **Approach**: Replace `CustomException` with standard `ValueError` / `RuntimeError` and `logger.exception()` / `logger.error()`.
  - **Rationale**: Python's standard library and logging infrastructure handle tracebacks natively without custom frame-parsing code.

## Risks / Trade-offs

- **[Risk] Test suite failures due to signature changes in component constructors**
  - **Mitigation**: Update all unit and integration tests (`tests/test_models.py`, `tests/test_integration.py`, `tests/test_features.py`) to pass `WattPredictorConfig` instances directly to components.
