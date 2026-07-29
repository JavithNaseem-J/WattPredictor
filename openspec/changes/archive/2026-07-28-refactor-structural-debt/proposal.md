## Why

The WattPredictor codebase currently suffers from significant structural debt, including dual configuration systems (hardcoded dataclasses vs. unused Pydantic models and YAML files), UI code coupling in `app.py` that duplicates model feature assembly and DST calculations, hardcoded artifact file paths across components, and duplicate API ingestion logic. These issues create maintainability confusion, risk silent runtime bugs during pipeline updates, and clutter the codebase with over-engineered and dead code.

## What Changes

- **Consolidate Configuration**: Unify `WattPredictorConfig` to parse `config_file/config.yaml` and `params.yaml` directly; delete the redundant Pydantic models in `src/WattPredictor/entity/config_entity.py`.
- **Decouple Web Application (`app.py`)**: Refactor `app.py` to delegate feature preparation and demand prediction directly to `Predictor` from `components/inference/predictor.py`. Replace custom DST offset loops with standard `pytz` / `zoneinfo` time zone conversion.
- **Standardize File Path Retrieval**: Eliminate hardcoded path string literals across `predictor.py`, `drift.py`, `monitoring.py`, `evaluator.py`, and `app.py` in favor of centralized properties on `self.config`.
- **Deduplicate Ingestion Logic**: Refactor `Ingestion` to utilize `EIAClient.fetch_day` and `EIAClient.fetch_range` directly instead of re-implementing HTTP request parameters.
- **Clean Up Over-Engineering & Dead Code**: Remove `CustomException` in favor of native Python exceptions and structured logging; delete unused root entry point `main.py`, empty notebook `WattPredictor_Project.ipynb`, and unreferenced helper functions/imports.

## Capabilities

### New Capabilities
- `config-management`: Consolidated, single source of truth configuration system combining YAML config files with typed Python dataclass accessors.
- `realtime-inference`: Decoupled real-time prediction interface utilizing pipeline components and standardized timezone handling.

### Modified Capabilities

## Impact

- **Affected Code**: `app.py`, `main.py`, `src/WattPredictor/config/config.py`, `src/WattPredictor/entity/config_entity.py`, `src/WattPredictor/components/features/ingestion.py`, `src/WattPredictor/components/inference/predictor.py`, `src/WattPredictor/components/monitor/drift.py`, `src/WattPredictor/components/monitor/monitoring.py`, `src/WattPredictor/components/training/trainer.py`, `src/WattPredictor/components/training/evaluator.py`, `src/WattPredictor/pipeline/feature_pipeline.py`, `src/WattPredictor/utils/exception.py`, `src/WattPredictor/utils/api_client.py`.
- **APIs & Dependencies**: No breaking API changes; utilizes existing `pytz` dependency for DST-aware Eastern Time calculations.
