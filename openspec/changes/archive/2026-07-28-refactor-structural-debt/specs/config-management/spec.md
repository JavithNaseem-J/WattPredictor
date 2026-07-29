## ADDED Requirements

### Requirement: Unified YAML Configuration Integration
The `WattPredictorConfig` class SHALL load project structure and hyperparameter configuration directly from `config_file/config.yaml` and `config_file/params.yaml`.

#### Scenario: Config initialization loads YAML settings
- **WHEN** `get_config()` or `WattPredictorConfig()` is initialized
- **THEN** configuration properties reflect values defined in `config_file/config.yaml` and `config_file/params.yaml`

### Requirement: Direct Config Object Dependency Injection
Component classes (`Ingestion`, `Validation`, `Engineering`, `Trainer`, `Evaluation`, `Predictor`, `Drift`, `Monitoring`) SHALL accept `WattPredictorConfig` directly instead of intermediate Pydantic entity classes.

#### Scenario: Component initialization with WattPredictorConfig
- **WHEN** a pipeline component is instantiated with a `WattPredictorConfig` object
- **THEN** the component successfully accesses all required directory paths and hyperparameters via the config instance
