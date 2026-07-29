## ADDED Requirements

### Requirement: MLflow Experiment & Model Tracking
The `Trainer` and `Evaluation` components SHALL record grid search hyperparameter runs, validation metrics, and model artifacts to MLflow.

#### Scenario: Logging hyperparameter grid search runs
- **WHEN** `trainer.train()` executes model tuning
- **THEN** hyperparameters and RMSE evaluation scores are recorded under an MLflow experiment run

#### Scenario: Model artifact registration
- **WHEN** training and evaluation complete successfully
- **THEN** the trained pipeline model object is saved and registered in MLflow tracking storage
