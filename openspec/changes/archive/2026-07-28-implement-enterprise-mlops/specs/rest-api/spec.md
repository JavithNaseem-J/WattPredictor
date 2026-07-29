## ADDED Requirements

### Requirement: FastAPI Microservice Endpoint
The system SHALL provide a `FastAPI` REST application exposing `/health`, `/predict`, and `/metrics` HTTP endpoints.

#### Scenario: Requesting system health status
- **WHEN** an HTTP GET request is sent to `/health`
- **THEN** the API returns HTTP 200 with JSON status `"healthy"` and model availability details

#### Scenario: Requesting demand predictions via REST API
- **WHEN** an HTTP POST request is sent to `/predict`
- **THEN** the API executes prediction via `Predictor` and returns JSON predictions per sub-region

#### Scenario: Fetching evaluation & business metrics
- **WHEN** an HTTP GET request is sent to `/metrics`
- **THEN** the API returns the latest validation metrics (RMSE, MAE, MAPE) and business financial savings JSON
