# realtime-inference Specification

## Purpose
TBD - created by archiving change refactor-structural-debt. Update Purpose after archive.
## Requirements
### Requirement: Decoupled Live Inference via Predictor Component
The Streamlit application `app.py` SHALL delegate real-time demand prediction logic directly to the `Predictor` component.

#### Scenario: Real-time prediction generation in web application
- **WHEN** `app.py` generates predictions for NYISO zones
- **THEN** it executes prediction using `Predictor.predict()` and uses the resulting DataFrame for UI map and metric displays

### Requirement: Standardized Eastern Time Zone Conversion
The application SHALL use `pytz` or `zoneinfo` standard library timezone modules to handle Eastern Time conversions and DST calculations.

#### Scenario: Display current New York Eastern time
- **WHEN** `app.py` calculates current Eastern Time and prediction target timestamps
- **THEN** it converts UTC timestamps to America/New_York timezone using standard DST-aware timezone objects

