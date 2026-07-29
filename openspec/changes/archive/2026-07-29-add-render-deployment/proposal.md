## Why

WattPredictor currently runs locally or inside local Docker containers, but lacks automated, cloud-based deployment infrastructure. Deploying to Render via a Blueprint configuration (`render.yaml`) and updated Dockerfile enables 1-click cloud deployment for both the Streamlit dashboard and the FastAPI REST API, complete with environment credential management and health monitoring.

## What Changes

- Create `render.yaml` Blueprint specification defining two web services (`wattpredictor-dashboard` and `wattpredictor-api`).
- Update `Dockerfile` start command to support dynamic `$PORT` binding assigned by cloud hosters like Render.
- Document deployment procedures, environment variable requirements (`ELEC_API_KEY`, `ELEC_API`, `WX_API`), and Render Blueprint setup in project documentation.

## Capabilities

### New Capabilities
- `cloud-deployment`: Infrastructure-as-code configuration (`render.yaml`) and dynamic port handling for deploying WattPredictor services to Render.

### Modified Capabilities

## Impact

- `Dockerfile`: Updated CMD to dynamically read `$PORT` with a local fallback to `8501`.
- `render.yaml`: New file at workspace root for Render Blueprint provisioning.
- `README.md`: Updated Deployment section with Render Blueprint instructions and URLs.
