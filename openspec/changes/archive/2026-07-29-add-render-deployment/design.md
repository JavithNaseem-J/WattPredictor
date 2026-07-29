# Design: Render Cloud Deployment

## Overview

This design details the infrastructure-as-code setup for deploying WattPredictor to Render via a Blueprint file (`render.yaml`) and Docker image dynamic port binding.

## Architecture & Services

```
                               ┌───────────────────────────┐
                               │     Render Blueprint      │
                               │        render.yaml        │
                               └─────────────┬─────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       │                                           │
                       ▼                                           ▼
         ┌───────────────────────────┐               ┌───────────────────────────┐
         │   wattpredictor-dashboard │               │      wattpredictor-api    │
         │      (Streamlit UI)       │               │      (FastAPI REST)       │
         ├───────────────────────────┤               ├───────────────────────────┤
         │ Docker runtime            │               │ Python runtime            │
         │ Reads $PORT dynamically   │               │ uvicorn start command     │
         │ Default fallback: 8501    │               │ Port: $PORT               │
         └───────────────────────────┘               └───────────────────────────┘
```

## Technical Specification

### 1. Dynamic Port Execution (`Dockerfile`)
- Modify `Dockerfile` CMD line from hardcoded `--server.port=8501` to shell evaluation:
  `CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]`
- This ensures local Docker containers continue defaulting to `8501` while cloud hosters (Render) passing dynamic `$PORT` environment variables bind cleanly.

### 2. Render Blueprint Specification (`render.yaml`)
- Define `wattpredictor-dashboard` (Docker web service)
- Define `wattpredictor-api` (Python web service using `uvicorn`)
- Environment variables: `ELEC_API`, `WX_API`, `ELEC_API_KEY` (sync: false)

### 3. Documentation & Verification
- Document 1-click Render setup in `README.md` under Deployment.
