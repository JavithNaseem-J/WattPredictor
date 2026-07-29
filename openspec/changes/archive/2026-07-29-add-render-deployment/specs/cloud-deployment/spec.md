# Capability: Cloud Deployment

## ADDED Requirements

### Requirement: Render Blueprint Infrastructure as Code
The project MUST include a `render.yaml` Blueprint file at the repository root defining web services for both the Streamlit Dashboard and the FastAPI REST API.

#### Scenario: Deploying via Render Blueprint
- GIVEN a GitHub repository containing `render.yaml`
- WHEN a user creates a new Blueprint instance in Render
- THEN Render MUST automatically provision both `wattpredictor-dashboard` and `wattpredictor-api` services.

### Requirement: Dynamic Port Binding in Docker Container
The Docker container MUST support dynamic port binding via the `$PORT` environment variable provided by cloud hosts, falling back to port `8501` when `$PORT` is not set.

#### Scenario: Running container in cloud hosting vs local Docker
- GIVEN a container environment with `PORT=10000`
- WHEN the container starts
- THEN Streamlit MUST listen on port `10000`.
- GIVEN a container environment without `PORT` set
- WHEN the container starts
- THEN Streamlit MUST listen on port `8501`.
