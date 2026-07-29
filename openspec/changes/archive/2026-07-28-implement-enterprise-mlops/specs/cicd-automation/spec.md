## ADDED Requirements

### Requirement: GitHub Actions Continuous Integration Workflow
The repository SHALL contain a `.github/workflows/ml_pipeline.yml` GitHub Actions workflow.

#### Scenario: Running automated CI test checks
- **WHEN** code is pushed or a pull request is submitted
- **THEN** GitHub Actions installs project dependencies via `uv` and executes pytest test suites
