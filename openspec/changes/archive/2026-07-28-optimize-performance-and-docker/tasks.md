## 1. Vectorized Feature Generator Optimization

- [x] 1.1 Refactor `features_and_target` in `src/WattPredictor/utils/ts_generator.py` to use `numpy.lib.stride_tricks.sliding_window_view` for 2D demand window generation.
- [x] 1.2 Fix pandas `SettingWithCopyWarning` in `average_demand_last_4_weeks()` by ensuring explicit copy initialization `X = X.copy()`.

## 2. Streamlit Prediction & UI Caching

- [x] 2.1 Wrap prediction execution in `app.py` with `@st.cache_data(ttl=300)` to cache predictions across user interactions.
- [x] 2.2 Wrap weather and electricity live API fetch calls in `app.py` with `@st.cache_data(ttl=600)` to prevent redundant HTTP roundtrips.

## 3. Container Runtime Alignment

- [x] 3.1 Update `Dockerfile` base image from `python:3.10` to `python:3.12-slim` for both builder and runtime stages.
- [x] 3.2 Update `Dockerfile` builder commands to utilize `uv sync --frozen` for reproducible production image builds.

## 4. Verification

- [x] 4.1 Run `.venv\Scripts\pytest` to verify feature generator vectorization and pipeline functionality pass all 74 test cases.
