## Context

After eliminating structural debt, profiling reveals performance optimization targets:
- Time-series feature extraction in `ts_generator.py` uses `.iloc[start:mid]` inside nested Python loops for each sliding window index. Vectorizing array generation using NumPy's `sliding_window_view` converts loop iterations into single 2D matrix slices.
- `app.py` computes live predictions synchronously on every Streamlit page rerun without caching prediction results.
- `Dockerfile` runs Python 3.10 and copies `requirements.txt` instead of using Python 3.12 and `uv.lock`.

## Goals / Non-Goals

**Goals:**
- Replace Python sequence slicing loops in `ts_generator.py` with NumPy `sliding_window_view` vectorization.
- Add Streamlit `@st.cache_data(ttl=300)` caching for real-time model inference and live data queries in `app.py`.
- Upgrade `Dockerfile` to `python:3.12-slim` and deploy via `uv sync --frozen`.

**Non-Goals:**
- Modifying underlying XGBoost / LightGBM hyperparameter search space.
- Changing EIA / Weather API endpoint schemas.

## Decisions

- **Decision 1: NumPy `sliding_window_view` Vectorization**
  - **Approach**: Replace the inner loop in `features_and_target()` with `numpy.lib.stride_tricks.sliding_window_view(demand_array, window_shape=input_seq_len)`.
  - **Rationale**: Operates directly on C-contiguous memory blocks without per-row pandas object instantiation overhead.

- **Decision 2: UI Result Caching**
  - **Approach**: Wrap real-time prediction invocation in a cached helper function `@st.cache_data(ttl=300)`.
  - **Rationale**: Prevents re-running model prediction matrices on minor UI events (e.g., resizing map view).

- **Decision 3: Python 3.12 & `uv.lock` Containerization**
  - **Approach**: Update `Dockerfile` base image to `python:3.12-slim` and use `uv sync --frozen` in builder stage.
  - **Rationale**: Aligns containerized runtime with local development environment and guarantees deterministic package builds.

## Risks / Trade-offs

- **[Risk] Memory allocation during large stride views**
  - **Mitigation**: Stride views create non-copying array views; only standard 2D arrays are instantiated upon pandas DataFrame construction.
