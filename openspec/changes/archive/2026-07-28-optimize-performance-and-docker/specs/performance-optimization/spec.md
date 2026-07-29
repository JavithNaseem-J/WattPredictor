## ADDED Requirements

### Requirement: Vectorized Time-Series Feature Extraction
The `features_and_target` function SHALL construct sequence feature matrices using vectorized NumPy window slicing rather than per-row DataFrame slicing loops.

#### Scenario: Generating features for time-series sequences
- **WHEN** `features_and_target(df, input_seq_len, step_size)` is called with a valid dataset
- **THEN** it generates feature matrices and targets matching the expected shape without pandas indexing warnings

### Requirement: Real-Time UI Inference Caching
The Streamlit application `app.py` SHALL cache prediction results with a TTL to prevent redundant model execution on UI reruns.

#### Scenario: Interacting with dashboard UI controls
- **WHEN** a user interacts with widgets on the Streamlit dashboard within the TTL window
- **THEN** predictions are returned instantly from cache without re-executing model inference
