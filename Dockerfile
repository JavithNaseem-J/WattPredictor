# Stage 1: Builder
FROM python:3.12-slim AS builder

# Install uv for ultra-fast package installation
RUN pip install --no-cache-dir uv

WORKDIR /build

# Copy dependency definition files for deterministic installation
COPY pyproject.toml uv.lock* requirements.txt ./

# Install packages into /install prefix using uv
RUN uv pip install --prefix=/install --no-cache -r pyproject.toml

# Stage 2: Runtime
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app

WORKDIR /app

# Install runtime system dependencies (curl for healthcheck, libgomp1 for LightGBM OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ src/
COPY app.py .
COPY config_file/ config_file/

# Create necessary directories
RUN mkdir -p artifacts/trainer artifacts/engineering artifacts/prediction \
    logs data/processed data/raw/elec_data data/raw/wx_data

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Run Streamlit dashboard (supports dynamic $PORT from cloud hosters like Render)
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]