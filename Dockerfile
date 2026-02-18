# Stage 1: Builder
FROM python:3.10 AS builder

# Install uv for ultra-fast package installation
RUN pip install --no-cache-dir uv

WORKDIR /build

# Copy requirements
COPY requirements.txt .

# Install to a specific prefix (we'll copy this to runtime)
RUN uv pip install --prefix=/install --no-cache -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app

WORKDIR /app

# Install ONLY curl for healthcheck (no build tools = -150MB)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
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
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]