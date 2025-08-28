FROM python:3.10-slim AS builder
ENV PYTHONUNBUFFERED=1

# Install build dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies in a virtual environment
RUN uv venv /venv && uv pip install --no-cache-dir -e . --no-deps && uv pip install --no-cache-dir .[prod]

FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1

COPY --from=builder /venv /venv

# Copy only necessary files
WORKDIR /app
COPY src/ src/
COPY app.py main.py config_file/ ./

# Create non-root user and directories
RUN useradd -m -u 1000 appuser && \
    mkdir -p artifacts/prediction logs data/raw && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py"]