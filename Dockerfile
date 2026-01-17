FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src:$PYTHONPATH

WORKDIR /app

# Install uv (10-100x faster than pip)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY src/ src/
COPY app.py ./
COPY config_file/ config_file/

# Copy pre-trained model and artifacts
COPY artifacts/trainer/model.joblib artifacts/trainer/
COPY artifacts/engineering/preprocessed.csv artifacts/engineering/

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p artifacts/prediction logs && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["python", "-m", "streamlit", "run", "app.py"]