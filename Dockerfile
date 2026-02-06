FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src

WORKDIR /app

# Install only curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for layer caching
COPY requirements.txt ./

# Install Python dependencies with retries and longer timeout
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt

# Copy application code
COPY src/ src/
COPY app.py ./
COPY config_file/ config_file/

# Copy artifacts (model and preprocessed data)
COPY artifacts/trainer/ artifacts/trainer/
COPY artifacts/engineering/ artifacts/engineering/

# Create non-root user and required directories
RUN useradd -m -u 1000 appuser && \
    mkdir -p artifacts/prediction logs data/processed && \
    chown -R appuser:appuser /app

USER appuser

# Environment variables
ENV ELEC_API=https://api.eia.gov/v2/electricity/rto/region-sub-ba-data/data/
ENV WX_API=https://api.open-meteo.com/v1/forecast

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0"]