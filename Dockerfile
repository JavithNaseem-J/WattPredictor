FROM python:3.12-slim AS builder
RUN pip install --no-cache-dir uv==0.11.15
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project --no-hashes -o /tmp/requirements.lock \
    && uv pip install --prefix=/install --no-cache -r /tmp/requirements.lock

FROM python:3.12-slim
ARG COMMIT_SHA=unknown
ARG RENDER_GIT_COMMIT=unknown
LABEL org.opencontainers.image.revision=$COMMIT_SHA
ENV BUILD_COMMIT_SHA=$COMMIT_SHA \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gettext-base libgomp1 nginx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ src/
COPY app.py ./
COPY config_file/ config_file/
COPY artifacts/ artifacts/
COPY deploy/ deploy/

# Render supplies the deployed commit at runtime; the file records image build time.
RUN printf '%s' "$RENDER_GIT_COMMIT" > /app/render_build_commit.txt \
    && python -c "from datetime import datetime, timezone; from pathlib import Path; Path('/app/build_time.txt').write_text(datetime.now(timezone.utc).isoformat())" \
    && mkdir -p artifacts/trainer artifacts/engineering artifacts/prediction logs data/processed data/raw/elec_data data/raw/wx_data \
    && useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:${PORT:-10000}/healthz > /dev/null || exit 1
CMD ["bash", "/app/deploy/start.sh"]
