FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.2

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    git \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

COPY pyproject.toml .
COPY poetry.lock* .

RUN poetry config virtualenvs.create false && \
    poetry config installer.max-workers 1 && \
    poetry config installer.parallel false && \
    poetry config cache-dir /tmp/poetry-cache

COPY src/ ./src

RUN poetry install --no-interaction --no-ansi --no-root || \
    poetry install --no-interaction --no-ansi --no-root

EXPOSE 8501

CMD ["streamlit", "run", "src/WattPredictor/app.py"]
