# 🐍 Base Python image
FROM python:3.10-slim

# 🌍 Environment setup
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.2

# 📂 Working directory
WORKDIR /app

# 🛠️ System dependencies (add more if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    git \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# 📦 Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# 📝 Copy only dependency files first (for caching)
COPY pyproject.toml .
COPY poetry.lock* .

# ⚙️ Configure Poetry to avoid virtualenvs & timeouts
RUN poetry config virtualenvs.create false && \
    poetry config installer.max-workers 1 && \
    poetry config installer.parallel false && \
    poetry config cache-dir /tmp/poetry-cache

# 📁 Copy your actual code
COPY src/ ./src

# 🧠 Install dependencies (with retry for flaky installs)
RUN poetry install --no-interaction --no-ansi --no-root || \
    poetry install --no-interaction --no-ansi --no-root

# 🚪 Expose app port (Streamlit)
EXPOSE 8501

# 🚀 Default command (change to main.py if needed)
CMD ["streamlit", "run", "src/WattPredictor/app.py"]
