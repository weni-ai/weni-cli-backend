FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_VERSION=1.8.5 \
    PYTHONPATH="$PYTHONPATH:/app"

# Set working directory
WORKDIR /app

# Install poetry
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s "$POETRY_HOME/bin/poetry" /usr/local/bin/poetry

RUN poetry self update 1.8.5

# Copy poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-root

# Copy the rest of the application
COPY . .

# Run as non-root user
RUN groupadd -r weni && useradd -r -g weni weni \
    && chown -R weni:weni /app
USER weni

# Expose the port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "app.server"]
