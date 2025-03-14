# Weni CLI Backend

[![CI](https://github.com/weni-ai/weni-cli-backend/actions/workflows/ci.yaml/badge.svg)](https://github.com/weni-ai/weni-cli-backend/actions/workflows/ci.yaml)
[![CD](https://github.com/weni-ai/weni-cli-backend/actions/workflows/build-push-tag.yaml/badge.svg)](https://github.com/weni-ai/weni-cli-backend/actions/workflows/build-push-tag.yaml)

FastAPI backend service for Weni CLI.

## Features

- FastAPI-based REST API
- Poetry for dependency management
- Type hints and validation with Pydantic
- Structured project layout
- Configuration management with environment variables
- Linting with Ruff and Mypy

## Prerequisites

- Python 3.12 or higher
- Poetry

## Setup

1. Clone the repository:

```bash
git clone https://github.com/weni-ai/weni-cli-backend.git
cd weni-cli-backend
```

2. Install dependencies with Poetry:

```bash
poetry install
```

3. Create a `.env` file based on the example:

```bash
cp .env.example .env
```

4. Configure environment variables in the `.env` file.

## Development

### Running the server for development

```bash
# Using poetry
poetry run uvicorn app.main:app --reload --host=0.0.0.0 --port=8000

# Or activate the virtual environment first
poetry shell
python -m app.server
```

The API will be available at `http://localhost:8000`.

## Testing

### Running tests

```bash
poetry run pytest
```

### Linting

```bash
# Run ruff linter
poetry run ruff check .

# Run mypy type checker
poetry run mypy .
```

## Docker

### Using Docker Compose

The project includes Docker configuration for both development and production environments.

#### Development Environment

Run the application with hot reload for development:

```bash
docker-compose up api-dev
```

The API will be available at `http://localhost:8001`.

#### Production Environment

Run the application in production mode:

```bash
docker-compose up api
```

The API will be available at `http://localhost:8000`.

### Building Docker Images

Build the production image:

```bash
docker build -t weni-cli-backend:latest -f docker/Dockerfile .
```

Build the development image:

```bash
docker build -t weni-cli-backend:dev -f docker/Dockerfile.dev .
```

### Running Docker Containers Directly

Run the production container:

```bash
docker run -p 8000:8000 --env-file .env weni-cli-backend:latest
```

Run the development container with mounted volume for hot reload:

```bash
docker run -p 8000:8000 -v $(pwd):/app --env-file .env weni-cli-backend:dev
``` 