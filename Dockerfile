# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install system dependencies, including Git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set working directory inside `src/`
WORKDIR /api/src

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Copy pyproject.toml and lockfile for dependency resolution
COPY pyproject.toml uv.lock ./

# Sync dependencies
RUN uv sync --locked --no-dev

# Copy application code + nltk stopwords
COPY ./src /api/src
COPY ./nltk_data /api/nltk_data

# Expose port 5000
EXPOSE 5000

# Start FastAPI application
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "5000"]
