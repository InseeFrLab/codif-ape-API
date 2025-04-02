FROM python:3.13

# Install system dependencies, including Git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set working directory inside `src/`
WORKDIR /api/src

# Copy pyproject.toml and lockfile for dependency resolution
COPY pyproject.toml uv.lock ./

# Install uv package manager
RUN pip install uv

# Sync dependencies
RUN uv sync

# Copy application code
COPY ./src /api/src

# Expose port 5000
EXPOSE 5000

# Start FastAPI application
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "5000"]
