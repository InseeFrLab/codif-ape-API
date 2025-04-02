# Use Python 3.12 as the base image
FROM python:3.12-slim

# Set environment variables
ARG API_USERNAME
ARG API_PASSWORD
ENV API_USERNAME=${API_USERNAME}
ENV API_PASSWORD=${API_PASSWORD}
ENV TIMEOUT=300

# Set working directory inside src/
WORKDIR /api/src

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml uv.lock ../

# Install dependencies using uv
RUN uv sync

# Copy the source code
COPY . .

# Download required NLTK stopwords
RUN uv run -m nltk.downloader stopwords

# Expose the API port
EXPOSE 80

# Command to run FastAPI with Uvicorn
CMD ["uvicorn", "api.main:codification_ape_app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80", "--timeout-graceful-shutdown", "300"]
