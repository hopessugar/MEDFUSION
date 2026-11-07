# Multi-stage Dockerfile for MedFusion-RAFT
# Stage 1: Build & Dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system compilation tools and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime Environment
FROM python:3.11-slim AS runner

WORKDIR /app

# Install minimal runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy source repository
COPY configs /app/configs
COPY src /app/src
COPY backend /app/backend
COPY frontend /app/frontend
COPY pyproject.toml /app/pyproject.toml

# Expose API and Streamlit ports
EXPOSE 8000
EXPOSE 8501

# Health check against API endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command launches FastAPI backend
CMD ["uvicorn", "backend.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
