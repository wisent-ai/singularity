# ============================================================================
# Singularity Agent - Docker Image
# ============================================================================
# Multi-stage build for minimal production image
#
# Build:   docker build -t singularity .
# Run:     docker run --env-file .env singularity
# Compose: docker compose up
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder - install dependencies
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy only dependency files first (cache layer)
COPY pyproject.toml ./
COPY singularity/__init__.py singularity/__init__.py

# Install the package with core dependencies
RUN pip install --no-cache-dir --prefix=/install .

# ---------------------------------------------------------------------------
# Stage 2: Runtime - minimal production image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Labels
LABEL org.opencontainers.image.title="Singularity Agent"
LABEL org.opencontainers.image.description="Autonomous AI agent framework"
LABEL org.opencontainers.image.source="https://github.com/wisent-ai/singularity"
LABEL org.opencontainers.image.licenses="MIT"

# Create non-root user for security
RUN groupadd --gid 1000 agent && \
    useradd --uid 1000 --gid agent --shell /bin/bash --create-home agent

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
WORKDIR /app
COPY singularity/ singularity/
COPY examples/ examples/
COPY pyproject.toml ./

# Install the package itself (deps already installed)
RUN pip install --no-cache-dir --no-deps .

# Create data directory for persistent state
RUN mkdir -p /app/agent_data && chown -R agent:agent /app/agent_data

# ---------------------------------------------------------------------------
# Environment defaults (override with --env-file or docker-compose)
# ---------------------------------------------------------------------------
ENV AGENT_NAME="Singularity" \
    AGENT_TICKER="SING" \
    AGENT_TYPE="general" \
    STARTING_BALANCE="10.0" \
    LLM_PROVIDER="anthropic" \
    LLM_MODEL="claude-sonnet-4-20250514" \
    PYTHONUNBUFFERED="1" \
    PYTHONDONTWRITEBYTECODE="1" \
    AGENT_DATA_DIR="/app/agent_data"

# Switch to non-root user
USER agent

# Health check - verify Python and singularity are importable
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from singularity import AutonomousAgent; print('healthy')" || exit 1

# Expose port for Service API (if used)
EXPOSE 8000

# Default entrypoint runs the agent
ENTRYPOINT ["python", "-m", "singularity.autonomous_agent"]
