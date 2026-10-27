# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.11-slim-bookworm
FROM ${PYTHON_IMAGE} AS dependencies
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY requirements-core.txt .
RUN python -m pip install --index-url "$PIP_INDEX_URL" --prefix=/install -r requirements-core.txt

FROM ${PYTHON_IMAGE} AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TZ=Asia/Shanghai \
    DATABASE_URL=sqlite:////data/lifeops.db KNOWLEDGE_DATABASE_URL=sqlite:////data/knowledge.db \
    CHECKPOINTER_DB_PATH=/data/checkpointer.db PERSONAL_MEMORY_DIR=/data/memory \
    HARNESS_TRACE_DIR=/data/harness DOCS_DIR=/opt/lifeops/examples/knowledge RAG_BACKEND=bm25
RUN test -f /etc/ssl/certs/ca-certificates.crt && test -f /usr/share/zoneinfo/Asia/Shanghai \
    && groupadd --gid 10001 lifeops && useradd --uid 10001 --gid lifeops --create-home lifeops \
    && mkdir -p /data /reports && chown lifeops:lifeops /data /reports
COPY --from=dependencies /install /usr/local
WORKDIR /opt/lifeops
COPY --chown=lifeops:lifeops app ./app
COPY --chown=lifeops:lifeops examples ./examples
COPY --chown=lifeops:lifeops scripts ./scripts
COPY requirements-core.txt requirements-ml.txt ./
USER lifeops
EXPOSE 5000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health/ready', timeout=2)"
CMD ["python", "-m", "app.src"]

FROM runtime AS test
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV RUFF_CACHE_DIR=/tmp/ruff PYTEST_ADDOPTS="-o cache_dir=/tmp/pytest-cache"
USER root
COPY requirements-test.txt ./
RUN python -m pip install --no-cache-dir --index-url "$PIP_INDEX_URL" -r requirements-test.txt
COPY --chown=lifeops:lifeops tests ./tests
COPY --chown=lifeops:lifeops pyproject.toml ./
USER lifeops
CMD ["python", "-m", "pytest", "tests", "-q", "--junitxml=/reports/unit.xml"]

FROM runtime AS ml
USER root
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install --no-cache-dir -r requirements-ml.txt
USER lifeops

FROM runtime AS production
