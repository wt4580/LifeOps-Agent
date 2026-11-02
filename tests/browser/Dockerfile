FROM python:3.11-slim-bookworm
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN python -m pip install --no-cache-dir --index-url "$PIP_INDEX_URL" playwright==1.64.1
RUN python -m playwright install --with-deps --only-shell chromium
WORKDIR /checks
COPY tests/browser/acceptance.py /checks/acceptance.py
ENTRYPOINT ["python", "/checks/acceptance.py"]
