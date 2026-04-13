# Python code evaluation sandbox.
# Uses the scientific dependency set from pyproject.toml.
#
# Build:
#   docker build -t nl-code/code-eval-scientific:v1 -f docker/scientific.Dockerfile .
FROM python:3.12-slim

LABEL org.opencontainers.image.title="nl-code/code-eval-scientific"
LABEL org.opencontainers.image.description="Python code evaluation sandbox (scientific)"
LABEL org.opencontainers.image.version="v1"

COPY pyproject.toml /tmp/nl-code/pyproject.toml

RUN python - <<'PY' > /tmp/bigcodebench-requirements.txt
from pathlib import Path
import tomllib

pyproject = tomllib.loads(Path("/tmp/nl-code/pyproject.toml").read_text())
requirements = pyproject["project"]["optional-dependencies"]["bigcodebench"]
print("\n".join(requirements))
PY

RUN pip install --no-cache-dir -r /tmp/bigcodebench-requirements.txt \
    && rm -rf /root/.cache/pip /tmp/bigcodebench-requirements.txt /tmp/nl-code

RUN useradd -m -s /bin/bash evaluser \
    && mkdir -p /sandbox \
    && chown evaluser:evaluser /sandbox

COPY --chown=evaluser:evaluser src/nl_code/code_execution/worker.py /sandbox/worker.py

USER evaluser
WORKDIR /tmp
