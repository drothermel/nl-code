# Python code evaluation sandbox with scientific libraries.
# For ClassEval and datasets that need numpy, pandas, etc.
#
# Build:
#   docker build -t nl-code/code-eval-scientific:v1 -f docker/scientific.Dockerfile .
FROM python:3.12-slim

LABEL org.opencontainers.image.title="nl-code/code-eval-scientific"
LABEL org.opencontainers.image.description="Python code evaluation sandbox (scientific)"
LABEL org.opencontainers.image.version="v1"

RUN pip install --no-cache-dir \
    numpy>=1.26 \
    pandas>=2.2 \
    scipy>=1.14 \
    scikit-learn>=1.5

RUN useradd -m -s /bin/bash evaluser

COPY src/nl_code/code_execution/worker.py /sandbox/code_eval_worker.py

USER evaluser
WORKDIR /sandbox
