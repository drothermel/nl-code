# Python-only code evaluation sandbox (slim).
# For HumanEval, HumanEval-Pro, MBPP-Pro, BigCodeBench Lite Pro.
#
# Build:
#   docker build -t nl-code/code-eval:v1 -f docker/slim.Dockerfile .
FROM python:3.12-slim

LABEL org.opencontainers.image.title="nl-code/code-eval"
LABEL org.opencontainers.image.description="Python code evaluation sandbox (slim)"
LABEL org.opencontainers.image.version="v1"

RUN useradd -m -s /bin/bash evaluser

COPY src/nl_code/code_execution/worker.py /sandbox/code_eval_worker.py

USER evaluser
WORKDIR /sandbox
