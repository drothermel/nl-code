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

RUN python - <<'PY' > /tmp/dr-docker-requirements.txt
from pathlib import Path
import tomllib

pyproject = tomllib.loads(Path("/tmp/nl-code/pyproject.toml").read_text())
requirements = [
    requirement
    for requirement in pyproject["project"]["dependencies"]
    if requirement.startswith("dr-docker")
]
if len(requirements) != 1:
    raise SystemExit(
        f"expected exactly one dr-docker requirement, found {len(requirements)}"
    )
print(requirements[0])
PY

RUN pip install --no-cache-dir \
        -r /tmp/bigcodebench-requirements.txt \
        -r /tmp/dr-docker-requirements.txt \
    && rm -rf \
        /root/.cache/pip \
        /tmp/bigcodebench-requirements.txt \
        /tmp/dr-docker-requirements.txt \
        /tmp/nl-code

# Preload NLTK resources needed by ClassEval tasks so evaluation does not
# attempt network downloads inside the sandbox at runtime.
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN python -m nltk.downloader -d "${NLTK_DATA}" \
    averaged_perceptron_tagger \
    averaged_perceptron_tagger_eng \
    punkt \
    punkt_tab \
    wordnet \
    omw-1.4

RUN useradd -m -s /bin/bash evaluser \
    && mkdir -p /sandbox \
    && chown evaluser:evaluser /sandbox

COPY --chown=evaluser:evaluser src/nl_code/code_execution/worker.py /sandbox/worker.py

USER evaluser
WORKDIR /tmp
