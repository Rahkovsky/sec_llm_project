# SEC Disclosure Intelligence Prototype — minimal, offline-capable image.
# The core platform needs only the standard library, so the base image stays
# small. Install extras at build time with --build-arg EXTRAS="embeddings,vectorstore".
FROM python:3.13-slim

ARG EXTRAS=""
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config

RUN pip install --upgrade pip \
    && if [ -n "$EXTRAS" ]; then pip install ".[$EXTRAS]"; else pip install .; fi

# Default: run the fully offline end-to-end demo.
ENTRYPOINT ["sec-intel"]
CMD ["demo"]
