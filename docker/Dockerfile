FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update -y && \
	apt-get install -y --no-install-recommends git && \
	rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir ".[serve]"

COPY routellm/ routellm/

RUN find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; exit 0

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update -y && \
	apt-get install -y --no-install-recommends --no-install-suggests ca-certificates && \
	rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/routellm /app/routellm
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

ARG ROUTERS=random
ARG PORT=6060
ARG STRONG_MODEL=gpt-4-1106-preview
ARG WEAK_MODEL=anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1

ENV ROUTERS=${ROUTERS}
ENV PORT=${PORT}
ENV STRONG_MODEL=${STRONG_MODEL}
ENV WEAK_MODEL=${WEAK_MODEL}

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
	CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

ENTRYPOINT ["sh", "-c", "exec python -m routellm.openai_server --port ${PORT} --routers ${ROUTERS} --strong-model ${STRONG_MODEL} --weak-model ${WEAK_MODEL} ${BASE_URL:+--base-url ${BASE_URL}} ${OPENAI_API_KEY:+--api-key ${OPENAI_API_KEY}} ${CONFIG_FILE:+--config ${CONFIG_FILE}}"]
