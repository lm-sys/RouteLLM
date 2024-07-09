# syntax = docker/dockerfile:1.4
FROM python:3.10

# Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt \
  apt-get update && \
  apt-get install -y --no-install-recommends net-tools nano

ENV PYTHONDONTWRITEBYTECODE=1

# Create and activate the routllm environment
RUN python -m venv /opt/routllm && \
  /opt/routllm/bin/pip install --upgrade pip

# Set the PATH to prioritize the virtualenv
ENV PATH="/opt/routllm/bin:$PATH"

COPY docker/rootfs /

WORKDIR /app

COPY . .

# Install additional requirements based on the target platform with apk package caching
ARG TARGET=cuda
RUN --mount=type=cache,target=/root/.cache/pip \
  if [ "$TARGET" = "cuda" ]; then \
  pip install torch --index-url https://download.pytorch.org/whl/cu121; \
  elif [ "$TARGET" = "rocm" ]; then \
  pip install torch --index-url https://download.pytorch.org/whl/rocm5.2; \
  else \
  pip install torch; \
  fi; \
  pip install -e .[serve,eval]

EXPOSE 6060
VOLUME [ "/root/.cache/huggingface" ]
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
