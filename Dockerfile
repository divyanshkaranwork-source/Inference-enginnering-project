# syntax=docker/dockerfile:1
# Hardened multi-stage image: build tools stay in the first stage, the runtime
# image runs as an unprivileged user and contains no compiler.
# For reproducible builds pin the base image by digest:
#   docker pull python:3.12-slim-bookworm && docker inspect --format '{{index .RepoDigests 0}}' python:3.12-slim-bookworm

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
COPY csrc ./csrc
COPY inferlab ./inferlab
# portable build (no -march=native) so the image runs on any x86-64 / arm64 host
RUN cmake -S csrc -B csrc/build -DCMAKE_BUILD_TYPE=Release -DINFERLAB_NATIVE=OFF \
 && cmake --build csrc/build -j \
 && ./csrc/build/test_kernels

FROM ${PYTHON_IMAGE}
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 10001 inferlab \
 && useradd --system --uid 10001 --gid inferlab --no-create-home --shell /usr/sbin/nologin inferlab
COPY --from=build /install /usr/local
WORKDIR /app
COPY --from=build --chown=root:root /src/inferlab ./inferlab
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    INFERLAB_HOST=0.0.0.0 \
    INFERLAB_PORT=8000 \
    INFERLAB_MODELS_DIR=/models \
    INFERLAB_KERNELS=cpp \
    INFERLAB_GENERATED_KEY_FILE=/tmp/inferlab_api_key
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status == 200 else 1)"]
ENTRYPOINT ["python", "-m", "inferlab", "serve"]
