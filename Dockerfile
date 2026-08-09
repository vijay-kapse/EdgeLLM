# syntax=docker/dockerfile:1.7
#
# EdgeLLM container image.
#
# Builds both native binaries against a pinned ONNX Runtime release, then ships
# them alongside the Python pipeline in a slim runtime image:
#
#   edgellm_infer  - C++17 harness, KV-cache greedy decode via the ORT C++ API
#   int8_gemm      - scalar vs. hand-vectorized INT8 GEMM microbenchmark
#
# The image is architecture-aware. On arm64 the kernel compiles the ARM NEON
# SDOT path; on amd64 it compiles the AVX2 path. That is deliberate: running the
# same image on an Apple Silicon host and on an x86 cloud instance produces two
# genuinely different SIMD backends to compare, rather than one emulated number.
#
# Build:
#   docker build -t edgellm .
#   docker build -t edgellm --build-arg ORT_VERSION=1.20.1 .
#
# See docker-compose.yml for the pipeline / inference / benchmark entry points.

ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------------- builder ----
FROM ubuntu:${UBUNTU_VERSION} AS builder

# Pinned rather than "latest": benchmark numbers are only comparable across runs
# if the inference runtime is held fixed.
ARG ORT_VERSION=1.22.0
ARG TARGETARCH

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ONNX Runtime publishes prebuilt Linux archives as x64 / aarch64; Docker's
# TARGETARCH is amd64 / arm64. Translate, with a uname fallback for the legacy
# (non-BuildKit) builder where TARGETARCH is not populated.
RUN set -eux; \
    arch="${TARGETARCH:-}"; \
    if [ -z "${arch}" ]; then \
        case "$(uname -m)" in \
            x86_64)         arch=amd64 ;; \
            aarch64|arm64)  arch=arm64 ;; \
        esac; \
    fi; \
    case "${arch}" in \
        amd64) ort_arch=x64 ;; \
        arm64) ort_arch=aarch64 ;; \
        *) echo "unsupported architecture: '${arch}'" >&2; exit 1 ;; \
    esac; \
    url="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ort_arch}-${ORT_VERSION}.tgz"; \
    echo "fetching ${url}"; \
    curl -fsSL -o /tmp/ort.tgz "${url}"; \
    mkdir -p /opt/onnxruntime; \
    tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1; \
    rm /tmp/ort.tgz; \
    test -f /opt/onnxruntime/include/onnxruntime_cxx_api.h; \
    test -e /opt/onnxruntime/lib/libonnxruntime.so

# cpp/CMakeLists.txt resolves ONNX Runtime from ORT_HOME (its documented escape
# hatch for non-Homebrew hosts), so no CMake changes are needed to build here.
ENV ORT_HOME=/opt/onnxruntime

WORKDIR /src
COPY cpp/ cpp/
COPY kernels/ kernels/

RUN cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build/cpp -j "$(nproc)"

RUN cmake -S kernels -B build/kernels -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build/kernels -j "$(nproc)"

# ---------------------------------------------------------------- runtime ----
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# libgomp is ONNX Runtime's OpenMP dependency; without it the .so fails to load.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/onnxruntime/lib/ /opt/onnxruntime/lib/
COPY --from=builder /src/build/cpp/edgellm_infer   /usr/local/bin/edgellm_infer
COPY --from=builder /src/build/kernels/int8_gemm   /usr/local/bin/int8_gemm

ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

# Dependency metadata first so the (slow) install layer caches independently of
# source edits. hatchling reads README.md at build time, so it has to come too.
COPY pyproject.toml README.md ./
COPY edgellm/ edgellm/

# CPU-only torch on x86 avoids pulling ~2.5 GB of CUDA wheels that this image
# has no GPU to use. The CPU index has no aarch64 wheels, so arm64 takes the
# ordinary PyPI build, which is already CPU-only.
RUN set -eux; \
    case "$(uname -m)" in \
        x86_64) pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu ;; \
        *)      pip install --no-cache-dir torch ;; \
    esac; \
    pip install --no-cache-dir ".[onnx,eval,report]"

COPY configs/ configs/
COPY scripts/ scripts/

# Written to by the pipeline; compose mounts host directories over both.
RUN mkdir -p /app/artifacts /app/results /cache/huggingface

CMD ["edgellm", "--help"]
