#!/usr/bin/env bash
# EdgeLLM benchmark host bootstrap.
#
# Runs once at first boot on the Deep Learning Base AMI, which already provides
# the NVIDIA driver, Docker, and the NVIDIA container toolkit. Everything here
# is logged to /var/log/edgellm-bootstrap.log and to the console.
#
# Deliberately does NOT run the full benchmark suite unattended: model downloads
# and quantization are long and failure-prone, and a number produced by an
# unattended script nobody watched is exactly the kind of result this repo
# refuses to publish. This gets the box to the point where you can SSH in and
# run the measurements yourself.

set -euo pipefail

exec > >(tee -a /var/log/edgellm-bootstrap.log) 2>&1
echo "=== EdgeLLM bootstrap starting: $(date -Is) ==="

TARGET_USER="ubuntu"
TARGET_HOME="/home/$${TARGET_USER}"
CHECKOUT="$${TARGET_HOME}/EdgeLLM"
RESULTS="$${TARGET_HOME}/edgellm-host-info"

mkdir -p "$${RESULTS}"

# --- record what this machine actually is -----------------------------------
# Every published number needs its hardware recorded next to it, or it is not
# reproducible. Capture that before anything else runs.
{
  echo "date:      $(date -Is)"
  echo "kernel:    $(uname -a)"
  echo "cpu:       $(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"
  echo "cores:     $(nproc)"
  echo "memory:    $(free -h | awk '/^Mem:/ {print $2}')"
} > "$${RESULTS}/host.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$${RESULTS}/nvidia-smi.txt" 2>&1 || echo "nvidia-smi failed" >> "$${RESULTS}/nvidia-smi.txt"
  echo "--- GPU ---"
  cat "$${RESULTS}/nvidia-smi.txt"
else
  echo "WARNING: nvidia-smi not found. This AMI may not be a GPU image." \
    | tee "$${RESULTS}/nvidia-smi.txt"
fi

# --- docker sanity ----------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker missing from the AMI; installing from the distro repo."
  apt-get update -y
  apt-get install -y --no-install-recommends docker.io git
  systemctl enable --now docker
fi

usermod -aG docker "$${TARGET_USER}" || true

# --- checkout ---------------------------------------------------------------
if [ ! -d "$${CHECKOUT}/.git" ]; then
  git clone --depth 1 "${repo_url}" "$${CHECKOUT}"
fi
chown -R "$${TARGET_USER}:$${TARGET_USER}" "$${CHECKOUT}" "$${RESULTS}"

# --- build the image --------------------------------------------------------
# Builds the AVX2 variant of the SIMD kernel here, since this is x86 -- a
# different code path from the NEON build measured on Apple Silicon.
cd "$${CHECKOUT}"
if docker build -t edgellm:local . ; then
  echo "image built"
  docker run --rm edgellm:local int8_gemm 256 256 512 50 \
    | tee "$${RESULTS}/int8_gemm_avx2.txt" || true
else
  echo "ERROR: image build failed -- see log above." | tee -a "$${RESULTS}/build-failed.txt"
fi

cat <<'NOTE' > "$${RESULTS}/NEXT_STEPS.txt"
Box is up. The CPU-side SIMD benchmark (AVX2 path) has already run; see
int8_gemm_avx2.txt.

What still needs doing by hand, because it is slow and worth watching:

  1. GPU-only INT4. The repo's INT4 path falls back to weight-only block-wise
     quantization on CPU. GPTQ/AWQ need CUDA and can run here:
         cd ~/EdgeLLM
         python3 -m venv .venv && . .venv/bin/activate
         pip install -e ".[onnx,eval,report]"
         edgellm quantize && edgellm benchmark && edgellm report

  2. ONNX Runtime CUDA execution provider. The container image ships CPU-only
     onnxruntime by design. For CUDA EP numbers install onnxruntime-gpu in the
     host venv above rather than in the image, and record the ORT version and
     driver version alongside any figure you publish.

  3. Copy results back before destroying the box:
         scp -r ubuntu@<ip>:~/EdgeLLM/results ./results-gpu
         scp -r ubuntu@<ip>:~/edgellm-host-info ./host-info-gpu

  4. terraform destroy. This instance bills by the hour.
NOTE

chown -R "$${TARGET_USER}:$${TARGET_USER}" "$${RESULTS}"
echo "=== EdgeLLM bootstrap finished: $(date -Is) ==="
