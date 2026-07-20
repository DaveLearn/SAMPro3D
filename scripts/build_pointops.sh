#!/usr/bin/env bash
set -euo pipefail

# Rebuilds the pointops CUDA extension into the pixi environment.
#
# `pixi install` already builds pointops, for the arch list pinned in
# [tool.pixi.activation.env] -- Turing through Blackwell, so the result is
# portable. Use this to rebuild for something else: a single arch to save build
# time, or a card that list does not cover (a load failing with "no kernel image
# is available for execution on the device").
#
#   pixi run build_pointops                            # auto-detect this machine's GPU
#   POINTOPS_CUDA_ARCH_LIST="8.0;9.0+PTX" pixi run build_pointops
#
# Auto-detect is this script's default because it produces native cubins for
# whatever card is present, including ones newer than any hardcoded list. Note
# it requires clearing TORCH_CUDA_ARCH_LIST: both the pixi activation above and
# conda's cuda-nvcc activation export one (conda's includes 10.1, which torch
# rejects outright with "Unknown CUDA arch").

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
POINTOPS_DIR="${PROJECT_ROOT}/libs/pointops"

if [[ -n "${POINTOPS_CUDA_ARCH_LIST:-}" ]]; then
    export TORCH_CUDA_ARCH_LIST="${POINTOPS_CUDA_ARCH_LIST}"
    printf '[build_pointops] building for arch list: %s\n' "${TORCH_CUDA_ARCH_LIST}"
else
    unset TORCH_CUDA_ARCH_LIST
    printf '[build_pointops] building for auto-detected GPU arch\n'
fi

# setuptools reuses object files in build/, and would silently relink the
# previously compiled archs instead of honouring a changed arch list.
rm -rf "${POINTOPS_DIR}/build" "${POINTOPS_DIR}/pointops.egg-info"

python -m pip install --no-build-isolation --no-deps --force-reinstall --no-cache-dir "${POINTOPS_DIR}"

python - <<'PY'
import torch  # noqa: F401  (loads libc10 before the extension)
import pointops._C as ext
print(f"[build_pointops] installed {ext.__file__}")
PY
