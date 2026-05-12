#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

: "${CUDA_ARCH:=sm_120}"

echo "[+] Building cuda_keccak_miner for ${CUDA_ARCH}"
nvcc -O3 -std=c++17 -arch=${CUDA_ARCH} cuda_keccak_miner.cu -o cuda_keccak_miner

echo "[+] Done: ./cuda_keccak_miner"
