# pfft-mining

NVIDIA CUDA miner for PFFT PoW minting.

## What this repo has

- `pfft_miner_gpu.py` — Python miner loop, wallet, RPC, tx submit
- `cuda_keccak_miner.cu` — CUDA brute-force kernel for `keccak256(challenge || nonce256be)`
- `build.sh` — compile CUDA binary
- `.env.example` — runtime config

## Target

- Vast.ai
- NVIDIA GPU
- User said RTX 5090 x1

## Setup

```bash
git clone https://github.com/keydeveloping/pfft-mining.git
cd pfft-mining
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Build

Try latest arch first:

```bash
CUDA_ARCH=sm_120 ./build.sh
```

If toolkit rejects `sm_120`, check supported arch from `nvcc --help`, then use closest supported arch.

Examples:

```bash
CUDA_ARCH=sm_90 ./build.sh
CUDA_ARCH=compute_90 ./build.sh
```

## Run

```bash
python3 pfft_miner_gpu.py
```

## Check GPU

```bash
nvidia-smi
nvcc --version
```

## Notes

- This code path is built for NVIDIA CUDA, not AMD.
- Python keeps Web3 + tx flow.
- CUDA binary does only PoW search.
- Current CUDA code uses one kernel launch over finite search window.
- If output says `status=NOT_FOUND`, tune bigger search space:
  - increase `CUDA_GRID_SIZE`
  - increase `CUDA_BATCHES`
- Default search space per run:

```txt
CUDA_GRID_SIZE * CUDA_BLOCK_SIZE * CUDA_BATCHES
```

With defaults:

```txt
65536 * 256 * 4096 = 68,719,476,736 nonces
```

## Tune suggestions for 5090

Start:

```env
CUDA_BLOCK_SIZE=256
CUDA_GRID_SIZE=65536
CUDA_BATCHES=4096
```

If VRAM/launch overhead OK, test bigger batches:

```env
CUDA_BATCHES=8192
```

## Important

This repo was written here without live CUDA compile/test because current machine has no NVIDIA GPU.

So first run on Vast.ai should be treated as bring-up phase:

1. build
2. run one round
3. if compile error, adjust `CUDA_ARCH`
4. if kernel logic bug, patch there

## Wallet

First run auto-creates `wallet.json` if missing.
Keep private key secret.
