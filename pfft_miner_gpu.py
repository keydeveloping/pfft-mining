#!/usr/bin/env python3
"""
PFFT GPU Miner Bot — NVIDIA CUDA backend
Ethereum Mainnet | Contract: 0xEFAd2Eab7172dDEbE5Ce7a41f5Ddf8fCcE4Ca0CB

Flow:
- Python handles RPC, wallet, tx submit
- External CUDA binary scans nonce space for valid PoW

Usage:
  python3 pfft_miner_gpu.py
  ETH_RPC=https://... python3 pfft_miner_gpu.py
  PFFT_THREADS=1 CUDA_WORKERS=65536 CUDA_BLOCK_SIZE=256 python3 pfft_miner_gpu.py
"""

import os
import sys
import json
import time
import signal
import subprocess
from pathlib import Path

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

CONTRACT = "0xEFAd2Eab7172dDEbE5Ce7a41f5Ddf8fCcE4Ca0CB"
CHAIN_ID = 1
RPC = os.environ.get("ETH_RPC", "https://ethereum-rpc.publicnode.com")
WALLET_FILE = os.environ.get("PFFT_WALLET", os.path.join(os.path.dirname(os.path.abspath(__file__)), "wallet.json"))
GAS_LIMIT = int(os.environ.get("PFFT_GAS_LIMIT", "200000"))
PAUSE_BETWEEN_ROUNDS = float(os.environ.get("PFFT_PAUSE_BETWEEN_ROUNDS", "0"))
PRIORITY_FEE_GWEI = float(os.environ.get("PFFT_PRIORITY_FEE_GWEI", "3"))
MAX_FEE_MULTIPLIER = float(os.environ.get("PFFT_MAX_FEE_MULTIPLIER", "2"))
MAX_FEE_EXTRA_GWEI = float(os.environ.get("PFFT_MAX_FEE_EXTRA_GWEI", "3"))
PRE_SUBMIT_CHALLENGE_CHECK = os.environ.get("PFFT_PRE_SUBMIT_CHALLENGE_CHECK", "1") != "0"
PREFLIGHT_CALL = os.environ.get("PFFT_PREFLIGHT_CALL", "1") != "0"
CUDA_BINARY = os.environ.get("PFFT_CUDA_BINARY", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda_keccak_miner"))
CUDA_BLOCK_SIZE = int(os.environ.get("CUDA_BLOCK_SIZE", "256"))
CUDA_GRID_SIZE = int(os.environ.get("CUDA_GRID_SIZE", "65536"))
CUDA_BATCHES = int(os.environ.get("CUDA_BATCHES", "4096"))
CUDA_GPU_ID = int(os.environ.get("CUDA_GPU_ID", "0"))

running = True
w3 = None


def handle_signal(sig, frame):
    global running
    print("\n  ⚠️  Stopping miner...")
    running = False


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def load_contract(w3):
    abi = [
        {"inputs": [], "name": "currentPowHexZeros", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
        {"inputs": [], "name": "totalMinted", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
        {"inputs": [], "name": "MAX_SUPPLY", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
        {"inputs": [{"name": "requested", "type": "uint256"}], "name": "calculateActualMint", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
        {"inputs": [{"name": "user", "type": "address"}], "name": "currentPowChallenge", "outputs": [{"type": "bytes32"}], "stateMutability": "view", "type": "function"},
        {"inputs": [{"name": "user", "type": "address"}, {"name": "powNonce", "type": "uint256"}], "name": "isValidPow", "outputs": [{"type": "bool"}], "stateMutability": "view", "type": "function"},
        {"inputs": [{"name": "powNonce", "type": "uint256"}], "name": "freeMint", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
        {"inputs": [{"name": "user", "type": "address"}], "name": "mintedByAddress", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
        {"inputs": [], "name": "getInfo", "outputs": [{"type": "uint256"}, {"type": "uint256"}, {"type": "uint256"}, {"type": "uint256"}], "stateMutability": "view", "type": "function"},
        {"inputs": [{"name": "account", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    ]
    return w3.eth.contract(address=w3.to_checksum_address(CONTRACT), abi=abi)


def get_status(contract, wallet_addr):
    hex_zeros = contract.functions.currentPowHexZeros().call()
    total_minted = contract.functions.totalMinted().call()
    max_supply = contract.functions.MAX_SUPPLY().call()
    next_mint = contract.functions.calculateActualMint(w3.to_wei(1000, 'ether')).call()
    wallet_minted = contract.functions.mintedByAddress(wallet_addr).call()
    wallet_bal = contract.functions.balanceOf(wallet_addr).call()
    target = (2**256 - 1) >> (hex_zeros * 4)
    progress = total_minted * 10000 / max_supply / 100
    return {
        "hex_zeros": hex_zeros,
        "difficulty_bits": hex_zeros * 4,
        "total_minted": total_minted,
        "max_supply": max_supply,
        "next_mint": next_mint,
        "wallet_minted": wallet_minted,
        "wallet_bal": wallet_bal,
        "target": target,
        "progress": progress,
    }


def get_challenge(contract, wallet_addr):
    c = contract.functions.currentPowChallenge(wallet_addr).call()
    return c if isinstance(c, bytes) else c.to_bytes(32, 'big')


def build_fee_fields(w3) -> dict:
    """Build aggressive EIP-1559 fee fields; fallback to legacy gasPrice if needed."""
    priority = w3.to_wei(PRIORITY_FEE_GWEI, 'gwei')
    extra = w3.to_wei(MAX_FEE_EXTRA_GWEI, 'gwei')
    try:
        latest = w3.eth.get_block('latest')
        base = latest.get('baseFeePerGas')
        if base is not None:
            max_fee = int(base * MAX_FEE_MULTIPLIER) + priority + extra
            return {
                'maxPriorityFeePerGas': int(priority),
                'maxFeePerGas': int(max_fee),
            }
    except Exception:
        pass
    return {'gasPrice': int(w3.eth.gas_price + priority + extra)}


def submit_mint(w3, wallet, contract, nonce: int) -> bool:
    try:
        fn = contract.functions.freeMint(nonce)
        fee_fields = build_fee_fields(w3)
        if PREFLIGHT_CALL:
            try:
                fn.call({'from': wallet.address})
            except Exception as e:
                print(f"  ⚠️  Preflight revert, skip tx: {e}")
                return False

        tx = fn.build_transaction({
            'from': wallet.address,
            'nonce': w3.eth.get_transaction_count(wallet.address),
            'chainId': CHAIN_ID,
            'gas': GAS_LIMIT,
            **fee_fields,
        })

        if 'maxFeePerGas' in tx:
            print(f"  ⛽ fee max={tx['maxFeePerGas']/1e9:.2f} gwei | tip={tx['maxPriorityFeePerGas']/1e9:.2f} gwei")
        else:
            print(f"  ⛽ gasPrice={tx['gasPrice']/1e9:.2f} gwei")

        signed = wallet.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        print(f"  📤 TX: https://etherscan.io/tx/0x{tx_hash.hex()}")
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        if receipt.status == 1:
            print(f"  ✅ MINT OK | Block {receipt.blockNumber} | Gas {receipt.gasUsed}")
            return True
        print(f"  ❌ REVERTED | Gas {receipt.gasUsed}")
        return False
    except Exception as e:
        print(f"  ❌ TX error: {e}")
        return False


def solve_pow_cuda(challenge: bytes, target: int):
    if not os.path.exists(CUDA_BINARY):
        raise FileNotFoundError(f"CUDA binary not found: {CUDA_BINARY}. Run ./build.sh first")

    cmd = [
        CUDA_BINARY,
        "--challenge", challenge.hex(),
        "--target", f"{target:064x}",
        "--block-size", str(CUDA_BLOCK_SIZE),
        "--grid-size", str(CUDA_GRID_SIZE),
        "--batches", str(CUDA_BATCHES),
        "--gpu-id", str(CUDA_GPU_ID),
    ]

    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        raise RuntimeError(f"CUDA miner failed rc={proc.returncode} stderr={stderr} stdout={stdout}")

    lines = [x.strip() for x in proc.stdout.splitlines() if x.strip()]
    result = {}
    for line in lines:
        if "=" in line:
            k, v = line.split("=", 1)
            result[k.strip()] = v.strip()

    if result.get("status") != "FOUND":
        raise RuntimeError(f"Unexpected CUDA output: {proc.stdout}")

    nonce = int(result["nonce"])
    attempts = int(result.get("attempts", "0"))
    elapsed = float(result.get("elapsed", "0"))
    rate = float(result.get("rate_hs", "0"))
    print(f"\n  ✅ GPU FOUND nonce={nonce} | {attempts:,} attempts | {elapsed:.3f}s | {rate/1000:,.0f} kH/s")
    return nonce, bytes.fromhex(result["hash"])


def main():
    from web3 import Web3
    from eth_account import Account

    print("=" * 60)
    print("  ⛏️  PFFT GPU Miner Bot — NVIDIA CUDA")
    print(f"  Contract: {CONTRACT}")
    print(f"  RPC: {RPC}")
    print(f"  CUDA binary: {CUDA_BINARY}")
    print("=" * 60)

    global w3
    w3 = Web3(Web3.HTTPProvider(RPC, request_kwargs={"timeout": 30}))
    if not w3.is_connected():
        print("❌ Cannot connect to RPC")
        sys.exit(1)
    print(f"✅ Connected | Block #{w3.eth.block_number}")

    wallet_path = Path(WALLET_FILE)
    if wallet_path.exists():
        with open(wallet_path) as f:
            wdata = json.load(f)
        pk = wdata.get('private_key_hex') or wdata.get('private_key')
        if not pk.startswith('0x'):
            pk = '0x' + pk
        wallet = Account.from_key(pk)
        print(f"✅ Wallet: {wallet.address}")
    else:
        wallet = Account.create()
        wdata = {
            "address": wallet.address,
            "private_key_hex": wallet.key.hex(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": "PFFT miner wallet — KEEP SECRET"
        }
        wallet_path.parent.mkdir(parents=True, exist_ok=True)
        with open(wallet_path, 'w') as f:
            json.dump(wdata, f, indent=2)
        os.chmod(wallet_path, 0o600)
        print(f"✅ New wallet: {wallet.address}")
        print(f"   Saved: {wallet_path}")

    eth_bal = w3.eth.get_balance(wallet.address) / 1e18
    print(f"💰 ETH: {eth_bal:.6f}")
    if eth_bal < 0.00005:
        print("⚠️  Low ETH! Need ~0.00005+ ETH for gas")

    contract = load_contract(w3)
    s = get_status(contract, wallet.address)
    print(f"\n📊 Contract:")
    print(f"   Minted: {s['total_minted']/1e18:,.0f} / {s['max_supply']/1e18:,.0f} PFFT ({s['progress']:.1f}%)")
    print(f"   Next mint: ~{s['next_mint']/1e18:,.2f} PFFT")
    print(f"   Difficulty: {s['hex_zeros']} hex zeros ({s['difficulty_bits']}-bit)")
    print(f"   Wallet minted: {s['wallet_minted']/1e18:,.2f} / 10,000 PFFT")
    print(f"   Wallet balance: {s['wallet_bal']/1e18:,.2f} PFFT")

    round_num = 0
    total_minted_count = 0
    total_pfft_earned = 0
    global_start = time.time()

    while running:
        round_num += 1
        print(f"\n{'─'*60}")
        print(f"  Round #{round_num}")
        print(f"{'─'*60}")

        try:
            s = get_status(contract, wallet.address)
            print(f"  Supply: {s['total_minted']/1e18:,.0f} ({s['progress']:.1f}%) | Next: ~{s['next_mint']/1e18:,.2f} PFFT | Diff: {s['difficulty_bits']}-bit")
            if s['total_minted'] >= s['max_supply']:
                print("  🏁 Max supply reached!")
                break
            if s['wallet_minted'] >= 10_000 * 1e18:
                print("  🏁 Wallet cap (10,000 PFFT) reached!")
                break
        except Exception as e:
            print(f"  ⚠️  Status error: {e}, retrying in 15s...")
            time.sleep(15)
            continue

        challenge = get_challenge(contract, wallet.address)
        print(f"  🎮 GPU mining ({s['difficulty_bits']}-bit)...")

        try:
            nonce, _ = solve_pow_cuda(challenge, s['target'])
        except Exception as e:
            print(f"  ❌ GPU solve error: {e}")
            time.sleep(5)
            continue

        if PRE_SUBMIT_CHALLENGE_CHECK:
            try:
                latest_challenge = get_challenge(contract, wallet.address)
                if latest_challenge != challenge:
                    print("  ⚠️  Challenge changed after solve, skip stale nonce and re-mine...")
                    continue
            except Exception as e:
                print(f"  ⚠️  Challenge refresh error: {e}, continuing...")

        try:
            is_valid = contract.functions.isValidPow(wallet.address, nonce).call()
            if not is_valid:
                print("  ⚠️  Nonce invalid on-chain (challenge changed?), re-mining...")
                continue
        except Exception as e:
            print(f"  ⚠️  Verify error: {e}, submitting anyway...")

        success = submit_mint(w3, wallet, contract, nonce)
        if success:
            total_minted_count += 1
            earned = s['next_mint'] / 1e18
            total_pfft_earned += earned
            print(f"  💰 +{earned:,.2f} PFFT | Total: {total_pfft_earned:,.2f} PFFT from {total_minted_count} mints")
            try:
                bal = contract.functions.balanceOf(wallet.address).call()
                print(f"  💰 PFFT balance: {bal/1e18:,.2f}")
            except Exception:
                pass

        elapsed = time.time() - global_start
        print(f"\n  📈 Session: {total_minted_count} mints | {total_pfft_earned:,.2f} PFFT | {elapsed/60:.1f} min")

        if running:
            print(f"  ⏳ {PAUSE_BETWEEN_ROUNDS}s cooldown...")
            time.sleep(PAUSE_BETWEEN_ROUNDS)

    print(f"\n{'='*60}")
    print("  Session Summary")
    print(f"  Mints: {total_minted_count}")
    print(f"  PFFT earned: {total_pfft_earned:,.2f}")
    print(f"  Runtime: {(time.time()-global_start)/60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
