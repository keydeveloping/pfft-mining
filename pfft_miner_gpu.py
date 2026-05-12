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
import secrets
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
RECEIPT_TIMEOUT = int(os.environ.get("PFFT_RECEIPT_TIMEOUT", "5"))
PRE_SUBMIT_CHALLENGE_CHECK = os.environ.get("PFFT_PRE_SUBMIT_CHALLENGE_CHECK", "1") != "0"
PREFLIGHT_CALL = os.environ.get("PFFT_PREFLIGHT_CALL", "1") != "0"
CUDA_BINARY = os.environ.get("PFFT_CUDA_BINARY", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda_keccak_miner"))
CUDA_BLOCK_SIZE = int(os.environ.get("CUDA_BLOCK_SIZE", "256"))
CUDA_GRID_SIZE = int(os.environ.get("CUDA_GRID_SIZE", "65536"))
CUDA_BATCHES = int(os.environ.get("CUDA_BATCHES", "4096"))
CUDA_GPU_ID = int(os.environ.get("CUDA_GPU_ID", "0"))

running = True
w3 = None


def log(msg: str = ""):
    print(msg, flush=True)


def short_hash(h) -> str:
    s = h.hex() if hasattr(h, 'hex') else str(h)
    if s.startswith('0x'):
        s = s[2:]
    return f"0x{s[:8]}...{s[-6:]}"


def fmt_rate(rate_hs: float) -> str:
    if rate_hs >= 1e9:
        return f"{rate_hs/1e9:.2f} GH/s"
    if rate_hs >= 1e6:
        return f"{rate_hs/1e6:.2f} MH/s"
    if rate_hs >= 1e3:
        return f"{rate_hs/1e3:.2f} kH/s"
    return f"{rate_hs:.0f} H/s"


def clean_error(e) -> str:
    text = str(e).replace("\n", " ")
    for marker in ("execution reverted:", "revert"):
        if marker in text:
            text = text.split(marker, 1)[1]
            break
    text = text.split("'0x", 1)[0].strip(" ()',")
    if len(text) > 100:
        text = text[:97] + "..."
    return text or e.__class__.__name__


def handle_signal(sig, frame):
    global running
    log("\n[stop] stopping miner...")
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
    priority = int(w3.to_wei(PRIORITY_FEE_GWEI, 'gwei'))
    extra = int(w3.to_wei(MAX_FEE_EXTRA_GWEI, 'gwei'))
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
        if PREFLIGHT_CALL:
            try:
                fn.call({'from': wallet.address})
            except Exception as e:
                log(f"[tx] preflight-skip reason={clean_error(e)}")
                return False

        fee_fields = build_fee_fields(w3)
        account_nonce = w3.eth.get_transaction_count(wallet.address, 'latest')
        tx = fn.build_transaction({
            'from': wallet.address,
            # Use latest nonce so each new round replaces any still-pending tx instead of queueing behind it.
            'nonce': account_nonce,
            'chainId': CHAIN_ID,
            'gas': GAS_LIMIT,
            **fee_fields,
        })

        if 'maxFeePerGas' in tx:
            log(f"[tx] nonce={account_nonce} fee max={tx['maxFeePerGas']/1e9:.2f} tip={tx['maxPriorityFeePerGas']/1e9:.2f} gwei")
        else:
            log(f"[tx] nonce={account_nonce} gasPrice={tx['gasPrice']/1e9:.2f} gwei")

        signed = wallet.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        log(f"[tx] sent {short_hash(tx_hash)}")

        try:
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=RECEIPT_TIMEOUT)
            if receipt.status == 1:
                log(f"[tx] success block={receipt.blockNumber} gas={receipt.gasUsed} hash={short_hash(tx_hash)}")
                return True
            log(f"[tx] reverted gas={receipt.gasUsed} hash={short_hash(tx_hash)}")
            return False
        except Exception as e:
            log(f"[tx] pending>{RECEIPT_TIMEOUT}s skip hash={short_hash(tx_hash)}")
            return False
    except Exception as e:
        log(f"[tx] error reason={clean_error(e)}")
        return False


def solve_pow_cuda(challenge: bytes, target: int):
    if not os.path.exists(CUDA_BINARY):
        raise FileNotFoundError(f"CUDA binary not found: {CUDA_BINARY}. Run ./build.sh first")

    start_nonce = secrets.randbits(63)
    cmd = [
        CUDA_BINARY,
        "--challenge", challenge.hex(),
        "--target", f"{target:064x}",
        "--start-nonce", str(start_nonce),
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
    log(f"[gpu] found nonce={nonce} tries={attempts:,} time={elapsed:.3f}s rate={fmt_rate(rate)}")
    return nonce, bytes.fromhex(result["hash"])


def main():
    from web3 import Web3
    from eth_account import Account

    log("PFFT GPU Miner — NVIDIA CUDA")
    log(f"contract={CONTRACT} cuda={CUDA_BINARY}")

    global w3
    w3 = Web3(Web3.HTTPProvider(RPC, request_kwargs={"timeout": 30}))
    if not w3.is_connected():
        print("❌ Cannot connect to RPC")
        sys.exit(1)
    log(f"connected block={w3.eth.block_number}")

    wallet_path = Path(WALLET_FILE)
    if wallet_path.exists():
        with open(wallet_path) as f:
            wdata = json.load(f)
        pk = wdata.get('private_key_hex') or wdata.get('private_key')
        if not pk.startswith('0x'):
            pk = '0x' + pk
        wallet = Account.from_key(pk)
        log(f"wallet={wallet.address}")
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
        log(f"new-wallet={wallet.address} saved={wallet_path}")

    eth_bal = w3.eth.get_balance(wallet.address) / 1e18
    log(f"eth={eth_bal:.6f}")
    if eth_bal < 0.00005:
        log("warn=low-eth need~0.00005+")

    contract = load_contract(w3)
    s = get_status(contract, wallet.address)
    log(f"status supply={s['total_minted']/1e18:,.0f}/{s['max_supply']/1e18:,.0f} progress={s['progress']:.1f}% diff={s['difficulty_bits']} wallet_minted={s['wallet_minted']/1e18:,.2f} bal={s['wallet_bal']/1e18:,.2f}")

    round_num = 0
    total_minted_count = 0
    total_pfft_earned = 0
    global_start = time.time()

    while running:
        round_num += 1
        try:
            s = get_status(contract, wallet.address)
            log(f"[r{round_num}] supply={s['total_minted']/1e18:,.0f} progress={s['progress']:.1f}% reward={s['next_mint']/1e18:,.2f} diff={s['difficulty_bits']}")
            if s['total_minted'] >= s['max_supply']:
                log(f"[r{round_num}] stop=max-supply")
                break
            if s['wallet_minted'] >= 10_000 * 1e18:
                log(f"[r{round_num}] stop=wallet-cap")
                break
        except Exception as e:
            log(f"[r{round_num}] status-error reason={clean_error(e)} retry=15s")
            time.sleep(15)
            continue

        challenge = get_challenge(contract, wallet.address)
        log(f"[r{round_num}] gpu-start")

        try:
            nonce, _ = solve_pow_cuda(challenge, s['target'])
        except Exception as e:
            log(f"[r{round_num}] gpu-error reason={clean_error(e)}")
            time.sleep(5)
            continue

        if PRE_SUBMIT_CHALLENGE_CHECK:
            try:
                latest_challenge = get_challenge(contract, wallet.address)
                if latest_challenge != challenge:
                    log(f"[r{round_num}] stale-challenge skip")
                    continue
            except Exception as e:
                log(f"[r{round_num}] challenge-refresh-error reason={clean_error(e)} continue")

        try:
            is_valid = contract.functions.isValidPow(wallet.address, nonce).call()
            if not is_valid:
                log(f"[r{round_num}] invalid-pow skip")
                continue
        except Exception as e:
            log(f"[r{round_num}] verify-error reason={clean_error(e)} submit-anyway")

        success = submit_mint(w3, wallet, contract, nonce)
        if success:
            total_minted_count += 1
            earned = s['next_mint'] / 1e18
            total_pfft_earned += earned
            log(f"[r{round_num}] reward +{earned:,.2f} total={total_pfft_earned:,.2f} mints={total_minted_count}")
            try:
                bal = contract.functions.balanceOf(wallet.address).call()
                log(f"[r{round_num}] balance={bal/1e18:,.2f}")
            except Exception:
                pass

        elapsed = time.time() - global_start
        log(f"[r{round_num}] session mints={total_minted_count} earned={total_pfft_earned:,.2f} runtime={elapsed/60:.1f}m")

        if running:
            if PAUSE_BETWEEN_ROUNDS > 0:
                log(f"[r{round_num}] cooldown={PAUSE_BETWEEN_ROUNDS}s")
            time.sleep(PAUSE_BETWEEN_ROUNDS)

    log(f"summary mints={total_minted_count} earned={total_pfft_earned:,.2f} runtime={(time.time()-global_start)/60:.1f}m")


if __name__ == "__main__":
    main()
