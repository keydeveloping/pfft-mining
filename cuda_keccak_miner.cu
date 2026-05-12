// PFFT CUDA Keccak256 nonce miner
// Builds: nvcc -O3 -arch=sm_120 cuda_keccak_miner.cu -o cuda_keccak_miner
// For RTX 5090, sm_120 likely. If nvcc too old, use sm_90 or native supported arch.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define KECCAK_ROUNDS 24

__device__ __constant__ uint64_t d_rndc[24] = {
    0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808aULL,0x8000000080008000ULL,
    0x000000000000808bULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,
    0x000000000000008aULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000aULL,
    0x000000008000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,
    0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800aULL,0x800000008000000aULL,
    0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL
};

__device__ __constant__ int d_rotc[24] = {1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44};
__device__ __constant__ int d_piln[24] = {10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1};

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int y) {
    return (x << y) | (x >> (64 - y));
}

__device__ void keccakf(uint64_t st[25]) {
    uint64_t bc[5];
    for (int round = 0; round < KECCAK_ROUNDS; round++) {
        for (int i = 0; i < 5; i++) bc[i] = st[i] ^ st[i+5] ^ st[i+10] ^ st[i+15] ^ st[i+20];
        for (int i = 0; i < 5; i++) {
            uint64_t t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
        }
        uint64_t t = st[1];
        for (int i = 0; i < 24; i++) {
            int j = d_piln[i];
            bc[0] = st[j];
            st[j] = rotl64(t, d_rotc[i]);
            t = bc[0];
        }
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; i++) bc[i] = st[j + i];
            for (int i = 0; i < 5; i++) st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }
        st[0] ^= d_rndc[round];
    }
}

__device__ void keccak256_64(const uint8_t msg[64], uint8_t out[32]) {
    uint64_t st[25];
    #pragma unroll
    for (int i = 0; i < 25; i++) st[i] = 0;

    // absorb 64 data bytes into little-endian lanes
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t lane = 0;
        #pragma unroll
        for (int b = 0; b < 8; b++) lane |= ((uint64_t)msg[i * 8 + b]) << (8 * b);
        st[i] ^= lane;
    }

    // Keccak padding for Ethereum: suffix 0x01, final rate byte xor 0x80. Rate=136.
    st[8] ^= 0x0000000000000001ULL;      // byte offset 64
    st[16] ^= 0x8000000000000000ULL;     // byte offset 135
    keccakf(st);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t lane = st[i];
        #pragma unroll
        for (int b = 0; b < 8; b++) out[i * 8 + b] = (uint8_t)((lane >> (8 * b)) & 0xff);
    }
}

__device__ __forceinline__ bool hash_le_target(const uint8_t h[32], const uint8_t target[32]) {
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        if (h[i] < target[i]) return true;
        if (h[i] > target[i]) return false;
    }
    return true;
}

__global__ void mine_kernel(const uint8_t *challenge, const uint8_t *target, uint64_t start_nonce, uint64_t stride, uint64_t iters_per_thread, unsigned int *found, uint64_t *found_nonce, uint8_t *found_hash) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = start_nonce + tid;
    uint8_t msg[64];
    uint8_t h[32];

    #pragma unroll
    for (int i = 0; i < 32; i++) msg[i] = challenge[i];
    #pragma unroll
    for (int i = 32; i < 56; i++) msg[i] = 0;

    for (uint64_t iter = 0; iter < iters_per_thread; iter++, nonce += stride) {
        if (atomicAdd(found, 0) != 0) return;

        // nonce as uint256 big-endian in last 32 bytes; support uint64 nonce in low 8 bytes
        msg[56] = (uint8_t)(nonce >> 56);
        msg[57] = (uint8_t)(nonce >> 48);
        msg[58] = (uint8_t)(nonce >> 40);
        msg[59] = (uint8_t)(nonce >> 32);
        msg[60] = (uint8_t)(nonce >> 24);
        msg[61] = (uint8_t)(nonce >> 16);
        msg[62] = (uint8_t)(nonce >> 8);
        msg[63] = (uint8_t)(nonce);

        keccak256_64(msg, h);
        if (hash_le_target(h, target)) {
            if (atomicCAS(found, 0, 1) == 0) {
                *found_nonce = nonce;
                #pragma unroll
                for (int i = 0; i < 32; i++) found_hash[i] = h[i];
            }
            return;
        }
    }
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int parse_hex32(const char *s, uint8_t out[32]) {
    if (strlen(s) != 64) return -1;
    for (int i = 0; i < 32; i++) {
        int hi = hexval(s[i*2]);
        int lo = hexval(s[i*2+1]);
        if (hi < 0 || lo < 0) return -1;
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return 0;
}

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    uint8_t challenge[32], target[32];
    memset(challenge, 0, 32);
    memset(target, 0xff, 32);
    int block_size = 256;
    int grid_size = 65536;
    uint64_t batches = 4096;
    int gpu_id = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--challenge") && i + 1 < argc) {
            if (parse_hex32(argv[++i], challenge) != 0) { fprintf(stderr, "bad challenge hex\n"); return 2; }
        } else if (!strcmp(argv[i], "--target") && i + 1 < argc) {
            if (parse_hex32(argv[++i], target) != 0) { fprintf(stderr, "bad target hex\n"); return 2; }
        } else if (!strcmp(argv[i], "--block-size") && i + 1 < argc) {
            block_size = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--grid-size") && i + 1 < argc) {
            grid_size = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--batches") && i + 1 < argc) {
            batches = strtoull(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--gpu-id") && i + 1 < argc) {
            gpu_id = atoi(argv[++i]);
        } else {
            fprintf(stderr, "usage: %s --challenge <64hex> --target <64hex> [--block-size 256] [--grid-size 65536] [--batches 4096] [--gpu-id 0]\n", argv[0]);
            return 2;
        }
    }

    cudaError_t err = cudaSetDevice(gpu_id);
    if (err != cudaSuccess) { fprintf(stderr, "cudaSetDevice: %s\n", cudaGetErrorString(err)); return 3; }

    uint8_t *d_challenge, *d_target, *d_hash;
    unsigned int *d_found;
    uint64_t *d_nonce;
    cudaMalloc(&d_challenge, 32);
    cudaMalloc(&d_target, 32);
    cudaMalloc(&d_hash, 32);
    cudaMalloc(&d_found, sizeof(unsigned int));
    cudaMalloc(&d_nonce, sizeof(uint64_t));
    cudaMemcpy(d_challenge, challenge, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, 32, cudaMemcpyHostToDevice);
    cudaMemset(d_found, 0, sizeof(unsigned int));

    uint64_t threads = (uint64_t)grid_size * (uint64_t)block_size;
    uint64_t iters_per_thread = batches;
    uint64_t window_attempts = threads * iters_per_thread;
    uint64_t start_nonce = 0;
    uint64_t total_attempts = 0;
    double t0 = now_sec();
    unsigned int found = 0;
    uint64_t nonce = 0;
    uint8_t hash[32];

    while (!found) {
        cudaMemset(d_found, 0, sizeof(unsigned int));
        mine_kernel<<<grid_size, block_size>>>(d_challenge, d_target, start_nonce, threads, iters_per_thread, d_found, d_nonce, d_hash);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { fprintf(stderr, "kernel: %s\n", cudaGetErrorString(err)); return 4; }

        total_attempts += window_attempts;
        cudaMemcpy(&found, d_found, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (found) break;

        start_nonce += window_attempts;
        if (start_nonce < window_attempts) { fprintf(stderr, "nonce overflow\n"); return 5; }
    }

    double t1 = now_sec();
    cudaMemcpy(&nonce, d_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(hash, d_hash, 32, cudaMemcpyDeviceToHost);

    double elapsed = t1 - t0;
    double rate = elapsed > 0 ? (double)total_attempts / elapsed : 0.0;

    printf("status=FOUND\n");
    printf("nonce=%llu\n", (unsigned long long)nonce);
    printf("hash=");
    for (int i = 0; i < 32; i++) printf("%02x", hash[i]);
    printf("\n");
    printf("attempts=%llu\n", (unsigned long long)total_attempts);
    printf("elapsed=%.6f\n", elapsed);
    printf("rate_hs=%.0f\n", rate);

    cudaFree(d_challenge); cudaFree(d_target); cudaFree(d_hash); cudaFree(d_found); cudaFree(d_nonce);
    return 0;
}
