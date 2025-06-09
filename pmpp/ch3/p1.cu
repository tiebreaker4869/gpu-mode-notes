#include "error.cuh"
#include <cstdlib>

__global__ void mm_by_row_kernel(float* a_d, float* b_d, float* out_d, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        for (int i = 0; i < N; i ++) {
            // out[row][i] = sum(a[row][k] * b[k][i])
            float val = 0;
            for (int k = 0; k < N; k ++) {
                val += a_d[row * N + k] * b_d[k * N + i];
            }
            out_d[row * N + i] = val;
        }
    }
}

__global__ void mm_by_col_kernel(float* a_d, float* b_d, float* out_d, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        for (int i = 0; i < N; i ++) {
            // out[i][col] = sum(a[i][k] * b[k][col])
            float val = 0;
            for (int k = 0; k < N; k ++) {
                val += a_d[i * N + k] * b_d[k * N + col];
            } 
            out_d[i * N + col] = val;
        }
    }
}


void mm_by_row(float* a_h, float* b_h, float* out_h, int N) {
    float* a_d, *b_d, *out_d;
    CHECK(cudaMalloc((void**)&a_d, N * N * sizeof(float)))
    CHECK(cudaMalloc((void**)&b_d, N * N * sizeof(float)))
    CHECK(cudaMalloc((void**)&out_d, N * N * sizeof(float)))
    CHECK(cudaMemcpy(a_d, a_h, N * N * sizeof(float), cudaMemcpyHostToDevice))
    CHECK(cudaMemcpy(b_d, b_h, N * N * sizeof(float), cudaMemcpyHostToDevice))
    mm_by_row_kernel<<<ceil(N / 128), 128>>>(a_d, b_d, out_d, N);
    cudaDeviceSynchronize();
    CHECK(cudaMemcpy(out_h, out_d, N * N * sizeof(float), cudaMemcpyDeviceToHost))
    CHECK(cudaFree(a_d))
    CHECK(cudaFree(b_d))
    CHECK(cudaFree(out_d))
}

void mm_by_col(float* a_h, float* b_h, float* out_h, int N) {
    float* a_d, *b_d, *out_d;
    CHECK(cudaMalloc((void**)&a_d, N * N * sizeof(float)))
    CHECK(cudaMalloc((void**)&b_d, N * N * sizeof(float)))
    CHECK(cudaMalloc((void**)&out_d, N * N * sizeof(float)))
    CHECK(cudaMemcpy(a_d, a_h, N * N * sizeof(float), cudaMemcpyHostToDevice))
    CHECK(cudaMemcpy(b_d, b_h, N * N * sizeof(float), cudaMemcpyHostToDevice))
    mm_by_col_kernel<<<ceil(N / 128), 128>>>(a_d, b_d, out_d, N);
    cudaDeviceSynchronize();
    CHECK(cudaMemcpy(out_h, out_d, N * N * sizeof(float), cudaMemcpyDeviceToHost))
    CHECK(cudaFree(a_d))
    CHECK(cudaFree(b_d))
    CHECK(cudaFree(out_d))
}

// in row major store, both kernel read 2 matrix with one consecutive read and another one non-consecutive
// but in by row mm the output is write in consecutive order.

int main() {

    const int N = 10000;

    float* a, *b, *out;

    a = (float*) malloc(N * N * sizeof(float));
    b = (float*) malloc(N * N * sizeof(float));
    out = (float*) malloc(N * N * sizeof(float));
    
    mm_by_row(a, b, out, N);

    mm_by_col(a, b, out, N);

    free(a);

    free(b);

    free(out);

    return 0;
}