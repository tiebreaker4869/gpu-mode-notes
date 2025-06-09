#include "error.cuh"
#include <cstdlib>

__global__ void mv_kernel(float* out_d, float* a_d, float* v_d, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = 0;
        for (int i = 0; i < N; i ++) {
            val += a_d[idx * N + i] * v_d[i];
        }
        out_d[idx] = val;
    }
}

void mv(float* out_h, float* a_h, float* v_h, int N) {
    float* a_d, *v_d, *out_d;
    CHECK(cudaMalloc((void**)&a_d, N * N * sizeof(float)))
    CHECK(cudaMalloc((void**)&v_d, N * sizeof(float)))
    CHECK(cudaMalloc((void**)&out_d, N * sizeof(float)))
    CHECK(cudaMemcpy(a_d, a_h, N * N * sizeof(float), cudaMemcpyHostToDevice))
    CHECK(cudaMemcpy(v_d, v_h, N * sizeof(float), cudaMemcpyHostToDevice))
    mv_kernel<<<ceil(N / 128), 128>>>(out_d, a_d, v_d, N);
    CHECK(cudaDeviceSynchronize())
    CHECK(cudaMemcpy(out_h, out_d, N * sizeof(float), cudaMemcpyDeviceToHost))
    CHECK(cudaFree(a_d))
    CHECK(cudaFree(v_d))
    CHECK(cudaFree(out_d))
}

int main() {

    float* a, *v, *out;

    const int N = 10000;

    a = (float*) malloc(N * N * sizeof(float));
    v = (float*) malloc(N * sizeof(float));
    out = (float*) malloc(N * sizeof(float));

    mv(out, a, v, N);

    free(a);

    free(v);

    free(out);

    return 0;
}