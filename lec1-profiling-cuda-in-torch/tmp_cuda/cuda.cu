#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void square_kernel(float *x, float * output, int H, int W) {
    int xid = blockIdx.x * blockDim.x + threadIdx.x;
    int yid = blockIdx.y * blockDim.y + threadIdx.y;
    if (xid < W && yid < H) {
        output[yid * W + xid] = x[yid * W + xid] * x[yid * W + xid];
    }
}

torch::Tensor square_matrix(torch::Tensor x) {
    const int height = x.size(0);
    const int width = x.size(1);
    auto result = torch::empty_like(x);
    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks((width + threads_per_block.x - 1) / threads_per_block.x,
                          (height + threads_per_block.y - 1) / threads_per_block.y);
    square_kernel<<<number_of_blocks, threads_per_block>>>(x.data_ptr<float>(), result.data_ptr<float>(), height, width);
    return result;
}
