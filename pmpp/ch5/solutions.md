# Chaper 5

1.
No. Because every element is used only once, so shared memory will not enable more data reuse.

3.

read-after-write write-after-write dependency will not be satisfied.

4.
some variables, such as local array will be stored in the local memory by default, which also have long latencies.

5.
32

6.
512000

7.
1000

8.
a. no tiling, N
b. with tiling, N / T

9.
AI = 36 / 28 = 1.2857
a. peak FLOPS by AI * peak bandwidth =  138 GFLOPS < peak flops, memory bound
b. peak FLOPS by AI * peak bandwidth = 345 GFLOPS > peak flops, compute bound

10.
a. when BLOCK_SIZE = BLOCK_WIDTH
b. __syncthreads should be used to do cooperative fetching.

11.
a. 1024
b. 1024
c. 8
d. 8
e. 129 * 4 = 516
f. 10 / (6 * 4) = 5 / 12 OP/B

12.
a.
limit factor is shared memory.

b.
limiting factor is register.