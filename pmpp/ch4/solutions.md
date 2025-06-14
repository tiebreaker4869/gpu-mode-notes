# Chapter 4

1. 
a. number of warp per block = 128 / 32 = 4
b. number of warp per grid = 4 * 8 = 32
c. 
i.
number of active warp = 3 * 8 = 24
ii. 
number of divergent warp per block = 2
number of divergent warp per grid = 2 * 8 = 16
iii.
SIMD efficiency = 100%
iv. 
SIMD efficiency = 1 - (63 - 40 + 1) / 32 = 1 - 24 / 32 = 1 - 0.75 = 25%
v.
SIMD efficiency = (127 - 104 + 1) / 32 = 24 / 32 = 75%
d.
i. number of active warps per grid = 32
ii. number of divergent warps per grid = 32
iii. SIMD efficiency = 50%
e.
i. 3
ii. 2

2.

2048 threads

3.

1 warp will be divergent

4.
(1 + 0.7 + 0.2 + 0.6 + 1.1 + 0.4 + 0.1) / 24 = 4.1 / 24 = 17.08%

5.
No. threads in a warp still can divergent, __syncwarp should be used for barierr synchronization.

6.
1536 threads = 48 warps

a. number of threads = 512
b. number of threads = 1024
c. number of threads = 1536
d. number of threads = 1024

we can see that increasing threads regardlessly will not always improve resource utilization. 

7.
maximum warps supported = 2048 / 32 = 64
a. possible. occupancy = 32 / 64 = 50%
b. possible. occupancy = 32 / 64 = 50%
c. possible. occupancy = 32 / 64 = 50%
d. possible. occupancy = 64 / 64 = 100%
e. possible. occupancy = 64 / 64 = 100%

8.
a. 
Yes.
b. 
No. Limiting factor is the number of blocks.
c. 
No. Limiting factor is the number of registers.

9.

32 * 32 = 1024, but the device only allows 512 threads per block.