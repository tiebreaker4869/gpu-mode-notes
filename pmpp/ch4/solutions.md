# Chapter 4

1. 
a. number of warp per block = 128 / 32 = 4
b. number of warp per grid = 4 * 8 = 32
c. 
i.
number of active threads per block = 39 - 0 + 1 + 127 - 104 + 1 = 40 + 24 = 64
number of active threads per grid = 64 * 8 = 512
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