#pragma once
#include <cstdio>

#define CHECK(x) \
do {    \
    cudaError_t error = x; \
    if (error != cudaSuccess) { \
        printf("%s in %s at line %d.\n", cudaGetErrorString(error), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0); \ 