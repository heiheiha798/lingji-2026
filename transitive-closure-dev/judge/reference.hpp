#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

void ReferenceSolve(const std::uint64_t *d_adjacency,
                    std::uint64_t *d_reachability, int vertices,
                    int words_per_row, cudaStream_t stream);
