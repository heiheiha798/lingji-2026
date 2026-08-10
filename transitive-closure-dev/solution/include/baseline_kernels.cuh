#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

void LaunchInitialize(std::uint64_t *reachability, int vertices, int words,
                      cudaStream_t stream);

void LaunchPivotBlock(std::uint64_t *reachability,
                      std::uint64_t *pivot_rows,
                      std::uint64_t *pivot_masks, int vertices, int words,
                      int block_start, cudaStream_t stream);
