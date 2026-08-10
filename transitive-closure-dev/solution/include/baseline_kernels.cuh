#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

void LaunchInitialize(const std::uint64_t *adjacency,
                      std::uint64_t *reachability, int vertices, int words,
                      cudaStream_t stream);

void LaunchPivot(std::uint64_t *reachability, int vertices, int words,
                 int pivot, cudaStream_t stream);
