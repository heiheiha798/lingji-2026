#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

std::size_t ReferenceWorkspaceSize(int rows, int cols);

void ReferenceSolve(const std::uint32_t *d_initial, std::uint8_t *d_stable,
                    std::uint64_t *d_odometer, int rows, int cols,
                    void *d_workspace, cudaStream_t stream,
                    std::uint64_t *iterations);

