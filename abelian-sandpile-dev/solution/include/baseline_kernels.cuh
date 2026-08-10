#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

void LaunchInitialize(const std::uint32_t *input, std::uint64_t *height,
                      std::uint64_t *odometer, std::size_t n,
                      cudaStream_t stream);

void LaunchSweep(const std::uint64_t *input, std::uint64_t *output,
                 std::uint64_t *odometer, int rows, int cols, int *active,
                 cudaStream_t stream);

void LaunchStore(const std::uint64_t *height, std::uint8_t *stable,
                 std::size_t n, cudaStream_t stream);
