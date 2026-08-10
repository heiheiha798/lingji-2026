#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

void LaunchInitialize(const std::uint32_t *input, std::uint32_t *height_a,
                      std::uint32_t *height_b, std::uint64_t *odometer,
                      std::size_t n, int rows, int cols, int *bounds,
                      bool find_bounds, cudaStream_t stream);

void LaunchSweep(const std::uint32_t *input, std::uint32_t *output,
                 std::uint64_t *odometer, int rows, int cols, int x_begin,
                 int y_begin, int x_end, int y_end, bool bounded, int *active,
                 cudaStream_t stream);

void LaunchStore(const std::uint32_t *height, std::uint8_t *stable,
                 std::size_t n, cudaStream_t stream);
