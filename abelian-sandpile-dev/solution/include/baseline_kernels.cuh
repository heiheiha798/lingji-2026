#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

void LaunchInitialize(const std::uint32_t *input, void *height_a,
                      void *height_b, std::uint64_t *odometer, std::size_t n,
                      int rows, int cols, int *bounds, bool find_bounds,
                      int height_width, cudaStream_t stream);

void LaunchSweep(const void *input, void *output,
                 std::uint64_t *odometer, int rows, int cols, int x_begin,
                 int y_begin, int x_end, int y_end, bool bounded, int *active,
                 int height_width, cudaStream_t stream);

void LaunchStore(const void *height, std::uint8_t *stable, std::size_t n,
                 int height_width, cudaStream_t stream);
